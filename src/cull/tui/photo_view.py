"""Kitty graphics protocol photo renderer with direct main-thread emission.

Approach:
1. ``display_photo`` runs on the main thread. It stores the request and
   dispatches a background worker (``run_worker(thread=True, exclusive=True)``)
   to do the heavy PNG encode + APC sequence build off the event loop.
2. When the worker finishes, it marshals the upload sequence back to the
   main thread via ``call_from_thread(_main_apply_upload, ...)``. All stdout
   writes happen on the main thread so APC sequences cannot race with
   Textual's own paint-cycle output.
3. ``_main_apply_upload`` writes the upload directly, then calls
   ``_maybe_emit_put`` inline — no refresh dance, no ``call_after_refresh``.
   We are guaranteed to be on the main thread, Textual is not mid-paint,
   and the put lands deterministically.
4. ``_maybe_emit_put`` moves the cursor (DEC save + absolute position) to
   the widget's top-left terminal cell, emits a nuclear ``a=d,d=a`` clear
   to remove any stale placements, then emits an ``a=p`` put referencing the
   freshly-uploaded image id.
5. ``render()`` is passive — returns the blank spacer text and never
   schedules any image emission. Emissions fire only from
   ``display_photo`` / ``on_mount`` / ``on_resize`` via the worker hand-off.

Requires a Kitty graphics protocol terminal (Kitty, Ghostty, wezterm).
"""

from __future__ import annotations

import base64
import itertools
import logging
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from rich.text import Text
from textual.widget import Widget

from cull.config import (
    OVERLAY_CROP_COLOR,
    OVERLAY_HORIZON_COLOR,
    OVERLAY_LABEL_OFFSET_PX,
    OVERLAY_LINE_THICKNESS,
)
from cull.models import CropProposal, GeometryScore

logger = logging.getLogger(__name__)

CACHE_MAX_ENTRIES: int = 6
PRECACHE_AHEAD: int = 2
PRECACHE_BEHIND: int = 1
PNG_TARGET_LONG_EDGE: int = 1024
PNG_COMPRESS_LEVEL: int = 1
KITTY_CHUNK_SIZE: int = 4096

DEBUG_LOG_PATH: Path = Path.home() / ".cache" / "cull_photoview_debug.log"
DEBUG_LOG_ENABLED: bool = bool(os.environ.get("CULL_PHOTOVIEW_DEBUG"))


def _dlog(msg: str) -> None:
    """Append a timestamped line to the debug log when enabled."""
    if not DEBUG_LOG_ENABLED:
        return
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.time():.3f} {msg}\n")
    except OSError:
        pass


class ViewportSize(BaseModel):
    """Terminal viewport dimensions in columns and rows."""

    cols: int
    rows: int


class _PngCache(OrderedDict[str, bytes]):
    """LRU cache of PNG-encoded image bytes keyed by RenderRequest.image_id."""

    def __init__(self, maxsize: int = CACHE_MAX_ENTRIES) -> None:
        super().__init__()
        self._maxsize = maxsize

    def put(self, key: str, value: bytes) -> None:
        """Insert or update a cache entry, evicting the oldest when full."""
        self[key] = value
        self.move_to_end(key)
        while len(self) > self._maxsize:
            self.popitem(last=False)


_png_cache = _PngCache()
_image_id_counter = itertools.count(1)
_STDOUT_WRITE_LOCK = threading.Lock()


def _next_image_id() -> int:
    """Return the next unique Kitty image ID for this session."""
    return next(_image_id_counter)


class _KittyUploadInput(BaseModel):
    """Bundle for a Kitty store-only upload APC sequence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    png_bytes: bytes
    image_id: int


def _build_upload_sequence(upload_in: _KittyUploadInput) -> str:
    """Build chunked APC sequences that upload a PNG to the image store only."""
    b64 = base64.b64encode(upload_in.png_bytes).decode("ascii")
    parts: list[str] = []
    remaining = b64
    first = True
    while remaining:
        chunk = remaining[:KITTY_CHUNK_SIZE]
        remaining = remaining[KITTY_CHUNK_SIZE:]
        more = 1 if remaining else 0
        if first:
            params = f"a=T,i={upload_in.image_id},f=100,q=2,m={more}"
            parts.append(f"\x1b_G{params};{chunk}\x1b\\")
            first = False
        else:
            parts.append(f"\x1b_Gm={more};{chunk}\x1b\\")
    return "".join(parts)


class _KittyPutInput(BaseModel):
    """Bundle for a Kitty put (display) APC sequence at an absolute position."""

    image_id: int
    col: int
    row: int
    cols: int
    rows: int


def _build_put_sequence(put_in: _KittyPutInput) -> str:
    """Build: save cursor, position absolute, put image, restore cursor.

    Uses a fixed placement id (p=1) so repeated puts for the same image id
    update the single placement in-place rather than stacking new placements.
    """
    save_cursor = "\x1b7"
    restore_cursor = "\x1b8"
    move_cursor = f"\x1b[{put_in.row + 1};{put_in.col + 1}H"
    put_apc = (
        f"\x1b_Ga=p,i={put_in.image_id},p=1,"
        f"c={put_in.cols},r={put_in.rows},q=2\x1b\\"
    )
    return f"{save_cursor}{move_cursor}{put_apc}{restore_cursor}"


def _write_raw(data: str) -> None:
    """Write a raw control/APC string directly to the real stdout under a lock.

    The lock serialises concurrent writes from background workers so APC
    sequences are never interleaved mid-chunk, which would corrupt them.
    """
    with _STDOUT_WRITE_LOCK:
        try:
            sys.__stdout__.write(data)
            sys.__stdout__.flush()
        except OSError as exc:
            _dlog(f"stdout write failed: {exc}")
            logger.warning("Kitty direct-write failed: %s", exc)


def _emit_kitty_upload(upload_in: _KittyUploadInput) -> None:
    """Emit a store-only upload APC sequence to the terminal."""
    seq = _build_upload_sequence(upload_in)
    _dlog(f"upload id={upload_in.image_id} png={len(upload_in.png_bytes)}B apc={len(seq)}B")
    _write_raw(seq)


def _emit_kitty_put(put_in: _KittyPutInput) -> None:
    """Emit a put (display) APC sequence at an absolute terminal cell position."""
    seq = _build_put_sequence(put_in)
    _dlog(
        f"put id={put_in.image_id} at ({put_in.col},{put_in.row}) "
        f"cells=({put_in.cols},{put_in.rows}) apc={len(seq)}B"
    )
    _write_raw(seq)


def _emit_kitty_delete(image_id: int) -> None:
    """Delete a previously uploaded Kitty image AND all its screen placements.

    Uses d=I (uppercase) so both stored image data AND currently-visible
    placements are removed; d=i (lowercase) only frees storage and leaves
    stale placements on screen.
    """
    _dlog(f"delete id={image_id}")
    _write_raw(f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\")


def _emit_kitty_clear_all() -> None:
    """Delete all visible Kitty placements regardless of image id."""
    _dlog("delete all placements")
    _write_raw("\x1b_Ga=d,d=a,q=2\x1b\\")


def _pil_from_bytes(image_bytes: bytes) -> object:
    """Open and reduce a PIL image via JPEG draft + thumbnail for fast PNG encode."""
    from PIL import Image  # noqa: PLC0415

    img = Image.open(BytesIO(image_bytes))
    target = (PNG_TARGET_LONG_EDGE, PNG_TARGET_LONG_EDGE)
    img.draft("RGB", target)
    img = img.convert("RGB")
    img.thumbnail(target, Image.LANCZOS)
    return img


def _pil_to_png_bytes(pil_img: object) -> bytes:
    """Encode a PIL image as PNG with fast compression and return the bytes."""
    buf = BytesIO()
    pil_img.save(  # type: ignore[attr-defined]
        buf, format="PNG", compress_level=PNG_COMPRESS_LEVEL, optimize=False,
    )
    return buf.getvalue()


class RenderRequest(BaseModel):
    """Input bundle for cached image rendering."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_id: str
    image_bytes: bytes
    viewport: ViewportSize
    geometry: GeometryScore | None = None
    crop: CropProposal | None = None


def _draw_horizon(img: object, geometry: GeometryScore) -> None:
    """Draw a horizon line and tilt label onto a PIL image in place."""
    from PIL import ImageDraw  # noqa: PLC0415

    if not geometry.has_horizon:
        return
    draw = ImageDraw.Draw(img)  # type: ignore[arg-type]
    w, h = img.size  # type: ignore[attr-defined]
    cx, cy = w // 2, h // 2
    rad = math.radians(geometry.tilt_degrees)
    dx = int(cx * math.cos(rad))
    dy = int(cx * math.sin(rad))
    x0, y0 = cx - dx, cy - dy
    x1, y1 = cx + dx, cy + dy
    draw.line([(x0, y0), (x1, y1)], fill=OVERLAY_HORIZON_COLOR, width=OVERLAY_LINE_THICKNESS)
    label = f"{geometry.tilt_degrees:+.1f}\u00b0"
    draw.text((x0 + OVERLAY_LABEL_OFFSET_PX, y0 - OVERLAY_LABEL_OFFSET_PX - 10), label, fill=OVERLAY_HORIZON_COLOR)


def _draw_crop_box(img: object, crop: CropProposal) -> None:
    """Draw a crop bounding box onto a PIL image in place."""
    from PIL import ImageDraw  # noqa: PLC0415

    draw = ImageDraw.Draw(img)  # type: ignore[arg-type]
    coords = [(crop.left, crop.top), (crop.right, crop.bottom)]
    draw.rectangle(coords, outline=OVERLAY_CROP_COLOR, width=OVERLAY_LINE_THICKNESS)


def _apply_overlays(img: object, request: RenderRequest) -> None:
    """Apply horizon and crop overlays onto the PIL image in place."""
    if request.geometry is not None:
        _draw_horizon(img, request.geometry)
    if request.crop is not None:
        _draw_crop_box(img, request.crop)


def _prepare_png_for_request(request: RenderRequest) -> bytes:
    """Decode, overlay, PNG-encode; return fresh PNG bytes."""
    pil_img = _pil_from_bytes(request.image_bytes)
    _apply_overlays(pil_img, request)
    return _pil_to_png_bytes(pil_img)


def _has_overlays(request: RenderRequest) -> bool:
    """Return True if the request carries any overlay data."""
    return request.geometry is not None or request.crop is not None


def _get_png_bytes(request: RenderRequest) -> bytes:
    """Return PNG bytes for a request; hit LRU cache when no overlays set."""
    if _has_overlays(request):
        return _prepare_png_for_request(request)
    key = request.image_id
    if key in _png_cache:
        return _png_cache[key]
    png = _prepare_png_for_request(request)
    _png_cache.put(key, png)
    return png


def render_cached(request: RenderRequest) -> bytes:
    """Return PNG bytes ready for Kitty upload; cached when possible."""
    return _get_png_bytes(request)


class PrecacheRequest(BaseModel):
    """Request to pre-cache PNG bytes around the current index."""

    paths: list[Path]
    current_index: int


def _precache_range(request: PrecacheRequest) -> list[int]:
    """Return indices to pre-cache around the current position."""
    total = len(request.paths)
    start = max(0, request.current_index - PRECACHE_BEHIND)
    end = min(total, request.current_index + PRECACHE_AHEAD + 1)
    return list(range(start, end))


def precache_images(request: PrecacheRequest, viewport: ViewportSize) -> None:
    """Pre-decode nearby photos into the PNG LRU cache for snappier navigation."""
    for idx in _precache_range(request):
        path = request.paths[idx]
        key = str(path)
        if key in _png_cache:
            continue
        try:
            data = path.read_bytes()
            req = RenderRequest(image_id=key, image_bytes=data, viewport=viewport)
            _get_png_bytes(req)
        except OSError:
            logger.warning("Failed to pre-cache %s", path)


class PhotoView(Widget):
    """Textual widget displaying a photo via Kitty direct-write and re-emit."""

    DEFAULT_CSS = """
    PhotoView {
        height: 1fr;
        width: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_request: RenderRequest | None = None
        self._kitty_id: int = 0
        self._blank_text: Text = Text("")
        self._last_put_state: tuple[int, int, int, int, int] | None = None

    def clear_terminal_image(self) -> None:
        """Remove any visible Kitty image placement and stored image state."""
        _emit_kitty_clear_all()
        if self._kitty_id != 0:
            _emit_kitty_delete(self._kitty_id)
        self._kitty_id = 0
        self._last_put_state = None
        self.refresh()

    def redisplay_current(self) -> None:
        """Re-upload and display the current request after an overlay closes."""
        if self._current_request is None:
            return
        request = self._current_request
        self._last_put_state = None
        self.refresh()
        self.run_worker(
            lambda: self._worker_prepare_photo(request),
            thread=True,
            exclusive=True,
            group="photo_emit",
        )

    def display_photo(self, request: RenderRequest) -> None:
        """Store the request and dispatch heavy encode+upload to a background worker."""
        _dlog(f"display_photo image_id={request.image_id}")
        _emit_kitty_clear_all()
        if self._kitty_id != 0:
            _emit_kitty_delete(self._kitty_id)
            self._kitty_id = 0
        self._current_request = request
        self._last_put_state = None
        self._rebuild_blank_text()
        self.refresh()
        self.run_worker(
            lambda: self._worker_prepare_photo(request),
            thread=True,
            exclusive=True,
            group="photo_emit",
        )

    def _worker_prepare_photo(self, request: RenderRequest) -> None:
        """Background worker: encode PNG and build the APC upload sequence ONLY.

        All stdout writes are marshalled back to the main thread via
        ``call_from_thread`` so that our APC sequences cannot race with
        Textual's own paint-cycle writes, which would corrupt the escape
        sequences and cause the base64 payload to render as raw text.
        """
        if self._current_request is not request:
            return
        try:
            png = _get_png_bytes(request)
        except OSError as exc:
            _dlog(f"worker: PNG encode failed for {request.image_id}: {exc}")
            return
        if self._current_request is not request:
            return
        image_id = _next_image_id()
        upload_seq = _build_upload_sequence(
            _KittyUploadInput(png_bytes=png, image_id=image_id)
        )
        _dlog(f"worker: built upload image_id={image_id} apc={len(upload_seq)}B")
        self.app.call_from_thread(
            self._main_apply_upload, request, image_id, upload_seq,
        )

    def _main_apply_upload(
        self, request: RenderRequest, image_id: int, upload_seq: str
    ) -> None:
        """Main thread: flush the APC upload and emit the put directly.

        We are guaranteed to be on the main thread here (via call_from_thread),
        which means Textual is not mid-paint. Emitting the put inline avoids
        racing with Textual's repaint cycle and the fragile
        refresh/call_after_refresh dance that was dropping emissions under
        rapid navigation.
        """
        if self._current_request is not request:
            _dlog(f"main_apply_upload: stale request for image_id={image_id}")
            return
        if self._kitty_id != 0:
            _emit_kitty_delete(self._kitty_id)
        _write_raw(upload_seq)
        self._kitty_id = image_id
        self._last_put_state = None
        self._maybe_emit_put()
        # Some terminals miss the immediate put when Textual repaints the same
        # cycle; queue one more emit after refresh for reliability.
        self.call_after_refresh(self._maybe_emit_put)

    def _rebuild_blank_text(self) -> None:
        """Refresh the blank-spacer Text to match the widget's current size."""
        rows = max(self.size.height, 1)
        cols = max(self.size.width, 1)
        self._blank_text = Text("\n".join(" " * cols for _ in range(rows)))

    def on_mount(self) -> None:
        """Emit the first Kitty put directly once the widget has real dimensions."""
        _dlog(f"on_mount size=({self.size.width},{self.size.height})")
        self._rebuild_blank_text()
        self._maybe_emit_put()

    def on_resize(self, event: object) -> None:  # noqa: ARG002
        """Re-emit the put directly at the new cell dimensions after a resize."""
        _dlog(f"on_resize size=({self.size.width},{self.size.height})")
        self._rebuild_blank_text()
        self._last_put_state = None  # force re-emit even if kitty_id unchanged
        self._maybe_emit_put()

    def render(self) -> Text:
        """Return blank text claiming the widget region. No Kitty emit per-frame."""
        return self._blank_text

    def _maybe_emit_put(self) -> None:
        """Emit a Kitty put only when image, position, or size has actually changed.

        Called from display_photo / on_mount / on_resize. NOT called from render(),
        which would cause per-frame clear+put cycles that flicker the footer and
        other widgets. Textual's virtual buffer holds blank cells for the widget
        region after the first paint, so subsequent refreshes diff empty and the
        image survives without re-emission.
        """
        if self._kitty_id == 0 or self._current_request is None:
            return
        cols = max(self.size.width, 1)
        rows = max(self.size.height, 1)
        if cols <= 1 or rows <= 1:
            return
        region = self.region
        state = (self._kitty_id, region.x, region.y, cols, rows)
        if state == self._last_put_state:
            return
        # State changed — clear every visible placement then put at the new region.
        _write_raw("\x1b_Ga=d,d=a,q=2\x1b\\")
        _dlog(f"nuclear clear + put for state {state}")
        _emit_kitty_put(_KittyPutInput(
            image_id=self._kitty_id,
            col=region.x,
            row=region.y,
            cols=cols,
            rows=rows,
        ))
        self._last_put_state = state
