"""EXIF-oriented, viewport-sized PNG previews cached on disk, rendered by a worker pool."""

from __future__ import annotations

import hashlib
import logging
import os
import struct
import threading
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

PREVIEW_CACHE_DIR: Path = Path.home() / ".cache" / "cull" / "previews"
# One 24 MP frame previewed at ~2500 px long edge is ~6-8 MB of fast-compressed
# PNG, so 3 GiB holds a full ~500-photo shoot at one viewport size.
PREVIEW_CACHE_MAX_BYTES: int = 3 * 1024**3
PREVIEW_BOX_STEP_PX: int = 64
PNG_COMPRESS_LEVEL: int = 1
PNG_HEADER_BYTES: int = 24
EXIF_ORIENTATION_TAG: int = 0x0112
# EXIF orientations 5-8 store the frame rotated by 90 degrees.
SWAPPED_ORIENTATIONS: frozenset[int] = frozenset({5, 6, 7, 8})
MIN_WORKERS: int = 2
MAX_WORKERS: int = 4


class PixelBox(BaseModel):
    """A target rectangle in pixels."""

    model_config = ConfigDict(frozen=True)

    width: int
    height: int


class PreviewSpec(BaseModel):
    """What to render: a source photo fitted inside a pixel box."""

    model_config = ConfigDict(frozen=True)

    source: Path
    box: PixelBox


class Preview(BaseModel):
    """A rendered preview PNG on disk and its pixel size."""

    model_config = ConfigDict(frozen=True)

    path: Path
    width: int
    height: int


class PreviewJob(BaseModel):
    """A queued render; urgent jobs may use the reserved worker."""

    model_config = ConfigDict(frozen=True)

    spec: PreviewSpec
    is_urgent: bool = False


def _round_up(value: float, step: int) -> int:
    """Round ``value`` up to a multiple of ``step`` (at least one step)."""
    return max(step, -(-int(value) // step) * step)


def make_spec(source: Path, box: PixelBox) -> PreviewSpec:
    """Build a spec whose box is bucketed so small resizes reuse the same cache entry."""
    bucketed = PixelBox(
        width=_round_up(box.width, PREVIEW_BOX_STEP_PX),
        height=_round_up(box.height, PREVIEW_BOX_STEP_PX),
    )
    return PreviewSpec(source=source, box=bucketed)


def cache_path(spec: PreviewSpec, stat: os.stat_result) -> Path:
    """Return the cache file for a spec, keyed by path, mtime, size and box."""
    key = f"{spec.source}|{stat.st_mtime_ns}|{stat.st_size}|{spec.box.width}x{spec.box.height}"
    return PREVIEW_CACHE_DIR / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.png"


def read_png_size(path: Path) -> tuple[int, int]:
    """Return (width, height) from a PNG's IHDR without decoding it."""
    with path.open("rb") as handle:
        header = handle.read(PNG_HEADER_BYTES)
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def _fit_pixels(size: tuple[int, int], box: PixelBox) -> tuple[int, int]:
    """Return ``size`` scaled down (never up) to fit ``box``, aspect preserved."""
    width, height = size
    scale = min(box.width / width, box.height / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _oriented_size(image: Image.Image) -> tuple[int, int]:
    """Return the image size after its EXIF orientation is applied."""
    orientation = image.getexif().get(EXIF_ORIENTATION_TAG, 1)
    width, height = image.size
    return (height, width) if orientation in SWAPPED_ORIENTATIONS else (width, height)


def decode_fitted(spec: PreviewSpec) -> Image.Image:
    """Decode a photo upright (EXIF transpose) and scaled to fit the spec box.

    JPEG draft mode decodes at the smallest DCT scale that still covers the
    target, which is what makes 24 MP frames cheap to preview.
    """
    with Image.open(spec.source) as image:
        oriented = _oriented_size(image)
        target = _fit_pixels(oriented, spec.box)
        is_swapped = oriented != image.size
        image.draft("RGB", (target[1], target[0]) if is_swapped else target)
        upright = ImageOps.exif_transpose(image)
    upright = upright.convert("RGB")
    if upright.size == target:
        return upright
    return upright.resize(target, Image.Resampling.BICUBIC, reducing_gap=2.0)


def _write_png_atomic(image: Image.Image, target: Path) -> None:
    """Write a PNG via a temp file so the terminal never reads a half-written file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
    image.save(temp, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    os.replace(temp, target)


def render_preview(spec: PreviewSpec) -> Preview:
    """Return the cached preview for a spec, rendering it first when missing."""
    target = cache_path(spec, spec.source.stat())
    if target.exists():
        os.utime(target)
        width, height = read_png_size(target)
        return Preview(path=target, width=width, height=height)
    image = decode_fitted(spec)
    _write_png_atomic(image, target)
    return Preview(path=target, width=image.width, height=image.height)


def prune_cache(max_bytes: int) -> int:
    """Delete least-recently-used previews until the cache fits; return files removed."""
    if not PREVIEW_CACHE_DIR.exists():
        return 0
    entries = sorted(
        (entry.stat().st_mtime, entry.stat().st_size, entry)
        for entry in PREVIEW_CACHE_DIR.glob("*.png")
    )
    total = sum(size for _, size, _ in entries)
    removed = 0
    for _, size, entry in entries:
        if total <= max_bytes:
            break
        entry.unlink(missing_ok=True)
        total -= size
        removed += 1
    return removed


def default_worker_count() -> int:
    """Return a worker count that leaves CPU for the UI and the terminal."""
    return max(MIN_WORKERS, min(MAX_WORKERS, (os.cpu_count() or 4) // 2))


PreviewDone = Callable[[PreviewSpec, "Preview | None"], None]


class PreviewScheduler:
    """Render previews on worker threads, always taking the nearest pending job first.

    Worker 0 only takes urgent jobs (the current photo and its neighbours),
    so a long background warm-up never delays the photo on screen by more
    than one in-flight render.
    """

    def __init__(self, on_done: PreviewDone) -> None:
        self._on_done = on_done
        self._condition = threading.Condition()
        self._pending: list[PreviewJob] = []
        self._claimed: set[PreviewSpec] = set()
        self._is_stopped = False

    def start(self, worker_count: int) -> None:
        """Start the worker threads."""
        for index in range(worker_count):
            thread = threading.Thread(
                target=self._worker_loop, args=(index == 0,), name=f"cull-preview-{index}", daemon=True,
            )
            thread.start()

    def prioritise(self, jobs: list[PreviewJob]) -> None:
        """Replace the pending queue with ``jobs`` (nearest first); skip claimed specs."""
        with self._condition:
            self._pending = [job for job in jobs if job.spec not in self._claimed]
            self._condition.notify_all()

    def stop(self) -> None:
        """Stop the workers after their current job."""
        with self._condition:
            self._is_stopped = True
            self._pending = []
            self._condition.notify_all()

    def _take(self, is_urgent_only: bool) -> PreviewJob | None:
        """Pop the first job this worker may run, or None if there is none yet."""
        for index, job in enumerate(self._pending):
            if job.is_urgent or not is_urgent_only:
                del self._pending[index]
                self._claimed.add(job.spec)
                return job
        return None

    def _wait_for_job(self, is_urgent_only: bool) -> PreviewJob | None:
        """Block until a job is available; return None once stopped."""
        with self._condition:
            while not self._is_stopped:
                job = self._take(is_urgent_only)
                if job is not None:
                    return job
                self._condition.wait()
        return None

    def _worker_loop(self, is_urgent_only: bool) -> None:
        """Render jobs until stopped; a failed render reports None and moves on."""
        while (job := self._wait_for_job(is_urgent_only)) is not None:
            try:
                result: Preview | None = render_preview(job.spec)
            except Exception as exc:  # noqa: BLE001 - a dead worker would stall every preview
                logger.warning("preview render failed for %s: %s", job.spec.source, exc)
                result = None
            self._on_done(job.spec, result)
