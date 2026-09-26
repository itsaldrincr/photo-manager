"""Terminal image service and the ImageCells widget that paints placeholder cells.

The service owns every kitty image this app uploads. All terminal writes go
through ``terminal_write`` on the UI thread, which hands them to Textual's
own driver writer, so our escape sequences queue behind (never inside)
Textual's frame output. Widgets never write escapes: they ask for an image,
keep painting the old one until the new one is uploaded, then swap cells.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections import Counter, OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from rich.segment import Segment
from rich.style import Style
from textual.app import App
from textual.strip import Strip
from textual.timer import Timer
from textual.widget import Widget

from cull.tui import kitty
from cull.tui.kitty import CellBox, DirectTransmit, FitRequest, PlaceholderRow
from cull.tui.previews import (
    PixelBox,
    Preview,
    PreviewJob,
    PreviewScheduler,
    PreviewSpec,
    default_worker_count,
    make_spec,
)

logger = logging.getLogger(__name__)

MAX_UPLOADED_IMAGES: int = 40
RESIZE_DEBOUNCE_SECONDS: float = 0.08


def terminal_write(app: App, data: str) -> None:
    """Queue raw escape data on Textual's driver writer (a no-op when headless)."""
    driver = app._driver
    if driver is not None:
        driver.write(data)


class ImageRequest(BaseModel):
    """A source photo to show inside a cell box."""

    model_config = ConfigDict(frozen=True)

    source: Path
    box: CellBox


class ShownImage(BaseModel):
    """An uploaded image and the aspect-fitted cell box its placement covers."""

    model_config = ConfigDict(frozen=True)

    image_id: int
    box: CellBox


ReadyCallback = Callable[[ShownImage], None]


class ImageTicket(BaseModel):
    """One widget's request; a newer ticket from the same owner replaces it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    owner: int
    request: ImageRequest
    on_ready: ReadyCallback


class PrefetchPlan(BaseModel):
    """What to render next, nearest first, and what to upload before it is needed."""

    urgent: list[ImageRequest] = Field(default_factory=list)
    upload_ahead: list[ImageRequest] = Field(default_factory=list)
    background: list[ImageRequest] = Field(default_factory=list)


class _UploadKey(BaseModel):
    """An uploaded preview at one fitted cell size."""

    model_config = ConfigDict(frozen=True)

    path: Path
    box: CellBox


class _Upload(BaseModel):
    """Upload state; direct (``t=d``) uploads are prepared off-thread first."""

    image_id: int
    is_ready: bool = False


class _DirectReady(BaseModel):
    """A prepared inline transmit, handed back to the UI thread."""

    key: _UploadKey
    sequence: str


def _read_direct_transmit(key: _UploadKey, image_id: int) -> _DirectReady:
    """Worker: read a preview PNG and build its inline transmit escape."""
    transmit = DirectTransmit(image_id=image_id, png_bytes=key.path.read_bytes())
    return _DirectReady(key=key, sequence=kitty.transmit_direct_sequence(transmit))


class ImageService:
    """Owns preview rendering, kitty uploads, and image lifetime for one app."""

    def __init__(self, app: App) -> None:
        self._app = app
        self._cell = kitty.query_cell_size()
        self._is_file_transmit = kitty.prefers_file_transmit()
        self._scheduler = PreviewScheduler(self._on_preview_done_threadsafe)
        self._direct_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cull-upload")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._previews: dict[PreviewSpec, Preview] = {}
        self._tickets: dict[int, ImageTicket] = {}
        self._ahead: dict[PreviewSpec, CellBox] = {}
        self._plan_jobs: list[PreviewJob] = []
        self._uploads: OrderedDict[_UploadKey, _Upload] = OrderedDict()
        self._in_use: Counter[int] = Counter()
        self._ids = itertools.count(1)

    def start(self) -> None:
        """Start render workers; must run on the UI event loop."""
        self._loop = asyncio.get_running_loop()
        self._scheduler.start(default_worker_count())

    def stop(self) -> None:
        """Stop workers and free every image this app uploaded."""
        self._scheduler.stop()
        self._direct_pool.shutdown(wait=False, cancel_futures=True)
        for upload in self._uploads.values():
            terminal_write(self._app, kitty.delete_image_sequence(upload.image_id))
        self._uploads.clear()

    def refresh_cell_size(self) -> None:
        """Re-read the cell pixel size (fonts or window may have changed)."""
        self._cell = kitty.query_cell_size()

    def spec_for(self, request: ImageRequest) -> PreviewSpec:
        """Return the preview spec that fills ``request.box`` at the cell pixel size."""
        pixels = PixelBox(
            width=int(request.box.cols * self._cell.width),
            height=int(request.box.rows * self._cell.height),
        )
        return make_spec(request.source, pixels)

    def want(self, ticket: ImageTicket) -> ShownImage | None:
        """Return the image now if ready, else call ``ticket.on_ready`` once it is."""
        self._tickets[ticket.owner] = ticket
        preview = self._previews.get(self.spec_for(ticket.request))
        if preview is None:
            self._reprioritise()
            return None
        shown = self._ensure_upload(preview, ticket.request.box)
        if shown is not None:
            self._tickets.pop(ticket.owner, None)
        return shown

    def cancel(self, owner: int) -> None:
        """Drop a widget's outstanding request."""
        self._tickets.pop(owner, None)

    def acquire(self, image_id: int) -> None:
        """Mark an image as on screen so eviction skips it."""
        self._in_use[image_id] += 1

    def release(self, image_id: int) -> None:
        """Mark an image as no longer on screen."""
        self._in_use[image_id] -= 1
        if self._in_use[image_id] <= 0:
            del self._in_use[image_id]

    def prefetch(self, plan: PrefetchPlan) -> None:
        """Queue renders nearest-first and upload the neighbours as soon as they exist."""
        self._ahead = {self.spec_for(request): request.box for request in plan.upload_ahead}
        for spec, box in self._ahead.items():
            if spec in self._previews:
                self._ensure_upload(self._previews[spec], box)
        urgent = [PreviewJob(spec=self.spec_for(r), is_urgent=True) for r in plan.urgent]
        background = [PreviewJob(spec=self.spec_for(r)) for r in plan.background]
        self._plan_jobs = urgent + background
        self._reprioritise()

    def _reprioritise(self) -> None:
        """Push wanted specs first, then the plan, skipping what is already rendered."""
        wanted = [PreviewJob(spec=self.spec_for(t.request), is_urgent=True) for t in self._tickets.values()]
        seen: set[PreviewSpec] = set()
        jobs: list[PreviewJob] = []
        for job in wanted + self._plan_jobs:
            if job.spec in seen or job.spec in self._previews:
                continue
            seen.add(job.spec)
            jobs.append(job)
        self._scheduler.prioritise(jobs)

    def _on_preview_done_threadsafe(self, spec: PreviewSpec, preview: Preview | None) -> None:
        """Worker thread: hand a finished render to the UI loop without blocking."""
        self._call_soon(self._on_preview_done, spec, preview)

    def _call_soon(self, callback: Callable[..., None], *args: object) -> None:
        """Schedule ``callback`` on the UI loop from any thread; ignore a closed loop."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            logger.debug("UI loop closed; dropping %s", getattr(callback, "__name__", callback))

    def _on_preview_done(self, spec: PreviewSpec, preview: Preview | None) -> None:
        """UI thread: record a render, upload it if needed, and wake waiting widgets."""
        if preview is None:
            return
        self._previews[spec] = preview
        if spec in self._ahead:
            self._ensure_upload(preview, self._ahead[spec])
        self._fulfil_tickets()

    def _fulfil_tickets(self) -> None:
        """Hand every outstanding request whose image is now uploaded to its widget."""
        for ticket in list(self._tickets.values()):
            preview = self._previews.get(self.spec_for(ticket.request))
            if preview is None:
                continue
            shown = self._ensure_upload(preview, ticket.request.box)
            if shown is None:
                continue
            self._tickets.pop(ticket.owner, None)
            ticket.on_ready(shown)

    def _fit(self, preview: Preview, box: CellBox) -> CellBox:
        """Return the aspect-correct cell box for a preview inside ``box``."""
        return kitty.fit_cells(FitRequest(
            image_width=preview.width, image_height=preview.height, box=box, cell=self._cell,
        ))

    def _ensure_upload(self, preview: Preview, box: CellBox) -> ShownImage | None:
        """Return the uploaded image for a preview at ``box``; start the upload if absent."""
        key = _UploadKey(path=preview.path, box=self._fit(preview, box))
        upload = self._uploads.get(key)
        if upload is not None:
            self._uploads.move_to_end(key)
            return ShownImage(image_id=upload.image_id, box=key.box) if upload.is_ready else None
        upload = _Upload(image_id=self._next_image_id())
        self._uploads[key] = upload
        if self._is_file_transmit:
            self._write_upload(key, kitty.transmit_file_sequence(upload.image_id, key.path))
            return ShownImage(image_id=upload.image_id, box=key.box)
        future = self._direct_pool.submit(_read_direct_transmit, key, upload.image_id)
        future.add_done_callback(partial(self._on_direct_future, key))
        return None

    def _next_image_id(self) -> int:
        """Return an id not held by any live upload, wrapping inside the modal-safe range."""
        live = {upload.image_id for upload in self._uploads.values()}
        while (image_id := next(self._ids) % kitty.MAX_IMAGE_ID + 1) in live:
            continue
        return image_id

    def _on_direct_future(self, key: _UploadKey, future: Future[_DirectReady]) -> None:
        """Pool thread: pass a prepared inline upload (or its failure) to the UI loop."""
        if future.cancelled():
            return
        try:
            ready = future.result()
        except OSError as exc:
            logger.warning("preview read for upload failed: %s", exc)
            self._call_soon(self._drop_upload, key)
            return
        self._call_soon(self._on_direct_ready, ready)

    def _drop_upload(self, key: _UploadKey) -> None:
        """UI thread: forget a failed upload so a later request can retry it."""
        self._uploads.pop(key, None)

    def _on_direct_ready(self, ready: _DirectReady) -> None:
        """UI thread: write a prepared inline upload and wake its waiters."""
        if ready.key not in self._uploads:
            return
        self._write_upload(ready.key, ready.sequence)
        self._fulfil_tickets()

    def _write_upload(self, key: _UploadKey, transmit: str) -> None:
        """Write transmit + virtual placement, mark ready, then evict old images."""
        upload = self._uploads[key]
        terminal_write(self._app, transmit + kitty.virtual_placement_sequence(upload.image_id, key.box))
        upload.is_ready = True
        self._evict()

    def _evict(self) -> None:
        """Free the least recently used uploads that are not on screen."""
        excess = len(self._uploads) - MAX_UPLOADED_IMAGES
        for key in list(self._uploads):
            if excess <= 0:
                return
            upload = self._uploads[key]
            if upload.image_id in self._in_use or not upload.is_ready:
                continue
            terminal_write(self._app, kitty.delete_image_sequence(upload.image_id))
            del self._uploads[key]
            excess -= 1


def image_service(app: App) -> ImageService | None:
    """Return the app's image service, if it has one."""
    return getattr(app, "images", None)


class ImageCells(Widget):
    """Paints an uploaded image as kitty placeholder cells, centred and aspect-fitted.

    Keeps painting the previous image until the requested one is uploaded, so
    navigation never flashes an empty frame.
    """

    DEFAULT_CSS = """
    ImageCells {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:  # noqa: A002
        super().__init__(id=id, classes=classes)
        self._source: Path | None = None
        self._shown: ShownImage | None = None
        self._is_suspended = False
        self._resize_timer: Timer | None = None

    @property
    def shown(self) -> ShownImage | None:
        """Return the image currently painted, if any."""
        return self._shown

    @property
    def source(self) -> Path | None:
        """Return the photo this widget is asked to show."""
        return self._source

    def show(self, source: Path | None) -> None:
        """Ask for ``source``; the old image stays until the new one is ready."""
        self._source = source
        self._request_current()

    def set_suspended(self, is_suspended: bool) -> None:
        """Stop painting cells while another screen covers this one."""
        self._is_suspended = is_suspended
        self.refresh()

    def cell_box(self) -> CellBox:
        """Return this widget's content box in cells."""
        return CellBox(cols=self.size.width, rows=self.size.height)

    def on_image_shown(self, shown: ShownImage) -> None:
        """Hook for subclasses; called after a new image replaces the old one."""

    def _request_current(self) -> None:
        """Request the current source at the current size."""
        service = image_service(self.app)
        if service is None:
            return
        if self._source is None:
            service.cancel(id(self))
            self._set_shown(None)
            return
        box = self.cell_box()
        if box.cols < 1 or box.rows < 1:
            return
        ticket = ImageTicket(owner=id(self), request=ImageRequest(source=self._source, box=box), on_ready=self._set_shown)
        shown = service.want(ticket)
        if shown is not None:
            self._set_shown(shown)

    def _set_shown(self, shown: ShownImage | None) -> None:
        """Swap the painted image and keep the service's on-screen counts right."""
        service = image_service(self.app)
        if service is not None and shown is not None:
            service.acquire(shown.image_id)
        if service is not None and self._shown is not None:
            service.release(self._shown.image_id)
        self._shown = shown
        self.refresh()
        if shown is not None:
            self.on_image_shown(shown)

    def on_resize(self) -> None:
        """Re-request at the new size once a resize storm settles."""
        if self._resize_timer is not None:
            self._resize_timer.stop()
        self._resize_timer = self.set_timer(RESIZE_DEBOUNCE_SECONDS, self._on_resize_settled)

    def _on_resize_settled(self) -> None:
        """Resize settled: re-read the cell size and request a better-fitting image."""
        self._resize_timer = None
        service = image_service(self.app)
        if service is not None:
            service.refresh_cell_size()
        self._request_current()

    def on_unmount(self) -> None:
        """Release the on-screen image and any pending request."""
        service = image_service(self.app)
        if service is None:
            return
        service.cancel(id(self))
        if self._shown is not None:
            service.release(self._shown.image_id)
            self._shown = None

    def render_line(self, y: int) -> Strip:
        """Return one row: padding, this row's placeholder cells, padding.

        The cells go out as one segment per row. Textual's cost grows with
        segment count, and a full-width photo is ~10k cells per frame.
        """
        width, height = self.size.width, self.size.height
        background = Style(bgcolor=self.rich_style.bgcolor)
        shown = self._shown
        if shown is None or self._is_suspended:
            return Strip.blank(width, background)
        row = y - (height - shown.box.rows) // 2
        if not 0 <= row < shown.box.rows:
            return Strip.blank(width, background)
        left = max(0, (width - shown.box.cols) // 2)
        right = max(0, width - left - shown.box.cols)
        cells = kitty.placeholder_text(PlaceholderRow(row=row, cols=shown.box.cols))
        segments = [
            Segment(" " * left, background),
            Segment(cells, background + kitty.placeholder_style(shown.image_id)),
            Segment(" " * right, background),
        ]
        return Strip(segments, left + shown.box.cols + right).crop(0, width)
