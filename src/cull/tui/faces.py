"""Face close-ups for the review TUI: detect on a cached preview, crop, grade.

Runs on one dedicated thread. MediaPipe's landmarker is a process-wide
singleton that is not safe to call from two threads at once, and the thread
keeps detection from ever competing with navigation for the UI loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cull.tui import previews
from cull.tui.previews import PixelBox, Preview

logger = logging.getLogger(__name__)

FACE_PREVIEW_BOX: PixelBox = PixelBox(width=1600, height=1600)
MAX_FACES: int = 4
FACE_CROP_PADDING: float = 0.25
SHARPNESS_SAMPLE_WIDTH_PX: int = 128
# Laplacian variance of the face resized to 128 px wide, so the grade does not
# depend on face size. Heuristic cut-offs, not calibrated on labelled faces.
SHARPNESS_GOOD_MIN: float = 120.0
SHARPNESS_SOFT_MIN: float = 40.0

EyeState = Literal["open", "squint", "closed"]
SharpnessLevel = Literal["good", "soft", "blurry"]


class FaceReading(BaseModel):
    """One detected face: its crop on disk and traffic-light grades."""

    crop: Preview
    eyes: EyeState
    sharpness: float
    sharpness_level: SharpnessLevel


class FaceReport(BaseModel):
    """All faces found in one photo, or why detection could not run."""

    source: Path
    faces: list[FaceReading] = Field(default_factory=list)
    error: str | None = None


class _FaceBox(BaseModel):
    """A padded face rectangle in preview pixels."""

    left: int
    top: int
    right: int
    bottom: int


def grade_sharpness(value: float) -> SharpnessLevel:
    """Map a normalised face sharpness to a traffic-light level."""
    if value >= SHARPNESS_GOOD_MIN:
        return "good"
    if value >= SHARPNESS_SOFT_MIN:
        return "soft"
    return "blurry"


def _eye_state(landmarks: list[Any]) -> EyeState:
    """Grade the eyes from the eye aspect ratio via the Stage 2 thresholds."""
    from cull.stage2 import portrait  # noqa: PLC0415

    ear = portrait.compute_ear(landmarks)
    if portrait.is_eyes_closed(ear):
        return "closed"
    return "squint" if portrait.is_squinting(ear, False) else "open"


def _face_box(landmarks: list[Any], shape: tuple[int, ...]) -> _FaceBox:
    """Return the landmark bounding box, padded and clamped to the image."""
    height, width = shape[:2]
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    pad_x = (max(xs) - min(xs)) * FACE_CROP_PADDING
    pad_y = (max(ys) - min(ys)) * FACE_CROP_PADDING
    return _FaceBox(
        left=max(0, int(min(xs) - pad_x)), top=max(0, int(min(ys) - pad_y)),
        right=min(width, int(max(xs) + pad_x)), bottom=min(height, int(max(ys) + pad_y)),
    )


def _sharpness(crop: Any) -> float:
    """Return Laplacian variance of a grey crop resized to a fixed width."""
    import cv2  # noqa: PLC0415

    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = SHARPNESS_SAMPLE_WIDTH_PX / max(1, grey.shape[1])
    resized = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(resized, cv2.CV_64F).var())


def _save_crop(crop: Any, key: str) -> Preview:
    """Write a face crop PNG into the preview cache and return it."""
    import cv2  # noqa: PLC0415

    name = hashlib.sha1(key.encode("utf-8")).hexdigest()
    target = previews.PREVIEW_CACHE_DIR / f"face-{name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        cv2.imwrite(str(target), crop)
    height, width = crop.shape[:2]
    return Preview(path=target, width=width, height=height)


class _DecodedPreview(BaseModel):
    """A decoded preview (BGR array) and the cache key its crops derive from."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Any
    key: str


def _read_face(frame: _DecodedPreview, landmarks: list[Any]) -> FaceReading:
    """Crop and grade one face."""
    box = _face_box(landmarks, frame.image.shape)
    crop = frame.image[box.top:box.bottom, box.left:box.right]
    sharpness = _sharpness(crop)
    return FaceReading(
        crop=_save_crop(crop, f"{frame.key}|{box.left},{box.top},{box.right},{box.bottom}"),
        eyes=_eye_state(landmarks),
        sharpness=sharpness,
        sharpness_level=grade_sharpness(sharpness),
    )


def analyse_faces(source: Path) -> FaceReport:
    """Detect, crop and grade up to MAX_FACES faces on a cached preview of ``source``."""
    import cv2  # noqa: PLC0415

    from cull.stage2 import portrait  # noqa: PLC0415

    preview = previews.render_preview(previews.make_spec(source, FACE_PREVIEW_BOX))
    image = cv2.imread(str(preview.path))
    if image is None:
        return FaceReport(source=source, error="preview unreadable")
    try:
        detected = portrait.detect_faces(image)
    except Exception as exc:  # noqa: BLE001 - missing model or MediaPipe failure: show, do not crash
        return FaceReport(source=source, error=f"face detector unavailable: {exc}")
    frame = _DecodedPreview(image=image, key=str(preview.path))
    faces = [_read_face(frame, landmarks) for landmarks in detected[:MAX_FACES]]
    return FaceReport(source=source, faces=faces)


FaceCallback = Callable[[FaceReport], None]


class FaceRequest(BaseModel):
    """Analyse ``source`` and report back on the UI loop."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Path
    on_done: FaceCallback


class FaceService:
    """Runs face analysis on one thread and caches reports per photo."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cull-faces")
        self._reports: dict[Path, FaceReport] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queued: Future[FaceReport] | None = None

    def start(self) -> None:
        """Bind to the running UI loop."""
        self._loop = asyncio.get_running_loop()

    def stop(self) -> None:
        """Drop queued work; a running detection finishes in the background."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def request(self, face_request: FaceRequest) -> FaceReport | None:
        """Return a cached report now, or analyse in the background and call back.

        A newer request cancels one still queued, so holding an arrow key
        never builds a backlog of detections for photos already left behind.
        """
        cached = self._reports.get(face_request.source)
        if cached is not None:
            return cached
        if self._queued is not None:
            self._queued.cancel()
        future = self._pool.submit(analyse_faces, face_request.source)
        self._queued = future
        future.add_done_callback(lambda done: self._deliver(done, face_request))
        return None

    def _deliver(self, future: Future[FaceReport], face_request: FaceRequest) -> None:
        """Pool thread: hand the finished report to the UI loop."""
        if future.cancelled():
            return
        try:
            report = future.result()
        except Exception as exc:  # noqa: BLE001 - surface any failure in the panel
            report = FaceReport(source=face_request.source, error=str(exc))
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._store_and_call, report, face_request.on_done)
        except RuntimeError:
            logger.debug("UI loop closed; dropping face report for %s", face_request.source)

    def _store_and_call(self, report: FaceReport, on_done: FaceCallback) -> None:
        """UI thread: cache the report and pass it on."""
        self._reports[report.source] = report
        on_done(report)
