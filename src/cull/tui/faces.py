"""Face close-ups for the review TUI: detect on a cached preview, crop, grade.

Detection runs in a child process (``cull.tui.face_worker``), never in the
TUI process. ``import mediapipe`` imports ``readline``, whose libedit init
puts the tty back into canonical/echo mode under Textual; MediaPipe, TFLite
and absl write native logs straight onto fd 2, the stream Textual's driver
paints on; and loading the model held the UI for seconds. The child runs in
its own session with no tty on any fd, and its stderr goes to
NATIVE_LOG_PATH.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from cull.tui import previews
from cull.tui.previews import PixelBox, Preview

logger = logging.getLogger(__name__)

FACE_PREVIEW_BOX: PixelBox = PixelBox(width=2400, height=2400)
MAX_FACES: int = 4
FACE_CROP_PADDING: float = 0.25
SHARPNESS_SAMPLE_WIDTH_PX: int = 128
# Laplacian variance of the face resized to 128 px wide, so the grade does not
# depend on face size. Heuristic cut-offs, not calibrated on labelled faces.
SHARPNESS_GOOD_MIN: float = 120.0
SHARPNESS_SOFT_MIN: float = 40.0
# Lower than Stage 2's scoring threshold: the panel is a viewing aid, and a
# missed face (dark indoor frames) costs the reviewer more than a stray one.
DETECTION_CONFIDENCE_MIN: float = 0.5
TILE_GRIDS: tuple[int, ...] = (1, 2, 3, 4)
TILE_OVERLAP: float = 0.25
DUPLICATE_FACE_IOU: float = 0.3
CLAHE_CLIP_LIMIT: float = 2.5
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)
NATIVE_LOG_PATH: Path = Path.home() / ".cache" / "cull" / "native.log"
QUIET_NATIVE_ENV: dict[str, str] = {"GLOG_minloglevel": "2", "TF_CPP_MIN_LOG_LEVEL": "3"}

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


class _DecodedPreview(BaseModel):
    """A decoded preview (BGR array) and the cache key its crops derive from."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Any
    key: str


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


def normalise_brightness(image: Any) -> Any:
    """Return a copy with CLAHE applied to luminance, so dark faces are detectable."""
    import cv2  # noqa: PLC0415

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@lru_cache(maxsize=1)
def _landmarker() -> Any:
    """Return this process's MediaPipe face landmarker (worker process only)."""
    import mediapipe as mp  # noqa: PLC0415

    from cull.config import FACE_LANDMARKER_FILENAME, ModelCacheConfig  # noqa: PLC0415

    model_path = ModelCacheConfig.from_env().mediapipe_dir / FACE_LANDMARKER_FILENAME
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=MAX_FACES,
        min_face_detection_confidence=DETECTION_CONFIDENCE_MIN,
        min_face_presence_confidence=DETECTION_CONFIDENCE_MIN,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


class _Point(NamedTuple):
    """A landmark in full-image normalised coordinates."""

    x: float
    y: float


class _Tile(NamedTuple):
    """A crop of the image, in pixels."""

    left: int
    top: int
    width: int
    height: int


def tiles_for(shape: tuple[int, ...]) -> list[_Tile]:
    """Return the whole frame plus overlapping 2x2 and 3x3 tiles.

    MediaPipe's landmarker uses the short-range face detector, which misses
    faces that are small in the frame (a guest across a room); in a tile the
    same face is several times larger relative to the input.
    """
    height, width = shape[:2]
    tiles: list[_Tile] = []
    for grid in TILE_GRIDS:
        tile_w = min(width, int(width / grid * (1 + TILE_OVERLAP)))
        tile_h = min(height, int(height / grid * (1 + TILE_OVERLAP)))
        for row in range(grid):
            for col in range(grid):
                left = 0 if grid == 1 else round(col * (width - tile_w) / (grid - 1))
                top = 0 if grid == 1 else round(row * (height - tile_h) / (grid - 1))
                tiles.append(_Tile(left, top, tile_w, tile_h))
    return tiles


def _detect_in_tile(image: Any, tile: _Tile) -> list[list[_Point]]:
    """Run the landmarker on one tile; return landmarks mapped to the full image."""
    import cv2  # noqa: PLC0415
    import mediapipe as mp  # noqa: PLC0415

    height, width = image.shape[:2]
    crop = image[tile.top:tile.top + tile.height, tile.left:tile.left + tile.width]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return [
        [_Point((tile.left + lm.x * tile.width) / width, (tile.top + lm.y * tile.height) / height) for lm in face]
        for face in (result.face_landmarks or [])
    ]


def _bounds(face: list[_Point]) -> tuple[float, float, float, float]:
    """Return (left, top, right, bottom) of a face in normalised coordinates."""
    xs, ys = [p.x for p in face], [p.y for p in face]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap(first: list[_Point], second: list[_Point]) -> float:
    """Return intersection-over-union of two faces' bounding boxes."""
    a, b = _bounds(first), _bounds(second)
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = inter_w * inter_h
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / area if area > 0 else 0.0


def detect_landmarks(image: Any) -> list[list[_Point]]:
    """Return up to MAX_FACES faces (largest first) found on a brightness-normalised copy."""
    normalised = normalise_brightness(image)
    found: list[list[_Point]] = []
    for tile in tiles_for(normalised.shape):
        for face in _detect_in_tile(normalised, tile):
            if all(_overlap(face, kept) < DUPLICATE_FACE_IOU for kept in found):
                found.append(face)
    found.sort(key=lambda face: -(_bounds(face)[2] - _bounds(face)[0]) * (_bounds(face)[3] - _bounds(face)[1]))
    return found[:MAX_FACES]


def analyse_faces(source: Path) -> FaceReport:
    """Worker: detect, crop and grade up to MAX_FACES faces on a cached preview."""
    import cv2  # noqa: PLC0415

    preview = previews.render_preview(previews.make_spec(source, FACE_PREVIEW_BOX))
    image = cv2.imread(str(preview.path))
    if image is None:
        return FaceReport(source=source, error="preview unreadable")
    try:
        detected = detect_landmarks(image)
    except Exception as exc:  # noqa: BLE001 - missing model or MediaPipe failure: show, do not crash
        return FaceReport(source=source, error=f"face detector unavailable: {exc}")
    frame = _DecodedPreview(image=image, key=str(preview.path))
    faces = [_read_face(frame, landmarks) for landmarks in detected[:MAX_FACES]]
    return FaceReport(source=source, faces=faces)


def worker_env() -> dict[str, str]:
    """Return the child's environment: quiet native logs, and our package importable."""
    package_root = str(Path(__file__).resolve().parents[2])
    python_path = os.pathsep.join(p for p in (package_root, os.environ.get("PYTHONPATH", "")) if p)
    return {**os.environ, **QUIET_NATIVE_ENV, "PYTHONPATH": python_path}


class FaceWorkerClient:
    """Talks to one long-lived face worker process over pipes (face thread only)."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None

    def analyse(self, source: Path) -> FaceReport:
        """Send one request and wait for its report."""
        process = self._ensure_started()
        request = {"source": str(source), "cache_dir": str(previews.PREVIEW_CACHE_DIR)}
        try:
            process.stdin.write(json.dumps(request) + "\n")  # type: ignore[union-attr]
            process.stdin.flush()  # type: ignore[union-attr]
            line = process.stdout.readline()  # type: ignore[union-attr]
        except (BrokenPipeError, OSError) as exc:
            line = ""
            logger.warning("face worker pipe failed: %s", exc)
        if not line:
            self.close()
            return FaceReport(source=source, error=f"face worker stopped (see {NATIVE_LOG_PATH})")
        return FaceReport.model_validate_json(line)

    def _ensure_started(self) -> subprocess.Popen[str]:
        """Start the worker detached from the tty: own session, pipes, stderr to a log."""
        if self._process is not None and self._process.poll() is None:
            return self._process
        NATIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with NATIVE_LOG_PATH.open("a", encoding="utf-8") as native_log:
            self._process = subprocess.Popen(
                [sys.executable, "-m", "cull.tui.face_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=native_log,
                start_new_session=True, close_fds=True, text=True, env=worker_env(),
            )
        return self._process

    def close(self) -> None:
        """Stop the worker process, if any."""
        if self._process is None:
            return
        self._process.kill()
        self._process.wait()
        self._process = None


FaceCallback = Callable[[FaceReport], None]


class FaceRequest(BaseModel):
    """Analyse ``source`` and report back on the UI loop."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Path
    on_done: FaceCallback


class FaceService:
    """Queues face analysis on one thread (which talks to the worker) and caches reports."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cull-faces")
        self._client = FaceWorkerClient()
        self._reports: dict[Path, FaceReport] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queued: Future[FaceReport] | None = None

    def start(self) -> None:
        """Bind to the running UI loop."""
        self._loop = asyncio.get_running_loop()

    def stop(self) -> None:
        """Drop queued work and stop the worker process."""
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._client.close()

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
        future = self._pool.submit(self._client.analyse, face_request.source)
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
