"""Stage 2b portrait-mode face and eye quality analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from cull.config import (
    FACE_LANDMARKER_FILENAME,
    PORTRAIT_NUM_FACES_MAX,
    PORTRAIT_FACE_DETECTION_CONFIDENCE_MIN,
    PORTRAIT_LANDMARK_VISIBILITY_THRESHOLD,
    PORTRAIT_EAR_CLOSED_MAX,
    PORTRAIT_FACE_OCCLUSION_MIN,
    CullConfig,
    ModelCacheConfig,
)
from cull.model_cache import ConfigError

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eye landmark indices (MediaPipe 478-point mesh, refined)
# ---------------------------------------------------------------------------

_LEFT_EYE_INDICES: list[int] = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_INDICES: list[int] = [33, 160, 158, 133, 153, 144]

EYE_CROP_PADDING: float = 0.20
TOTAL_LANDMARK_COUNT: int = 468

_CACHE: ModelCacheConfig = ModelCacheConfig.from_env()
_face_landmarker: Any | None = None


# ---------------------------------------------------------------------------
# Pydantic result model (defined here, NOT in models.py)
# ---------------------------------------------------------------------------


class PortraitResult(BaseModel):
    """Full portrait-mode assessment result for a single image."""

    face_count: int = 0
    face_bbox: tuple[int, int, int, int] | None = None
    eye_sharpness_left: float | None = None
    eye_sharpness_right: float | None = None
    ear_left: float | None = None
    ear_right: float | None = None
    eyes_closed: bool = False
    face_occluded: bool = False
    occlusion_ratio: float | None = None
    dominant_emotion: str | None = None

    @property
    def has_face(self) -> bool:
        """Return True if a face was detected in the image."""
        return self.face_count > 0


# ---------------------------------------------------------------------------
# Internal parameter containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EyeRegion:
    """Pairs landmark index list with a landmark list for bbox computation."""

    indices: list[int]
    landmarks: list[Any]


@dataclass(frozen=True)
class _FaceContext:
    """Carries image + detected landmark list into portrait assembly."""

    image: np.ndarray
    landmarks: list[Any]


@dataclass(frozen=True)
class _AssemblyInput:
    """Groups FaceContext, total face count, and source path for assembly."""

    ctx: _FaceContext
    face_count: int
    image_path: Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_face_landmarker_path(cache: ModelCacheConfig) -> Path:
    """Return the on-disk face_landmarker.task path or raise ConfigError."""
    path = cache.mediapipe_dir / FACE_LANDMARKER_FILENAME
    if not path.exists():
        raise ConfigError(
            f"face_landmarker.task not found at {path}. "
            "Run 'cull setup --allow-network' to populate the model cache."
        )
    return path


def _get_face_landmarker() -> Any:
    """Return module-level FaceLandmarker singleton, initialising on first call."""
    global _face_landmarker
    if _face_landmarker is not None:
        return _face_landmarker
    import mediapipe as mp  # noqa: PLC0415

    model_path = _resolve_face_landmarker_path(_CACHE)
    base_opts = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=PORTRAIT_NUM_FACES_MAX,
        min_face_detection_confidence=PORTRAIT_FACE_DETECTION_CONFIDENCE_MIN,
    )
    _face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    return _face_landmarker


def _to_px(val: float, dim: int) -> int:
    """Convert a normalised coordinate to a pixel index."""
    return int(val * dim)


def _eye_bbox(region: _EyeRegion, image: np.ndarray) -> tuple[int, int, int, int]:
    """Return padded bounding box (x1, y1, x2, y2) around indexed landmarks."""
    h, w = image.shape[:2]
    xs = [_to_px(region.landmarks[i].x, w) for i in region.indices]
    ys = [_to_px(region.landmarks[i].y, h) for i in region.indices]
    pad_x = max(1, int((max(xs) - min(xs)) * EYE_CROP_PADDING))
    pad_y = max(1, int((max(ys) - min(ys)) * EYE_CROP_PADDING))
    x1 = max(0, min(xs) - pad_x)
    y1 = max(0, min(ys) - pad_y)
    x2 = min(w, max(xs) + pad_x)
    y2 = min(h, max(ys) + pad_y)
    return x1, y1, x2, y2


def _tenengrad(crop: np.ndarray) -> float:
    """Compute Tenengrad sharpness score on a grayscale crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


def _ear_from_pts(pts: list[tuple[float, float]]) -> float:
    """Compute Eye Aspect Ratio from 6 (x, y) landmark points."""
    p1, p2, p3, p4, p5, p6 = pts
    vert_a = float(np.linalg.norm(np.array(p2) - np.array(p6)))
    vert_b = float(np.linalg.norm(np.array(p3) - np.array(p5)))
    horiz = float(np.linalg.norm(np.array(p1) - np.array(p4)))
    if horiz < 1e-6:
        return 0.0
    return (vert_a + vert_b) / (2.0 * horiz)


def _crop_sharpness(ctx: _FaceContext, indices: list[int]) -> float:
    """Return Tenengrad sharpness of the eye crop at given landmark indices."""
    region = _EyeRegion(indices=indices, landmarks=ctx.landmarks)
    x1, y1, x2, y2 = _eye_bbox(region, ctx.image)
    crop = ctx.image[y1:y2, x1:x2]
    return _tenengrad(crop) if crop.size > 0 else 0.0


def _ear_pair(landmarks: list[Any]) -> tuple[float, float]:
    """Return (ear_left, ear_right) EAR values from a landmark list."""
    pts_l = [(landmarks[i].x, landmarks[i].y) for i in _LEFT_EYE_INDICES]
    pts_r = [(landmarks[i].x, landmarks[i].y) for i in _RIGHT_EYE_INDICES]
    return _ear_from_pts(pts_l), _ear_from_pts(pts_r)


def _face_bbox_from_landmarks(ctx: _FaceContext) -> tuple[int, int, int, int]:
    """Return tight pixel bbox enclosing all visible face landmarks."""
    h, w = ctx.image.shape[:2]
    xs = [_to_px(lm.x, w) for lm in ctx.landmarks[:TOTAL_LANDMARK_COUNT]]
    ys = [_to_px(lm.y, h) for lm in ctx.landmarks[:TOTAL_LANDMARK_COUNT]]
    return (max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys)))


def _assemble_result(assembly: _AssemblyInput) -> PortraitResult:
    """Build PortraitResult from AssemblyInput."""
    ctx = assembly.ctx
    ear_l, ear_r = _ear_pair(ctx.landmarks)
    mean_ear = (ear_l + ear_r) / 2.0
    occlusion = detect_occlusion(ctx.landmarks)
    sharp_l = _crop_sharpness(ctx, _LEFT_EYE_INDICES)
    sharp_r = _crop_sharpness(ctx, _RIGHT_EYE_INDICES)
    emotion = detect_expression(assembly.image_path)
    return PortraitResult(
        face_count=assembly.face_count,
        face_bbox=_face_bbox_from_landmarks(ctx),
        eye_sharpness_left=sharp_l,
        eye_sharpness_right=sharp_r,
        ear_left=ear_l,
        ear_right=ear_r,
        eyes_closed=is_eyes_closed(mean_ear),
        face_occluded=occlusion < PORTRAIT_FACE_OCCLUSION_MIN,
        occlusion_ratio=occlusion,
        dominant_emotion=emotion or None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def unload_face_landmarker() -> None:
    """Release the MediaPipe FaceLandmarker singleton and reset module state."""
    global _face_landmarker
    if _face_landmarker is None:
        return
    try:
        _face_landmarker.close()
    except Exception as exc:
        log.warning("Error closing FaceLandmarker: %s", exc)
    _face_landmarker = None


def detect_faces(image: np.ndarray) -> list[Any]:
    """Return list of face landmark lists detected by MediaPipe FaceLandmarker."""
    import mediapipe as mp  # noqa: PLC0415

    landmarker = _get_face_landmarker()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return []
    return list(result.face_landmarks)


def compute_eye_sharpness(image: np.ndarray, landmarks: list[Any]) -> float:
    """Return Tenengrad sharpness score on left-eye crop with 20% padding."""
    ctx = _FaceContext(image=image, landmarks=landmarks)
    return _crop_sharpness(ctx, _LEFT_EYE_INDICES)


def compute_ear(landmarks: list[Any]) -> float:
    """Return mean Eye Aspect Ratio across both eyes."""
    ear_l, ear_r = _ear_pair(landmarks)
    return (ear_l + ear_r) / 2.0


def is_eyes_closed(ear_value: float) -> bool:
    """Return True if EAR is below the closed-eye threshold."""
    return ear_value < PORTRAIT_EAR_CLOSED_MAX


def detect_occlusion(landmarks: list[Any]) -> float:
    """Return ratio of landmarks with visibility > threshold (MediaPipe visibility attribute)."""
    visible = sum(1 for lm in landmarks[:TOTAL_LANDMARK_COUNT] if lm.visibility > PORTRAIT_LANDMARK_VISIBILITY_THRESHOLD)
    return visible / TOTAL_LANDMARK_COUNT


def detect_expression(image_path: Path) -> str:
    """Return dominant emotion string via DeepFace; log errors, return empty."""
    try:
        from deepface import DeepFace  # type: ignore[import]  # noqa: PLC0415

        result = DeepFace.analyze(
            str(image_path), actions=["emotion"], enforce_detection=False
        )
        emotions = result[0] if isinstance(result, list) else result
        return str(emotions.get("dominant_emotion", ""))
    except Exception as exc:
        log.warning("DeepFace emotion detection failed: %s", exc)
        return ""


def assess_portrait(image_path: Path, config: CullConfig) -> PortraitResult:
    """Load image, detect faces, and return full PortraitResult."""
    if not config.is_portrait:
        return PortraitResult(face_count=0)
    image = cv2.imread(str(image_path))
    if image is None:
        log.error("Could not read image: %s", image_path)
        return PortraitResult(face_count=0)
    faces = detect_faces(image)
    if not faces:
        return PortraitResult(face_count=0)
    ctx = _FaceContext(image=image, landmarks=faces[0])
    assembly = _AssemblyInput(ctx=ctx, face_count=len(faces), image_path=image_path)
    return _assemble_result(assembly)
