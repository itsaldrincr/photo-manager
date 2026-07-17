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
    PORTRAIT_EAR_CLOSED_MAX,
    PORTRAIT_EAR_SQUINT_MAX,
    PORTRAIT_EMOTION_CROP_MARGIN_FRACTION,
    PORTRAIT_FACE_OCCLUSION_MIN,
    PORTRAIT_OCCLUSION_PATCH_IOD_FRACTION,
    PORTRAIT_OCCLUSION_PATCH_MIN_HALF_PX,
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

# Outer eye corners, reused to measure inter-ocular distance (occlusion patch sizing).
_IOD_RIGHT_OUTER_IDX: int = _RIGHT_EYE_INDICES[0]
_IOD_LEFT_OUTER_IDX: int = _LEFT_EYE_INDICES[3]

# Key face regions sampled for occlusion texture analysis (see detect_occlusion).
_OCCLUSION_LANDMARK_GROUPS: dict[str, tuple[int, ...]] = {
    "left_eye": tuple(_LEFT_EYE_INDICES),
    "right_eye": tuple(_RIGHT_EYE_INDICES),
    "left_brow": (70, 63, 105, 66, 107),
    "right_brow": (300, 293, 334, 296, 336),
    "nose": (1, 2, 98, 327),
    "mouth": (61, 291, 0, 17, 78, 308, 13, 14),
}

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
    is_squinting: bool = False
    face_occluded: bool = False
    occlusion_ratio: float | None = None
    dominant_emotion: str | None = None
    valence: float | None = None
    arousal: float | None = None

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
    """Groups FaceContext and total face count for result assembly."""

    ctx: _FaceContext
    face_count: int


@dataclass(frozen=True)
class _EmotionReading:
    """Mapped 7-class emotion label plus EmotiEffLib valence/arousal scores."""

    label: str
    valence: float | None
    arousal: float | None


# EmotiEffLib's enet_b0_8_va_mtl model emits 8 classes; DeepFace-era consumers
# (Stage 2/3/4 scoring, VLM prompt hints) expect the classic 7-class FER
# vocabulary. "Contempt" has no direct match, so it folds into "disgust" —
# the nearest facial-signature neighbour (curled lip / nose-wrinkle overlap)
# and DeepFace's own historical closest bucket for contempt-like expressions.
_EMOTIEFF_TO_CONSUMER_LABEL: dict[str, str] = {
    "anger": "angry",
    "contempt": "disgust",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}


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


def _emotion_crop(ctx: _FaceContext) -> np.ndarray:
    """Return the margin-padded face crop fed to EmotiEffLib (no internal detector)."""
    x1, y1, x2, y2 = _face_bbox_from_landmarks(ctx)
    h, w = ctx.image.shape[:2]
    pad_x = int((x2 - x1) * PORTRAIT_EMOTION_CROP_MARGIN_FRACTION)
    pad_y = int((y2 - y1) * PORTRAIT_EMOTION_CROP_MARGIN_FRACTION)
    return ctx.image[
        max(0, y1 - pad_y):min(h, y2 + pad_y),
        max(0, x1 - pad_x):min(w, x2 + pad_x),
    ]


@dataclass(frozen=True)
class _FaceMetrics:
    """Aggregated per-face measurements feeding PortraitResult assembly."""

    ear_left: float
    ear_right: float
    occlusion: float
    sharp_left: float
    sharp_right: float
    emotion: _EmotionReading


def _compute_face_metrics(ctx: _FaceContext) -> _FaceMetrics:
    """Compute EAR, occlusion, eye sharpness, and emotion for a detected face."""
    ear_l, ear_r = _ear_pair(ctx.landmarks)
    return _FaceMetrics(
        ear_left=ear_l,
        ear_right=ear_r,
        occlusion=detect_occlusion(ctx.image, ctx.landmarks),
        sharp_left=_crop_sharpness(ctx, _LEFT_EYE_INDICES),
        sharp_right=_crop_sharpness(ctx, _RIGHT_EYE_INDICES),
        emotion=_detect_emotion_reading(_emotion_crop(ctx)),
    )


def _assemble_result(assembly: _AssemblyInput) -> PortraitResult:
    """Build PortraitResult from AssemblyInput."""
    ctx = assembly.ctx
    metrics = _compute_face_metrics(ctx)
    mean_ear = (metrics.ear_left + metrics.ear_right) / 2.0
    eyes_closed = is_eyes_closed(mean_ear)
    return PortraitResult(
        face_count=assembly.face_count,
        face_bbox=_face_bbox_from_landmarks(ctx),
        eye_sharpness_left=metrics.sharp_left,
        eye_sharpness_right=metrics.sharp_right,
        ear_left=metrics.ear_left,
        ear_right=metrics.ear_right,
        eyes_closed=eyes_closed,
        is_squinting=is_squinting(mean_ear, eyes_closed),
        face_occluded=metrics.occlusion < PORTRAIT_FACE_OCCLUSION_MIN,
        occlusion_ratio=metrics.occlusion,
        dominant_emotion=metrics.emotion.label or None,
        valence=metrics.emotion.valence,
        arousal=metrics.emotion.arousal,
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


def is_squinting(ear_value: float, eyes_closed: bool) -> bool:
    """Return True if eyes are open but EAR is below squint threshold."""
    return (not eyes_closed) and (ear_value < PORTRAIT_EAR_SQUINT_MAX)


@dataclass(frozen=True)
class _TexturePatch:
    """Pixel-space patch centre + half-width for a local-texture sample."""

    cx: int
    cy: int
    half: int


def _texture_patch_variance(gray: np.ndarray, patch: _TexturePatch) -> float:
    """Return Laplacian variance (local texture strength) of a small patch."""
    h, w = gray.shape[:2]
    x1, x2 = max(0, patch.cx - patch.half), min(w, patch.cx + patch.half)
    y1, y2 = max(0, patch.cy - patch.half), min(h, patch.cy + patch.half)
    crop = gray[y1:y2, x1:x2]
    if crop.size < 4:
        return 0.0
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def _inter_ocular_distance_px(ctx: _FaceContext) -> float:
    """Return pixel distance between the outer corners of both eyes."""
    h, w = ctx.image.shape[:2]
    left = ctx.landmarks[_IOD_LEFT_OUTER_IDX]
    right = ctx.landmarks[_IOD_RIGHT_OUTER_IDX]
    dx = (left.x - right.x) * w
    dy = (left.y - right.y) * h
    return float(np.hypot(dx, dy))


def _occlusion_patch_half_px(ctx: _FaceContext) -> int:
    """Return face-size-scaled patch half-width, floored for stability."""
    scaled = int(_inter_ocular_distance_px(ctx) * PORTRAIT_OCCLUSION_PATCH_IOD_FRACTION)
    return max(PORTRAIT_OCCLUSION_PATCH_MIN_HALF_PX, scaled)


@dataclass(frozen=True)
class _RegionSample:
    """Bundles a grayscale face image with one region's landmark indices."""

    gray: np.ndarray
    indices: tuple[int, ...]


def _region_texture_variance(ctx: _FaceContext, sample: _RegionSample) -> float:
    """Return mean local-texture variance across one face region's landmarks."""
    h, w = ctx.image.shape[:2]
    half = _occlusion_patch_half_px(ctx)
    patches = [
        _TexturePatch(cx=_to_px(ctx.landmarks[i].x, w), cy=_to_px(ctx.landmarks[i].y, h), half=half)
        for i in sample.indices
    ]
    return float(np.mean([_texture_patch_variance(sample.gray, p) for p in patches]))


def _occlusion_region_variances(ctx: _FaceContext) -> dict[str, float]:
    """Return per-face-region local-texture variance (eyes, brows, nose, mouth)."""
    gray = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)
    return {
        region: _region_texture_variance(ctx, _RegionSample(gray=gray, indices=indices))
        for region, indices in _OCCLUSION_LANDMARK_GROUPS.items()
    }


def detect_occlusion(image: np.ndarray, landmarks: list[Any]) -> float:
    """Return occlusion visibility ratio: min region texture over median region texture.

    A hand, object, or other occluder flattens local pixel texture in the
    region it covers (MediaPipe still fits landmarks there, but the pixels
    underneath are uniform). Comparing the flattest region against this same
    face's own median keeps the ratio valid across arbitrary photo
    resolutions and stays distinct from whole-face blur (which lowers every
    region together, leaving the ratio near 1.0). Lower = more occluded.
    """
    variances = list(_occlusion_region_variances(_FaceContext(image=image, landmarks=landmarks)).values())
    median = float(np.median(variances))
    if median < 1e-6:
        return 1.0
    return min(variances) / median


def _map_emotiefflib_label(label: str) -> str:
    """Map an EmotiEffLib 8-class label into the legacy 7-class FER vocabulary."""
    return _EMOTIEFF_TO_CONSUMER_LABEL.get(label.strip().lower(), "")


def _emotieff_reading(image_rgb: np.ndarray) -> _EmotionReading:
    """Run the EmotiEffLib recognizer on an RGB array; return mapped label + VA."""
    from cull.emotieff_loader import get_emotieff_recognizer  # noqa: PLC0415

    recognizer = get_emotieff_recognizer()
    labels, scores = recognizer.predict_emotions(image_rgb, logits=True)
    return _EmotionReading(
        label=_map_emotiefflib_label(labels[0]),
        valence=float(scores[0, -2]),
        arousal=float(scores[0, -1]),
    )


def _detect_emotion_reading(image: np.ndarray) -> _EmotionReading:
    """Return mapped emotion + valence/arousal for a BGR array; log errors, return empty."""
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return _emotieff_reading(rgb)
    except Exception as exc:
        log.warning("EmotiEffLib emotion detection failed: %s", exc)
        return _EmotionReading(label="", valence=None, arousal=None)


def detect_expression_from_array(image: np.ndarray) -> str:
    """Return dominant emotion for an already face-cropped BGR array; log errors, return empty."""
    return _detect_emotion_reading(image).label


def detect_expression(image_path: Path) -> str:
    """Return dominant emotion string for a face-cropped image path; log errors, return empty."""
    image = cv2.imread(str(image_path))
    if image is None:
        log.error("Could not read image: %s", image_path)
        return ""
    return detect_expression_from_array(image)


def assess_portrait_from_array(image: np.ndarray, config: CullConfig) -> PortraitResult:
    """Detect faces and return full PortraitResult for an already-decoded image.

    Resolution note: the caller must pass a FULL-RESOLUTION decode (not the
    downscaled pil_1280 used elsewhere in Stage 2). Eye-crop Tenengrad
    sharpness is resolution-sensitive, and the historical cv2.imread-based
    path always ran on the original file resolution — downscaling here
    would silently shift the eye-sharpness scale and any calibrated
    thresholds built against it.
    """
    if not config.is_portrait:
        return PortraitResult(face_count=0)
    faces = detect_faces(image)
    if not faces:
        return PortraitResult(face_count=0)
    ctx = _FaceContext(image=image, landmarks=faces[0])
    assembly = _AssemblyInput(ctx=ctx, face_count=len(faces))
    return _assemble_result(assembly)


def assess_portrait(image_path: Path, config: CullConfig) -> PortraitResult:
    """Load image at full resolution and return full PortraitResult (thin path wrapper)."""
    if not config.is_portrait:
        return PortraitResult(face_count=0)
    image = cv2.imread(str(image_path))
    if image is None:
        log.error("Could not read image: %s", image_path)
        return PortraitResult(face_count=0)
    return assess_portrait_from_array(image, config)
