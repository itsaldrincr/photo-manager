"""Stage 4 peak-portrait selection via MediaPipe face blendshapes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cull.config import (
    CullConfig,
    EYE_OPEN_THRESHOLD,
    ModelCacheConfig,
    PORTRAIT_FACE_DETECTION_CONFIDENCE_MIN,
    PORTRAIT_NUM_FACES_MAX,
    SMILE_THRESHOLD,
)
from cull.models import PeakMomentScore
from cull.stage2.portrait import _resolve_face_landmarker_path

_PEAK_CACHE: ModelCacheConfig = ModelCacheConfig.from_env()

log = logging.getLogger(__name__)

_blendshape_landmarker: Any | None = None

BLINK_LEFT: str = "eyeBlinkLeft"
BLINK_RIGHT: str = "eyeBlinkRight"
SMILE_LEFT: str = "mouthSmileLeft"
SMILE_RIGHT: str = "mouthSmileRight"
GAZE_LEFT: str = "eyeLookOutLeft"
GAZE_RIGHT: str = "eyeLookOutRight"
EYE_OPEN_WEIGHT: float = 0.5
SMILE_WEIGHT: float = 0.3
GAZE_WEIGHT: float = 0.2
BONUS_EYES_OPEN: float = 0.05
BONUS_SMILE: float = 0.05
PEAK_TYPE_PORTRAIT: str = "portrait"


class PeakPortraitInput(BaseModel):
    """Input bundle for peak portrait selection."""

    burst_members: list[Path]
    config: CullConfig


def _get_blendshape_landmarker() -> Any:
    """Return module-level blendshape-enabled FaceLandmarker singleton."""
    global _blendshape_landmarker
    if _blendshape_landmarker is not None:
        return _blendshape_landmarker
    import mediapipe as mp  # noqa: PLC0415

    model_path = _resolve_face_landmarker_path(_PEAK_CACHE)
    base_opts = mp.tasks.BaseOptions(
        model_asset_path=str(model_path)
    )
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=PORTRAIT_NUM_FACES_MAX,
        min_face_detection_confidence=PORTRAIT_FACE_DETECTION_CONFIDENCE_MIN,
        output_face_blendshapes=True,
    )
    _blendshape_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    return _blendshape_landmarker


def _detect_blendshapes(image_path: Path) -> dict[str, float]:
    """Return blendshape name→score dict for the first face found."""
    import cv2  # noqa: PLC0415
    import mediapipe as mp  # noqa: PLC0415

    image = cv2.imread(str(image_path))
    if image is None:
        log.warning("Could not read image: %s", image_path)
        return {}
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker = _get_blendshape_landmarker()
    result = landmarker.detect(mp_image)
    if not result.face_blendshapes:
        return {}
    return {bs.category_name: bs.score for bs in result.face_blendshapes[0]}


def _combine_blendshapes(blendshapes: dict[str, float]) -> PeakMomentScore:
    """Compute PeakMomentScore from a blendshape name→score dict."""
    blink_l = blendshapes.get(BLINK_LEFT, 1.0)
    blink_r = blendshapes.get(BLINK_RIGHT, 1.0)
    eyes_open = max(0.0, 1.0 - (blink_l + blink_r) / 2.0)
    smile_l = blendshapes.get(SMILE_LEFT, 0.0)
    smile_r = blendshapes.get(SMILE_RIGHT, 0.0)
    smile = (smile_l + smile_r) / 2.0
    gaze_l = blendshapes.get(GAZE_LEFT, 0.0)
    gaze_r = blendshapes.get(GAZE_RIGHT, 0.0)
    gaze = max(0.0, 1.0 - (gaze_l + gaze_r) / 2.0)
    return PeakMomentScore(
        eyes_open_score=eyes_open,
        smile_score=smile,
        gaze_score=gaze,
        motion_peak_score=0.0,
        peak_type=PEAK_TYPE_PORTRAIT,
    )


def _score_member(member: Path) -> tuple[float, PeakMomentScore]:
    """Return (combined_score, PeakMomentScore) for a single burst member."""
    blendshapes = _detect_blendshapes(member)
    peak = _combine_blendshapes(blendshapes)
    is_eyes_open = peak.eyes_open_score >= EYE_OPEN_THRESHOLD
    is_smiling = peak.smile_score >= SMILE_THRESHOLD
    combined = (
        EYE_OPEN_WEIGHT * peak.eyes_open_score
        + SMILE_WEIGHT * peak.smile_score
        + GAZE_WEIGHT * peak.gaze_score
        + (BONUS_EYES_OPEN if is_eyes_open else 0.0)
        + (BONUS_SMILE if is_smiling else 0.0)
    )
    return combined, peak


def pick_winner(input: PeakPortraitInput) -> tuple[Path, PeakMomentScore]:
    """Return the burst member with the highest blendshape peak score."""
    if not input.burst_members:
        raise ValueError("burst_members must not be empty")
    best_path = input.burst_members[0]
    best_score, best_peak = _score_member(best_path)
    for member in input.burst_members[1:]:
        score, peak = _score_member(member)
        if score > best_score:
            best_score = score
            best_path = member
            best_peak = peak
    return best_path, best_peak
