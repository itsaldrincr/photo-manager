"""Shared models and pure decision logic for the blendshapes-vs-DeepFace eval.

No heavy imports (mediapipe/deepface) live here so both the lightweight
orchestrator and the subprocess runner can import it cheaply.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ExpressionBucket = Literal["happy", "neutral", "negative", "surprised", "unknown"]

# ---------------------------------------------------------------------------
# Blendshape decision thresholds (heuristic — mirrors how PORTRAIT_EAR_CLOSED_MAX
# etc. are hand-tuned constants in cull/config.py, not a calibrated classifier).
# ---------------------------------------------------------------------------

BLENDSHAPE_EYE_BLINK_CLOSED_MIN: float = 0.5
BLENDSHAPE_SMILE_HAPPY_MIN: float = 0.4
BLENDSHAPE_BROW_DOWN_NEGATIVE_MIN: float = 0.4
BLENDSHAPE_FROWN_NEGATIVE_MIN: float = 0.2
BLENDSHAPE_JAW_OPEN_SURPRISE_MIN: float = 0.3
BLENDSHAPE_BROW_INNER_UP_SURPRISE_MIN: float = 0.3

# ---------------------------------------------------------------------------
# Swap/no-swap rubric thresholds
# ---------------------------------------------------------------------------

SWAP_AGREEMENT_MIN: float = 0.90
SWAP_SPEEDUP_MIN: float = 1.5
SWAP_RSS_SAVINGS_MIN_MB: float = 200.0
MIN_AGREEMENT_SAMPLE_SIZE: int = 10

_DEEPFACE_BUCKETS: dict[str, ExpressionBucket] = {
    "happy": "happy",
    "neutral": "neutral",
    "sad": "negative",
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "surprise": "surprised",
}


def bucket_from_deepface_label(label: str) -> ExpressionBucket:
    """Map a raw DeepFace dominant_emotion label to a decision-relevant bucket."""
    return _DEEPFACE_BUCKETS.get(label.strip().lower(), "unknown")


class BlendshapeScores(BaseModel):
    """Named subset of the 52 MediaPipe blendshape scores used by the rubric."""

    eye_blink_left: float = 0.0
    eye_blink_right: float = 0.0
    mouth_smile_left: float = 0.0
    mouth_smile_right: float = 0.0
    brow_down_left: float = 0.0
    brow_down_right: float = 0.0
    mouth_frown_left: float = 0.0
    mouth_frown_right: float = 0.0
    jaw_open: float = 0.0
    brow_inner_up: float = 0.0


def blendshape_eyes_closed(scores: BlendshapeScores) -> bool:
    """Return True if the mean eye-blink blendshape score exceeds the closed threshold."""
    mean_blink = (scores.eye_blink_left + scores.eye_blink_right) / 2.0
    return mean_blink > BLENDSHAPE_EYE_BLINK_CLOSED_MIN


def bucket_from_blendshapes(scores: BlendshapeScores) -> ExpressionBucket:
    """Derive a decision-relevant expression bucket from blendshape scores."""
    if _is_surprised(scores):
        return "surprised"
    if _is_happy(scores):
        return "happy"
    if _is_negative(scores):
        return "negative"
    return "neutral"


def _is_happy(scores: BlendshapeScores) -> bool:
    """Return True if the mean smile score clears the happy threshold."""
    mean_smile = (scores.mouth_smile_left + scores.mouth_smile_right) / 2.0
    return mean_smile > BLENDSHAPE_SMILE_HAPPY_MIN


def _is_negative(scores: BlendshapeScores) -> bool:
    """Return True if brow-down and mouth-frown both clear their thresholds."""
    mean_brow_down = (scores.brow_down_left + scores.brow_down_right) / 2.0
    mean_frown = (scores.mouth_frown_left + scores.mouth_frown_right) / 2.0
    return (
        mean_brow_down > BLENDSHAPE_BROW_DOWN_NEGATIVE_MIN
        and mean_frown > BLENDSHAPE_FROWN_NEGATIVE_MIN
    )


def _is_surprised(scores: BlendshapeScores) -> bool:
    """Return True if jaw-open and brow-inner-up both clear their thresholds."""
    return (
        scores.jaw_open > BLENDSHAPE_JAW_OPEN_SURPRISE_MIN
        and scores.brow_inner_up > BLENDSHAPE_BROW_INNER_UP_SURPRISE_MIN
    )


# ---------------------------------------------------------------------------
# Stage I/O models (written/read as JSON between subprocess stages)
# ---------------------------------------------------------------------------


class StageTiming(BaseModel):
    """Load time and RSS footprint measured around one model's initialisation."""

    load_seconds: float
    rss_before_load_mb: float
    rss_after_load_mb: float

    @property
    def rss_delta_mb(self) -> float:
        """Return the RSS growth attributable to loading this model."""
        return round(self.rss_after_load_mb - self.rss_before_load_mb, 1)


class MediapipeReading(BaseModel):
    """One photo's MediaPipe face-landmarker + blendshapes result."""

    name: str
    path: str
    has_face: bool
    latency_seconds: float = 0.0
    ear_eyes_closed: bool | None = None
    blendshape_eyes_closed: bool | None = None
    expression_bucket: ExpressionBucket = "unknown"


class DeepfaceReading(BaseModel):
    """One photo's DeepFace emotion-analysis result."""

    name: str
    latency_seconds: float
    dominant_emotion: str
    expression_bucket: ExpressionBucket


class MediapipeStageOutput(BaseModel):
    """Full output of the MediaPipe blendshapes benchmark stage."""

    timing: StageTiming
    readings: list[MediapipeReading]


class DeepfaceStageOutput(BaseModel):
    """Full output of the DeepFace emotion benchmark stage."""

    timing: StageTiming
    readings: list[DeepfaceReading]


def list_image_paths(directory: Path) -> list[Path]:
    """Return sorted JPEG paths directly inside a directory."""
    suffixes = (".jpg", ".jpeg")
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )
