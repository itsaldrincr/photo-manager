"""Shared models and pure decision logic for the py-feat / EmotiEffLib eval.

No heavy imports (torch/feat/emotiefflib) live here so both the lightweight
orchestrator and the subprocess runner can import it cheaply. Mirrors the
structure of portrait_eval_models.py.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from portrait_eval_models import StageTiming, list_image_paths  # noqa: F401

VaQuadrant = Literal[
    "pos_valence_pos_arousal",
    "pos_valence_neg_arousal",
    "neg_valence_pos_arousal",
    "neg_valence_neg_arousal",
]

# ---------------------------------------------------------------------------
# AU / VA decision thresholds (heuristic — mirrors BLENDSHAPE_* constants in
# portrait_eval_models.py, not a calibrated classifier).
# ---------------------------------------------------------------------------

AU43_EYES_CLOSED_MIN: float = 0.5

# ---------------------------------------------------------------------------
# REPLACE/KEEP rubric thresholds (per the task's decision rubric)
# ---------------------------------------------------------------------------

REPLACE_SPEEDUP_MIN: float = 5.0
REPLACE_RSS_SAVINGS_MIN_MB: float = 2000.0
REPLACE_AU43_AGREEMENT_MIN: float = 0.85


def au43_eyes_closed(intensity: float) -> bool:
    """Return True if the AU43 (eyes-closed) intensity clears the closed threshold."""
    return intensity > AU43_EYES_CLOSED_MIN


def va_quadrant(valence: float, arousal: float) -> VaQuadrant:
    """Bucket a valence-arousal pair into one of four dimensional quadrants."""
    if valence >= 0.0 and arousal >= 0.0:
        return "pos_valence_pos_arousal"
    if valence >= 0.0 and arousal < 0.0:
        return "pos_valence_neg_arousal"
    if valence < 0.0 and arousal >= 0.0:
        return "neg_valence_pos_arousal"
    return "neg_valence_neg_arousal"


# ---------------------------------------------------------------------------
# Stage I/O models (written/read as JSON between subprocess stages)
# ---------------------------------------------------------------------------


class PyFeatReading(BaseModel):
    """One photo's py-feat Detectorv2 result for its highest-confidence face."""

    name: str
    has_face: bool
    latency_seconds: float = 0.0
    face_score: float | None = None
    action_units: dict[str, float] = Field(default_factory=dict)
    dominant_emotion: str | None = None
    valence: float | None = None
    arousal: float | None = None
    au43_eyes_closed: bool | None = None
    va_quadrant: VaQuadrant | None = None


class EmotiEfflibReading(BaseModel):
    """One photo's EmotiEffLib (enet_b0_8_va_mtl) result for its cropped face."""

    name: str
    has_face: bool
    latency_seconds: float = 0.0
    dominant_emotion: str | None = None
    valence: float | None = None
    arousal: float | None = None
    va_quadrant: VaQuadrant | None = None


class PyFeatStageOutput(BaseModel):
    """Full output of the py-feat Detectorv2 benchmark stage."""

    timing: StageTiming
    readings: list[PyFeatReading]


class EmotiEfflibStageOutput(BaseModel):
    """Full output of the EmotiEffLib benchmark stage."""

    timing: StageTiming
    readings: list[EmotiEfflibReading]
