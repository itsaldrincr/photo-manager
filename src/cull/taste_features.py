"""Canonical taste-model feature row — single source of truth for train + serve.

Both taste_trainer.py (fitting from OverrideEntry logs) and
_pipeline/stage2_scoring.py (scoring a photo mid-Stage-2) must build
byte-identical rows for the same logical photo, or a profile trained on one
shape silently mis-scores on the other. This module is the only place the
row layout and missing-value transforms are defined; both call sites import
build_taste_feature_row and never construct a row by hand.

Row layout: sorted stage1_scores values, then five scalars reconstructible
from a logged OverrideEntry — stage2_composite, composition composite,
thirds_alignment, negative_space_balance, subject_blur tenengrad (log-scaled).
No CLIP embedding: the 602 real override records never carried one, so a
profile trained on it would be untestable against real history.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from cull.models import Stage1Result

# Stage1Result fields flattened into the leading part of the row. Shared by
# override_log.py (log-time extraction) and stage2_scoring.py (serve-time
# extraction) so both read the exact same 10 keys off the same object shape.
_STAGE1_BLUR_KEYS: tuple[str, ...] = ("tenengrad", "fft_ratio")
_STAGE1_EXPOSURE_KEYS: tuple[str, ...] = (
    "dr_score", "clipping_highlight", "clipping_shadow", "midtone_pct", "color_cast_score",
)

# Neutral fallbacks for missing scalars (older override records predating a
# field, or a photo with no detected subject) — keep row length stable
# without biasing the classifier toward either class.
NEUTRAL_COMPOSITE: float = 0.5
NEUTRAL_COMPOSITION_METRIC: float = 0.5
NEUTRAL_SUBJECT_BLUR: float = 0.5

# Tenengrad is unbounded and right-skewed; log1p compresses outliers so a
# handful of very sharp frames don't dominate the row's scale before the
# trainer's StandardScaler normalizes it further.
SUBJECT_BLUR_LOG_DIVISOR: float = 10.0


class TasteFeatureInputs(BaseModel):
    """Canonical scalar inputs consumed by the taste feature-row builder."""

    stage1_scores: dict[str, float]
    stage2_composite: float | None = None
    composition_composite: float | None = None
    thirds_alignment: float | None = None
    negative_space_balance: float | None = None
    subject_blur_tenengrad: float | None = None


def _sanitize(value: float | None, neutral: float) -> float:
    """Return value if present, else the neutral fallback."""
    return neutral if value is None else float(value)


def _normalize_subject_blur(tenengrad: float | None) -> float:
    """Log-compress a raw Tenengrad value into a small positive range, or neutral if absent."""
    if tenengrad is None:
        return NEUTRAL_SUBJECT_BLUR
    return float(np.log1p(max(0.0, tenengrad)) / SUBJECT_BLUR_LOG_DIVISOR)


def _stage1_row(stage1_scores: dict[str, float]) -> list[float]:
    """Flatten a stage1_scores dict into a stable, sorted-key vector."""
    return [float(stage1_scores[key]) for key in sorted(stage1_scores)]


def flatten_stage1_scores(stage1: Stage1Result | None) -> dict[str, float]:
    """Flatten a Stage1Result's blur + exposure + noise + geometry into a flat dict."""
    if stage1 is None:
        return {}
    scores: dict[str, float] = {}
    for key in _STAGE1_BLUR_KEYS:
        value = getattr(stage1.blur, key, None)
        if value is not None:
            scores[key] = float(value)
    for key in _STAGE1_EXPOSURE_KEYS:
        value = getattr(stage1.exposure, key, None)
        if value is not None:
            scores[key] = float(value)
    scores["noise_score"] = float(stage1.noise_score)
    if stage1.geometry is not None:
        scores["tilt_degrees"] = float(stage1.geometry.tilt_degrees)
        scores["keystone_degrees"] = float(stage1.geometry.keystone_degrees)
    return scores


def build_taste_feature_row(inputs: TasteFeatureInputs) -> np.ndarray:
    """Build the canonical taste feature row shared by trainer and inference."""
    stage2_row = [
        _sanitize(inputs.stage2_composite, NEUTRAL_COMPOSITE),
        _sanitize(inputs.composition_composite, NEUTRAL_COMPOSITION_METRIC),
        _sanitize(inputs.thirds_alignment, NEUTRAL_COMPOSITION_METRIC),
        _sanitize(inputs.negative_space_balance, NEUTRAL_COMPOSITION_METRIC),
        _normalize_subject_blur(inputs.subject_blur_tenengrad),
    ]
    row = _stage1_row(inputs.stage1_scores) + stage2_row
    return np.asarray(row, dtype=np.float32)
