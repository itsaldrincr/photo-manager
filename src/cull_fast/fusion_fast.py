"""Fast-mode fusion bridge: IqaScores builder + weight reallocation + composite.

Spec §5C ambiguity (`keep exposure as-is` vs `renormalize all three to sum to 1.0`)
was resolved as RENORMALIZE-ALL-THREE by the architect. All three of
topiq + laion_aesthetic + exposure are proportionally scaled so they sum to 1.0.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from cull.config import (
    GENRE_WEIGHTS,
    PRESET_QUALITY_POLICY,
    ROUTING_AMBIGUOUS_MIN,
    ROUTING_KEEPER_MIN,
)
from cull.stage2.fusion import (
    COMPOSITE_PRECISION,
    FusionResult,
    IqaScores,
    RoutingLabel,
)
from cull.models import Stage2Result
from cull_fast.musiq import MusiQScorePair

logger = logging.getLogger(__name__)

FAST_CLIPIQA_SENTINEL: float = 0.0
FAST_WEIGHT_SUM_TARGET: float = 1.0


class _FusionFastInput(BaseModel):
    """Input bundle for fast-mode fusion."""

    musiq_scores: MusiQScorePair
    exposure_score: float


class _FastComputeInput(BaseModel):
    """Input bundle for computing a fast-mode composite + routing for one photo."""

    scores: IqaScores
    weights: dict[str, float]
    preset: str


class _FastBuildInput(BaseModel):
    """Input bundle for building a FusionResult from precomputed composite + preset."""

    scores: IqaScores
    composite: float
    preset: str


class _FastCompositeValueInput(BaseModel):
    """Input bundle for raw fast composite calculation."""

    scores: IqaScores
    weights: dict[str, float]
    bokeh_bonus: float


def _rescale_preset_weights(preset: str) -> dict[str, float]:
    """Return fast-mode weights with core quality terms renormalized to 1.0."""
    if preset not in GENRE_WEIGHTS:
        raise ValueError(f"Unknown preset: {preset!r}")
    original = GENRE_WEIGHTS[preset]
    base_sum = original["topiq"] + original["laion_aesthetic"] + original["exposure"]
    if base_sum == 0.0:
        raise ValueError(
            f"Preset {preset!r} has zero weight for topiq+laion_aesthetic+exposure "
            f"— cannot renormalize"
        )
    scale = FAST_WEIGHT_SUM_TARGET / base_sum
    return {
        "topiq": original["topiq"] * scale,
        "laion_aesthetic": original["laion_aesthetic"] * scale,
        "exposure": original["exposure"] * scale,
        "clipiqa": 0.0,
        "composition": 0.0,
        "taste": 0.0,
        "tilt_penalty": original.get("tilt_penalty", 0.0),
    }


def build_iqa_from_musiq(fast_in: _FusionFastInput) -> IqaScores:
    """Build an IqaScores from a MUSIQ pair + exposure score."""
    return IqaScores(
        photo_path=fast_in.musiq_scores.photo_path,
        topiq=fast_in.musiq_scores.technical,
        laion_aesthetic=fast_in.musiq_scores.aesthetic,
        clipiqa=FAST_CLIPIQA_SENTINEL,
        exposure=fast_in.exposure_score,
    )


def _fast_composite_value(value_in: _FastCompositeValueInput) -> float:
    """Compute the fast-mode composite with light geometry and preset bokeh relief."""
    raw = (
        value_in.weights["topiq"] * value_in.scores.topiq
        + value_in.weights["laion_aesthetic"] * value_in.scores.laion_aesthetic
        + value_in.weights["exposure"] * value_in.scores.exposure
        - value_in.weights["tilt_penalty"] * (value_in.scores.tilt_penalty or 0.0)
        + value_in.bokeh_bonus
    )
    return round(raw, COMPOSITE_PRECISION)


def _fast_route(composite: float) -> RoutingLabel:
    """Map composite to a routing label using config thresholds."""
    if composite >= ROUTING_KEEPER_MIN:
        return "KEEPER"
    if composite >= ROUTING_AMBIGUOUS_MIN:
        return "AMBIGUOUS"
    return "REJECT"


def _fast_build_result(build_in: _FastBuildInput) -> FusionResult:
    """Build a FusionResult from scores + composite + preset."""
    stage2 = Stage2Result(
        photo_path=build_in.scores.photo_path,
        topiq=build_in.scores.topiq,
        laion_aesthetic=build_in.scores.laion_aesthetic,
        clipiqa=build_in.scores.clipiqa,
        composite=build_in.composite,
        preset_used=build_in.preset,
    )
    return FusionResult(stage2=stage2, routing=_fast_route(build_in.composite))


def _compute_composite_fast(compute_in: _FastComputeInput) -> FusionResult:
    """Fuse fast-mode scores into a composite and route the image."""
    policy = PRESET_QUALITY_POLICY.get(
        compute_in.preset, PRESET_QUALITY_POLICY["general"]
    )
    bokeh_bonus = policy.get("bokeh_bonus", 0.0) if compute_in.scores.is_bokeh else 0.0
    value_in = _FastCompositeValueInput(
        scores=compute_in.scores,
        weights=compute_in.weights,
        bokeh_bonus=bokeh_bonus,
    )
    composite = _fast_composite_value(value_in)
    composite = max(0.0, min(1.0, round(composite, COMPOSITE_PRECISION)))
    logger.debug("Fast composite for %s: %.4f", compute_in.scores.photo_path, composite)
    build_in = _FastBuildInput(
        scores=compute_in.scores, composite=composite, preset=compute_in.preset,
    )
    return _fast_build_result(build_in)
