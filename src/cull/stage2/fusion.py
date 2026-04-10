"""Stage 2 score fusion — weighted composite and routing decision."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from cull.config import (
    GENRE_WEIGHTS,
    PRESET_QUALITY_POLICY,
    ROUTING_AMBIGUOUS_MIN,
    ROUTING_KEEPER_MIN,
    TASTE_RAMP_LABELS,
    CullConfig,
)
from cull.models import (
    CompositionScore,
    CropProposal,
    PortraitScores,
    ShootStatsScore,
    Stage2Result,
    SubjectBlurScore,
    TasteScore,
)

logger = logging.getLogger(__name__)

RoutingLabel = Literal["KEEPER", "AMBIGUOUS", "REJECT"]
SUBJECT_BLUR_NORM_DIVISOR: float = 1000.0


class IqaScores(BaseModel):
    """Raw per-metric scores fed into the fusion step."""

    photo_path: Path
    topiq: float
    laion_aesthetic: float
    clipiqa: float
    exposure: float
    composition: float | None = None
    composition_score: CompositionScore | None = None
    crop: CropProposal | None = None
    taste: float | None = None
    taste_score: TasteScore | None = None
    portrait: PortraitScores | None = None
    subject_blur: float | None = None
    subject_blur_score: SubjectBlurScore | None = None
    is_bokeh: bool = False
    tilt_penalty: float | None = None
    palette_outlier: float | None = None
    exposure_drift: float | None = None
    exif_anomaly: float | None = None
    scene_start_bonus: float | None = None


class FusionResult(BaseModel):
    """Stage2Result enriched with a routing decision."""

    stage2: Stage2Result
    routing: RoutingLabel


def _route(composite: float) -> RoutingLabel:
    """Map a composite score to a routing label using config thresholds."""
    if composite >= ROUTING_KEEPER_MIN:
        return "KEEPER"
    if composite >= ROUTING_AMBIGUOUS_MIN:
        return "AMBIGUOUS"
    return "REJECT"


COMPOSITE_PRECISION: int = 9
TILT_PENALTY_DEFAULT_WEIGHT: float = 0.0


def _tilt_term(scores: IqaScores, weights: dict[str, float]) -> float:
    """Return the tilt-penalty subtraction term, gated by weight presence."""
    if scores.tilt_penalty is None:
        return 0.0
    weight = weights.get("tilt_penalty", TILT_PENALTY_DEFAULT_WEIGHT)
    return weight * scores.tilt_penalty


def _taste_term(scores: IqaScores, weights: dict[str, float]) -> float:
    """Apply the ramped taste weight to the taste probability."""
    if scores.taste is None:
        return 0.0
    label_count = scores.taste_score.label_count_at_score if scores.taste_score else 0
    ramp = min(label_count / TASTE_RAMP_LABELS, 1.0)
    taste_weight = ramp * weights.get("taste", 0.0)
    return taste_weight * scores.taste


def _normalize_subject_blur(tenengrad: float) -> float:
    """Map raw Tenengrad value into [0.0, 1.0] for fusion compatibility."""
    return min(1.0, max(0.0, tenengrad / SUBJECT_BLUR_NORM_DIVISOR))


def _quality_policy(config: CullConfig) -> dict[str, float]:
    """Return the active preset quality policy, defaulting to general."""
    return PRESET_QUALITY_POLICY.get(config.preset, PRESET_QUALITY_POLICY["general"])


def _clamp_unit(value: float) -> float:
    """Clamp a scalar into the closed [0, 1] interval."""
    return min(1.0, max(0.0, value))


def _topiq_term(scores: IqaScores, config: CullConfig) -> float:
    """Blend global IQA with local subject sharpness, with optional bokeh relief."""
    if scores.subject_blur is None:
        return scores.topiq
    policy = _quality_policy(config)
    blend = _clamp_unit(policy.get("subject_blur_blend", 0.0))
    subject_score = _normalize_subject_blur(scores.subject_blur)
    blended = ((1.0 - blend) * scores.topiq) + (blend * subject_score)
    if scores.is_bokeh:
        blended += policy.get("bokeh_bonus", 0.0)
    return _clamp_unit(blended)


def _portrait_adjustment(scores: IqaScores, config: CullConfig) -> float:
    """Return preset-aware portrait quality adjustment for face-containing frames."""
    portrait = scores.portrait
    if portrait is None:
        return 0.0
    policy = _quality_policy(config)
    delta = 0.0
    eye_values = [
        sharpness
        for sharpness in (portrait.eye_sharpness_left, portrait.eye_sharpness_right)
        if sharpness is not None
    ]
    if eye_values:
        normalized_eye = _normalize_subject_blur(sum(eye_values) / len(eye_values))
        delta += policy.get("portrait_sharpness_bonus", 0.0) * normalized_eye
    if portrait.is_eyes_closed:
        delta -= policy.get("eyes_closed_penalty", 0.0)
    if portrait.is_face_occluded:
        delta -= policy.get("face_occlusion_penalty", 0.0)
    return delta


def _reducer_term(scores: IqaScores, weights: dict[str, float]) -> float:
    """Combine palette / exposure / EXIF / scene reducer scores into one delta."""
    palette = (scores.palette_outlier or 0.0) * weights.get("palette_outlier", 0.0)
    drift = (scores.exposure_drift or 0.0) * weights.get("exposure_drift", 0.0)
    anomaly = (scores.exif_anomaly or 0.0) * weights.get("exif_anomaly", 0.0)
    bonus = (scores.scene_start_bonus or 0.0) * weights.get("scene_start_bonus", 0.0)
    return bonus - palette - drift - anomaly


def _weighted_sum(scores: IqaScores, weights: dict[str, float]) -> float:
    """Compute weighted composite from global IQA, aesthetics, and reducers."""
    composition_value = scores.composition if scores.composition is not None else 0.0
    raw = (
        weights["topiq"] * scores.topiq
        + weights["laion_aesthetic"] * scores.laion_aesthetic
        + weights["clipiqa"] * scores.clipiqa
        + weights["exposure"] * scores.exposure
        + weights.get("composition", 0.0) * composition_value
        + _taste_term(scores, weights)
        - _tilt_term(scores, weights)
        + _reducer_term(scores, weights)
    )
    return round(raw, COMPOSITE_PRECISION)


def _build_stage2_result(scores: IqaScores, composite: float, preset: str) -> Stage2Result:
    """Build the Stage 2 result model from fused scores and preset."""
    return Stage2Result(
        photo_path=scores.photo_path,
        topiq=scores.topiq,
        laion_aesthetic=scores.laion_aesthetic,
        clipiqa=scores.clipiqa,
        composite=composite,
        portrait=scores.portrait,
        preset_used=preset,
        composition=scores.composition_score,
        crop=scores.crop,
        taste=scores.taste_score,
        subject_blur=scores.subject_blur_score,
    )


def _compute_clamped_composite(scores: IqaScores, config: CullConfig) -> float:
    """Compute the final rounded composite after portrait adjustment and clamping."""
    weights = GENRE_WEIGHTS[config.preset]
    base = _weighted_sum(scores, weights)
    adjusted = base + _portrait_adjustment(scores, config)
    return round(_clamp_unit(adjusted), COMPOSITE_PRECISION)


def compute_composite(scores: IqaScores, config: CullConfig) -> FusionResult:
    """Fuse per-metric scores into a composite and route the image."""
    working_scores = scores.model_copy(update={"topiq": _topiq_term(scores, config)})
    composite = _compute_clamped_composite(working_scores, config)
    logger.debug(
        "Composite for %s: %.4f (preset=%s)", scores.photo_path, composite, config.preset
    )
    stage2 = _build_stage2_result(scores, composite, config.preset)
    return FusionResult(stage2=stage2, routing=_route(composite))


class ReducerPatchInput(BaseModel):
    """Input bundle for patch_reducer_scores — fusion map plus per-photo reducer scores."""

    model_config = {"arbitrary_types_allowed": True}

    fusion_results: dict[str, FusionResult]
    reducer_scores: dict[str, ShootStatsScore]
    config: CullConfig


def _stage2_to_iqa(stage2: Stage2Result, reducer: ShootStatsScore) -> IqaScores:
    """Rebuild an IqaScores from a Stage2Result + the new reducer scores."""
    return IqaScores(
        photo_path=stage2.photo_path,
        topiq=stage2.topiq,
        laion_aesthetic=stage2.laion_aesthetic,
        clipiqa=stage2.clipiqa,
        exposure=stage2.composite,
        composition=stage2.composition.composite if stage2.composition else None,
        composition_score=stage2.composition,
        crop=stage2.crop,
        taste=stage2.taste.probability if stage2.taste else None,
        taste_score=stage2.taste,
        portrait=stage2.portrait,
        subject_blur=stage2.subject_blur.tenengrad if stage2.subject_blur else None,
        subject_blur_score=stage2.subject_blur,
        palette_outlier=reducer.palette_outlier_score,
        exposure_drift=reducer.exposure_drift_score,
        exif_anomaly=reducer.exif_anomaly_score,
        scene_start_bonus=reducer.scene_start_bonus,
    )


class _PatchOneInput(BaseModel):
    """Bundle for patching one FusionResult with its reducer scores."""

    model_config = {"arbitrary_types_allowed": True}

    fusion: FusionResult
    reducer: ShootStatsScore
    config: CullConfig


def _patch_one(patch_one_in: _PatchOneInput) -> None:
    """Mutate one FusionResult so its Stage2Result carries the reducer scores + new composite."""
    patch_one_in.fusion.stage2.shoot_stats = patch_one_in.reducer
    iqa = _stage2_to_iqa(patch_one_in.fusion.stage2, patch_one_in.reducer)
    updated = compute_composite(iqa, patch_one_in.config)
    patch_one_in.fusion.stage2.composite = updated.stage2.composite
    patch_one_in.fusion.routing = updated.routing


def patch_reducer_scores(patch_in: ReducerPatchInput) -> None:
    """Write reducer scores into every fusion result and re-route in place."""
    for key, reducer in patch_in.reducer_scores.items():
        fusion = patch_in.fusion_results.get(key)
        if fusion is None:
            continue
        _patch_one(_PatchOneInput(
            fusion=fusion, reducer=reducer, config=patch_in.config,
        ))
