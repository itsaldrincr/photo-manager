"""Stage 2 cross-photo reducer — palette / exposure / EXIF outliers and scene boundaries."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from cull.config import (
    BURST_GAP_DEFAULT_SECONDS,
    EXIF_ANOMALY_SIGMA,
    EXPOSURE_DRIFT_SIGMA,
    PALETTE_OUTLIER_SIGMA,
)
from cull.models import ShootStatsScore

logger = logging.getLogger(__name__)

SCENE_BOUNDARY_GAP_MULTIPLIER: float = 4.0
SCENE_BOUNDARY_BONUS: float = 1.0
OUTLIER_FALLBACK_SCORE: float = 0.0
OUTLIER_FULL_SCORE: float = 1.0
DISPERSION_FLOOR: float = 1e-6
EXIF_ANOMALY_KEYS: tuple[str, ...] = ("iso", "shutter", "aperture", "focal_length_mm")


class ShootStatsBundle(BaseModel):
    """Module-private rolling statistics for the current shoot."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    palette_median_lab: np.ndarray
    palette_max_distance: float
    exposure_median: float
    exposure_max_distance: float
    exif_modes: dict[str, Any]
    scene_boundaries: list[int]


class ShootStatsInput(BaseModel):
    """Public input bundle for compute()."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage2_results: list[Any]
    stage1_results: list[Any]


def _palette_lab_for_result(result: Any) -> np.ndarray:
    """Extract a 3-vector LAB centroid from a Stage 2 result; zero on miss."""
    palette = getattr(result, "palette_lab", None)
    if palette is None:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(palette, dtype=np.float32)


def _exposure_value_for_s1(s1: Any) -> float:
    """Pull the dynamic-range proxy as the exposure scalar for drift maths."""
    return float(s1.exposure.dr_score)


def _exif_dict_for_s1(s1: Any) -> dict[str, Any]:
    """Return a dict of EXIF-derived fields, falling back to empty when absent."""
    raw = getattr(s1, "exif", None)
    if raw is None:
        return {}
    return {key: getattr(raw, key, None) for key in EXIF_ANOMALY_KEYS}


def _max_palette_distance(palette_rows: np.ndarray, median_lab: np.ndarray) -> float:
    """Maximum per-photo LAB distance from the shoot median — scale for outlier ratio."""
    if palette_rows.shape[0] == 0:
        return 0.0
    distances = np.linalg.norm(palette_rows - median_lab, axis=1)
    return float(np.max(distances))


def _max_scalar_distance(values: np.ndarray, median: float) -> float:
    """Maximum absolute distance of any scalar from the shoot median."""
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values - median)))


def _modes_for_keys(exif_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the most common value per EXIF key across the shoot."""
    modes: dict[str, Any] = {}
    for key in EXIF_ANOMALY_KEYS:
        values = [d.get(key) for d in exif_dicts if d.get(key) is not None]
        if not values:
            modes[key] = None
            continue
        counts: dict[Any, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        modes[key] = max(counts.items(), key=lambda kv: kv[1])[0]
    return modes


def _build_bundle(input_in: ShootStatsInput) -> ShootStatsBundle:
    """Compute palette / exposure / EXIF / scene aggregates for the entire shoot."""
    palette_rows = np.stack(
        [_palette_lab_for_result(r) for r in input_in.stage2_results], axis=0
    )
    exposure_values = np.asarray(
        [_exposure_value_for_s1(s1) for s1 in input_in.stage1_results], dtype=np.float32
    )
    median_lab = np.median(palette_rows, axis=0)
    exposure_median = float(np.median(exposure_values))
    exif_dicts = [_exif_dict_for_s1(s1) for s1 in input_in.stage1_results]
    return ShootStatsBundle(
        palette_median_lab=median_lab,
        palette_max_distance=_max_palette_distance(palette_rows, median_lab),
        exposure_median=exposure_median,
        exposure_max_distance=_max_scalar_distance(exposure_values, exposure_median),
        exif_modes=_modes_for_keys(exif_dicts),
        scene_boundaries=_detect_scene_boundaries(input_in.stage1_results),
    )


class _OutlierInput(BaseModel):
    """Bundle for normalising one photo's deviation against shoot scale."""

    distance: float
    scale: float
    sigma: float


def _normalised_outlier(outlier_in: _OutlierInput) -> float:
    """Map (distance / scale) * sigma to a [0, 1] outlier score with saturation."""
    if outlier_in.scale <= DISPERSION_FLOOR:
        return OUTLIER_FALLBACK_SCORE
    ratio = outlier_in.distance / outlier_in.scale
    if ratio <= 0.0:
        return OUTLIER_FALLBACK_SCORE
    return float(min(OUTLIER_FULL_SCORE, ratio * outlier_in.sigma))


def _compute_palette_outlier(result: Any, bundle: ShootStatsBundle) -> float:
    """Score how far one photo's palette LAB centroid lies from the shoot median."""
    lab = _palette_lab_for_result(result)
    distance = float(np.linalg.norm(lab - bundle.palette_median_lab))
    return _normalised_outlier(_OutlierInput(
        distance=distance, scale=bundle.palette_max_distance, sigma=PALETTE_OUTLIER_SIGMA,
    ))


def _compute_exposure_drift(s1: Any, bundle: ShootStatsBundle) -> float:
    """Score how far one photo's exposure scalar drifts from the shoot median."""
    distance = abs(_exposure_value_for_s1(s1) - bundle.exposure_median)
    return _normalised_outlier(_OutlierInput(
        distance=distance, scale=bundle.exposure_max_distance, sigma=EXPOSURE_DRIFT_SIGMA,
    ))


def _compute_exif_anomaly(s1: Any, bundle: ShootStatsBundle) -> float:
    """Score the fraction of EXIF keys whose value differs from the shoot mode."""
    exif = _exif_dict_for_s1(s1)
    if not bundle.exif_modes:
        return OUTLIER_FALLBACK_SCORE
    mismatches = 0
    valid_keys = 0
    for key, mode_value in bundle.exif_modes.items():
        if mode_value is None:
            continue
        valid_keys += 1
        if exif.get(key) != mode_value:
            mismatches += 1
    if valid_keys == 0:
        return OUTLIER_FALLBACK_SCORE
    fraction = mismatches / valid_keys
    return float(min(OUTLIER_FULL_SCORE, fraction * (EXIF_ANOMALY_SIGMA / valid_keys)))


def _capture_seconds(s1: Any) -> float | None:
    """Return the capture timestamp as POSIX seconds, or None when missing."""
    raw = getattr(s1, "capture_time", None)
    if raw is None:
        return None
    if hasattr(raw, "timestamp"):
        return float(raw.timestamp())
    return float(raw)


def _detect_scene_boundaries(stage1_results: list[Any]) -> list[int]:
    """Mark photo indices that follow a temporal gap larger than the burst gap."""
    boundaries: list[int] = [0] if stage1_results else []
    threshold = BURST_GAP_DEFAULT_SECONDS * SCENE_BOUNDARY_GAP_MULTIPLIER
    previous: float | None = None
    for index, s1 in enumerate(stage1_results):
        current = _capture_seconds(s1)
        if current is None:
            previous = current
            continue
        if previous is not None and (current - previous) > threshold and index not in boundaries:
            boundaries.append(index)
        previous = current
    return boundaries


def _scene_id_for_index(index: int, boundaries: list[int]) -> int:
    """Return the scene ID for a photo by counting boundaries up to its index."""
    scene_id = 0
    for boundary in boundaries:
        if boundary <= index:
            scene_id = boundaries.index(boundary)
    return scene_id


def _bonus_for_scene_start(index: int, boundaries: list[int]) -> float:
    """Return SCENE_BOUNDARY_BONUS when this index sits at a scene boundary."""
    if index in boundaries and index != 0:
        return SCENE_BOUNDARY_BONUS
    return OUTLIER_FALLBACK_SCORE


def _build_score(index: int, ctx: "_ScoreCtx") -> ShootStatsScore:
    """Assemble a ShootStatsScore for one photo at the given index."""
    s1 = ctx.input_in.stage1_results[index]
    result = ctx.input_in.stage2_results[index]
    return ShootStatsScore(
        palette_outlier_score=_compute_palette_outlier(result, ctx.bundle),
        exposure_drift_score=_compute_exposure_drift(s1, ctx.bundle),
        exif_anomaly_score=_compute_exif_anomaly(s1, ctx.bundle),
        scene_start_bonus=_bonus_for_scene_start(index, ctx.bundle.scene_boundaries),
        scene_id=_scene_id_for_index(index, ctx.bundle.scene_boundaries),
    )


class _ScoreCtx(BaseModel):
    """Internal bundle binding the input and computed bundle for per-photo scoring."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_in: ShootStatsInput
    bundle: ShootStatsBundle


def compute(input_in: ShootStatsInput) -> dict[str, ShootStatsScore]:
    """Compute per-photo shoot-statistics scores keyed by str(photo_path)."""
    if not input_in.stage1_results:
        return {}
    bundle = _build_bundle(input_in)
    ctx = _ScoreCtx(input_in=input_in, bundle=bundle)
    return {
        str(input_in.stage2_results[i].photo_path): _build_score(i, ctx)
        for i in range(len(input_in.stage1_results))
    }
