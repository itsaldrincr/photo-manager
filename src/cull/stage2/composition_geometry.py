"""Composition geometry — rule-of-thirds, edge clearance, and balance metrics."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from cull.saliency import SaliencyResult

THIRDS_FRACTIONS: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
# Max fractional distance to a thirds intersection used to normalise the score.
THIRDS_NORMALIZER: float = 0.5
EDGE_CLEARANCE_NORMALIZER: float = 0.5
SYMMETRY_NORMALIZER: float = 0.5
_COMPOSITE_DECIMAL_PLACES: int = 6
COMPOSITION_WEIGHT_THIRDS: float = 0.35
COMPOSITION_WEIGHT_EDGES: float = 0.25
COMPOSITION_WEIGHT_BALANCE: float = 0.20
COMPOSITION_WEIGHT_TOPIQ: float = 0.20


class _GeometryMetrics(BaseModel):
    """Intermediate geometric measurements derived from saliency."""

    thirds_alignment: float
    edge_clearance: float
    negative_space_balance: float


def _compute_geometry_metrics(
    saliency: SaliencyResult, frame_size: tuple[int, int]
) -> _GeometryMetrics:
    """Bundle thirds, edge, and balance metrics from saliency on the given frame."""
    heatmap_h, heatmap_w = saliency.heatmap.shape
    return _GeometryMetrics(
        thirds_alignment=_thirds_alignment(saliency.peak_xy, (heatmap_w, heatmap_h)),
        edge_clearance=_edge_clearance(saliency.bbox, (heatmap_w, heatmap_h)),
        negative_space_balance=_negative_space_balance(saliency.heatmap),
    )


def _thirds_alignment(peak_xy: tuple[float, float], frame: tuple[int, int]) -> float:
    """Score how close the saliency peak is to a rule-of-thirds intersection."""
    width, height = frame
    if width <= 0 or height <= 0:
        return 0.0
    px = float(peak_xy[0])
    py = float(peak_xy[1])
    distances = [
        ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
        for tx in THIRDS_FRACTIONS
        for ty in THIRDS_FRACTIONS
    ]
    nearest = min(distances)
    return max(0.0, 1.0 - nearest / THIRDS_NORMALIZER)


def _edge_clearance(
    bbox: tuple[float, float, float, float], frame: tuple[int, int]
) -> float:
    """Score how far the saliency bbox sits from the frame edges."""
    width, height = frame
    if width <= 0 or height <= 0:
        return 0.0
    left = float(bbox[0])
    top = float(bbox[1])
    right = 1.0 - float(bbox[2])
    bottom = 1.0 - float(bbox[3])
    min_margin = min(left, top, right, bottom)
    return max(0.0, min(1.0, min_margin / EDGE_CLEARANCE_NORMALIZER))


def _negative_space_balance(heatmap: np.ndarray) -> float:
    """Score symmetry of saliency mass between left/right and top/bottom halves."""
    height, width = heatmap.shape
    if height == 0 or width == 0:
        return 0.0
    half_w = width // 2
    half_h = height // 2
    left_mass = float(heatmap[:, :half_w].sum())
    right_mass = float(heatmap[:, half_w:].sum())
    top_mass = float(heatmap[:half_h, :].sum())
    bottom_mass = float(heatmap[half_h:, :].sum())
    horizontal = _balance_ratio(left_mass, right_mass)
    vertical = _balance_ratio(top_mass, bottom_mass)
    return (horizontal + vertical) / 2.0


def _balance_ratio(side_a: float, side_b: float) -> float:
    """Return symmetry ratio in [0, 1] between two non-negative masses."""
    total = side_a + side_b
    if total <= 0.0:
        return 1.0
    diff = abs(side_a - side_b) / total
    return max(0.0, 1.0 - diff)


def _composite_score(metrics: _GeometryMetrics, topiq_iaa: float) -> float:
    """Combine geometry metrics and topiq_iaa into a single composition composite."""
    raw = (
        COMPOSITION_WEIGHT_THIRDS * metrics.thirds_alignment
        + COMPOSITION_WEIGHT_EDGES * metrics.edge_clearance
        + COMPOSITION_WEIGHT_BALANCE * metrics.negative_space_balance
        + COMPOSITION_WEIGHT_TOPIQ * topiq_iaa
    )
    return round(raw, _COMPOSITE_DECIMAL_PLACES)
