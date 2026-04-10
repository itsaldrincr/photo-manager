"""Stage 1 geometry — LSD-based horizon tilt and vertical keystone detection."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from cull.models import GeometryScore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

MIN_LINES_FOR_RANSAC: int = 4
RANSAC_ITERATIONS: int = 50
RANSAC_INLIER_TOLERANCE_DEG: float = 2.0
HORIZON_ANGLE_BAND_DEG: float = 30.0
VERTICAL_ANGLE_BAND_DEG: float = 30.0
DEGREES_PER_RADIAN: float = 180.0 / np.pi
RIGHT_ANGLE_DEG: float = 90.0
STRAIGHT_ANGLE_DEG: float = 180.0
LSD_GRAYSCALE_FLAG: int = 0
EMPTY_TILT_DEG: float = 0.0
EMPTY_KEYSTONE_DEG: float = 0.0
EMPTY_CONFIDENCE: float = 0.0


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class GeometryResult(BaseModel):
    """Complete geometry assessment result for one image."""

    model_config = ConfigDict(frozen=True)

    path: Path
    scores: GeometryScore


class _LineGroups(BaseModel):
    """Horizontal and vertical angle samples extracted from line segments."""

    horizontal_angles: list[float]
    vertical_angles: list[float]


# ---------------------------------------------------------------------------
# Empty / failure result helpers
# ---------------------------------------------------------------------------


def _empty_scores() -> GeometryScore:
    """Return zeroed GeometryScore for empty/failed detections."""
    return GeometryScore(
        tilt_degrees=EMPTY_TILT_DEG,
        keystone_degrees=EMPTY_KEYSTONE_DEG,
        confidence=EMPTY_CONFIDENCE,
        has_horizon=False,
        has_verticals=False,
    )


# ---------------------------------------------------------------------------
# Private helpers (≤ 20 LOC each, ≤ 2 params)
# ---------------------------------------------------------------------------


def _load_gray(image_path: Path) -> np.ndarray | None:
    """Load image from disk as grayscale; return None on failure."""
    raw = cv2.imread(str(image_path), LSD_GRAYSCALE_FLAG)
    if raw is None:
        logger.debug("geometry: cannot read %s", image_path)
        return None
    return raw


def _detect_lines(gray: np.ndarray) -> np.ndarray:
    """Run LSD on a grayscale image and return Nx4 line endpoints."""
    detector = cv2.createLineSegmentDetector()
    detected = detector.detect(gray)[0]
    if detected is None:
        return np.empty((0, 4), dtype=np.float32)
    return detected.reshape(-1, 4)


def _line_angle_deg(segment: np.ndarray) -> float:
    """Return signed angle of one line segment in degrees from horizontal."""
    x1, y1, x2, y2 = segment
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def _normalize_horizontal(angle_deg: float) -> float:
    """Fold angle into the [-90, 90) horizontal range."""
    folded = ((angle_deg + RIGHT_ANGLE_DEG) % STRAIGHT_ANGLE_DEG) - RIGHT_ANGLE_DEG
    return float(folded)


def _normalize_vertical(angle_deg: float) -> float:
    """Return angular deviation from vertical (90deg) in [-90, 90)."""
    folded = _normalize_horizontal(angle_deg)
    return float(folded - RIGHT_ANGLE_DEG if folded >= 0 else folded + RIGHT_ANGLE_DEG)


def _classify_segment(segment: np.ndarray, groups: _LineGroups) -> None:
    """Classify one segment into horizontal or vertical bucket by angle band."""
    horizontal = _normalize_horizontal(_line_angle_deg(segment))
    if abs(horizontal) <= HORIZON_ANGLE_BAND_DEG:
        groups.horizontal_angles.append(horizontal)
        return
    vertical = _normalize_vertical(_line_angle_deg(segment))
    if abs(vertical) <= VERTICAL_ANGLE_BAND_DEG:
        groups.vertical_angles.append(vertical)


def _group_lines(lines: np.ndarray) -> _LineGroups:
    """Split detected lines into horizontal and vertical bands."""
    groups = _LineGroups(horizontal_angles=[], vertical_angles=[])
    for segment in lines:
        _classify_segment(segment, groups)
    return groups


def _ransac_horizon(angles: list[float]) -> float:
    """Return RANSAC consensus tilt angle in degrees from a horizontal sample."""
    if len(angles) < MIN_LINES_FOR_RANSAC:
        return EMPTY_TILT_DEG
    arr = np.asarray(angles, dtype=np.float64)
    rng = np.random.default_rng(seed=arr.size)
    best_count = 0
    best_angle = 0.0
    for _ in range(RANSAC_ITERATIONS):
        candidate = float(rng.choice(arr))
        inliers = arr[np.abs(arr - candidate) <= RANSAC_INLIER_TOLERANCE_DEG]
        if inliers.size > best_count:
            best_count = int(inliers.size)
            best_angle = float(np.median(inliers))
    return best_angle


def _vertical_angle(angles: list[float]) -> float:
    """Return median deviation-from-vertical in degrees, or 0 if too few lines."""
    if len(angles) < MIN_LINES_FOR_RANSAC:
        return EMPTY_KEYSTONE_DEG
    return float(np.median(np.asarray(angles, dtype=np.float64)))


def _confidence_from_groups(groups: _LineGroups) -> float:
    """Compute a [0,1] confidence proportional to evidence size."""
    total = len(groups.horizontal_angles) + len(groups.vertical_angles)
    if total == 0:
        return EMPTY_CONFIDENCE
    return float(min(1.0, total / (MIN_LINES_FOR_RANSAC * 4.0)))


def _scores_from_groups(groups: _LineGroups) -> GeometryScore:
    """Build a GeometryScore from horizontal/vertical line groupings."""
    has_horizon = len(groups.horizontal_angles) >= MIN_LINES_FOR_RANSAC
    has_verticals = len(groups.vertical_angles) >= MIN_LINES_FOR_RANSAC
    return GeometryScore(
        tilt_degrees=_ransac_horizon(groups.horizontal_angles),
        keystone_degrees=_vertical_angle(groups.vertical_angles),
        confidence=_confidence_from_groups(groups),
        has_horizon=has_horizon,
        has_verticals=has_verticals,
    )


# ---------------------------------------------------------------------------
# Public top-level entry (picklable, no closures)
# ---------------------------------------------------------------------------


def assess_geometry(image_path: Path) -> GeometryResult:
    """Assess geometry: detect tilt and vertical keystone via LSD + RANSAC."""
    logger.debug("assess_geometry: %s", image_path)
    gray = _load_gray(image_path)
    if gray is None:
        return GeometryResult(path=image_path, scores=_empty_scores())
    lines = _detect_lines(gray)
    if lines.size == 0:
        return GeometryResult(path=image_path, scores=_empty_scores())
    groups = _group_lines(lines)
    return GeometryResult(path=image_path, scores=_scores_from_groups(groups))
