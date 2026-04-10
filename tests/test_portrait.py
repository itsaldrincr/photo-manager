"""Tests for cull.stage2.portrait — EAR computation and face detection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cull.stage2.portrait import (
    PortraitResult,
    _ear_from_pts,
    compute_ear,
    is_eyes_closed,
)

# ---------------------------------------------------------------------------
# Named constants (no magic numbers in tests)
# ---------------------------------------------------------------------------

EAR_CLOSED_THRESHOLD: float = 0.20
EAR_OPEN_THRESHOLD: float = 0.25
BLANK_IMAGE_SIZE: int = 256
LANDMARK_COUNT: int = 468


# ---------------------------------------------------------------------------
# Helpers — synthetic landmark builders
# ---------------------------------------------------------------------------


def _make_landmark(x: float, y: float) -> SimpleNamespace:
    """Return a minimal landmark-like object with x, y coordinates."""
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=1.0)


def _closed_eye_points() -> list[tuple[float, float]]:
    """Return 6 points with near-zero vertical separation (closed eye)."""
    # EAR formula: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    # horizontal = 0.10, verticals ≈ 0.002 → EAR ≈ 0.01
    return [
        (0.40, 0.50),   # p1 left corner
        (0.43, 0.501),  # p2 upper-left
        (0.47, 0.501),  # p3 upper-right
        (0.50, 0.50),   # p4 right corner
        (0.47, 0.499),  # p5 lower-right
        (0.43, 0.499),  # p6 lower-left
    ]


def _open_eye_points() -> list[tuple[float, float]]:
    """Return 6 points with significant vertical separation (open eye)."""
    # horizontal = 0.10, verticals ≈ 0.04 → EAR ≈ 0.40
    return [
        (0.40, 0.50),  # p1
        (0.43, 0.52),  # p2
        (0.47, 0.52),  # p3
        (0.50, 0.50),  # p4
        (0.47, 0.48),  # p5
        (0.43, 0.48),  # p6
    ]


def _build_landmarks_for_ear(eye_pts: list[tuple[float, float]]) -> list[Any]:
    """Build minimal face landmark list with both eyes using same geometry."""
    left_indices = [362, 385, 387, 263, 373, 380]
    right_indices = [33, 160, 158, 133, 153, 144]
    base = [_make_landmark(0.5, 0.5) for _ in range(LANDMARK_COUNT)]
    for slot, idx in enumerate(left_indices):
        base[idx] = _make_landmark(*eye_pts[slot])
    for slot, idx in enumerate(right_indices):
        base[idx] = _make_landmark(*eye_pts[slot])
    return base


# ---------------------------------------------------------------------------
# _ear_from_pts formula tests (pure arithmetic)
# ---------------------------------------------------------------------------


def test_ear_formula_closed_is_low() -> None:
    """_ear_from_pts must return a low value for nearly-closed eye geometry."""
    ear = _ear_from_pts(_closed_eye_points())
    assert ear < EAR_CLOSED_THRESHOLD, (
        f"Expected EAR < {EAR_CLOSED_THRESHOLD} for closed eye, got {ear:.4f}"
    )


def test_ear_formula_open_is_high() -> None:
    """_ear_from_pts must return a high value for open eye geometry."""
    ear = _ear_from_pts(_open_eye_points())
    assert ear > EAR_OPEN_THRESHOLD, (
        f"Expected EAR > {EAR_OPEN_THRESHOLD} for open eye, got {ear:.4f}"
    )


def test_ear_formula_non_negative() -> None:
    """_ear_from_pts must always return a non-negative value."""
    assert _ear_from_pts(_closed_eye_points()) >= 0.0
    assert _ear_from_pts(_open_eye_points()) >= 0.0


# ---------------------------------------------------------------------------
# compute_ear tests (uses landmark objects)
# ---------------------------------------------------------------------------


def test_compute_ear_closed_eye_below_threshold() -> None:
    """compute_ear must return value below closed-eye threshold for closed eye."""
    lm = _build_landmarks_for_ear(_closed_eye_points())
    ear = compute_ear(lm)
    assert ear < EAR_CLOSED_THRESHOLD, (
        f"Expected EAR < {EAR_CLOSED_THRESHOLD}, got {ear:.4f}"
    )


def test_compute_ear_open_eye_above_squint() -> None:
    """compute_ear must return value above squint threshold for open eye."""
    lm = _build_landmarks_for_ear(_open_eye_points())
    ear = compute_ear(lm)
    assert ear > EAR_OPEN_THRESHOLD, (
        f"Expected EAR > {EAR_OPEN_THRESHOLD}, got {ear:.4f}"
    )


# ---------------------------------------------------------------------------
# is_eyes_closed tests
# ---------------------------------------------------------------------------


def test_is_eyes_closed_true_below_threshold() -> None:
    """is_eyes_closed must return True when EAR < 0.20."""
    assert is_eyes_closed(0.10) is True
    assert is_eyes_closed(EAR_CLOSED_THRESHOLD - 0.001) is True


def test_is_eyes_closed_false_above_squint() -> None:
    """is_eyes_closed must return False when EAR > 0.25."""
    assert is_eyes_closed(EAR_OPEN_THRESHOLD + 0.001) is False
    assert is_eyes_closed(0.40) is False


def test_is_eyes_closed_false_at_exact_threshold() -> None:
    """is_eyes_closed must return False when EAR equals threshold exactly."""
    assert is_eyes_closed(EAR_CLOSED_THRESHOLD) is False


# ---------------------------------------------------------------------------
# detect_faces with mocked landmarker (blank image → no faces)
# ---------------------------------------------------------------------------


def test_detect_faces_returns_empty_on_blank_image() -> None:
    """detect_faces must return empty list when landmarker finds no faces."""
    from cull.stage2.portrait import detect_faces

    blank = np.full(
        (BLANK_IMAGE_SIZE, BLANK_IMAGE_SIZE, 3), fill_value=255, dtype=np.uint8
    )
    mock_result = MagicMock()
    mock_result.face_landmarks = []

    mock_landmarker = MagicMock()
    mock_landmarker.detect.return_value = mock_result

    with patch("cull.stage2.portrait._get_face_landmarker", return_value=mock_landmarker):
        faces = detect_faces(blank)

    assert isinstance(faces, list)
    assert len(faces) == 0, f"Expected 0 faces, got {len(faces)}"


# ---------------------------------------------------------------------------
# PortraitResult model construction
# ---------------------------------------------------------------------------


def test_portrait_result_default_construction() -> None:
    """PortraitResult must construct with correct defaults."""
    result = PortraitResult()
    assert result.face_count == 0
    assert result.eyes_closed is False
    assert result.face_occluded is False
    assert result.dominant_emotion is None
