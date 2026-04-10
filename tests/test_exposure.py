"""Tests for Stage 1 exposure analysis functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from cull.stage1.exposure import (
    ClippingResult,
    ColorCastResult,
    ExposureResult,
    assess_exposure,
    compute_clipping,
    compute_color_cast,
    compute_dynamic_range,
    compute_midtone_population,
)
from cull.config import CullConfig

_WHITE_VALUE: int = 255
_BLACK_VALUE: int = 0
_MID_VALUE: int = 128
_IMAGE_SIZE: int = 100
_TINT_A_VALUE: int = 200
_NARROW_RANGE_LOW: int = 100
_NARROW_RANGE_HIGH: int = 110
_LAB_NEUTRAL: int = 128

_WHITE_BGR: tuple[int, int, int] = (_WHITE_VALUE, _WHITE_VALUE, _WHITE_VALUE)
_BLACK_BGR: tuple[int, int, int] = (_BLACK_VALUE, _BLACK_VALUE, _BLACK_VALUE)
_MID_BGR: tuple[int, int, int] = (_MID_VALUE, _MID_VALUE, _MID_VALUE)
_RED_BGR: tuple[int, int, int] = (_BLACK_VALUE, _BLACK_VALUE, _WHITE_VALUE)


def _make_bgr_image(bgr: tuple[int, int, int]) -> np.ndarray:
    """Create a solid-colour BGR image of fixed size."""
    return np.full((_IMAGE_SIZE, _IMAGE_SIZE, 3), list(bgr), dtype=np.uint8)


def _save_temp_image(img: np.ndarray) -> Path:
    """Save image to a temp file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, img)
    return Path(tmp.name)


def _make_lab_image(lab: tuple[int, int, int]) -> np.ndarray:
    """Create a solid Lab image (8-bit uint, a/b offset by 128)."""
    return np.full((_IMAGE_SIZE, _IMAGE_SIZE, 3), list(lab), dtype=np.uint8)


# ---------------------------------------------------------------------------
# compute_clipping
# ---------------------------------------------------------------------------


def test_white_image_has_highlight_clip() -> None:
    """Near-white image should show high highlight clipping."""
    img = _make_bgr_image(_WHITE_BGR)
    result = compute_clipping(img)
    assert result.highlight_pct > 0.90


def test_black_image_has_shadow_clip() -> None:
    """Near-black image should show high shadow clipping."""
    img = _make_bgr_image(_BLACK_BGR)
    result = compute_clipping(img)
    assert result.shadow_pct > 0.90


def test_midtone_image_has_no_clip() -> None:
    """Midtone grey image should have no significant clipping."""
    img = _make_bgr_image(_MID_BGR)
    result = compute_clipping(img)
    assert result.highlight_pct < 0.01
    assert result.shadow_pct < 0.01


# ---------------------------------------------------------------------------
# compute_dynamic_range
# ---------------------------------------------------------------------------


def test_full_range_image_has_high_dr() -> None:
    """Image spanning full 0-255 range should have high DR."""
    flat = np.linspace(_BLACK_VALUE, _WHITE_VALUE, _IMAGE_SIZE * _IMAGE_SIZE, dtype=np.uint8)
    luminance = flat.reshape((_IMAGE_SIZE, _IMAGE_SIZE))
    dr = compute_dynamic_range(luminance)
    assert dr > 0.90


def test_narrow_range_image_has_low_dr() -> None:
    """Image with narrow tonal range should have low DR."""
    luminance = np.full((_IMAGE_SIZE, _IMAGE_SIZE), _NARROW_RANGE_LOW, dtype=np.uint8)
    luminance[0, 0] = _NARROW_RANGE_HIGH
    dr = compute_dynamic_range(luminance)
    assert dr < 0.10


# ---------------------------------------------------------------------------
# compute_midtone_population
# ---------------------------------------------------------------------------


def test_midtone_grey_has_high_midtone_pct() -> None:
    """Solid midtone grey should have 100% midtone population."""
    luminance = np.full((_IMAGE_SIZE, _IMAGE_SIZE), _MID_VALUE, dtype=np.uint8)
    pct = compute_midtone_population(luminance)
    assert pct == pytest.approx(1.0)


def test_pure_black_has_zero_midtone_pct() -> None:
    """Pure black image has no midtone pixels."""
    luminance = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE), dtype=np.uint8)
    pct = compute_midtone_population(luminance)
    assert pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_color_cast
# ---------------------------------------------------------------------------


def test_neutral_image_has_no_cast() -> None:
    """Neutral grey image (Lab a=128, b=128) has near-zero cast."""
    lab = _make_lab_image((_LAB_NEUTRAL, _LAB_NEUTRAL, _LAB_NEUTRAL))
    result = compute_color_cast(lab)
    assert abs(result.mean_a) < 1.0
    assert abs(result.mean_b) < 1.0
    assert result.cast_score < 2.0


def test_tinted_image_has_color_cast() -> None:
    """Strongly tinted image should produce high cast_score."""
    lab = _make_lab_image((_LAB_NEUTRAL, _TINT_A_VALUE, _LAB_NEUTRAL))
    result = compute_color_cast(lab)
    assert abs(result.mean_a) > 50.0
    assert result.cast_score > 50.0


# ---------------------------------------------------------------------------
# assess_exposure (end-to-end with real image files)
# ---------------------------------------------------------------------------


def test_assess_exposure_white_image_flags_highlight() -> None:
    """assess_exposure on a white image sets has_highlight_clip=True."""
    img = _make_bgr_image(_WHITE_BGR)
    path = _save_temp_image(img)
    result = assess_exposure(path)
    assert isinstance(result, ExposureResult)
    assert result.has_highlight_clip is True


def test_assess_exposure_black_image_flags_shadow() -> None:
    """assess_exposure on a black image sets has_shadow_clip=True."""
    img = _make_bgr_image(_BLACK_BGR)
    path = _save_temp_image(img)
    result = assess_exposure(path)
    assert result.has_shadow_clip is True


def test_assess_exposure_tinted_image_flags_cast() -> None:
    """assess_exposure on a strongly red-tinted image sets has_color_cast=True."""
    img = _make_bgr_image(_RED_BGR)
    path = _save_temp_image(img)
    result = assess_exposure(path)
    assert result.has_color_cast is True


def test_assess_exposure_import() -> None:
    """Confirms assess_exposure is importable from cull.stage1.exposure."""
    from cull.stage1.exposure import assess_exposure as fn
    assert callable(fn)
