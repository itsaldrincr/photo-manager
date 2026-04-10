"""Tests for cull.stage1.blur — synthetic image blur detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from cull.stage1.blur import (
    BlurResult,
    assess_blur,
    compute_fft_ratio,
    compute_tenengrad,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SHARP_SIZE: int = 256
BLUR_KERNEL_SIZE: int = 61
BOKEH_CENTER_FRACTION: float = 0.3


def _make_sharp_gray() -> np.ndarray:
    """Create a synthetic sharp grayscale image with high-frequency content."""
    rng = np.random.default_rng(seed=42)
    base = np.zeros((SHARP_SIZE, SHARP_SIZE), dtype=np.uint8)
    # Chessboard pattern: very high contrast edges
    tile = 16
    for r in range(0, SHARP_SIZE, tile):
        for c in range(0, SHARP_SIZE, tile):
            if (r // tile + c // tile) % 2 == 0:
                base[r : r + tile, c : c + tile] = 255
    base = base + rng.integers(0, 5, base.shape, dtype=np.uint8)
    return base


def _make_blurry_gray() -> np.ndarray:
    """Apply heavy Gaussian blur to the sharp image."""
    sharp = _make_sharp_gray()
    return cv2.GaussianBlur(sharp, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)


def _make_bokeh_bgr() -> np.ndarray:
    """Create BGR image: sharp chessboard center, heavily blurred edges."""
    size = SHARP_SIZE
    sharp_gray = _make_sharp_gray()
    blurred_gray = cv2.GaussianBlur(sharp_gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    bgr = cv2.cvtColor(blurred_gray, cv2.COLOR_GRAY2BGR)
    # Paste sharp region in the center
    half = int(size * BOKEH_CENTER_FRACTION)
    center_start = size // 2 - half
    center_end = size // 2 + half
    bgr[center_start:center_end, center_start:center_end] = cv2.cvtColor(
        sharp_gray[center_start:center_end, center_start:center_end],
        cv2.COLOR_GRAY2BGR,
    )
    return bgr


def _save_bgr_tmp(bgr: np.ndarray) -> Path:
    """Write BGR array to a lossless temp PNG and return the Path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = Path(f.name)
    cv2.imwrite(str(tmp_path), bgr)
    return tmp_path


def _save_gray_tmp(gray: np.ndarray) -> Path:
    """Write grayscale array to a lossless temp PNG and return the Path."""
    return _save_bgr_tmp(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


# ---------------------------------------------------------------------------
# Unit tests — metric functions
# ---------------------------------------------------------------------------


def test_tenengrad_sharp_higher_than_blurred() -> None:
    """Tenengrad score must be larger for a sharp image than a blurred one."""
    sharp = _make_sharp_gray()
    blurry = _make_blurry_gray()
    score_sharp = compute_tenengrad(sharp)
    score_blurry = compute_tenengrad(blurry)
    assert score_sharp > score_blurry, (
        f"Expected sharp ({score_sharp:.1f}) > blurry ({score_blurry:.1f})"
    )


def test_fft_ratio_sharp_higher_than_blurred() -> None:
    """FFT high-freq ratio must be larger for a sharp image than a blurred one."""
    sharp = _make_sharp_gray()
    blurry = _make_blurry_gray()
    ratio_sharp = compute_fft_ratio(sharp)
    ratio_blurry = compute_fft_ratio(blurry)
    assert ratio_sharp > ratio_blurry, (
        f"Expected sharp ({ratio_sharp:.4f}) > blurry ({ratio_blurry:.4f})"
    )


# ---------------------------------------------------------------------------
# Integration tests — assess_blur
# ---------------------------------------------------------------------------


def test_assess_blur_returns_blur_result() -> None:
    """assess_blur must return a BlurResult Pydantic model."""
    gray = _make_sharp_gray()
    tmp = _save_gray_tmp(gray)
    try:
        result = assess_blur(tmp, config=None)
        assert isinstance(result, BlurResult)
    finally:
        tmp.unlink(missing_ok=True)


def test_assess_blur_blurry_image_is_blurry() -> None:
    """assess_blur must mark a synthetically blurred image as is_blurry=True."""
    blurry = _make_blurry_gray()
    tmp = _save_gray_tmp(blurry)
    try:
        result = assess_blur(tmp, config=None)
        assert result.is_blurry, (
            f"Expected is_blurry=True, got scores: {result.scores}"
        )
    finally:
        tmp.unlink(missing_ok=True)


def test_assess_blur_bokeh_not_rejected() -> None:
    """Bokeh images (sharp center, blurred edges) must NOT be marked as blurry."""
    bokeh = _make_bokeh_bgr()
    tmp = _save_bgr_tmp(bokeh)
    try:
        result = assess_blur(tmp, config=None)
        assert not result.is_blurry, (
            f"Expected is_blurry=False for bokeh, got scores: {result.scores}"
        )
    finally:
        tmp.unlink(missing_ok=True)
