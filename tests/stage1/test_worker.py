"""Verify Stage 1 decode-once worker matches per-assessor path-based results."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from cull.config import CullConfig
from cull.stage1.blur import assess_blur
from cull.stage1.exposure import assess_exposure
from cull.stage1.geometry import assess_geometry
from cull.stage1.noise import assess_noise
from cull.stage1.worker import assess_one

CANVAS_SIZE: int = 256
TILE_SIZE: int = 16
LINE_COLOR: tuple[int, int, int] = (0, 0, 0)
TILT_DX: int = 200
TILT_DY: int = 20


def _make_photo_bgr() -> np.ndarray:
    """Build a synthetic BGR image with a chessboard texture and a tilted line."""
    rng = np.random.default_rng(seed=7)
    base = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    for row in range(0, CANVAS_SIZE, TILE_SIZE):
        for col in range(0, CANVAS_SIZE, TILE_SIZE):
            if (row // TILE_SIZE + col // TILE_SIZE) % 2 == 0:
                base[row : row + TILE_SIZE, col : col + TILE_SIZE] = (200, 180, 160)
    base = base + rng.integers(0, 5, base.shape, dtype=np.uint8)
    for offset in range(0, 40, 8):
        start = (10, 40 + offset)
        end = (10 + TILT_DX, 40 + offset + TILT_DY)
        cv2.line(base, start, end, LINE_COLOR, 2)
    return base


def _save_png_tmp(bgr: np.ndarray) -> Path:
    """Write BGR array to a lossless temp PNG and return the Path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = Path(f.name)
    cv2.imwrite(str(tmp_path), bgr)
    return tmp_path


@pytest.fixture()
def photo_path() -> Path:
    """Yield a synthetic lossless PNG photo path, cleaned up after the test."""
    tmp = _save_png_tmp(_make_photo_bgr())
    yield tmp
    tmp.unlink(missing_ok=True)


def test_decode_once_matches_path_based_blur(photo_path: Path) -> None:
    """assess_one's blur result must match the path-based assess_blur exactly."""
    expected = assess_blur(photo_path, config=None)
    actual = assess_one(photo_path, CullConfig(is_portrait=False))
    assert actual.blur.scores.tenengrad == pytest.approx(expected.scores.tenengrad)
    assert actual.blur.scores.fft_ratio == pytest.approx(expected.scores.fft_ratio)
    assert actual.blur.is_blurry == expected.is_blurry


def test_decode_once_matches_path_based_exposure(photo_path: Path) -> None:
    """assess_one's exposure result must match the path-based assess_exposure exactly."""
    expected = assess_exposure(photo_path)
    actual = assess_one(photo_path, CullConfig(is_portrait=False))
    assert actual.exposure.dynamic_range == pytest.approx(expected.dynamic_range)
    assert actual.exposure.midtone_pct == pytest.approx(expected.midtone_pct)
    assert actual.exposure.has_highlight_clip == expected.has_highlight_clip


def test_decode_once_matches_path_based_noise(photo_path: Path) -> None:
    """assess_one's noise result must match the path-based assess_noise exactly."""
    expected = assess_noise(photo_path)
    actual = assess_one(photo_path, CullConfig(is_portrait=False))
    assert actual.noise.noise_score == pytest.approx(expected.noise_score)
    assert actual.noise.is_noisy == expected.is_noisy


def test_decode_once_matches_path_based_geometry_on_lossless_source(
    photo_path: Path,
) -> None:
    """geometry's dedicated IMREAD_GRAYSCALE read must match the path-based result exactly."""
    expected = assess_geometry(photo_path)
    actual = assess_one(photo_path, CullConfig(is_portrait=False))
    assert actual.geometry.scores.tilt_degrees == pytest.approx(
        expected.scores.tilt_degrees
    )
    assert actual.geometry.scores.confidence == pytest.approx(
        expected.scores.confidence
    )
    assert actual.geometry.scores.has_horizon == expected.scores.has_horizon
