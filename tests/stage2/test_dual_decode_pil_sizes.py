"""Verify dual-decode PIL batch has correct 224 + 1280 sizes and aspect."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from cull.config import SHARED_DECODE_CLIP_PX, SHARED_DECODE_PIXEL_PX
from cull.pipeline import _DualLoadInput, _load_dual_pil_batch

FIXTURE_WIDTH: int = 3000
FIXTURE_HEIGHT: int = 2000
ASPECT_TOLERANCE_PX: int = 1


@pytest.fixture()
def wide_jpeg(tmp_path: Path) -> Path:
    """Create a synthetic 3000x2000 JPEG on disk."""
    img = Image.new("RGB", (FIXTURE_WIDTH, FIXTURE_HEIGHT), color=(16, 64, 128))
    jpeg_path = tmp_path / "wide.jpg"
    img.save(str(jpeg_path), format="JPEG")
    return jpeg_path


def test_pil_224_is_square_and_correct_size(wide_jpeg: Path) -> None:
    """pil_224 must be a SHARED_DECODE_CLIP_PX square."""
    batch = _load_dual_pil_batch(_DualLoadInput(paths=[wide_jpeg], device="cpu"))
    assert batch.pil_224[0].size == (SHARED_DECODE_CLIP_PX, SHARED_DECODE_CLIP_PX)


def test_pil_1280_long_edge_and_aspect(wide_jpeg: Path) -> None:
    """pil_1280 long edge == SHARED_DECODE_PIXEL_PX; aspect preserved within 1 px."""
    batch = _load_dual_pil_batch(_DualLoadInput(paths=[wide_jpeg], device="cpu"))
    w, h = batch.pil_1280[0].size
    assert max(w, h) == SHARED_DECODE_PIXEL_PX
    expected_h = int(round(SHARED_DECODE_PIXEL_PX * FIXTURE_HEIGHT / FIXTURE_WIDTH))
    assert abs(h - expected_h) <= ASPECT_TOLERANCE_PX


def test_tensor_1280_matches_pil_1280_shape(wide_jpeg: Path) -> None:
    """tensor_1280 must have shape (1, 3, H, W) matching pil_1280 dimensions."""
    batch = _load_dual_pil_batch(_DualLoadInput(paths=[wide_jpeg], device="cpu"))
    w, h = batch.pil_1280[0].size
    assert batch.tensor_1280.shape == (1, 3, h, w)
