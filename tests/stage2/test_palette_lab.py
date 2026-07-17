"""Tests for cull._pipeline.stage2_scoring palette LAB centroid computation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cull._pipeline.stage2_scoring import _palette_lab_centroid, score_palette_lab_batch

IMAGE_SIDE_PX: int = 64
RED_RGB: tuple[int, int, int] = (200, 30, 30)
BLUE_RGB: tuple[int, int, int] = (30, 30, 200)
GRAY_RGB: tuple[int, int, int] = (128, 128, 128)


def _solid_image(rgb: tuple[int, int, int]) -> Image.Image:
    """Build a solid-color RGB PIL image for deterministic centroid tests."""
    return Image.new("RGB", (IMAGE_SIDE_PX, IMAGE_SIDE_PX), color=rgb)


def test_palette_lab_centroid_returns_three_floats() -> None:
    """A solid-color image yields a 3-tuple of finite LAB floats."""
    centroid = _palette_lab_centroid(_solid_image(GRAY_RGB))
    assert len(centroid) == 3
    assert all(np.isfinite(v) for v in centroid)


def test_palette_lab_centroid_distinguishes_different_colors() -> None:
    """Distinctly colored images must produce distinct LAB centroids."""
    red_centroid = _palette_lab_centroid(_solid_image(RED_RGB))
    blue_centroid = _palette_lab_centroid(_solid_image(BLUE_RGB))
    distance = np.linalg.norm(np.array(red_centroid) - np.array(blue_centroid))
    assert distance > 10.0


def test_palette_lab_centroid_stable_for_same_color() -> None:
    """The same solid color must produce (near-)identical centroids across calls."""
    first = _palette_lab_centroid(_solid_image(RED_RGB))
    second = _palette_lab_centroid(_solid_image(RED_RGB))
    assert first == pytest.approx(second)


def test_score_palette_lab_batch_matches_per_image_calls() -> None:
    """Batch scoring must return one centroid per input image, in order."""
    images = [_solid_image(RED_RGB), _solid_image(BLUE_RGB), _solid_image(GRAY_RGB)]
    batch = score_palette_lab_batch(images)
    assert len(batch) == 3
    for image, centroid in zip(images, batch):
        assert centroid == pytest.approx(_palette_lab_centroid(image))
