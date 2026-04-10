"""Parity test: PIL-grayscale tenengrad vs cv2-grayscale tenengrad < 2% relative error."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from cull.stage1.blur import compute_tenengrad
from cull.stage2.subject_blur import pil_to_gray

IMAGE_SIDE: int = 256
RELATIVE_ERROR_THRESHOLD: float = 0.02


def _make_gradient_array() -> np.ndarray:
    """Build a 256x256 RGB gradient image as a numpy array."""
    x = np.arange(IMAGE_SIDE, dtype=np.uint8)
    row = np.tile(x, (IMAGE_SIDE, 1))
    col = row.T
    r = row
    g = col
    b = (row.astype(np.uint16) + col.astype(np.uint16)).clip(0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=2)


def _cv2_gray(rgb_array: np.ndarray) -> np.ndarray:
    """Convert RGB numpy array to grayscale via cv2 (BGR path)."""
    bgr = rgb_array[:, :, ::-1]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def test_pil_gray_tenengrad_matches_cv2() -> None:
    """PIL and cv2 grayscale produce tenengrad scores within 2% relative error."""
    rgb_array = _make_gradient_array()
    pil_img = Image.fromarray(rgb_array)

    pil_gray = pil_to_gray(pil_img)
    cv2_gray = _cv2_gray(rgb_array)

    pil_score = compute_tenengrad(pil_gray)
    cv2_score = compute_tenengrad(cv2_gray)

    assert cv2_score > 0.0, "cv2 reference score must be positive"
    relative_error = abs(pil_score - cv2_score) / cv2_score
    assert relative_error < RELATIVE_ERROR_THRESHOLD, (
        f"Relative error {relative_error:.4f} exceeds {RELATIVE_ERROR_THRESHOLD}"
    )
