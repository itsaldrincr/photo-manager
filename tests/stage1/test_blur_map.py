"""cull.stage1.blur_map must reproduce blur_detector.detectBlur bit for bit."""

from __future__ import annotations

import blur_detector
import cv2
import numpy as np

from cull.stage1.blur_map import DOWNSAMPLING_FACTOR, NUM_SCALES, detect_blur_map

FRAME_HEIGHT: int = 150
FRAME_WIDTH: int = 220
BLUR_KSIZE: tuple[int, int] = (15, 15)
SHARP_COLUMNS: int = 110


def _half_blurred_frame() -> np.ndarray:
    """Return a noise texture that is sharp on the left and blurred on the right."""
    rng = np.random.default_rng(seed=7)
    frame = rng.integers(0, 256, (FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    blurred = cv2.GaussianBlur(frame, BLUR_KSIZE, 0)
    frame[:, SHARP_COLUMNS:] = blurred[:, SHARP_COLUMNS:]
    return frame


def _library_map(gray: np.ndarray) -> np.ndarray:
    """Return the reference map from the blur_detector package."""
    return blur_detector.detectBlur(
        gray, downsampling_factor=DOWNSAMPLING_FACTOR, num_scales=NUM_SCALES, show_progress=False,
    )


def test_blur_map_is_bitwise_equal_to_blur_detector() -> None:
    """The vectorised map equals the library's exactly, not just approximately."""
    gray = _half_blurred_frame()
    assert np.array_equal(detect_blur_map(gray), _library_map(gray))


def test_blur_map_ranks_sharp_side_above_blurred_side() -> None:
    """The map is higher on the sharp half than on the blurred half."""
    blur_map = detect_blur_map(_half_blurred_frame())
    assert blur_map[:, :SHARP_COLUMNS].mean() > blur_map[:, SHARP_COLUMNS:].mean()
