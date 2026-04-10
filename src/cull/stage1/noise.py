"""Stage 1 noise estimation using Laplacian-of-Gaussian on low-gradient patches."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from cull.config import IMAGE_LONG_EDGE_PX, NOISE_SCORE_REJECT_MIN

log = logging.getLogger(__name__)

_PATCH_SIZE: int = 16
_LOG_KSIZE: int = 5
_LOG_SIGMA: float = 1.0
_GRADIENT_LOW_PCT: float = 25.0
_NOISE_SCALE: float = 1000.0


class NoiseResult(BaseModel):
    """Noise assessment output for a single image."""

    noise_score: float
    is_noisy: bool


def _resize_to_long_edge(image: np.ndarray) -> np.ndarray:
    """Resize image so longest edge equals IMAGE_LONG_EDGE_PX."""
    h, w = image.shape[:2]
    scale = IMAGE_LONG_EDGE_PX / max(h, w)
    if scale >= 1.0:
        return image
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _extract_patch_variances(gray: np.ndarray) -> tuple[list[float], list[float]]:
    """Extract LoG variance and gradient magnitude per non-overlapping patch."""
    log_kernel = cv2.Laplacian(gray, cv2.CV_64F)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    h, w = gray.shape
    log_variances: list[float] = []
    grad_means: list[float] = []
    for row in range(0, h - _PATCH_SIZE + 1, _PATCH_SIZE):
        for col in range(0, w - _PATCH_SIZE + 1, _PATCH_SIZE):
            patch_log = log_kernel[row:row + _PATCH_SIZE, col:col + _PATCH_SIZE]
            patch_grad = gradient_mag[row:row + _PATCH_SIZE, col:col + _PATCH_SIZE]
            log_variances.append(float(np.var(patch_log)))
            grad_means.append(float(np.mean(patch_grad)))
    return log_variances, grad_means


def estimate_noise(gray: np.ndarray) -> float:
    """Estimate noise as mean LoG variance on low-gradient 16x16 patches."""
    log_variances, grad_means = _extract_patch_variances(gray)
    if not log_variances:
        return 0.0
    threshold = float(np.percentile(grad_means, _GRADIENT_LOW_PCT))
    low_grad_vars = [
        v for v, g in zip(log_variances, grad_means) if g <= threshold
    ]
    if not low_grad_vars:
        return 0.0
    raw = float(np.mean(low_grad_vars))
    return min(raw / _NOISE_SCALE, 1.0)


def assess_noise(image_path: Path) -> NoiseResult:
    """Assess noise level for a single image."""
    log.debug("Assessing noise: %s", image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = _resize_to_long_edge(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_score = estimate_noise(gray)
    return NoiseResult(
        noise_score=noise_score,
        is_noisy=noise_score >= NOISE_SCORE_REJECT_MIN,
    )
