"""Stage 1a — Three-tier blur detection: Tenengrad, FFT, spatial map, motion."""

from __future__ import annotations

import logging
from pathlib import Path

import blur_detector
import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from cull.config import (
    BLUR_MOTION_ANISOTROPY_RATIO,
    BLUR_SPATIAL_CENTER_FRACTION,
    IMAGE_LONG_EDGE_PX,
)
from cull.models import BlurScores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

TENENGRAD_THRESHOLD: float = 100.0
FFT_RATIO_THRESHOLD: float = 0.05
FFT_RADIUS_FRACTION: float = 0.6
SOBEL_DIRECTIONS: int = 8
SOBEL_KSIZE: int = 3
DOWNSAMPLING_FACTOR: int = 4
NUM_SCALES: int = 4
BOKEH_SHARPNESS_RATIO: float = 2.0


# ---------------------------------------------------------------------------
# Result model (defined here, not in models.py)
# ---------------------------------------------------------------------------


class BlurResult(BaseModel):
    """Complete blur assessment result for one image."""

    path: Path
    scores: BlurScores
    is_blurry: bool


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resize_to_long_edge(image: np.ndarray) -> np.ndarray:
    """Resize image so its long edge equals IMAGE_LONG_EDGE_PX."""
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= IMAGE_LONG_EDGE_PX:
        return image
    scale = IMAGE_LONG_EDGE_PX / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale if needed."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _fft_mask(shape: tuple[int, int]) -> np.ndarray:
    """Build a centered circular mask for low-frequency FFT energy."""
    h, w = shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * FFT_RADIUS_FRACTION)
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    return dist > radius


def _center_slice(h: int, w: int) -> tuple[slice, slice]:
    """Return slice indices for the center BLUR_SPATIAL_CENTER_FRACTION area."""
    frac = BLUR_SPATIAL_CENTER_FRACTION
    pad_h = int(h * (1.0 - frac) / 2)
    pad_w = int(w * (1.0 - frac) / 2)
    return slice(pad_h, h - pad_h), slice(pad_w, w - pad_w)


# ---------------------------------------------------------------------------
# Public metric functions (≤ 2 params each)
# ---------------------------------------------------------------------------


def compute_tenengrad(gray: np.ndarray) -> float:
    """Compute Tenengrad sharpness: mean squared Sobel gradient energy."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
    return float(np.mean(gx ** 2 + gy ** 2))


def compute_fft_ratio(gray: np.ndarray) -> float:
    """Compute ratio of high-frequency energy to total FFT energy."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32))))
    high_freq_mask = _fft_mask(gray.shape)
    total = float(np.sum(spectrum))
    if total == 0.0:
        return 0.0
    return float(np.sum(spectrum[high_freq_mask])) / total


def compute_spatial_blur_map(image: np.ndarray) -> np.ndarray:
    """Return pixel-level blur map via blur_detector (lower = more blurred)."""
    gray = _to_gray(image)
    return blur_detector.detectBlur(
        gray,
        downsampling_factor=DOWNSAMPLING_FACTOR,
        num_scales=NUM_SCALES,
        show_progress=False,
    )


def detect_motion_blur(gray: np.ndarray) -> bool:
    """Return True if oriented Sobel anisotropy indicates motion blur."""
    angles_deg = [i * (180 // SOBEL_DIRECTIONS) for i in range(SOBEL_DIRECTIONS)]
    energies: list[float] = []
    for deg in angles_deg:
        rad = np.deg2rad(deg)
        kx = np.cos(rad)
        ky = np.sin(rad)
        kernel = np.array(
            [[kx + ky, ky, -kx + ky], [kx, 0.0, -kx], [kx - ky, -ky, -kx - ky]],
            dtype=np.float32,
        )
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        energies.append(float(np.mean(filtered ** 2)))
    mean_energy = float(np.mean(energies))
    if mean_energy == 0.0:
        return False
    return max(energies) / mean_energy > BLUR_MOTION_ANISOTROPY_RATIO


# ---------------------------------------------------------------------------
# Bokeh detection helpers
# ---------------------------------------------------------------------------


def _subject_background_sharpness(
    blur_map: np.ndarray,
) -> tuple[float, float]:
    """Return (subject_sharpness, background_sharpness) from blur map."""
    h, w = blur_map.shape
    row_sl, col_sl = _center_slice(h, w)
    subject = float(np.mean(blur_map[row_sl, col_sl]))
    mask = np.ones_like(blur_map, dtype=bool)
    mask[row_sl, col_sl] = False
    background = float(np.mean(blur_map[mask]))
    return subject, background


def _is_bokeh(subject_sharpness: float, background_sharpness: float) -> bool:
    """Return True when center is sharp but background is blurred (bokeh)."""
    return subject_sharpness > background_sharpness * BOKEH_SHARPNESS_RATIO


# ---------------------------------------------------------------------------
# Top-level assessment
# ---------------------------------------------------------------------------


def _load_and_resize(image_path: Path) -> np.ndarray:
    """Load image from disk and resize to long-edge limit."""
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return _resize_to_long_edge(raw)


class _MetricPair(BaseModel):
    """Paired Tenengrad and FFT-ratio metrics for a single image."""

    tenengrad: float
    fft_ratio: float


class _SpatialCtx(BaseModel):
    """Intermediate spatial analysis results passed between tier functions."""

    subject_sharp: float
    bg_sharp: float
    bokeh: bool


class _TierDecision(BaseModel):
    """Tier classification with flag fields."""

    tier: int
    is_bokeh: bool
    is_motion_blur: bool
    is_blurry: bool


class _TierInput(BaseModel):
    """Input bundle for _classify_tier function."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: _MetricPair
    ctx: _SpatialCtx
    image: np.ndarray


def _analyze_spatial(image: np.ndarray) -> _SpatialCtx:
    """Compute spatial blur map and derive subject/background sharpness."""
    blur_map = compute_spatial_blur_map(image)
    subject_sharp, bg_sharp = _subject_background_sharpness(blur_map)
    return _SpatialCtx(
        subject_sharp=subject_sharp,
        bg_sharp=bg_sharp,
        bokeh=_is_bokeh(subject_sharp, bg_sharp),
    )


def _classify_tier(tier_in: _TierInput) -> _TierDecision:
    """Run three-tier logic and return tier classification."""
    tier1_blurry = (
        tier_in.metrics.tenengrad < TENENGRAD_THRESHOLD
        and tier_in.metrics.fft_ratio < FFT_RATIO_THRESHOLD
    )
    if tier_in.ctx.bokeh:
        return _TierDecision(tier=2, is_bokeh=True, is_motion_blur=False, is_blurry=False)
    if tier1_blurry:
        motion = detect_motion_blur(_to_gray(tier_in.image))
        return _TierDecision(tier=3 if motion else 1, is_bokeh=False, is_motion_blur=motion, is_blurry=True)
    return _TierDecision(tier=2, is_bokeh=False, is_motion_blur=False, is_blurry=False)


def _build_scores(metrics: _MetricPair, image: np.ndarray) -> tuple[BlurScores, bool]:
    """Assemble BlurScores and is_blurry flag from metrics and image."""
    ctx = _analyze_spatial(image)
    decision = _classify_tier(_TierInput(metrics=metrics, ctx=ctx, image=image))
    scores = BlurScores(
        tenengrad=metrics.tenengrad,
        fft_ratio=metrics.fft_ratio,
        blur_tier=decision.tier,
        subject_sharpness=ctx.subject_sharp,
        background_sharpness=ctx.bg_sharp,
        is_bokeh=decision.is_bokeh,
        is_motion_blur=decision.is_motion_blur,
    )
    return scores, decision.is_blurry


def assess_blur(image_path: Path, config: object) -> BlurResult:
    """Load image, resize, run three-tier blur assessment, return BlurResult."""
    logger.debug("assess_blur: %s", image_path)
    image = _load_and_resize(image_path)
    gray = _to_gray(image)
    metrics = _MetricPair(
        tenengrad=compute_tenengrad(gray),
        fft_ratio=compute_fft_ratio(gray),
    )
    scores, is_blurry = _build_scores(metrics, image)
    return BlurResult(path=image_path, scores=scores, is_blurry=is_blurry)
