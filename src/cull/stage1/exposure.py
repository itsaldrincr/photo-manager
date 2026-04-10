"""Stage 1 exposure analysis: clipping, dynamic range, color cast."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from cull.config import (
    CullConfig,
    EXPOSURE_CAST_A_MAX,
    EXPOSURE_CAST_B_MAX,
    EXPOSURE_CAST_COMBINED_MAX,
    EXPOSURE_DR_MIN,
    EXPOSURE_HIGHLIGHT_CLIP_PCT_MAX,
    EXPOSURE_HIGHLIGHT_CLIP_VALUE,
    EXPOSURE_MIDTONE_L_HIGH,
    EXPOSURE_MIDTONE_L_LOW,
    EXPOSURE_MIDTONE_PCT_MIN,
    EXPOSURE_SHADOW_CLIP_PCT_MAX,
    EXPOSURE_SHADOW_CLIP_VALUE,
    IMAGE_LONG_EDGE_PX,
)

log = logging.getLogger(__name__)

_PIXEL_MAX: int = 255
_PIXEL_SCALE: float = 254.0
_P1_PERCENTILE: float = 1.0
_P99_PERCENTILE: float = 99.0


class ClippingResult(BaseModel):
    """Per-channel highlight and shadow clipping percentages."""

    highlight_r: float
    highlight_g: float
    highlight_b: float
    shadow_r: float
    shadow_g: float
    shadow_b: float
    highlight_pct: float
    shadow_pct: float


class ColorCastResult(BaseModel):
    """Lab color cast metrics."""

    mean_a: float
    mean_b: float
    cast_score: float


class ExposureResult(BaseModel):
    """Full exposure assessment output."""

    clipping: ClippingResult
    dynamic_range: float
    midtone_pct: float
    color_cast: ColorCastResult
    has_highlight_clip: bool
    has_shadow_clip: bool
    has_color_cast: bool
    has_low_dr: bool


def _resize_to_long_edge(image: np.ndarray) -> np.ndarray:
    """Resize image so longest edge equals IMAGE_LONG_EDGE_PX."""
    h, w = image.shape[:2]
    scale = IMAGE_LONG_EDGE_PX / max(h, w)
    if scale >= 1.0:
        return image
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _channel_clip_pct(channel: np.ndarray, threshold: int) -> float:
    """Return fraction of pixels at or above threshold."""
    total = channel.size
    if total == 0:
        return 0.0
    return float(np.sum(channel >= threshold)) / total


def _channel_shadow_pct(channel: np.ndarray, threshold: int) -> float:
    """Return fraction of pixels at or below threshold."""
    total = channel.size
    if total == 0:
        return 0.0
    return float(np.sum(channel <= threshold)) / total


def compute_clipping(image: np.ndarray) -> ClippingResult:
    """Compute per-channel highlight and shadow clipping percentages."""
    b_ch, g_ch, r_ch = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    hi_r = _channel_clip_pct(r_ch, EXPOSURE_HIGHLIGHT_CLIP_VALUE)
    hi_g = _channel_clip_pct(g_ch, EXPOSURE_HIGHLIGHT_CLIP_VALUE)
    hi_b = _channel_clip_pct(b_ch, EXPOSURE_HIGHLIGHT_CLIP_VALUE)
    sh_r = _channel_shadow_pct(r_ch, EXPOSURE_SHADOW_CLIP_VALUE)
    sh_g = _channel_shadow_pct(g_ch, EXPOSURE_SHADOW_CLIP_VALUE)
    sh_b = _channel_shadow_pct(b_ch, EXPOSURE_SHADOW_CLIP_VALUE)
    highlight_pct = max(hi_r, hi_g, hi_b)
    shadow_pct = max(sh_r, sh_g, sh_b)
    return ClippingResult(
        highlight_r=hi_r, highlight_g=hi_g, highlight_b=hi_b,
        shadow_r=sh_r, shadow_g=sh_g, shadow_b=sh_b,
        highlight_pct=highlight_pct, shadow_pct=shadow_pct,
    )


def compute_dynamic_range(luminance: np.ndarray) -> float:
    """Compute dynamic range as (p99 - p1) / PIXEL_SCALE."""
    p1 = float(np.percentile(luminance, _P1_PERCENTILE))
    p99 = float(np.percentile(luminance, _P99_PERCENTILE))
    return (p99 - p1) / _PIXEL_SCALE


def compute_midtone_population(luminance: np.ndarray) -> float:
    """Compute fraction of pixels in midtone luminance range."""
    total = luminance.size
    if total == 0:
        return 0.0
    mask = (luminance >= EXPOSURE_MIDTONE_L_LOW) & (luminance <= EXPOSURE_MIDTONE_L_HIGH)
    return float(np.sum(mask)) / total


def compute_color_cast(lab_image: np.ndarray) -> ColorCastResult:
    """Compute mean Lab a*/b* channels and cast_score = sqrt(a^2 + b^2)."""
    a_channel = lab_image[:, :, 1].astype(np.float32) - 128.0
    b_channel = lab_image[:, :, 2].astype(np.float32) - 128.0
    mean_a = float(np.mean(a_channel))
    mean_b = float(np.mean(b_channel))
    cast_score = math.sqrt(mean_a ** 2 + mean_b ** 2)
    return ColorCastResult(mean_a=mean_a, mean_b=mean_b, cast_score=cast_score)


def _load_and_resize(image_path: Path) -> np.ndarray:
    """Load image from path and resize to long edge."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return _resize_to_long_edge(img)


class _ExposureMetrics(BaseModel):
    """Internal bundle of raw exposure metrics for flag derivation."""

    clipping: ClippingResult
    dynamic_range: float
    color_cast: ColorCastResult


def _derive_flags(metrics: _ExposureMetrics) -> tuple[bool, bool, bool, bool]:
    """Compute boolean exposure flags from metric bundle."""
    has_highlight_clip = metrics.clipping.highlight_pct > EXPOSURE_HIGHLIGHT_CLIP_PCT_MAX
    has_shadow_clip = metrics.clipping.shadow_pct > EXPOSURE_SHADOW_CLIP_PCT_MAX
    has_color_cast = (
        abs(metrics.color_cast.mean_a) > EXPOSURE_CAST_A_MAX
        or abs(metrics.color_cast.mean_b) > EXPOSURE_CAST_B_MAX
        or metrics.color_cast.cast_score > EXPOSURE_CAST_COMBINED_MAX
    )
    has_low_dr = metrics.dynamic_range < EXPOSURE_DR_MIN
    return has_highlight_clip, has_shadow_clip, has_color_cast, has_low_dr


def assess_exposure(image_path: Path) -> ExposureResult:
    """Assess exposure quality for a single image."""
    log.debug("Assessing exposure: %s", image_path)
    image = _load_and_resize(image_path)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = lab_image[:, :, 0]
    clipping = compute_clipping(image)
    dynamic_range = compute_dynamic_range(luminance)
    midtone_pct = compute_midtone_population(luminance)
    color_cast = compute_color_cast(lab_image)
    metrics = _ExposureMetrics(clipping=clipping, dynamic_range=dynamic_range, color_cast=color_cast)
    hi_clip, sh_clip, cast, low_dr = _derive_flags(metrics)
    return ExposureResult(
        clipping=clipping, dynamic_range=dynamic_range, midtone_pct=midtone_pct,
        color_cast=color_cast, has_highlight_clip=hi_clip, has_shadow_clip=sh_clip,
        has_color_cast=cast, has_low_dr=low_dr,
    )
