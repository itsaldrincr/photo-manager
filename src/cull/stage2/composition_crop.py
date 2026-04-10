"""Composition crop — smartcrop dispatch, downsample, bbox upscale, thirds fallback."""

# Torch/smartcrop weights are pinned via TORCH_HOME, set in cull.env_bootstrap.
# No loader kwargs are needed here — smartcrop reads from the same torch hub dir.

from __future__ import annotations

import logging

from PIL import Image
from pydantic import BaseModel, ConfigDict

from cull.models import CropProposal
from cull.saliency import SaliencyResult

logger = logging.getLogger(__name__)

SMARTCROP_DEFAULT_RATIO: float = 1.0
SMARTCROP_DOWNSAMPLE_LONG_EDGE: int = 512


def _build_crop(image: Image.Image, saliency: SaliencyResult) -> CropProposal:
    """Build a crop proposal via smartcrop, falling back to saliency thirds."""
    from cull.stage2 import composition  # noqa: PLC0415 — lazy monkeypatch seam

    smartcrop_box = composition._try_smartcrop(image)
    if smartcrop_box is not None:
        return CropProposal(
            top=smartcrop_box[1],
            left=smartcrop_box[0],
            bottom=smartcrop_box[3],
            right=smartcrop_box[2],
            source="smartcrop",
        )
    return _saliency_thirds_crop(image, saliency)


def _try_smartcrop(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Run smartcrop on a 512px-downsampled copy; scale bbox back to native dims."""
    native_w, native_h = image.size
    small, scale = _downsample_for_smartcrop(image)
    try:
        import smartcrop  # noqa: PLC0415

        cropper = smartcrop.SmartCrop()
        result = cropper.crop(small, small.width, small.height)
        raw_box = result.get("top_crop") if isinstance(result, dict) else None
        if not raw_box:
            return None
        box = _SmartcropBox.model_validate(raw_box)
    except Exception as exc:  # noqa: BLE001
        logger.warning("smartcrop unavailable: %s", exc)
        return None
    return _upscale_box(box, _UpscaleParams(scale=scale, native=(native_w, native_h)))


def _downsample_for_smartcrop(
    image: Image.Image,
) -> tuple[Image.Image, float]:
    """Return (downsampled_copy, scale_factor) where scale = native / small."""
    long_edge = max(image.width, image.height)
    if long_edge <= SMARTCROP_DOWNSAMPLE_LONG_EDGE:
        return image, 1.0
    scale = long_edge / float(SMARTCROP_DOWNSAMPLE_LONG_EDGE)
    new_w = max(1, int(round(image.width / scale)))
    new_h = max(1, int(round(image.height / scale)))
    small = image.resize((new_w, new_h), Image.LANCZOS)
    return small, scale


class _UpscaleParams(BaseModel):
    """Bundle of scale factor + native (width, height) for bbox upscaling."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scale: float
    native: tuple[int, int]


class _SmartcropBox(BaseModel):
    """Pixel-space smartcrop bbox dict, validated."""

    x: int
    y: int
    width: int
    height: int


def _upscale_box(
    box: _SmartcropBox, params: _UpscaleParams
) -> tuple[int, int, int, int]:
    """Scale a smartcrop bbox back to native dimensions, clamped to frame."""
    left = int(round(box.x * params.scale))
    top = int(round(box.y * params.scale))
    right = int(round((box.x + box.width) * params.scale))
    bottom = int(round((box.y + box.height) * params.scale))
    native_w, native_h = params.native
    left = max(0, min(left, native_w))
    top = max(0, min(top, native_h))
    right = max(0, min(right, native_w))
    bottom = max(0, min(bottom, native_h))
    return (left, top, right, bottom)


def _saliency_thirds_crop(
    image: Image.Image, saliency: SaliencyResult
) -> CropProposal:
    """Fallback crop centered on the saliency bbox scaled to image dimensions."""
    img_w, img_h = image.width, image.height
    if img_w <= 0 or img_h <= 0:
        return CropProposal(
            top=0, left=0, bottom=img_h, right=img_w,
            source="saliency_thirds",
        )
    return CropProposal(
        top=int(saliency.bbox[1] * img_h),
        left=int(saliency.bbox[0] * img_w),
        bottom=int(saliency.bbox[3] * img_h),
        right=int(saliency.bbox[2] * img_w),
        source="saliency_thirds",
    )
