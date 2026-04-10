"""Stage 2 subject-region blur — face/saliency-cropped Tenengrad scoring."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from cull.models import SubjectBlurScore
from cull.saliency import SaliencyResult
from cull.stage1.blur import compute_tenengrad
from cull.stage2.portrait import PortraitResult

logger = logging.getLogger(__name__)

PixelBBox = tuple[int, int, int, int]


class SubjectBlurInput(BaseModel):
    """Inputs for one subject-region blur measurement."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pil_1280: Image.Image
    portrait: PortraitResult | None = None
    saliency: SaliencyResult | None = None


def pil_to_gray(pil: Image.Image) -> np.ndarray:
    """Convert a PIL image to a uint8 grayscale numpy array."""
    return np.asarray(pil.convert("L"), dtype=np.uint8)


def compute_subject_blur(sb_input: SubjectBlurInput) -> SubjectBlurScore:
    """Compute subject-region Tenengrad with face/saliency/global fallback."""
    gray = pil_to_gray(sb_input.pil_1280)
    bbox, source, has_subject = _resolve_region(sb_input, gray.shape)
    crop = _crop(gray, bbox)
    score = compute_tenengrad(crop) if crop.size > 0 else 0.0
    return SubjectBlurScore(tenengrad=score, subject_region_source=source, has_subject=has_subject)


def score_one(sb_input: SubjectBlurInput) -> SubjectBlurScore:
    """Thin alias for compute_subject_blur — kept for backward compatibility."""
    return compute_subject_blur(sb_input)


def _resolve_region(
    sb_input: SubjectBlurInput, shape: tuple[int, int]
) -> tuple[PixelBBox, str, bool]:
    """Pick the cropping region from face / saliency / global fallback."""
    if sb_input.portrait is not None and sb_input.portrait.has_face and sb_input.portrait.face_bbox is not None:
        return sb_input.portrait.face_bbox, "face", True
    if sb_input.saliency is not None:
        return _saliency_to_pixels(sb_input.saliency, shape), "saliency_peak", True
    return (0, 0, shape[1], shape[0]), "global", False


def _saliency_to_pixels(saliency: SaliencyResult, shape: tuple[int, int]) -> PixelBBox:
    """Scale fractional saliency bbox to image pixel coordinates."""
    h_img, w_img = shape
    x0, y0, x1, y1 = saliency.bbox
    return (
        int(x0 * w_img),
        int(y0 * h_img),
        max(int(x1 * w_img), int(x0 * w_img) + 1),
        max(int(y1 * h_img), int(y0 * h_img) + 1),
    )


def _crop(image: np.ndarray, bbox: PixelBBox) -> np.ndarray:
    """Crop a grayscale image to the given pixel bbox, clipped to bounds."""
    h, w = image.shape
    x0 = max(0, min(bbox[0], w))
    y0 = max(0, min(bbox[1], h))
    x1 = max(x0, min(bbox[2], w))
    y1 = max(y0, min(bbox[3], h))
    return image[y0:y1, x0:x1]
