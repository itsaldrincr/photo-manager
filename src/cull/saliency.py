"""gScoreCAM-style saliency map generator using the shared CLIP singleton."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict

from cull.clip_loader import get_clip_model, get_clip_processor
from cull.config import SALIENCY_TARGET_PX

logger = logging.getLogger(__name__)

SALIENCY_THRESHOLD: float = 0.5
PATCH_TOKEN_COUNT: int = 256
PATCH_EMBED_DIM: int = 1024


class SaliencyRequest(BaseModel):
    """Input parameters for a saliency computation."""

    image_path: Path
    target_px: int = SALIENCY_TARGET_PX


class SaliencyFromTokensRequest(BaseModel):
    """Input for token-based saliency: raw patch tokens from a vision encoder."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    patch_tokens: torch.Tensor
    grid_size: int = 16


class SaliencyResult(BaseModel):
    """Output of a saliency map computation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    heatmap: np.ndarray
    peak_xy: tuple[float, float]
    bbox: tuple[float, float, float, float]


class _ActivationInput(BaseModel):
    """Bundle of image + CLIP handles for activation extraction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image
    clip_model: object
    clip_processor: object


def compute_saliency(request: SaliencyRequest) -> SaliencyResult:
    """Compute a gScoreCAM saliency map for the image at request.image_path."""
    act_input = _ActivationInput(
        image=_load_image(request.image_path, request.target_px),
        clip_model=get_clip_model(),
        clip_processor=get_clip_processor(),
    )
    patch_tokens = _extract_patch_tokens(act_input)
    grid_size = int(patch_tokens.shape[0] ** 0.5)
    tokens_request = SaliencyFromTokensRequest(patch_tokens=patch_tokens, grid_size=grid_size)
    return compute_saliency_from_tokens(tokens_request)


def compute_saliency_from_tokens(request: SaliencyFromTokensRequest) -> SaliencyResult:
    """Compute saliency heatmap from raw patch tokens via L2 norm reshape."""
    heatmap = _tokens_to_heatmap(request.patch_tokens, request.grid_size)
    return SaliencyResult(
        heatmap=heatmap,
        peak_xy=_peak_from_heatmap_fractional(heatmap),
        bbox=_bbox_from_heatmap_fractional(heatmap),
    )


def _load_image(image_path: Path, target_px: int) -> Image.Image:
    """Open and resize image so its long edge equals target_px."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = target_px / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)


def _extract_patch_tokens(act_input: _ActivationInput) -> torch.Tensor:
    """Run vision encoder and return patch tokens (no CLS) as tensor."""
    inputs = act_input.clip_processor(images=act_input.image, return_tensors="pt")
    device = next(act_input.clip_model.parameters()).device
    pixel_values: torch.Tensor = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vision_out = act_input.clip_model.vision_model(pixel_values=pixel_values)
    return vision_out.last_hidden_state[0, 1:, :]


def _tokens_to_heatmap(patch_tokens: torch.Tensor, grid_size: int) -> np.ndarray:
    """Compute per-patch L2 norm, reshape to grid, and min-max normalize."""
    activation = patch_tokens.norm(dim=-1).cpu().float().numpy()
    heatmap = activation.reshape(grid_size, grid_size)
    denom = heatmap.max() - heatmap.min() + 1e-8
    return (heatmap - heatmap.min()) / denom


def _peak_from_heatmap_fractional(heatmap: np.ndarray) -> tuple[float, float]:
    """Return (x, y) of the highest-activation cell as fractional [0, 1] coords."""
    flat_idx = int(np.argmax(heatmap))
    row, col = divmod(flat_idx, heatmap.shape[1])
    return (col / heatmap.shape[1], row / heatmap.shape[0])


def _bbox_from_heatmap_fractional(
    heatmap: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (x0, y0, x1, y1) bounding box as fractional [0, 1] coords."""
    mask = heatmap >= SALIENCY_THRESHOLD
    h, w = heatmap.shape
    if not mask.any():
        return (0.0, 0.0, 1.0, 1.0)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return (cols[0] / w, rows[0] / h, cols[-1] / w, rows[-1] / h)
