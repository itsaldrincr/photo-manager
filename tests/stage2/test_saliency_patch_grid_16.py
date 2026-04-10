"""Tests for compute_saliency_from_tokens — synthetic tokens, no real CLIP model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cull.saliency import (
    SaliencyFromTokensRequest,
    SaliencyResult,
    compute_saliency_from_tokens,
)

PATCH_COUNT: int = 256
EMBED_DIM: int = 1024
GRID_SIZE: int = 16
SEED: int = 7


def _make_patch_tokens() -> torch.Tensor:
    """Return a synthetic (256, 1024) float tensor."""
    rng = torch.Generator()
    rng.manual_seed(SEED)
    return torch.randn(PATCH_COUNT, EMBED_DIM, generator=rng)


def test_heatmap_shape_is_16x16() -> None:
    """compute_saliency_from_tokens must return a (16, 16) heatmap."""
    tokens = _make_patch_tokens()
    request = SaliencyFromTokensRequest(patch_tokens=tokens)
    result = compute_saliency_from_tokens(request)
    assert isinstance(result, SaliencyResult)
    assert result.heatmap.shape == (GRID_SIZE, GRID_SIZE)


def test_peak_xy_fractional() -> None:
    """peak_xy values must both be in [0.0, 1.0]."""
    tokens = _make_patch_tokens()
    request = SaliencyFromTokensRequest(patch_tokens=tokens)
    result = compute_saliency_from_tokens(request)
    px, py = result.peak_xy
    assert 0.0 <= px <= 1.0, f"peak_xy x={px} out of [0, 1]"
    assert 0.0 <= py <= 1.0, f"peak_xy y={py} out of [0, 1]"


def test_bbox_fractional() -> None:
    """All four bbox values must be in [0.0, 1.0]."""
    tokens = _make_patch_tokens()
    request = SaliencyFromTokensRequest(patch_tokens=tokens)
    result = compute_saliency_from_tokens(request)
    for i, val in enumerate(result.bbox):
        assert 0.0 <= val <= 1.0, f"bbox[{i}]={val} out of [0, 1]"


def test_heatmap_normalized() -> None:
    """Heatmap values must be in [0.0, 1.0] after min-max normalization."""
    tokens = _make_patch_tokens()
    request = SaliencyFromTokensRequest(patch_tokens=tokens)
    result = compute_saliency_from_tokens(request)
    assert float(result.heatmap.min()) >= 0.0
    assert float(result.heatmap.max()) <= 1.0 + 1e-6
