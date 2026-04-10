"""Test saliency cache population from pre-computed patch tokens."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from cull.config import CLIP_PATCH_GRID, CullConfig
from cull.pipeline import (
    _BatchCtx,
    _build_composition_inputs,
    _build_subject_blur_input,
    _CompositionBuildInput,
    _DualPilBatch,
    _Stage2LoopInput,
    _populate_saliency_cache,
)

IMAGE_SIZE: int = 90
PATCH_TOKEN_COUNT: int = CLIP_PATCH_GRID * CLIP_PATCH_GRID
PATCH_EMBED_DIM: int = 1024


def _make_patch_tokens(batch: int) -> torch.Tensor:
    """Build a deterministic (batch, 256, 1024) patch-token tensor."""
    torch.manual_seed(0)
    return torch.randn(batch, PATCH_TOKEN_COUNT, PATCH_EMBED_DIM)


@pytest.fixture()
def test_jpeg(tmp_path: Path) -> Path:
    """Create a test JPEG on disk."""
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 64, 32))
    jpeg_path = tmp_path / "test.jpg"
    img.save(str(jpeg_path), format="JPEG")
    return jpeg_path


def _build_batch_ctx(jpeg: Path) -> _BatchCtx:
    """Assemble a _BatchCtx with a 1-photo dual_pil batch."""
    pil_full = Image.open(jpeg).convert("RGB")
    dual = _DualPilBatch(
        pil_224=[pil_full.resize((224, 224))],
        pil_1280=[pil_full],
        tensor_1280=torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        paths=[jpeg],
    )
    loop_in = _Stage2LoopInput(
        survivors=[jpeg], config=CullConfig(), s1_results={}
    )
    return _BatchCtx(loop_in=loop_in, cache=None, dual_pil=dual)


def test_saliency_cache_populated_from_tokens(test_jpeg: Path) -> None:
    """_populate_saliency_cache must fill one entry per dual_pil path."""
    ctx = _build_batch_ctx(test_jpeg)
    tokens = _make_patch_tokens(batch=1)
    _populate_saliency_cache(tokens, ctx)
    assert str(test_jpeg) in ctx.saliency_cache
    assert ctx.saliency_cache[str(test_jpeg)] is not None


def test_downstream_consumers_hit_cache_without_recompute(
    test_jpeg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After populating via tokens, composition and subject_blur must NOT recompute."""
    compute_calls: list[object] = []

    def _spy(*_args: object, **_kwargs: object) -> object:
        compute_calls.append(_args)
        raise AssertionError("compute_saliency must not be called after pre-populate")

    monkeypatch.setattr("cull.pipeline.compute_saliency", _spy)
    ctx = _build_batch_ctx(test_jpeg)
    _populate_saliency_cache(_make_patch_tokens(batch=1), ctx)

    build_in = _CompositionBuildInput(paths=[test_jpeg], skip_flags=[False])
    _build_composition_inputs(build_in, ctx)
    _build_subject_blur_input(test_jpeg, ctx)

    assert compute_calls == []
