"""Verify shared CLIP forward yields both patch tokens and image embeddings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from cull import clip_loader
from cull.pipeline import _run_shared_clip_forward

BATCH_SIZE: int = 3
CLIP_HIDDEN_DIM: int = 1024
PROJECTION_DIM: int = 768
TOTAL_TOKENS: int = 257
PATCH_TOKENS: int = 256
PIL_SIZE: int = 224


class _StubInputs:
    """Stand-in for HuggingFace BatchEncoding with a device-moving `.to()`."""

    def __init__(self, pixel_values: torch.Tensor) -> None:
        self.pixel_values = pixel_values

    def to(self, _device: str) -> "_StubInputs":
        return self

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.pixel_values


class _StubProcessor:
    """CLIPProcessor stub returning zero-filled pixel values."""

    def __call__(self, images: list[object], return_tensors: str) -> _StubInputs:
        return _StubInputs(torch.zeros(len(images), 3, PIL_SIZE, PIL_SIZE))


class _StubVisionModel:
    """Vision model stub that records its call count."""

    def __init__(self) -> None:
        self.call_count: int = 0

    def __call__(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.call_count += 1
        batch = pixel_values.shape[0]
        return SimpleNamespace(
            last_hidden_state=torch.randn(batch, TOTAL_TOKENS, CLIP_HIDDEN_DIM),
            pooler_output=torch.randn(batch, CLIP_HIDDEN_DIM),
        )


class _StubClipModel:
    """CLIPModel stub exposing a vision_model and visual_projection."""

    def __init__(self) -> None:
        self.vision_model = _StubVisionModel()
        self.visual_projection = nn.Linear(CLIP_HIDDEN_DIM, PROJECTION_DIM)


def _install_stub_clip(monkeypatch: pytest.MonkeyPatch) -> _StubClipModel:
    """Patch clip_loader singletons with deterministic stubs."""
    stub_model = _StubClipModel()
    monkeypatch.setattr(clip_loader, "get_clip_model", lambda: stub_model)
    monkeypatch.setattr(clip_loader, "get_clip_processor", lambda: _StubProcessor())
    return stub_model


def _make_pil_batch() -> list[Image.Image]:
    """Build a small PIL image batch at the CLIP input size."""
    return [Image.new("RGB", (PIL_SIZE, PIL_SIZE)) for _ in range(BATCH_SIZE)]


def test_shared_forward_calls_vision_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_run_shared_clip_forward must call vision_model exactly once."""
    stub_model = _install_stub_clip(monkeypatch)
    _run_shared_clip_forward(_make_pil_batch(), "cpu")
    assert stub_model.vision_model.call_count == 1


def test_shared_forward_extracts_patch_tokens_and_embeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward output must carry (N,256,1024) tokens and (N,768) image embeds."""
    _install_stub_clip(monkeypatch)
    out = _run_shared_clip_forward(_make_pil_batch(), "cpu")
    assert out.patch_tokens_batch.shape == (BATCH_SIZE, PATCH_TOKENS, CLIP_HIDDEN_DIM)
    assert out.image_embeds.shape == (BATCH_SIZE, PROJECTION_DIM)
