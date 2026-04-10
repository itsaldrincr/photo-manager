"""Tightened invariant test: batch path calls vision_model once, get_image_features zero."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from cull import clip_loader
from cull.config import CullConfig
from cull.pipeline import (
    _BatchCtx,
    _ChunkInput,
    _Stage2LoopInput,
    _process_batch,
)
from cull.stage2 import aesthetic

EMBEDDING_DIM: int = 768
CLIP_HIDDEN_DIM: int = 1024
TOTAL_TOKENS: int = 257
PIL_INPUT_SIZE: int = 224
FIXTURE_IMAGE_SIZE: int = 256


class _StubInputs:
    """Stand-in for HuggingFace BatchEncoding with a `to(device)` method."""

    def __init__(self, pixel_values: torch.Tensor) -> None:
        self.pixel_values = pixel_values

    def to(self, _device: str) -> "_StubInputs":
        return self

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.pixel_values


class _StubProcessor:
    """CLIPProcessor stub returning a deterministic pixel batch."""

    def __call__(self, images: list[object], return_tensors: str) -> _StubInputs:
        return _StubInputs(torch.zeros(len(images), 3, PIL_INPUT_SIZE, PIL_INPUT_SIZE))


class _RecordingVisionModel:
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


class _RecordingClipModel:
    """CLIPModel stub exposing vision_model + visual_projection + a trip-wire."""

    def __init__(self) -> None:
        self.vision_model = _RecordingVisionModel()
        self.visual_projection = nn.Linear(CLIP_HIDDEN_DIM, EMBEDDING_DIM)
        self.get_image_features_calls: int = 0

    def get_image_features(self, **_kwargs: torch.Tensor) -> torch.Tensor:
        self.get_image_features_calls += 1
        raise AssertionError(
            "Batch path must not call clip_model.get_image_features"
        )


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    """Drop aesthetic + clip_loader caches between tests."""
    aesthetic.unload_predictor()
    clip_loader.unload()


def _wire_recorders(monkeypatch: pytest.MonkeyPatch) -> _RecordingClipModel:
    """Patch clip_loader + aesthetic head with recorders; return clip stub."""
    clip_stub = _RecordingClipModel()
    monkeypatch.setattr(clip_loader, "get_clip_model", lambda: clip_stub)
    monkeypatch.setattr(clip_loader, "get_clip_processor", lambda: _StubProcessor())
    head = aesthetic.AestheticHead(layers=nn.Linear(EMBEDDING_DIM, 1))
    monkeypatch.setattr("cull.pipeline._get_head", lambda _device: (head, object()))
    return clip_stub


def _write_fixture_jpeg(tmp_path: Path, name: str) -> Path:
    """Save a synthetic JPEG under tmp_path and return the path."""
    img = Image.new("RGB", (FIXTURE_IMAGE_SIZE, FIXTURE_IMAGE_SIZE), color=(32, 64, 96))
    jpeg_path = tmp_path / name
    img.save(str(jpeg_path), format="JPEG")
    return jpeg_path


def _patch_iqa_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch pyiqa metrics + taste + composition to no-ops for this test."""
    monkeypatch.setattr(
        "cull.pipeline.score_topiq_batch",
        lambda tensor_batch, device: [0.5] * tensor_batch.shape[0],
    )
    monkeypatch.setattr(
        "cull.pipeline.score_clipiqa_batch",
        lambda tensor_batch, device: [0.5] * tensor_batch.shape[0],
    )
    monkeypatch.setattr(
        "cull.pipeline._apply_composition_to_scores",
        lambda iqa_list, paths, ctx: None,
    )
    monkeypatch.setattr(
        "cull.pipeline._apply_taste_to_scores",
        lambda iqa_list, paths: None,
    )
    monkeypatch.setattr(
        "cull.pipeline._apply_subject_blur_to_scores",
        lambda iqa_list, sb_ctx, batch_ctx: {},
    )


def test_batch_path_calls_vision_model_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_scorers: None
) -> None:
    """_process_batch must invoke clip_model.vision_model exactly once per batch."""
    clip_stub = _wire_recorders(monkeypatch)
    _patch_iqa_metrics(monkeypatch)
    paths = [_write_fixture_jpeg(tmp_path, f"img_{i}.jpg") for i in range(3)]
    loop_in = _Stage2LoopInput(
        survivors=paths, config=CullConfig(is_portrait=False), s1_results={}
    )
    batch_ctx = _BatchCtx(loop_in=loop_in, cache=None)
    chunk_in = _ChunkInput(paths=paths, device="cpu")
    _process_batch(chunk_in, batch_ctx)
    assert clip_stub.vision_model.call_count == 1
    assert clip_stub.get_image_features_calls == 0
