"""Structural test: stored predictor is head-only with no CLIP backbone."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from cull.stage2 import aesthetic

EMBEDDING_DIM: int = 768


class _FakeFullPredictor:
    """Stand-in for AestheticsPredictorV2Linear with .layers + a dummy backbone."""

    def __init__(self) -> None:
        self.layers = nn.Sequential(nn.Linear(EMBEDDING_DIM, 1))
        self.vision_model = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

    @classmethod
    def from_pretrained(cls, _model_id: str) -> "_FakeFullPredictor":
        return cls()


class _FakePredictorModule:
    """Fake `aesthetics_predictor` module exposing AestheticsPredictorV2Linear."""

    AestheticsPredictorV2Linear = _FakeFullPredictor


@pytest.fixture(autouse=True)
def _reset_aesthetic_caches() -> None:
    """Drop any cached heads/processors between tests."""
    aesthetic.unload_predictor()


def _install_fake_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the lazy `aesthetics_predictor` import + processor loader."""
    monkeypatch.setitem(
        __import__("sys").modules, "aesthetics_predictor", _FakePredictorModule
    )
    monkeypatch.setattr(
        "transformers.CLIPProcessor.from_pretrained",
        lambda _id: object(),
    )


def test_stored_predictor_is_head_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_head returns an AestheticHead with .layers and NO CLIP vision backbone."""
    _install_fake_predictor(monkeypatch)
    head, _ = aesthetic._get_head("cpu")
    assert hasattr(head, "layers")
    assert isinstance(head.layers, nn.Module)
    assert not hasattr(head, "vision_model")
    fields: dict[str, Any] = dict(head)
    assert set(fields.keys()) == {"layers"}


def test_extracted_head_drops_full_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cached head must NOT carry a reference to the full predictor."""
    _install_fake_predictor(monkeypatch)
    head = aesthetic._extract_head("cpu")
    out = head.layers(torch.zeros(1, EMBEDDING_DIM))
    assert out.shape == (1, 1)
    assert not hasattr(head, "vision_model")
    assert not hasattr(head, "config")


BATCH_SIZE: int = 3


def test_score_from_embeddings_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """score_from_embeddings returns length-N floats in [0, 1] for synthetic embeddings."""
    _install_fake_predictor(monkeypatch)
    head = aesthetic._extract_head("cpu")
    embeddings = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    scores = aesthetic.score_from_embeddings(head, embeddings)
    assert isinstance(scores, list)
    assert len(scores) == BATCH_SIZE
    for score in scores:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
