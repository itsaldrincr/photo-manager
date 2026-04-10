"""Plumbing test: score_from_embeddings feeds the head without touching CLIP."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cull import clip_loader
from cull.stage2 import aesthetic

EMBEDDING_DIM: int = 768
BATCH_SIZE: int = 2


class _RecordingHead(nn.Module):
    """Linear head that records the embeddings it was fed."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(EMBEDDING_DIM, 1)
        self.received: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.received = x.detach().clone()
        return self.linear(x)


class _ExplodingClipModel:
    """Stand-in for CLIPModel whose vision pathways all raise."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(
            f"score_from_embeddings must not touch clip_model.{name}"
        )


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    """Drop aesthetic + clip_loader caches between tests."""
    aesthetic.unload_predictor()
    clip_loader.unload()


def _install_exploding_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch clip_loader with a stub that blows up on any access."""
    monkeypatch.setattr(clip_loader, "get_clip_model", lambda: _ExplodingClipModel())
    monkeypatch.setattr(
        clip_loader, "get_clip_processor", lambda: _ExplodingClipModel()
    )


def _install_stub_head(monkeypatch: pytest.MonkeyPatch) -> _RecordingHead:
    """Replace aesthetic head loader with a stub recording linear head."""
    head_module = _RecordingHead()
    head = aesthetic.AestheticHead(layers=head_module)
    monkeypatch.setattr(aesthetic, "_get_head", lambda _device: (head, object()))
    return head_module


def _make_sentinel_embeddings() -> torch.Tensor:
    """Build a deterministic (BATCH_SIZE, EMBEDDING_DIM) sentinel batch."""
    return torch.arange(
        BATCH_SIZE * EMBEDDING_DIM, dtype=torch.float32
    ).reshape(BATCH_SIZE, EMBEDDING_DIM)


def test_score_from_embeddings_does_not_touch_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """score_from_embeddings must not load or call CLIP — head-only path."""
    _install_exploding_clip(monkeypatch)
    head_module = _install_stub_head(monkeypatch)
    embeddings = _make_sentinel_embeddings()
    scores = aesthetic.score_from_embeddings(
        aesthetic.AestheticHead(layers=head_module), embeddings
    )
    assert len(scores) == BATCH_SIZE
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert head_module.received is not None


def test_score_from_embeddings_feeds_exact_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The head must receive the identity of the embeddings tensor (no copy/reshape)."""
    head_module = _install_stub_head(monkeypatch)
    embeddings = _make_sentinel_embeddings()
    aesthetic.score_from_embeddings(
        aesthetic.AestheticHead(layers=head_module), embeddings
    )
    assert head_module.received is not None
    assert torch.equal(head_module.received, embeddings)
