"""Mocked unit tests for cull_fast.musiq normalization, env gate, fp16 gate, and unload.

No real ML models are loaded. All metric calls go through monkeypatched stubs.
Tests are authored for inspection; see user no_test_runs memory note.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

import cull_fast.musiq as musiq_mod
from cull_fast.musiq import (
    MUSIQ_AVA_MAX_MOS,
    MUSIQ_AVA_MIN_MOS,
    MUSIQ_KONIQ_MAX_MOS,
    MUSIQ_USE_FP16,
    MPS_FALLBACK_ENABLED,
    MPS_FALLBACK_ENV_VAR,
    _MusiQBatchRequest,
    _MUSIQ_METRICS,
    _normalize_musiq_ava,
    _normalize_musiq_koniq,
    unload_musiq,
)

KONIQ_SCORE_LOW: float = 42.0
KONIQ_SCORE_HIGH: float = 76.0
AVA_SCORE_LOW: float = 3.4
AVA_SCORE_HIGH: float = 5.7
BATCH_SIZE: int = 2
IMAGE_CHANNELS: int = 3
IMAGE_DIM: int = 224

# ---------------------------------------------------------------------------
# KonIQ normalization boundary tests
# ---------------------------------------------------------------------------


def test_normalize_koniq_at_zero() -> None:
    """Zero raw score maps to 0.0."""
    assert _normalize_musiq_koniq(0.0) == 0.0


def test_normalize_koniq_at_max() -> None:
    """Max raw score maps to exactly 1.0."""
    assert _normalize_musiq_koniq(MUSIQ_KONIQ_MAX_MOS) == 1.0


def test_normalize_koniq_overshoot() -> None:
    """Raw scores above max are clamped to 1.0."""
    assert _normalize_musiq_koniq(150.0) == 1.0


def test_normalize_koniq_undershoot() -> None:
    """Negative raw scores are clamped to 0.0."""
    assert _normalize_musiq_koniq(-5.0) == 0.0


# ---------------------------------------------------------------------------
# AVA normalization boundary tests
# ---------------------------------------------------------------------------


def test_normalize_ava_at_min() -> None:
    """Minimum AVA score maps to 0.0."""
    assert _normalize_musiq_ava(MUSIQ_AVA_MIN_MOS) == 0.0


def test_normalize_ava_at_max() -> None:
    """Maximum AVA score maps to 1.0."""
    assert _normalize_musiq_ava(MUSIQ_AVA_MAX_MOS) == 1.0


def test_normalize_ava_midpoint() -> None:
    """Midpoint of AVA range maps to approximately 0.5."""
    midpoint = (MUSIQ_AVA_MIN_MOS + MUSIQ_AVA_MAX_MOS) / 2.0
    assert _normalize_musiq_ava(midpoint) == pytest.approx(0.5)


def test_normalize_ava_overshoot() -> None:
    """AVA scores above max are clamped to 1.0."""
    assert _normalize_musiq_ava(15.0) == 1.0


def test_normalize_ava_undershoot() -> None:
    """AVA scores at or below zero (below min) are clamped to 0.0."""
    assert _normalize_musiq_ava(0.0) == 0.0


# ---------------------------------------------------------------------------
# Lifecycle and safety tests
# ---------------------------------------------------------------------------


def test_mps_fallback_env_var_set() -> None:
    """Importing cull_fast.musiq sets PYTORCH_ENABLE_MPS_FALLBACK=1."""
    assert os.environ.get(MPS_FALLBACK_ENV_VAR) == MPS_FALLBACK_ENABLED


def test_fp16_gate_off() -> None:
    """MUSIQ_USE_FP16 is False (fp32-only, guards against MPS softmax overflow)."""
    assert MUSIQ_USE_FP16 is False


def test_unload_musiq_clears_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """unload_musiq() empties _MUSIQ_METRICS dict."""
    monkeypatch.setitem(_MUSIQ_METRICS, "x", object())
    unload_musiq()
    assert _MUSIQ_METRICS == {}


def _make_fixed_metric(raw_tensor: torch.Tensor) -> MagicMock:
    """Return a callable mock that yields raw_tensor when called."""
    metric = MagicMock()
    metric.return_value = raw_tensor
    return metric


def test_score_musiq_batch_returns_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    """score_musiq_batch normalizes both metrics; results are in [0, 1]."""
    koniq_tensor = torch.tensor([[KONIQ_SCORE_LOW], [KONIQ_SCORE_HIGH]])
    ava_tensor = torch.tensor([[AVA_SCORE_LOW], [AVA_SCORE_HIGH]])

    def _fake_get_musiq_metric(name: str) -> MagicMock:
        if name == "musiq":
            return _make_fixed_metric(koniq_tensor)
        return _make_fixed_metric(ava_tensor)

    monkeypatch.setattr(musiq_mod, "_get_musiq_metric", _fake_get_musiq_metric)

    paths = [Path("/fake/a.jpg"), Path("/fake/b.jpg")]
    batch = torch.zeros(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_DIM, IMAGE_DIM)
    req = _MusiQBatchRequest(tensor_batch=batch, photo_paths=paths, device="cpu")

    result = musiq_mod.score_musiq_batch(req)

    assert len(result) == BATCH_SIZE
    for pair in result:
        assert 0.0 <= pair.technical <= 1.0
        assert 0.0 <= pair.aesthetic <= 1.0
