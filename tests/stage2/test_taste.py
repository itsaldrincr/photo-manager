"""Tests for cull.stage2.taste — warm-start + scored-path behaviour with mocks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cull.stage2.taste as taste
from cull.stage2.taste import TasteScoreInput

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMSTART_PROBABILITY: float = 0.5
WARMSTART_WEIGHT: float = 0.0
SCALAR_FEATURE_VECTOR: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
TRAINED_LABEL_COUNT: int = 25
PROBE_PROBABILITY: float = 0.7
PROFILE_VERSION: str = "logreg-v25"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubEstimator:
    """Stub LogisticRegression-like estimator with a fixed predict_proba."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        rows = matrix.shape[0]
        return np.tile([1.0 - self.probability, self.probability], (rows, 1))


def _make_input(tmp_path: Path) -> TasteScoreInput:
    """Build a TasteScoreInput with a synthetic scalar feature row."""
    return TasteScoreInput(
        image_path=tmp_path / "fake.jpg",
        scalar_features=np.asarray(SCALAR_FEATURE_VECTOR, dtype=np.float32),
    )


def _patch_no_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the taste profile path at a non-existent file and reset cache."""
    monkeypatch.setattr(taste, "TASTE_PROFILE_PATH", tmp_path / "missing.joblib")
    taste._reset_profile_cache()


def _patch_with_profile(
    monkeypatch: pytest.MonkeyPatch, profile: taste._TasteProfile
) -> None:
    """Inject a fake taste profile and stub joblib + CLIP loaders."""
    monkeypatch.setattr(taste, "_profile_cache", profile, raising=True)
    monkeypatch.setattr(taste, "_load_attempted", True, raising=True)

    def _fake_embed(_path: Path) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(taste, "_embed_clip", _fake_embed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_warmstart_when_profile_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """score_one returns the warm-start TasteScore when no profile exists."""
    _patch_no_profile(monkeypatch, tmp_path)
    result = taste.score_one(_make_input(tmp_path))
    assert result.probability == WARMSTART_PROBABILITY
    assert result.weight_applied == WARMSTART_WEIGHT
    assert result.model_version == "warmstart"


def test_warmstart_when_label_count_below_min(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """score_one returns warm-start when loaded profile has too few labels."""
    profile = taste._TasteProfile(
        estimator=_StubEstimator(PROBE_PROBABILITY), label_count=1, version="cold"
    )
    _patch_with_profile(monkeypatch, profile)
    result = taste.score_one(_make_input(tmp_path))
    assert result.probability == WARMSTART_PROBABILITY
    assert result.weight_applied == WARMSTART_WEIGHT


def test_score_in_unit_interval_with_trained_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """score_one returns probability in [0, 1] when a trained profile is loaded."""
    profile = taste._TasteProfile(
        estimator=_StubEstimator(PROBE_PROBABILITY),
        label_count=TRAINED_LABEL_COUNT,
        version=PROFILE_VERSION,
    )
    _patch_with_profile(monkeypatch, profile)
    result = taste.score_one(_make_input(tmp_path))
    assert 0.0 <= result.probability <= 1.0
    assert math.isclose(result.probability, PROBE_PROBABILITY, abs_tol=1e-6)
    assert result.label_count_at_score == TRAINED_LABEL_COUNT
    assert result.weight_applied == 1.0


def test_score_batch_returns_one_per_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """score_batch returns one TasteScore per input even in warm-start mode."""
    _patch_no_profile(monkeypatch, tmp_path)
    inputs = [_make_input(tmp_path), _make_input(tmp_path), _make_input(tmp_path)]
    results = taste.score_batch(inputs)
    assert len(results) == len(inputs)
    for result in results:
        assert result.probability == WARMSTART_PROBABILITY
