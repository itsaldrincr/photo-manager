"""Tests for cull.taste_trainer — batch retrain produces a usable model."""

from __future__ import annotations

import random
from pathlib import Path

import joblib
import pytest

from cull.models import OverrideEntry
from cull.taste_trainer import TasteTrainerInput, maybe_retrain, retrain

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEEPER_LABEL: str = "keeper"
REJECT_LABEL: str = "rejected"
TOTAL_OVERRIDES: int = 60
HELDOUT_SIZE: int = 10
KEEPER_FEATURE_MEAN: float = 0.8
REJECT_FEATURE_MEAN: float = 0.2
FEATURE_NOISE: float = 0.05
RANDOM_SEED: int = 1337
ACCURACY_FLOOR: float = 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(label: str, mean: float) -> OverrideEntry:
    """Build an OverrideEntry with a tiny scalar feature dict near `mean`."""
    return OverrideEntry(
        photo_path="/tmp/x.jpg",
        filename="x.jpg",
        original_decision="uncertain",
        user_decision=label,
        stage1_scores={
            "feat_a": mean + random.uniform(-FEATURE_NOISE, FEATURE_NOISE),
            "feat_b": mean + random.uniform(-FEATURE_NOISE, FEATURE_NOISE),
            "feat_c": mean + random.uniform(-FEATURE_NOISE, FEATURE_NOISE),
        },
        session_source="test",
        override_origin="unit",
    )


def _make_balanced_corpus() -> list[OverrideEntry]:
    """Build a 50/50 keeper/reject corpus separable by feature mean."""
    random.seed(RANDOM_SEED)
    half = TOTAL_OVERRIDES // 2
    keepers = [_make_entry(KEEPER_LABEL, KEEPER_FEATURE_MEAN) for _ in range(half)]
    rejects = [_make_entry(REJECT_LABEL, REJECT_FEATURE_MEAN) for _ in range(half)]
    corpus = keepers + rejects
    random.shuffle(corpus)
    return corpus


def _holdout_accuracy(estimator: object, holdout: list[OverrideEntry]) -> float:
    """Score the trained estimator against a held-out OverrideEntry list."""
    import numpy as np  # noqa: PLC0415

    correct = 0
    for entry in holdout:
        keys = sorted(entry.stage1_scores.keys())
        row = np.asarray([[entry.stage1_scores[k] for k in keys]], dtype=np.float32)
        predicted = int(estimator.predict(row)[0])
        actual = 1 if entry.user_decision == KEEPER_LABEL else 0
        if predicted == actual:
            correct += 1
    return correct / max(len(holdout), 1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retrain_beats_chance_on_holdout(tmp_path: Path) -> None:
    """retrain produces a model that beats 0.5 accuracy on a held-out split."""
    corpus = _make_balanced_corpus()
    train, holdout = corpus[:-HELDOUT_SIZE], corpus[-HELDOUT_SIZE:]
    profile_path = tmp_path / "taste.joblib"

    written = retrain(TasteTrainerInput(overrides=train, profile_path=profile_path))
    payload = joblib.load(written)

    accuracy = _holdout_accuracy(payload["estimator"], holdout)
    assert accuracy > ACCURACY_FLOOR
    assert payload["label_count"] == len(train)
    assert payload["version"].startswith("logreg-v")


def test_maybe_retrain_skips_until_batch_threshold(tmp_path: Path) -> None:
    """maybe_retrain returns None before TASTE_RETRAIN_BATCH labels accumulate."""
    profile_path = tmp_path / "taste.joblib"
    tiny = _make_balanced_corpus()[:2]
    result = maybe_retrain(TasteTrainerInput(overrides=tiny, profile_path=profile_path))
    assert result is None
    assert not profile_path.exists()


def test_maybe_retrain_persists_when_batch_ready(tmp_path: Path) -> None:
    """maybe_retrain triggers retrain once the counter exceeds TASTE_RETRAIN_BATCH."""
    profile_path = tmp_path / "taste.joblib"
    full = _make_balanced_corpus()  # 60 entries > TASTE_RETRAIN_BATCH (50)
    result = maybe_retrain(TasteTrainerInput(overrides=full, profile_path=profile_path))
    assert result is not None
    assert profile_path.exists()
