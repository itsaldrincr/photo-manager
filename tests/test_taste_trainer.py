"""Tests for cull.taste_trainer — batch retrain produces a usable model."""

from __future__ import annotations

import random
from pathlib import Path

import joblib
import pytest

from cull.models import OverrideEntry
from cull.taste_trainer import TasteTrainerInput, _features_for, maybe_retrain, retrain

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
    correct = 0
    for entry in holdout:
        row = _features_for(entry).reshape(1, -1)
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


def test_maybe_retrain_persists_when_batch_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """maybe_retrain triggers retrain once the counter exceeds TASTE_RETRAIN_BATCH."""
    profile_path = tmp_path / "taste.joblib"
    full = _make_balanced_corpus()  # 60 entries > TASTE_RETRAIN_BATCH (50)
    monkeypatch.setattr("cull.taste_trainer.load_overrides", lambda: full)

    result = maybe_retrain(TasteTrainerInput(overrides=full, profile_path=profile_path))

    assert result is not None
    assert profile_path.exists()


def test_maybe_retrain_trains_on_full_history_not_just_the_trip_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single new override that trips the counter must train on the full log.

    Regression test for the bug where app.py's _trigger_retrain passed only
    the newest override into a training call, fitting LogisticRegression on
    one sample and silently never producing a usable profile.
    """
    profile_path = tmp_path / "taste.joblib"
    full_history = _make_balanced_corpus()  # 60 real entries sit on disk
    monkeypatch.setattr("cull.taste_trainer.load_overrides", lambda: full_history)
    newest_entry = full_history[-1]

    # Simulate TASTE_RETRAIN_BATCH already having accumulated via the counter
    # file, and the current call only carrying the single newest override —
    # exactly what tui/app.py._trigger_retrain does per decision.
    counter_path = profile_path.with_suffix(profile_path.suffix + ".counter")
    counter_path.write_text("49", encoding="utf-8")

    result = maybe_retrain(TasteTrainerInput(overrides=[newest_entry], profile_path=profile_path))

    assert result is not None
    # Safe: loading a joblib file this same test just wrote to tmp_path, not
    # an untrusted/external artifact.
    payload = joblib.load(result)
    assert payload["label_count"] == len(full_history)


def test_retrain_skips_single_class_history_without_raising(tmp_path: Path) -> None:
    """A single-class override history (all-keeper or all-reject) skips cleanly."""
    profile_path = tmp_path / "taste.joblib"
    all_keepers = [_make_entry(KEEPER_LABEL, KEEPER_FEATURE_MEAN) for _ in range(5)]

    result = retrain(TasteTrainerInput(overrides=all_keepers, profile_path=profile_path))

    assert result is None
    assert not profile_path.exists()


def test_retrain_persists_scaler_in_the_same_artifact(tmp_path: Path) -> None:
    """The persisted estimator is a scaler+classifier Pipeline in one joblib file."""
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    corpus = _make_balanced_corpus()
    profile_path = tmp_path / "taste.joblib"

    written = retrain(TasteTrainerInput(overrides=corpus, profile_path=profile_path))
    payload = joblib.load(written)
    estimator = payload["estimator"]

    assert isinstance(estimator, Pipeline)
    scaler = estimator.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    # Scaler is fitted (mean_/scale_ populated) and travels inside the one
    # persisted artifact — predict_proba on a raw, unscaled row must work
    # without any separate scaling step at call time.
    assert scaler.mean_ is not None
    row = _features_for(corpus[0]).reshape(1, -1)
    probability = estimator.predict_proba(row)[0, 1]
    assert 0.0 <= probability <= 1.0


def test_retrain_does_not_raise_convergence_warning(tmp_path: Path) -> None:
    """Standardizing features before fitting avoids sklearn's ConvergenceWarning."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning  # noqa: PLC0415

    corpus = _make_balanced_corpus()
    profile_path = tmp_path / "taste.joblib"

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        retrain(TasteTrainerInput(overrides=corpus, profile_path=profile_path))


def test_label_for_treats_select_as_keeper() -> None:
    """The curated 'select' queue counts as a positive taste label, like 'keeper'."""
    from cull.taste_trainer import _label_for  # noqa: PLC0415

    select_entry = _make_entry("select", KEEPER_FEATURE_MEAN)
    reject_entry = _make_entry(REJECT_LABEL, REJECT_FEATURE_MEAN)

    assert _label_for(select_entry) == 1
    assert _label_for(reject_entry) == 0
