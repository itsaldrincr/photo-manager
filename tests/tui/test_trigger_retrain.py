"""Tests for CullApp._trigger_retrain — real end-to-end taste-trainer wiring.

Regression coverage for the bug where a single override that tripped the
TASTE_RETRAIN_BATCH counter fit LogisticRegression on one sample (the newest
override only), raised, and was swallowed as a warning — leaving
~/.cull/taste_profile.joblib never created despite hundreds of real overrides.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import joblib
import pytest

from cull.models import OverrideEntry
from cull.tui.app import CullApp

KEEPER_MEAN: float = 0.8
REJECT_MEAN: float = 0.2
HISTORY_SIZE: int = 60  # > TASTE_RETRAIN_BATCH (50)

_photo_ids = itertools.count()


def _make_entry(label: str, mean: float) -> OverrideEntry:
    """Build a minimal OverrideEntry for taste-trainer feature extraction."""
    photo_id = next(_photo_ids)
    return OverrideEntry(
        photo_path=f"/tmp/{photo_id}.jpg",
        filename=f"{photo_id}.jpg",
        original_decision="uncertain",
        user_decision=label,
        stage1_scores={"feat_a": mean, "feat_b": mean},
        session_source="test",
        override_origin="single",
    )


def _make_history() -> list[OverrideEntry]:
    """Build a balanced keeper/reject override history longer than one batch."""
    half = HISTORY_SIZE // 2
    keepers = [_make_entry("keeper", KEEPER_MEAN) for _ in range(half)]
    rejects = [_make_entry("rejected", REJECT_MEAN) for _ in range(half, HISTORY_SIZE)]
    return keepers + rejects


def test_trigger_retrain_trains_on_full_history_when_counter_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single newest override tripping the counter must persist a profile
    trained on the full on-disk override log, not just that one entry."""
    profile_path = tmp_path / "taste.joblib"
    monkeypatch.setattr("cull.tui.app.TASTE_PROFILE_PATH", profile_path)
    history = _make_history()
    monkeypatch.setattr("cull.taste_trainer.load_overrides", lambda: history)

    counter_path = profile_path.with_suffix(profile_path.suffix + ".counter")
    counter_path.write_text(str(HISTORY_SIZE - 1), encoding="utf-8")

    CullApp._trigger_retrain(None, history[-1])

    assert profile_path.exists()
    # Safe: joblib file written by this test to tmp_path, not untrusted input.
    payload = joblib.load(profile_path)
    assert payload["label_count"] == HISTORY_SIZE


def test_trigger_retrain_does_not_raise_on_single_class_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An all-keeper history that trips the counter skips cleanly, no exception."""
    profile_path = tmp_path / "taste.joblib"
    monkeypatch.setattr("cull.tui.app.TASTE_PROFILE_PATH", profile_path)
    all_keepers = [_make_entry("keeper", KEEPER_MEAN) for _ in range(HISTORY_SIZE)]
    monkeypatch.setattr("cull.taste_trainer.load_overrides", lambda: all_keepers)

    counter_path = profile_path.with_suffix(profile_path.suffix + ".counter")
    counter_path.write_text(str(HISTORY_SIZE - 1), encoding="utf-8")

    CullApp._trigger_retrain(None, all_keepers[-1])

    assert not profile_path.exists()
