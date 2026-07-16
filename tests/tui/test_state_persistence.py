"""Regression tests for TUI state persistence: atomic writes and .bak recovery."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cull.tui.app import (
    STATE_BACKUP_SUFFIX,
    STATE_TEMP_SUFFIX,
    TuiState,
    _load_state,
    _save_state,
    _state_backup_path,
)


def _state_path(tmp_path: Path) -> Path:
    """Return a scratch state file path under tmp_path."""
    return tmp_path / ".cull_tui_state.json"


def test_save_state_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    """A successful save should not leave a stray .tmp sibling on disk."""
    path = _state_path(tmp_path)
    state = TuiState(overrides={"a.jpg": "keeper"}, current_index=1, current_queue=3)

    _save_state(state, path)

    tmp_sibling = path.with_name(path.name + STATE_TEMP_SUFFIX)
    assert path.exists()
    assert not tmp_sibling.exists()
    assert TuiState.model_validate(json.loads(path.read_text())) == state


def test_save_state_uses_replace_not_truncating_write(tmp_path: Path, monkeypatch) -> None:
    """Simulate an interrupt: if os.replace never runs, the primary file is untouched."""
    path = _state_path(tmp_path)
    original = TuiState(overrides={"a.jpg": "keeper"}, current_index=0, current_queue=0)
    _save_state(original, path)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated crash before replace")

    monkeypatch.setattr("cull.tui.app.os.replace", _boom)
    updated = TuiState(overrides={"b.jpg": "rejected"}, current_index=5, current_queue=1)
    try:
        _save_state(updated, path)
    except RuntimeError:
        pass

    # Primary file must still hold the last fully-committed (pre-crash) state.
    recovered = _load_state(path)
    assert recovered == original


def test_save_state_rotates_prior_valid_state_to_bak(tmp_path: Path) -> None:
    """A second save should copy the previous valid state into a .bak sibling."""
    path = _state_path(tmp_path)
    first = TuiState(overrides={"a.jpg": "keeper"}, current_index=0, current_queue=0)
    second = TuiState(overrides={"b.jpg": "rejected"}, current_index=2, current_queue=1)

    _save_state(first, path)
    _save_state(second, path)

    backup_path = _state_backup_path(path)
    assert backup_path == path.with_name(path.name + STATE_BACKUP_SUFFIX)
    assert backup_path.exists()
    assert TuiState.model_validate(json.loads(backup_path.read_text())) == first
    assert TuiState.model_validate(json.loads(path.read_text())) == second


def test_load_state_recovers_from_bak_when_primary_corrupt(
    tmp_path: Path,
    caplog,
) -> None:
    """A corrupt primary file should fall back to a valid .bak with a logged warning."""
    path = _state_path(tmp_path)
    good = TuiState(overrides={"a.jpg": "keeper"}, current_index=3, current_queue=2)
    backup_path = _state_backup_path(path)
    backup_path.write_text(good.model_dump_json(), encoding="utf-8")
    path.write_text("{not valid json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="cull.tui.app"):
        recovered = _load_state(path)

    assert recovered == good
    assert any("recovered from backup" in record.message for record in caplog.records)


def test_load_state_returns_none_when_both_primary_and_bak_corrupt(
    tmp_path: Path,
    caplog,
) -> None:
    """If both the primary and .bak are corrupt, load should warn and start fresh."""
    path = _state_path(tmp_path)
    backup_path = _state_backup_path(path)
    path.write_text("{not valid json", encoding="utf-8")
    backup_path.write_text("also not valid", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="cull.tui.app"):
        recovered = _load_state(path)

    assert recovered is None
    assert any("no valid backup" in record.message for record in caplog.records)


def test_load_state_missing_file_returns_none(tmp_path: Path) -> None:
    """A missing state file should simply return None, no error."""
    path = _state_path(tmp_path)

    assert _load_state(path) is None
