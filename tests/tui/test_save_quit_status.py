"""Tests for review save feedback in the TUI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from cull.config import CullConfig
from cull.pipeline import SessionResult, SessionSummary
from cull.tui.app import (
    AppInput,
    CullApp,
    SAVE_COMPLETE_MESSAGE,
    SAVE_IN_PROGRESS_MESSAGE,
)


def _make_session(source: Path) -> SessionResult:
    """Build a minimal session for TUI unit tests."""
    return SessionResult(
        source_path=str(source),
        total_photos=1,
        summary=SessionSummary(),
    )


def test_action_save_quit_sets_banner_and_defers_persist(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Save+quit should render a saving banner before running blocking persistence."""
    app = CullApp(AppInput(session=_make_session(tmp_path), config=CullConfig()))
    scheduled: dict[str, object] = {}

    monkeypatch.setattr(app, "_update_info_bar", lambda decision: None)

    def fake_call_after_refresh(callback, *args, **kwargs):  # type: ignore[no-untyped-def]
        scheduled["callback"] = callback
        scheduled["args"] = args
        scheduled["kwargs"] = kwargs
        return True

    monkeypatch.setattr(app, "call_after_refresh", fake_call_after_refresh)

    app.action_save_quit()

    assert app._status_message == SAVE_IN_PROGRESS_MESSAGE
    assert app._save_in_progress is True
    assert scheduled["callback"].__name__ == "_commit_save_and_exit"


def test_commit_save_and_exit_sets_complete_banner_before_exit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Successful saves should show completion feedback before the app exits."""
    app = CullApp(AppInput(session=_make_session(tmp_path), config=CullConfig()))
    state_path = tmp_path / ".cull_tui_state.json"
    state_path.write_text("{}", encoding="utf-8")
    scheduled: dict[str, object] = {}

    monkeypatch.setattr(app, "_update_info_bar", lambda decision: None)
    monkeypatch.setattr("cull.tui.app.execute_moves", lambda decisions, config: None)
    monkeypatch.setattr(
        "cull.tui.app.write_report",
        lambda session, overwrite=True: None,
    )
    monkeypatch.setattr("cull.tui.app._state_path", lambda session: state_path)

    def fake_set_timer(delay, callback, *args, **kwargs):  # type: ignore[no-untyped-def]
        scheduled["delay"] = delay
        scheduled["callback"] = callback
        scheduled["args"] = args
        scheduled["kwargs"] = kwargs
        return True

    monkeypatch.setattr(app, "set_timer", fake_set_timer)

    app._commit_save_and_exit()

    assert app._status_message == SAVE_COMPLETE_MESSAGE
    assert not state_path.exists()
    assert scheduled["delay"] > 0
    assert scheduled["callback"].__name__ == "exit"
