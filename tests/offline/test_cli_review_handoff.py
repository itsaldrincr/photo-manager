from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cull.cli_review import (
    REVIEW_EXIT_HANDOFF_ERROR,
    ReviewLaunchInput,
    _launch_review_entry,
    _launch_review_session_file,
)
from cull.cli_results import _launch_review_after
from cull.config import CullConfig
from cull.pipeline import SessionResult, SessionSummary
from cull.review_handoff import ReviewHandoffError


def _make_session(source: Path) -> SessionResult:
    return SessionResult(
        source_path=str(source),
        total_photos=4,
        summary=SessionSummary(uncertain=1),
    )


def test_launch_review_entry_runs_local_review_when_handoff_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    session = _make_session(tmp_path)
    (tmp_path / "session_report.json").write_text("{}", encoding="utf-8")
    review_in = ReviewLaunchInput(config=CullConfig(), source=tmp_path)

    monkeypatch.setattr("cull.cli_review.show_tui_handoff", lambda ctx: None)
    monkeypatch.setattr("cull.cli_review.should_handoff_review", lambda: False)
    monkeypatch.setattr(
        "cull.cli_review._load_session_from_report",
        lambda report_path: session,
    )
    run_app = MagicMock()
    monkeypatch.setattr("cull.cli_review._run_cull_app", run_app)

    _launch_review_entry(review_in)

    run_app.assert_called_once_with(session, review_in.config)


def test_launch_review_entry_handoffs_temp_session_and_cleans_up(
    monkeypatch,
    tmp_path: Path,
) -> None:
    session = _make_session(tmp_path)
    review_in = ReviewLaunchInput(config=CullConfig(), session=session)
    observed: dict[str, Path] = {}

    monkeypatch.setattr("cull.cli_review.show_tui_handoff", lambda ctx: None)
    monkeypatch.setattr("cull.cli_review.should_handoff_review", lambda: True)

    def fake_launch(handoff_in) -> None:  # type: ignore[no-untyped-def]
        observed["session_path"] = handoff_in.session_path
        observed["cwd"] = handoff_in.cwd
        assert handoff_in.session_path.exists()
        loaded = SessionResult.model_validate_json(
            handoff_in.session_path.read_text(encoding="utf-8")
        )
        assert loaded.source_path == str(tmp_path)

    monkeypatch.setattr("cull.cli_review.launch_review_handoff", fake_launch)
    run_app = MagicMock()
    monkeypatch.setattr("cull.cli_review._run_cull_app", run_app)

    _launch_review_entry(review_in)

    assert observed["cwd"] == tmp_path
    assert not observed["session_path"].exists()
    run_app.assert_not_called()


def test_launch_review_session_file_runs_local_review(
    monkeypatch,
    tmp_path: Path,
) -> None:
    session = _make_session(tmp_path)
    session_path = tmp_path / "handoff-session.json"
    session_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    run_app = MagicMock()

    monkeypatch.setattr("cull.cli_review._run_cull_app", run_app)

    _launch_review_session_file(session_path, CullConfig())

    run_app.assert_called_once()
    assert run_app.call_args.args[0].source_path == str(tmp_path)


def test_launch_review_entry_reports_handoff_error_cleanly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    session = _make_session(tmp_path)
    review_in = ReviewLaunchInput(config=CullConfig(), session=session)
    observed: dict[str, Path] = {}
    show_error = MagicMock()

    monkeypatch.setattr("cull.cli_review.show_tui_handoff", lambda ctx: None)
    monkeypatch.setattr("cull.cli_review.show_general_error", show_error)
    monkeypatch.setattr("cull.cli_review.should_handoff_review", lambda: True)

    def fake_launch(handoff_in) -> None:  # type: ignore[no-untyped-def]
        observed["session_path"] = handoff_in.session_path
        raise ReviewHandoffError("Ghostty launch failed.")

    monkeypatch.setattr("cull.cli_review.launch_review_handoff", fake_launch)

    with pytest.raises(SystemExit) as excinfo:
        _launch_review_entry(review_in)

    assert excinfo.value.code == REVIEW_EXIT_HANDOFF_ERROR
    assert not observed["session_path"].exists()
    show_error.assert_called_once_with("Review handoff failed", "Ghostty launch failed.")


def test_launch_review_after_routes_session_to_unified_launcher(
    monkeypatch,
    tmp_path: Path,
) -> None:
    session = _make_session(tmp_path)
    config = CullConfig()
    captured: dict[str, object] = {}

    def fake_launch(review_in: ReviewLaunchInput) -> None:
        captured["review_in"] = review_in

    monkeypatch.setattr("cull.cli_results._launch_review_entry", fake_launch)

    _launch_review_after(session, config)

    review_in = captured["review_in"]
    assert isinstance(review_in, ReviewLaunchInput)
    assert review_in.session == session
    assert review_in.config == config
    assert review_in.source is None
