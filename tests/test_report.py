"""Regression tests for cull.report — atomic write_report behaviour."""

from __future__ import annotations

from pathlib import Path

from cull.pipeline import SessionResult, SessionSummary
from cull.report import REPORT_FILENAME, _atomic_write_text, write_report


def _make_session(source: Path) -> SessionResult:
    """Build a minimal session for report-writer unit tests."""
    return SessionResult(
        source_path=str(source),
        total_photos=0,
        summary=SessionSummary(),
    )


def test_write_report_produces_valid_json(tmp_path: Path) -> None:
    """write_report should produce a parseable session_report.json."""
    session = _make_session(tmp_path)

    target = write_report(session, overwrite=True)

    assert target == tmp_path / REPORT_FILENAME
    assert SessionResult.model_validate_json(target.read_text()) == session


def test_write_report_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    """A successful write should not leave a stray .tmp sibling on disk."""
    session = _make_session(tmp_path)

    target = write_report(session, overwrite=True)

    tmp_sibling = target.with_name(target.name + ".tmp")
    assert not tmp_sibling.exists()


def test_write_report_overwrite_replaces_existing_content(tmp_path: Path) -> None:
    """overwrite=True should atomically replace prior report content."""
    first = _make_session(tmp_path)
    write_report(first, overwrite=True)

    second = _make_session(tmp_path)
    second.total_photos = 5
    target = write_report(second, overwrite=True)

    assert SessionResult.model_validate_json(target.read_text()).total_photos == 5


def test_atomic_write_text_survives_interrupted_replace(tmp_path: Path, monkeypatch) -> None:
    """If os.replace is interrupted, the target file must retain its last-good content."""
    target = tmp_path / REPORT_FILENAME
    _atomic_write_text(target, '{"version": 1}')

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated crash before replace")

    monkeypatch.setattr("cull.report.os.replace", _boom)
    try:
        _atomic_write_text(target, '{"version": 2, "corrupt')
    except RuntimeError:
        pass

    assert target.read_text() == '{"version": 1}'
