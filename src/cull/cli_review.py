"""Review launch helpers for the cull CLI — extracted from cli.py in the 600-series split."""

from __future__ import annotations

import sys
from pathlib import Path

from cull.config import CullConfig
from cull.dashboard import (
    TuiHandoffCtx,
    show_general_error,
    show_tui_handoff,
)
from cull.pipeline import SessionResult

REVIEW_EXIT_MISSING_REPORT: int = 1


def _load_session_from_report(report_path: Path) -> SessionResult:
    """Parse session_report.json into a SessionResult."""
    data = report_path.read_text(encoding="utf-8")
    return SessionResult.model_validate_json(data)


def _run_cull_app(result: SessionResult, config: CullConfig) -> None:
    """Instantiate and run the Textual review TUI."""
    from cull.tui.app import AppInput, CullApp  # noqa: PLC0415

    app = CullApp(AppInput(session=result, config=config))
    app.run()


def _launch_review(source: Path, config: CullConfig) -> None:
    """Load existing session_report.json and launch the TUI."""
    report_path = source / "session_report.json"
    if not report_path.exists():
        show_general_error(
            "No session report",
            f"Expected {report_path} — run `cull {source}` first.",
        )
        sys.exit(REVIEW_EXIT_MISSING_REPORT)
    result = _load_session_from_report(report_path)
    ctx = TuiHandoffCtx(
        uncertain_count=result.summary.uncertain,
        total_count=result.total_photos,
    )
    show_tui_handoff(ctx)
    _run_cull_app(result, config)
