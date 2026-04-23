"""Results display, file moves, report, and TUI launch helpers — extracted from cli.py in the 600-series split."""

from __future__ import annotations

from pathlib import Path

# one-way dep: cli_results -> cli_review
from cull.cli_review import ReviewLaunchInput, _launch_review_entry
from cull.config import CullConfig
from cull.dashboard import (
    DryRunSummary,
    ResultsSummary,
    show_dry_run_results,
    show_move_complete,
    show_report_writing,
    show_results_card,
)
from cull.pipeline import SessionResult
from cull.report import write_report
from cull.router import MoveEntry


def _show_dry_run(result: SessionResult) -> None:
    """Display dry-run results via dashboard."""
    summary = DryRunSummary(
        keepers=result.summary.keepers,
        rejected=result.summary.rejected,
        duplicates=result.summary.duplicates,
        uncertain=result.summary.uncertain,
        total=result.total_photos,
    )
    show_dry_run_results(summary)


def _show_results(result: SessionResult, config: CullConfig) -> None:
    """Display pipeline results card via dashboard."""
    summary = ResultsSummary(
        keepers=result.summary.keepers,
        rejected=result.summary.rejected,
        duplicates=result.summary.duplicates,
        uncertain=result.summary.uncertain,
        selected=result.summary.selected,
        total=result.total_photos,
        elapsed_seconds=result.timing.total_seconds,
        stages_run=config.stages,
    )
    show_results_card(summary)


def _move_files(result: SessionResult, config: CullConfig) -> None:
    """Execute file moves silently, then show a completion line."""
    movable = [d for d in result.decisions if d.decision != "keeper"]
    if not movable:
        return
    error_count = 0
    for decision in movable:
        entry = _execute_single_move(decision, config)
        if entry and not entry.is_success:
            error_count += 1
    show_move_complete(len(movable), error_count)


def _execute_single_move(decision: object, config: CullConfig) -> MoveEntry | None:
    """Move one non-keeper file using the router module."""
    from cull.router import process_single_move  # noqa: PLC0415

    return process_single_move(decision, config)


def _write_report(result: SessionResult) -> None:
    """Write session report with dashboard feedback."""
    source_dir = Path(result.source_path)
    report_path = source_dir / "session_report.json"
    write_report(result)
    show_report_writing(report_path)


def _launch_review_after(result: SessionResult, config: CullConfig) -> None:
    """Launch review on the final pipeline session via the unified entry path."""
    _launch_review_entry(ReviewLaunchInput(config=config, session=result))
