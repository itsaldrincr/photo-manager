"""Review launch helpers for the cull CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, ConfigDict

from cull.config import CullConfig
from cull.dashboard import TuiHandoffCtx, show_general_error, show_tui_handoff
from cull.pipeline import SessionResult
from cull.report import REPORT_FILENAME
from cull.review_handoff import (
    ReviewHandoffInput,
    ReviewHandoffError,
    ReviewHandoffUnavailable,
    ensure_handoff_available,
    launch_review_handoff,
    should_handoff_review,
)

REVIEW_EXIT_MISSING_REPORT: int = 1
REVIEW_EXIT_HANDOFF_ERROR: int = 1


class ReviewHandoffPayload(BaseModel):
    """What the Ghostty child needs: the session and the parent's config.

    The child's own CLI flags are defaults only, so without the config here
    --model, --preset and --no-sidecars would be lost across the handoff.
    """

    session: SessionResult
    config: CullConfig


class ReviewLaunchInput(BaseModel):
    """Inputs for launching review from disk or an in-memory session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: CullConfig
    source: Path | None = None
    session: SessionResult | None = None


def _load_session_from_report(report_path: Path) -> SessionResult:
    """Parse ``session_report.json`` into a SessionResult."""
    data = report_path.read_text(encoding="utf-8")
    return SessionResult.model_validate_json(data)


def _load_handoff_file(session_path: Path, default_config: CullConfig) -> ReviewHandoffPayload:
    """Parse a handoff file; a bare session (older format) keeps the CLI config."""
    data = json.loads(session_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and {"session", "config"} <= data.keys():
        return ReviewHandoffPayload.model_validate(data)
    return ReviewHandoffPayload(session=SessionResult.model_validate(data), config=default_config)


def _run_cull_app(result: SessionResult, config: CullConfig) -> None:
    """Instantiate and run the Textual review TUI."""
    from cull.tui.app import AppInput, CullApp  # noqa: PLC0415

    app = CullApp(AppInput(session=result, config=config))
    app.run()


def _load_review_session(review_in: ReviewLaunchInput) -> SessionResult:
    """Resolve the review session from disk or an in-memory pipeline result."""
    if review_in.session is not None:
        return review_in.session
    if review_in.source is None:
        raise ValueError("Review launch requires either a source path or a session.")
    report_path = review_in.source / REPORT_FILENAME
    if not report_path.exists():
        show_general_error(
            "No session report",
            f"Expected {report_path} — run `cull {review_in.source}` first.",
        )
        sys.exit(REVIEW_EXIT_MISSING_REPORT)
    return _load_session_from_report(report_path)


def _write_temp_review_session(payload: ReviewHandoffPayload) -> Path:
    """Write a temporary serialized review session + config for the Ghostty child."""
    with NamedTemporaryFile(
        mode="w",
        suffix="-review-session.json",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(payload.model_dump_json(indent=2))
        return Path(handle.name)


def _resolve_review_cwd(review_in: ReviewLaunchInput, result: SessionResult) -> Path:
    """Choose the working directory used for Ghostty handoff."""
    if review_in.source is not None:
        return review_in.source
    return Path(result.source_path)


def _build_tui_handoff_ctx(result: SessionResult) -> TuiHandoffCtx:
    """Build the dashboard banner context for review launch."""
    return TuiHandoffCtx(
        uncertain_count=result.summary.uncertain,
        total_count=result.total_photos,
    )


def _launch_review_session_file(session_path: Path, config: CullConfig) -> None:
    """Launch the review TUI from a serialized session file."""
    payload = _load_handoff_file(session_path, config)
    _run_cull_app(payload.session, payload.config)


def _can_handoff() -> bool:
    """Return True when Ghostty and `cull` exist; otherwise say so and return False."""
    try:
        ensure_handoff_available()
    except ReviewHandoffUnavailable as exc:
        show_general_error("Ghostty handoff unavailable", f"{exc} Reviewing in this terminal instead.")
        return False
    return True


def _handoff_to_ghostty(review_in: ReviewLaunchInput, result: SessionResult) -> None:
    """Run review in a new Ghostty window and wait for it to close."""
    session_path = _write_temp_review_session(ReviewHandoffPayload(session=result, config=review_in.config))
    try:
        launch_review_handoff(ReviewHandoffInput(
            cwd=_resolve_review_cwd(review_in, result),
            session_path=session_path,
        ))
    except ReviewHandoffError as exc:
        show_general_error("Review handoff failed", str(exc))
        sys.exit(REVIEW_EXIT_HANDOFF_ERROR)
    finally:
        session_path.unlink(missing_ok=True)


def _launch_review_entry(review_in: ReviewLaunchInput) -> None:
    """Launch review locally or hand it off to Ghostty when needed."""
    result = _load_review_session(review_in)
    show_tui_handoff(_build_tui_handoff_ctx(result))
    if should_handoff_review() and _can_handoff():
        _handoff_to_ghostty(review_in, result)
        return
    _run_cull_app(result, review_in.config)


def _launch_review(source: Path, config: CullConfig) -> None:
    """Load an existing report and launch the review flow."""
    _launch_review_entry(ReviewLaunchInput(config=config, source=source))


def _launch_review_session(session_path: Path, config: CullConfig) -> None:
    """Launch a serialized internal review session without re-handoff."""
    _launch_review_session_file(session_path, config)
