"""Pipeline runners and post-processing helpers — extracted from cli.py in the 600-series split."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from cull.cli_config import _build_config
from cull.cli_help import FAST_MODE_AVAILABLE
from cull.cli_results import (
    _launch_review_after,
    _move_files,
    _show_dry_run,
    _show_results,
    _write_report,
)
from cull.config import CullConfig, VLM_MODELS_ROOT
from cull.dashboard import show_general_error, show_vlm_load_error
from cull.pipeline import SessionResult, _PipelineRunInput, run_pipeline
from cull.vlm_session import VlmLoadError

BYTES_PER_GB: float = 1024 * 1024 * 1024
VALIDATION_EXIT_CODE: int = 1
SUBCOMMAND_FLAGS: tuple[str, ...] = (
    "overrides", "search", "similar", "explain", "report_card", "calibrate",
    "bake_manifest",
)
REVIEW_FLAGS: tuple[str, ...] = ("review", "review_all", "review_after")
CURATE_REQUIRED_STAGES: tuple[int, ...] = (1, 2)


class PostPipelineInput(BaseModel):
    """Inputs for post-pipeline handling (results, moves, report, TUI)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    result: SessionResult
    config: CullConfig
    report_flag: bool = False
    review_after: bool = False


def _require_source(kwargs: dict) -> None:
    """Exit with error if no source argument was supplied."""
    if not kwargs.get("source"):
        show_general_error(
            "Missing source",
            "cull requires a SOURCE directory argument (or a subcommand flag).",
        )
        sys.exit(VALIDATION_EXIT_CODE)


def _reject_subcommand_conflict(active_review: list[str]) -> None:
    """Emit review-vs-subcommand conflict error and exit."""
    show_general_error(
        "Conflicting flags",
        f"--{active_review[0].replace('_', '-')} cannot be combined with subcommand flags.",
    )
    sys.exit(VALIDATION_EXIT_CODE)


def _validate_subcommand_flags(kwargs: dict) -> None:
    """Ensure at most one subcommand flag is set; exit with error if more."""
    active = [f for f in SUBCOMMAND_FLAGS if kwargs.get(f)]
    if len(active) > 1:
        show_general_error(
            "Conflicting flags",
            f"Only one subcommand allowed at a time. Got: {active}",
        )
        sys.exit(VALIDATION_EXIT_CODE)
    active_review = [f for f in REVIEW_FLAGS if kwargs.get(f)]
    if active_review and active:
        _reject_subcommand_conflict(active_review)


def _validate_fast_conflicts(kwargs: dict) -> None:
    """Refuse --fast combined with any subcommand or review flag (spec §9 Q2)."""
    if not kwargs.get("fast"):
        return
    conflicts = [f for f in SUBCOMMAND_FLAGS if kwargs.get(f)]
    conflicts.extend(f for f in REVIEW_FLAGS if kwargs.get(f))
    if conflicts:
        show_general_error(
            "Conflicting flags",
            f"--fast cannot be combined with: {conflicts}",
        )
        sys.exit(VALIDATION_EXIT_CODE)


def _validate_curate_stages(config: CullConfig) -> None:
    """Stage 4 requires Stages 1 and 2 to run — fail fast with a clear error."""
    if config.curate_target is None:
        return
    missing = [s for s in CURATE_REQUIRED_STAGES if s not in config.stages]
    if not missing:
        return
    show_general_error(
        "Invalid stage combination",
        f"--curate requires Stages 1 and 2 to run. Missing: {missing}. "
        f"Remove --stage, or include at least stages 1 and 2.",
    )
    sys.exit(VALIDATION_EXIT_CODE)


def _compute_file_size_gb(source: Path) -> float:
    """Sum the size of all files under source in gigabytes."""
    total = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    return total / BYTES_PER_GB


def _run_with_dashboard(config: CullConfig, source: Path) -> SessionResult:
    """Run pipeline with error handling and dashboard panels."""
    file_size_gb = _compute_file_size_gb(source)
    try:
        run_in = _PipelineRunInput(
            config=config, source_path=source, file_size_gb=file_size_gb,
        )
        return run_pipeline(run_in)
    except VlmLoadError as exc:
        alias = getattr(exc, "alias", str(exc))
        show_vlm_load_error(alias, VLM_MODELS_ROOT)
        sys.exit(1)
    except Exception as exc:
        show_general_error("Pipeline Error", str(exc))
        raise


def _run_fast_dispatch(kwargs: dict) -> None:
    """Lazy-import and dispatch to the fast-mode pipeline (Pattern B)."""
    if not FAST_MODE_AVAILABLE:
        show_general_error(
            "Fast mode unavailable",
            "The cull_fast package is not installed. Install it to enable --fast.",
        )
        sys.exit(1)
    from cull_fast.cli_hook import run_fast_pipeline  # noqa: PLC0415

    _require_source(kwargs)
    config = _build_config(kwargs)
    source = Path(kwargs["source"])
    run_in = _PipelineRunInput(
        config=config, source_path=source, file_size_gb=_compute_file_size_gb(source),
    )
    result = run_fast_pipeline(run_in)
    post_in = PostPipelineInput(
        result=result,
        config=config,
        report_flag=bool(kwargs["report"]),
        review_after=bool(kwargs.get("review_after", False)),
    )
    _post_pipeline(post_in)


def _post_pipeline(post_in: PostPipelineInput) -> None:
    """Handle results display, file moves, report, and TUI handoff."""
    result = post_in.result
    config = post_in.config
    if config.is_dry_run:
        _show_dry_run(result)
        return
    _show_results(result, config)
    _move_files(result, config)
    if post_in.report_flag:
        _write_report(result)
    if post_in.review_after:
        _launch_review_after(result, config)
