"""Slim hub for the ``cull`` CLI.

Keeps the Click decorator stack and ``main()`` entry point lightweight by
lazy-loading the heavy pipeline/review modules only when a command path
actually needs them.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

_root = logging.getLogger()
_root.setLevel(logging.CRITICAL)
_root.handlers.clear()
_root.addHandler(logging.NullHandler())

# Bootstrap offline model cache env vars BEFORE any cull.* import that
# transitively pulls torch / transformers / pyiqa / mediapipe / deepface.
from cull.env_bootstrap import bootstrap_default  # noqa: E402
from cull.config import ModelCacheConfig  # noqa: E402

_CACHE: ModelCacheConfig = bootstrap_default()

import click  # noqa: E402

from cull.config import (  # noqa: E402
    CURATE_DEFAULT_TARGET, CURATE_VLM_TIEBREAK_THRESHOLD, CullConfig, VLM_DEFAULT_ALIAS,
)
from cull.model_cache import require_bootstrap_valid  # noqa: E402, F401
from cull.cli_config import (  # noqa: E402, F401
    DEFAULT_BURST_GAP, DEFAULT_STAGES, DEFAULT_THRESHOLD, PRESET_CHOICES,
    _build_config, _build_stages,
)
from cull.cli_help import CullHelp, FAST_MODE_AVAILABLE  # noqa: E402, F401
from cull.cli_bootstrap import (  # noqa: E402, F401
    SETUP_SUBCOMMAND_TOKEN, _is_setup_invocation, setup,
)

BOOTSTRAP_EXIT_CODE: int = 1


def _get_cli_pipeline_module() -> Any:
    """Return the cli_pipeline module on demand."""
    from cull import cli_pipeline  # noqa: PLC0415

    return cli_pipeline


def _get_cli_review_module() -> Any:
    """Return the cli_review module on demand."""
    from cull import cli_review  # noqa: PLC0415

    return cli_review


def _get_cli_subcommands_module() -> Any:
    """Return the cli_subcommands module on demand."""
    from cull import cli_subcommands  # noqa: PLC0415

    return cli_subcommands


def _enforce_bootstrap_gate() -> None:
    """Refuse to run the pipeline unless the offline cache is verified."""
    if _is_setup_invocation():
        return
    result = require_bootstrap_valid(_CACHE)
    if result.is_ok:
        return
    click.echo(f"error: {result.message}", err=True)
    click.echo("run: cull setup --allow-network", err=True)
    sys.exit(BOOTSTRAP_EXIT_CODE)


def _dispatch_subcommand(kwargs: dict) -> bool:
    """Dispatch any subcommand flags via the lazy-loaded subcommand module."""
    return _get_cli_subcommands_module()._dispatch_subcommand(kwargs)


def _launch_review(source: Path, config: CullConfig) -> None:
    """Launch review via the lazy-loaded review module."""
    _get_cli_review_module()._launch_review(source, config)


def _launch_review_session(session_path: Path, config: CullConfig) -> None:
    """Launch an internal serialized review session via the lazy-loaded module."""
    _get_cli_review_module()._launch_review_session(session_path, config)


def _post_pipeline(post_in: object) -> None:
    """Handle post-pipeline results via the lazy-loaded pipeline module."""
    _get_cli_pipeline_module()._post_pipeline(post_in)


def _require_source(kwargs: dict) -> None:
    """Require a SOURCE path via the lazy-loaded pipeline module."""
    _get_cli_pipeline_module()._require_source(kwargs)


def _run_with_dashboard(config: CullConfig, source: Path) -> object:
    """Run the full pipeline with dashboard/error handling."""
    return _get_cli_pipeline_module()._run_with_dashboard(config, source)


def _validate_curate_stages(config: CullConfig) -> None:
    """Validate Stage 4 prerequisites via the lazy-loaded pipeline module."""
    _get_cli_pipeline_module()._validate_curate_stages(config)


def _validate_fast_conflicts(kwargs: dict) -> None:
    """Validate fast-mode flag conflicts via the lazy-loaded pipeline module."""
    _get_cli_pipeline_module()._validate_fast_conflicts(kwargs)


def _validate_subcommand_flags(kwargs: dict) -> None:
    """Validate subcommand flags via the lazy-loaded pipeline module."""
    _get_cli_pipeline_module()._validate_subcommand_flags(kwargs)


def _run_fast_dispatch(kwargs: dict) -> None:
    """Run the fast pipeline without importing its stack during CLI import."""
    if not FAST_MODE_AVAILABLE:
        from cull.dashboard import show_general_error  # noqa: PLC0415

        show_general_error(
            "Fast mode unavailable",
            "The cull_fast package is not installed. Install it to enable --fast.",
        )
        sys.exit(1)
    from cull.pipeline import _PipelineRunInput  # noqa: PLC0415
    from cull_fast.cli_hook import run_fast_pipeline  # noqa: PLC0415

    pipe = _get_cli_pipeline_module()
    _require_source(kwargs)
    config = _build_config(kwargs)
    source = Path(kwargs["source"])
    run_in = _PipelineRunInput(
        config=config, source_path=source, file_size_gb=pipe._compute_file_size_gb(source),
    )
    result = run_fast_pipeline(run_in)
    _post_pipeline(pipe.PostPipelineInput(
        result=result,
        config=config,
        report_flag=bool(kwargs["report"]),
        review_after=bool(kwargs.get("review_after", False)),
    ))


def _run_standard_pipeline(kwargs: dict, config: CullConfig) -> None:
    """Run default pipeline path: source check, dashboard, post-processing."""
    pipe = _get_cli_pipeline_module()
    _require_source(kwargs)
    _validate_curate_stages(config)
    result = _run_with_dashboard(config, Path(kwargs["source"]))
    post_in = pipe.PostPipelineInput(
        result=result,
        config=config,
        report_flag=bool(kwargs["report"]),
        review_after=bool(kwargs["review_after"]),
    )
    _post_pipeline(post_in)


@click.command(cls=CullHelp)
@click.argument("source", type=click.Path(exists=True), required=False, default=None)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--stage", type=int, multiple=True)
@click.option("--no-vlm", is_flag=True, default=False)
@click.option("--portrait", is_flag=True, default=False)
@click.option(
    "--model",
    type=str,
    default=VLM_DEFAULT_ALIAS,
    help="VLM alias (see VLM_ALIASES)",
)
@click.option("--threshold", type=float, default=DEFAULT_THRESHOLD)
@click.option("--burst-gap", type=float, default=DEFAULT_BURST_GAP)
@click.option(
    "--preset",
    type=click.Choice(PRESET_CHOICES),
    default="general",
)
@click.option("--review", is_flag=True, default=False)
@click.option("--review-all", is_flag=True, default=False)
@click.option(
    "--review-after",
    is_flag=True,
    default=False,
    help="Run the pipeline, then launch the review TUI on the final session.",
)
@click.option(
    "--review-session",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    hidden=True,
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Write session_report.json (default: on). Use --no-report to skip.",
)
@click.option(
    "--calibrate",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Auto-bake manifest then bake golden baselines from a fixture corpus.",
)
@click.option(
    "--no-rebake",
    "no_rebake",
    is_flag=True,
    default=False,
    help="With --calibrate: skip the manifest auto-bake (use existing manifest as-is).",
)
@click.option(
    "--bake-manifest",
    "bake_manifest",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Bake manifest.json only (inspection — no ML scoring).",
)
@click.option(
    "--curate",
    is_flag=False,
    flag_value=CURATE_DEFAULT_TARGET,
    default=None,
    type=click.IntRange(min=1),
    help=(
        "Stage 4 curator: peak-moment portrait + peak-moment action + "
        "diversity (MMR) + pairwise tournament + narrative-flow regulariser. "
        "--curate (alone) → top 30; --curate N → top N; absent → disabled."
    ),
)
@click.option(
    "--curate-vlm-threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=CURATE_VLM_TIEBREAK_THRESHOLD,
    help=(
        "Composite-score gap below which Stage 4 calls the VLM to break "
        "a cluster-winner tie (default 0.02). Lower = VLM fires less often."
    ),
)
@click.option(
    "--overrides",
    is_flag=True,
    default=False,
    help="Dump override log to stdout and exit.",
)
@click.option(
    "--vlms",
    is_flag=True,
    default=False,
    help="List discovered VLMs (alias, directory, source, default) and exit.",
)
@click.option(
    "--search",
    type=str,
    default=None,
    help="Semantic text search across photos: 'bride laughing'",
)
@click.option(
    "--similar",
    type=click.Path(exists=True),
    default=None,
    help="Find photos similar to this reference image",
)
@click.option(
    "--explain",
    type=click.Path(exists=True),
    default=None,
    help="Explain why a photo is weak/strong via VLM",
)
@click.option(
    "--report-card",
    is_flag=True,
    default=False,
    help="Generate diagnostic report from session_report.json",
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=None,
    help="Number of search results (default 10)",
)
@click.option(
    "--no-sidecars",
    "no_sidecars",
    is_flag=True,
    default=False,
    help="Skip writing XMP sidecar files alongside source images.",
)
@click.option(
    "--fast",
    is_flag=True,
    default=False,
    hidden=not FAST_MODE_AVAILABLE,
    help="Single-pass MUSIQ scoring (ablation surface; no speed claim until benchmarked).",
)
def _cull_pipeline_command(**kwargs: object) -> None:
    """Run the photo cull pipeline on SOURCE directory."""
    logging.getLogger("cull").setLevel(logging.INFO)
    if kwargs.get("review_session"):
        _launch_review_session(kwargs["review_session"], _build_config(kwargs))
        return
    if kwargs.get("vlms") or kwargs.get("bake_manifest"):
        _dispatch_subcommand(kwargs)
        return
    _enforce_bootstrap_gate()
    _validate_subcommand_flags(kwargs)
    _validate_fast_conflicts(kwargs)
    if _dispatch_subcommand(kwargs):
        return
    if kwargs.get("fast"):
        _run_fast_dispatch(kwargs)
        return
    config = _build_config(kwargs)
    if kwargs["review"] or kwargs["review_all"]:
        _require_source(kwargs)
        _launch_review(Path(kwargs["source"]), config)
        return
    _run_standard_pipeline(kwargs, config)


def main() -> None:
    """Top-level entry point: dispatch `setup` subcommand or pipeline command."""
    if _is_setup_invocation():
        sys.argv = [sys.argv[0]] + [
            arg for arg in sys.argv[1:] if arg != SETUP_SUBCOMMAND_TOKEN
        ]
        setup()
        return
    _cull_pipeline_command()
