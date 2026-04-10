"""Slim hub for the ``cull`` CLI — re-exports every helper from sibling
``cli_*`` modules, keeps the frozen Click decorator stack and ``main()``
entry point, and owns ``_enforce_bootstrap_gate`` locally so tests that
monkeypatch ``cli.require_bootstrap_valid`` intercept correctly.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

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
from cull.pipeline import run_pipeline  # noqa: E402, F401
from cull.cli_config import (  # noqa: E402, F401
    DEFAULT_BURST_GAP, DEFAULT_STAGES, DEFAULT_THRESHOLD, PRESET_CHOICES,
    _build_config, _build_stages,
)
from cull.cli_help import CullHelp, FAST_MODE_AVAILABLE  # noqa: E402, F401
from cull.cli_review import _launch_review, _load_session_from_report  # noqa: E402, F401
from cull.cli_bootstrap import (  # noqa: E402, F401
    SETUP_SUBCOMMAND_TOKEN, _is_setup_invocation, setup,
)
from cull.cli_results import (  # noqa: E402, F401
    _execute_single_move, _launch_review_after, _move_files, _run_cull_app,
    _show_dry_run, _show_results, _write_report,
)
from cull.cli_subcommands import (  # noqa: E402, F401
    _dispatch_subcommand, _run_explain, _run_overrides_dump, _run_report_card,
    _run_search_similar, _run_search_text,
)
from cull.cli_pipeline import (  # noqa: E402, F401
    BYTES_PER_GB, PostPipelineInput, _compute_file_size_gb, _post_pipeline,
    _require_source, _run_fast_dispatch, _run_with_dashboard,
    _validate_curate_stages, _validate_fast_conflicts, _validate_subcommand_flags,
)

BOOTSTRAP_EXIT_CODE: int = 1


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


def _run_standard_pipeline(kwargs: dict, config: CullConfig) -> None:
    """Run default pipeline path: source check, dashboard, post-processing."""
    _require_source(kwargs)
    _validate_curate_stages(config)
    result = _run_with_dashboard(config, Path(kwargs["source"]))
    post_in = PostPipelineInput(
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
