"""CLI subcommand dispatch helpers for the photo cull pipeline.

Extracted from ``cull.cli`` as part of the 600-series CLI hub split.
Owns ``_dispatch_subcommand`` and the per-subcommand runners
(overrides, search, similar, explain, report-card).

One-way dependency: this module may import from ``cull.cli_config`` but
``cull.cli_config`` must never import from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cull.config import (
    SEARCH_TOP_K_DEFAULT,
    VLM_ALIASES,
    VLM_DEFAULT_ALIAS,
    VLM_MODELS_ROOT,
)
from cull.dashboard import show_general_error
from cull.models import ExplainResult
from cull.override_log import load_overrides
from cull.report_card import build_report_card, render_report_card
from cull.vlm_registry import VLMEntry, VLMRegistry, discover_vlms

OVERRIDE_LOG_TAIL_LIMIT: int = 50
OVERRIDE_TIME_FORMAT: str = "%Y-%m-%d %H:%M"
VLM_SOURCE_OVERRIDE: str = "override"
VLM_SOURCE_AUTO_SLUG: str = "auto-slug"
VLM_DEFAULT_MARKER: str = "*"


def _dispatch_subcommand(kwargs: dict) -> bool:
    """Route to a subcommand helper if any subcommand flag is set; return True if handled."""
    if kwargs.get("vlms"):
        _run_vlms_list()
        return True
    if kwargs.get("bake_manifest"):
        _run_bake_manifest(kwargs)
        return True
    if kwargs.get("calibrate"):
        _run_calibrate(kwargs)
        return True
    if kwargs.get("overrides"):
        _run_overrides_dump()
        return True
    if kwargs.get("search"):
        _run_search_text(kwargs)
        return True
    if kwargs.get("similar"):
        _run_search_similar(kwargs)
        return True
    if kwargs.get("explain"):
        _run_explain(kwargs)
        return True
    if kwargs.get("report_card"):
        _run_report_card(kwargs)
        return True
    return False


def _build_override_table(entries: list) -> Table:
    """Construct a Rich Table summarizing the most recent override entries."""
    table = Table(
        title=f"Override Log ({len(entries)} entries)",
        border_style="bright_blue",
    )
    table.add_column("Time")
    table.add_column("File")
    table.add_column("From")
    table.add_column("To")
    table.add_column("Origin")
    for entry in entries[-OVERRIDE_LOG_TAIL_LIMIT:]:
        table.add_row(
            entry.timestamp.strftime(OVERRIDE_TIME_FORMAT),
            entry.filename,
            entry.original_decision,
            entry.user_decision,
            entry.override_origin,
        )
    return table


def _run_overrides_dump() -> None:
    """Print override log as a Rich table and exit."""
    entries = load_overrides()
    console = Console()
    if not entries:
        console.print("[dim]No overrides logged yet.[/]")
        return
    console.print(_build_override_table(entries))


def _entry_source(entry: VLMEntry) -> str:
    """Return 'override' if alias is in VLM_ALIASES, else 'auto-slug'."""
    return VLM_SOURCE_OVERRIDE if entry.alias in VLM_ALIASES else VLM_SOURCE_AUTO_SLUG


def _build_vlms_table(registry: VLMRegistry) -> Table:
    """Construct a Rich table of discovered VLMs with alias / dir / source / default."""
    table = Table(
        title=f"Discovered VLMs under {VLM_MODELS_ROOT}",
        border_style="bright_blue",
    )
    table.add_column("Alias", style="bold")
    table.add_column("Directory")
    table.add_column("Source")
    table.add_column("Default", justify="center")
    for entry in registry.entries:
        marker = VLM_DEFAULT_MARKER if entry.alias == VLM_DEFAULT_ALIAS else ""
        table.add_row(entry.alias, entry.display_name, _entry_source(entry), marker)
    return table


def _run_vlms_list() -> None:
    """Discover VLMs under VLM_MODELS_ROOT and print them as a Rich table."""
    console = Console()
    registry = discover_vlms()
    if not registry.entries:
        console.print(f"[red]No VLMs found under {VLM_MODELS_ROOT}[/]")
        console.print(f"[dim]Override the search root via PHOTO_MANAGER_VLM_ROOT.[/]")
        return
    console.print(_build_vlms_table(registry))


def _format_calibration_summary(result: "CalibrationResult") -> Panel:
    """Render a Rich panel summarising a CalibrationResult."""
    body = (
        f"[bold]Corpus:[/] {result.corpus_name}\n"
        f"[bold]Photos scored:[/] {result.photo_count}\n"
        f"[bold]Duration:[/] {result.duration_seconds:.1f}s\n\n"
        f"[bold green]p1 baseline:[/] {result.p1_baseline_path}\n"
        f"[bold green]p4lite baseline:[/] {result.p4lite_baseline_path}"
    )
    return Panel(body, title="Calibration complete", border_style="green")


def _maybe_rebake_manifest(corpus_dir: Path, no_rebake: bool) -> None:
    """Rebake the corpus manifest unless the caller passed --no-rebake."""
    from cull.manifest_baker import (  # noqa: PLC0415
        ManifestBakeError, ManifestBakeRequest, bake_manifest,
    )

    if no_rebake:
        return
    try:
        bake_manifest(ManifestBakeRequest(corpus_dir=corpus_dir))
    except ManifestBakeError as exc:
        show_general_error("Manifest auto-bake failed", str(exc))
        sys.exit(1)


def _run_calibrate(kwargs: dict) -> None:
    """Auto-bake manifest then bake golden baselines for the calibrate target."""
    from cull.calibrate import (  # noqa: PLC0415
        CalibrationError, CalibrationRequest, CalibrationResult, run_calibration,
    )

    console = Console()
    corpus_dir = Path(kwargs["calibrate"])
    _maybe_rebake_manifest(corpus_dir, bool(kwargs.get("no_rebake")))
    try:
        result: CalibrationResult = run_calibration(CalibrationRequest(corpus_dir=corpus_dir))
    except CalibrationError as exc:
        show_general_error("Calibration failed", str(exc))
        sys.exit(1)
    console.print(_format_calibration_summary(result))


def _format_manifest_summary(result: "ManifestBakeResult") -> Panel:
    """Render a Rich panel summarising a ManifestBakeResult with category counts."""
    cats = sorted(result.category_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    cat_lines = "\n".join(f"  [bold]{cat}:[/] {count}" for cat, count in cats)
    body = (
        f"[bold]Corpus:[/] {result.corpus_dir.name}\n"
        f"[bold]Total entries:[/] {result.entry_count}\n"
        f"[bold green]Manifest:[/] {result.manifest_path}\n\n"
        f"[bold]Categories:[/]\n{cat_lines}"
    )
    return Panel(body, title="Manifest baked", border_style="green")


def _run_bake_manifest(kwargs: dict) -> None:
    """Bake manifest.json for the corpus directory passed via --bake-manifest."""
    from cull.manifest_baker import (  # noqa: PLC0415
        ManifestBakeError, ManifestBakeRequest, ManifestBakeResult, bake_manifest,
    )

    console = Console()
    request = ManifestBakeRequest(corpus_dir=Path(kwargs["bake_manifest"]))
    try:
        result: ManifestBakeResult = bake_manifest(request)
    except ManifestBakeError as exc:
        show_general_error("Manifest bake failed", str(exc))
        sys.exit(1)
    console.print(_format_manifest_summary(result))


def _resolve_top_k(kwargs: dict) -> int:
    """Return user-supplied --top-k or the default."""
    top_k = kwargs.get("top_k")
    return top_k if top_k is not None else SEARCH_TOP_K_DEFAULT


def _require_subcommand_source(kwargs: dict, flag_name: str) -> None:
    """Exit with error if a subcommand requires source but none is set."""
    if not kwargs.get("source"):
        show_general_error(
            "Missing source",
            f"--{flag_name} requires a SOURCE directory.",
        )
        sys.exit(1)


def _run_search_text(kwargs: dict) -> None:
    """Run text-based semantic search and print top-K results."""
    from cull.models import SearchRequest  # noqa: PLC0415
    from cull.search import search_by_text  # noqa: PLC0415

    _require_subcommand_source(kwargs, "search")
    top_k = _resolve_top_k(kwargs)
    req = SearchRequest(query_text=kwargs["search"], source=Path(kwargs["source"]), top_k=top_k)
    results = search_by_text(req)
    _render_search_results(results, f"text: '{kwargs['search']}'")


def _run_search_similar(kwargs: dict) -> None:
    """Run image-similarity search and print top-K results."""
    from cull.models import SearchRequest  # noqa: PLC0415
    from cull.search import search_by_similarity  # noqa: PLC0415

    _require_subcommand_source(kwargs, "similar")
    top_k = _resolve_top_k(kwargs)
    ref_path = Path(kwargs["similar"])
    req = SearchRequest(reference_path=ref_path, source=Path(kwargs["source"]), top_k=top_k)
    results = search_by_similarity(req)
    _render_search_results(results, f"similar to: {ref_path.name}")


def _render_search_results(results: list, title: str) -> None:
    """Render search results as a Rich table."""
    console = Console()
    table = Table(title=title, border_style="bright_blue")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Photo")
    for result in results:
        table.add_row(str(result.rank), f"{result.similarity:.3f}", result.path.name)
    console.print(table)


def _build_explain_request(kwargs: dict) -> "ExplainRequest":  # noqa: F821
    """Construct an ExplainRequest from CLI kwargs (no prior stage data)."""
    from cull.models import ExplainRequest  # noqa: PLC0415

    return ExplainRequest(
        image_path=Path(kwargs["explain"]),
        stage1_result=None,
        stage2_composite=None,
        stage3_result=None,
        model=str(kwargs["model"]),
    )


def _run_explain(kwargs: dict) -> None:
    """Run VLM explanation on a single photo and print a panel."""
    from cull.stage3.vlm_explain import ExplainCallInput, explain_photo  # noqa: PLC0415
    from cull.vlm_session import vlm_session  # noqa: PLC0415

    request = _build_explain_request(kwargs)
    with vlm_session(str(kwargs["model"])) as session:
        call_in = ExplainCallInput(request=request, session=session)
        result = explain_photo(call_in)
    _render_explanation(result)


def _format_strengths(strengths: list[str]) -> list[str]:
    """Return Rich-styled lines for the strengths section."""
    lines = ["[bold green]Strengths[/]"]
    for item in strengths:
        lines.append(f"  [green]●[/] {item}")
    return lines


def _format_weaknesses(weaknesses: list[str]) -> list[str]:
    """Return Rich-styled lines for the weaknesses section."""
    lines = ["[bold red]Weaknesses[/]"]
    for item in weaknesses:
        lines.append(f"  [red]○[/] {item}")
    return lines


def _build_explanation_lines(result: ExplainResult) -> list[str]:
    """Assemble the full explanation text body for the Rich panel."""
    lines: list[str] = []
    lines.extend(_format_strengths(result.strengths))
    lines.append("")
    lines.extend(_format_weaknesses(result.weaknesses))
    lines.append("")
    lines.append(f"[bold bright_white]Summary[/]\n  {result.explanation}")
    lines.append(f"\n[dim]confidence: {result.confidence:.2f}[/]")
    return lines


def _render_explanation(result: ExplainResult) -> None:
    """Print an ExplainResult as a Rich Panel with strengths/weaknesses/summary."""
    console = Console()
    if result.is_parse_error:
        console.print(Panel("[red]Unable to parse VLM response.[/]", border_style="red"))
        return
    panel = Panel(
        "\n".join(_build_explanation_lines(result)),
        title=result.photo_path.name,
        border_style="bright_blue",
    )
    console.print(panel)


def _run_report_card(kwargs: dict) -> None:
    """Build and render the shoot report card."""
    _require_subcommand_source(kwargs, "report-card")
    try:
        card = build_report_card(Path(kwargs["source"]))
    except FileNotFoundError as exc:
        show_general_error("No session report", str(exc))
        sys.exit(1)
    render_report_card(card)
