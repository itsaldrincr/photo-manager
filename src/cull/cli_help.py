"""Rich help rendering for the cull CLI.

Houses the ``CullHelp`` Click command subclass, the ``_show_help_tui``
entry point, and the ``_print_*`` section helpers. Extracted from
``cull.cli`` as part of the 600-series CLI hub split.
"""

from __future__ import annotations

import importlib.util

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cull.config import SEARCH_TOP_K_DEFAULT, VLM_DEFAULT_ALIAS

FAST_MODE_AVAILABLE: bool = importlib.util.find_spec("cull_fast") is not None

_BASE_FLAG_ROWS: list[tuple[str, str, str]] = [
    ("--dry-run", "Preview decisions without moving any files", "off"),
    ("--no-vlm", "Skip Stage 3 (no VLM needed, faster)", "off"),
    ("--portrait", "Face/eye quality analysis (blink, sharpness, expression)", "on"),
    ("--model TEXT", "VLM alias \u2014 see VLM_ALIASES", VLM_DEFAULT_ALIAS),
    ("--threshold FLOAT", "Keeper confidence threshold (higher = stricter)", "0.65"),
    ("--burst-gap FLOAT", "Max seconds between shots to group as burst", "0.5"),
    ("--preset NAME", "Genre scoring weights (see below)", "general"),
    ("--stage INT", "Run specific stage(s) only (1, 2, or 3)", "all"),
    ("--review", "Open TUI to review existing _uncertain/ results", "off"),
    ("--review-all", "Open TUI showing all photos with decisions", "off"),
    ("--review-after", "Run pipeline first, then open review on the final session", "off"),
    ("--no-report", "Skip session_report.json (default: written)", "on"),
    ("--calibrate PATH", "Auto-bake manifest then bake golden baselines for a corpus", "off"),
    ("--no-rebake", "With --calibrate: keep existing manifest (skip auto-bake)", "off"),
    ("--bake-manifest PATH", "Bake manifest.json only — no ML scoring (inspection)", "off"),
    ("--vlms", "List discovered VLMs (alias, directory, source, default)", "off"),
    (
        "--curate [N]",
        (
            "Stage 4 curator: peak-moment portrait, peak-moment action, "
            "diversity (MMR), pairwise tournament, narrative-flow regulariser. "
            "Default N=30."
        ),
        "off",
    ),
    (
        "--curate-vlm-threshold F",
        "Stage 4 VLM tiebreak gap: lower = fewer VLM calls (default 0.02)",
        "0.02",
    ),
    ("--overrides", "Dump override log as a table", "off"),
    ("--search TEXT", "Semantic text search across photos", "off"),
    ("--similar PATH", "Find photos visually similar to reference", "off"),
    ("--explain PATH", "VLM explanation for why a photo is weak", "off"),
    ("--report-card", "Diagnostic report from session_report.json", "off"),
    ("--top-k N", "Number of search results to return", str(SEARCH_TOP_K_DEFAULT)),
    ("--no-sidecars", "Skip writing XMP sidecar files alongside source images", "off"),
]
_FAST_FLAG_ROW: tuple[str, str, str] = (
    "--fast",
    "Single-pass MUSIQ scoring (ablation; no speed claim until benchmarked)",
    "off",
)
_PRESET_INFO: list[tuple[str, str]] = [
    ("general", "Balanced scoring across all dimensions"),
    ("wedding", "Conservative - keeps more faces, errs toward keeping"),
    ("documentary", "Composition weighted heavily"),
    ("wildlife", "Sharpness is king, strict blur rejection"),
    ("landscape", "Exposure weighted more, composition matters"),
    ("street", "Motion blur tolerance slightly higher"),
    ("holiday", "Documentary + landscape blend for holiday photos"),
]


def _show_help_tui() -> None:
    """Display a rich help screen with command descriptions."""
    console = Console()
    console.print()

    title = Text("cull", style="bold bright_white")
    title.append(" - ", style="dim")
    title.append("AI photo culling manager", style="italic bright_white")

    console.print(Panel(
        title,
        border_style="bright_blue",
        padding=(0, 2),
    ))
    console.print()

    _print_usage(console)
    _print_pipeline(console)
    _print_flags(console)
    _print_presets(console)
    _print_examples(console)


def _print_usage(console: Console) -> None:
    """Print usage line."""
    console.print("  [dim]usage:[/]  cull [bright_cyan][OPTIONS][/] [bright_green]SOURCE[/]")
    console.print()


def _build_pipeline_text() -> str:
    """Return the rich-formatted pipeline overview body string."""
    fast_line = (
        "\n[bright_magenta]Fast[/]    [dim]|[/] Single-pass MUSIQ  [dim]-[/] "
        "musiq + musiq-ava ablation in place of Stage 2  [dim]via --fast[/]"
        if FAST_MODE_AVAILABLE else ""
    )
    return (
        "[bright_white]Stage 1[/] [dim]|[/] Classical filters  [dim]-[/] "
        "blur, exposure, noise, burst/duplicate detection\n"
        "[bright_white]Stage 2[/] [dim]|[/] IQA scoring        [dim]-[/] "
        "TOPIQ + CLIP-IQA+ + LAION Aesthetics + face/eye analysis\n"
        "[bright_white]Stage 3[/] [dim]|[/] VLM tiebreaker     [dim]-[/] "
        "local vision model scores ambiguous photos via oMLX\n"
        "[bright_cyan]Stage 4[/] [dim]|[/] Curator  [dim](opt-in)[/]  [dim]-[/] "
        "peak-moment portrait + action, diversity (MMR), pairwise tournament, "
        "narrative-flow regulariser  [dim]via --curate[/]"
        f"{fast_line}"
    )


def _print_pipeline(console: Console) -> None:
    """Print pipeline overview card."""
    console.print(Panel(
        _build_pipeline_text(),
        title="[bold bright_blue]Pipeline[/]",
        border_style="blue",
        padding=(1, 2),
    ))
    console.print()


def _build_flag_rows() -> list[tuple[str, str, str]]:
    """Return the static flag metadata rows, plus the conditional --fast row."""
    rows = list(_BASE_FLAG_ROWS)
    if FAST_MODE_AVAILABLE:
        rows.append(_FAST_FLAG_ROW)
    return rows


def _build_flags_table() -> Table:
    """Construct the rich Table listing every CLI flag."""
    table = Table(
        show_header=True,
        header_style="bold bright_blue",
        border_style="bright_black",
        padding=(0, 2),
        expand=True,
    )
    table.add_column("Flag", style="bright_cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Default", style="dim", justify="right")
    for flag, desc, default in _build_flag_rows():
        table.add_row(flag, desc, default)
    return table


def _print_flags(console: Console) -> None:
    """Print flags table."""
    table = _build_flags_table()
    console.print(Panel(
        table,
        title="[bold bright_blue]Options[/]",
        border_style="blue",
        padding=(1, 0),
    ))
    console.print()


def _build_presets_table() -> Table:
    """Construct the rich Table listing all cull presets."""
    presets = Table(
        show_header=False,
        border_style="bright_black",
        padding=(0, 2),
        expand=True,
    )
    presets.add_column("Preset", style="bright_green", no_wrap=True)
    presets.add_column("Style", style="white")
    for name, desc in _PRESET_INFO:
        presets.add_row(name, desc)
    return presets


def _print_presets(console: Console) -> None:
    """Print preset descriptions."""
    console.print(Panel(
        _build_presets_table(),
        title="[bold bright_blue]Presets[/]",
        border_style="blue",
        padding=(1, 0),
    ))
    console.print()


def _print_examples(console: Console) -> None:
    """Print example commands."""
    examples = (
        "[dim]# sort photos on an SD card[/]\n"
        "  cull /Volumes/SD_CARD/DCIM\n\n"
        "[dim]# preview without moving files[/]\n"
        "  cull --dry-run ~/Photos/shoot\n\n"
        "[dim]# wedding preset, skip VLM for speed[/]\n"
        "  cull --preset wedding --no-vlm /path/to/photos\n\n"
        "[dim]# review uncertain photos from a previous run[/]\n"
        "  cull --review /path/to/photos\n\n"
        "[dim]# run the pipeline, then drop straight into review[/]\n"
        "  cull --review-after /path/to/photos"
    )
    console.print(Panel(
        examples,
        title="[bold bright_blue]Examples[/]",
        border_style="blue",
        padding=(1, 2),
    ))


class CullHelp(click.Command):
    """Custom Click command that shows rich help instead of plain text."""

    def get_help(self, ctx: click.Context) -> str:
        """Override to show rich TUI help."""
        _show_help_tui()
        return ""
