"""One-shot Rich panel printers for the cull CLI.

Extracted from ``cull.dashboard`` to isolate the public ``show_*`` surface
from the live ``Dashboard`` class. Every function here creates its own
``Console`` instance and prints once — they hold no state and never touch
the live dashboard. Color constants and spacing come from ``cull.dashboard``.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cull.dashboard import (
    CLR_BORDER,
    CLR_DIM,
    CLR_KEEPER,
    CLR_PROGRESS,
    CLR_REJECT,
    CLR_SELECT,
    CLR_UNCERTAIN,
    TUI_HANDOFF_SLEEP_SECONDS,
)


# ---------------------------------------------------------------------------
# Disk selection panel
# ---------------------------------------------------------------------------


class DiskDisplayInfo(BaseModel):
    """Serializable disk info for rendering the selection panel."""

    name: str
    mount_point: str
    jpeg_count: int
    size_label: str = ""


def _build_disk_table(disks: list[DiskDisplayInfo]) -> Table:
    """Build the disk selection table."""
    table = Table(
        show_header=True,
        header_style="bold bright_blue",
        border_style="bright_black",
        padding=(0, 2),
        expand=True,
    )
    table.add_column("#", style=CLR_PROGRESS, no_wrap=True, width=4)
    table.add_column("Disk", style="bright_white")
    table.add_column("Path", style=CLR_DIM)
    table.add_column("JPEGs", style=CLR_KEEPER, justify="right")
    table.add_column("Size", style=CLR_DIM, justify="right")
    for idx, disk in enumerate(disks, start=1):
        table.add_row(
            str(idx), disk.name, disk.mount_point,
            str(disk.jpeg_count), disk.size_label,
        )
    return table


def show_disk_selection(disks: list[DiskDisplayInfo]) -> None:
    """Render the disk selection list as a Rich panel."""
    console = Console()
    table = _build_disk_table(disks)
    console.print()
    console.print(Panel(
        table,
        title="[bold bright_blue]External Disks[/]",
        border_style=CLR_BORDER,
        padding=(1, 0),
    ))


# ---------------------------------------------------------------------------
# Dry-run results panel
# ---------------------------------------------------------------------------


class DryRunSummary(BaseModel):
    """Summary of what a dry run would do."""

    keepers: int = 0
    rejected: int = 0
    duplicates: int = 0
    uncertain: int = 0
    total: int = 0


def _build_dry_run_table(summary: DryRunSummary) -> Table:
    """Build the dry-run results table."""
    body = Table(
        show_header=False,
        border_style="bright_black",
        padding=(0, 2),
        expand=True,
    )
    body.add_column("Label", style="bright_white")
    body.add_column("Count", justify="right")
    body.add_column("Action", style=CLR_DIM)
    body.add_row(
        f"[{CLR_KEEPER}]Keepers[/]",
        str(summary.keepers),
        "would stay in place",
    )
    body.add_row(
        f"[{CLR_REJECT}]Rejected[/]",
        str(summary.rejected),
        "would move to _review/_rejected/",
    )
    body.add_row(
        f"[{CLR_DIM}]Duplicates[/]",
        str(summary.duplicates),
        "would move to _review/_duplicates/",
    )
    body.add_row(
        f"[{CLR_UNCERTAIN}]Uncertain[/]",
        str(summary.uncertain),
        "would move to _review/_uncertain/",
    )
    return body


def show_dry_run_results(summary: DryRunSummary) -> None:
    """Display a prominent DRY RUN results card."""
    console = Console()
    header = Text("DRY RUN -- no files moved", style="bold yellow on dark_red")
    body = _build_dry_run_table(summary)
    console.print()
    console.print(Panel(
        header, border_style=CLR_UNCERTAIN, padding=(0, 2),
    ))
    console.print(Panel(
        body,
        title="[bold bright_blue]Results[/]",
        border_style=CLR_BORDER,
        padding=(1, 0),
    ))
    console.print()


# ---------------------------------------------------------------------------
# File moving phase panel
# ---------------------------------------------------------------------------


def show_move_complete(moved: int, errors: int) -> None:
    """Show a brief completion message after file moves."""
    console = Console()
    if errors > 0:
        msg = Text(f"  Moved {moved} files ({errors} errors)", style=CLR_REJECT)
    else:
        msg = Text(f"  Moved {moved} files", style=CLR_KEEPER)
    console.print(msg)


# ---------------------------------------------------------------------------
# Report writing panel
# ---------------------------------------------------------------------------


def show_report_writing(report_path: Path) -> None:
    """Show report-writing status with checkmark on completion."""
    console = Console()
    console.print(
        f"  [green]\u2713[/] Writing [bright_white]{report_path.name}[/]... "
        f"[dim]{report_path.parent}[/]"
    )


# ---------------------------------------------------------------------------
# TUI handoff panel
# ---------------------------------------------------------------------------


class TuiHandoffCtx(BaseModel):
    """Context for the TUI handoff transition panel."""

    uncertain_count: int = 0
    total_count: int = 0


def show_tui_handoff(ctx: TuiHandoffCtx) -> None:
    """Show the styled panel before transitioning to the TUI."""
    console = Console()
    body = Text()
    body.append(f"  {ctx.uncertain_count}", style="bold yellow")
    body.append(" photos need your review. ", style="bright_white")
    body.append("Opening TUI...", style="bold bright_blue")
    if ctx.total_count > 0:
        body.append(f"\n  {ctx.total_count} total photos processed", style=CLR_DIM)
    console.print()
    console.print(Panel(
        body,
        title="[bold bright_blue]Review[/]",
        border_style=CLR_BORDER,
        padding=(0, 1),
    ))
    console.print()
    time.sleep(TUI_HANDOFF_SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Error state panels
# ---------------------------------------------------------------------------


def show_vlm_load_error(alias: str, root: Path) -> None:
    """Show a styled error panel when a VLM alias fails to load from the registry."""
    console = Console()
    body = Text()
    body.append(f"  VLM alias '{alias}' failed to load\n", style="bold red")
    body.append(f"  Registry root: {root}\n\n", style="bright_white")
    body.append("  Suggestions:\n", style=CLR_DIM)
    body.append("    1. Run ", style=CLR_DIM)
    body.append("cull setup --allow-network", style="bright_cyan")
    body.append(" to validate the registry\n", style=CLR_DIM)
    body.append("    2. Check the alias exists under the registry root\n", style=CLR_DIM)
    body.append("    3. Run with ", style=CLR_DIM)
    body.append("--no-vlm", style="bright_cyan")
    body.append(" to skip Stage 3", style=CLR_DIM)
    console.print()
    console.print(Panel(
        body,
        title="[bold red]VLM Load Error[/]",
        border_style=CLR_REJECT,
        padding=(1, 1),
    ))
    console.print()


def show_general_error(title: str, message: str) -> None:
    """Show a styled error panel for any unrecoverable error."""
    console = Console()
    body = Text(f"  {message}", style="bright_white")
    console.print()
    console.print(Panel(
        body,
        title=f"[bold red]{title}[/]",
        border_style=CLR_REJECT,
        padding=(0, 1),
    ))
    console.print()


# ---------------------------------------------------------------------------
# Results card (non-dry-run)
# ---------------------------------------------------------------------------


class ResultsSummary(BaseModel):
    """Summary for the post-pipeline results card."""

    keepers: int = 0
    rejected: int = 0
    duplicates: int = 0
    uncertain: int = 0
    selected: int = 0
    total: int = 0
    elapsed_seconds: float = 0.0
    stages_run: list[int] = Field(default_factory=list)


def _build_results_table(summary: ResultsSummary) -> Table:
    """Build the results summary table."""
    table = Table(
        show_header=False,
        border_style="bright_black",
        padding=(0, 2),
        expand=True,
    )
    table.add_column("Label", style="bright_white")
    table.add_column("Count", justify="right")
    table.add_row(f"[{CLR_KEEPER}]Keepers[/]", str(summary.keepers))
    table.add_row(f"[{CLR_REJECT}]Rejected[/]", str(summary.rejected))
    table.add_row(f"[{CLR_DIM}]Duplicates[/]", str(summary.duplicates))
    table.add_row(f"[{CLR_UNCERTAIN}]Uncertain[/]", str(summary.uncertain))
    if summary.selected:
        table.add_row(f"[{CLR_SELECT}]Selected[/]", str(summary.selected))
    table.add_row("", "")
    table.add_row("[bright_white]Total[/]", str(summary.total))
    return table


def show_results_card(summary: ResultsSummary) -> None:
    """Display the pipeline results card."""
    console = Console()
    table = _build_results_table(summary)
    elapsed = f"{summary.elapsed_seconds:.1f}s"
    stages_str = ", ".join(str(s) for s in summary.stages_run)
    subtitle = f"[dim]stages {stages_str} in {elapsed}[/]"
    console.print()
    console.print(Panel(
        table,
        title="[bold bright_blue]Results[/]",
        subtitle=subtitle,
        border_style=CLR_BORDER,
        padding=(1, 0),
    ))
    console.print()
