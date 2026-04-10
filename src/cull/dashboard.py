"""Rich TUI dashboard panels for every stage of the cull experience."""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from cull.models import Stage1Result, Stage3Result


# ---------------------------------------------------------------------------
# Byte size constants
# ---------------------------------------------------------------------------

BYTES_PER_KB: int = 1024
BYTES_PER_MB: int = 1024 * 1024
BYTES_PER_GB: int = 1024 * 1024 * 1024

# ---------------------------------------------------------------------------
# Time and rendering constants
# ---------------------------------------------------------------------------

SECONDS_PER_MINUTE: int = 60
PERCENT_SCALE: int = 100
MS_PER_SECOND: int = 1000

# ---------------------------------------------------------------------------
# Live dashboard color theme (Claude Code dark)
# ---------------------------------------------------------------------------

CLR_BORDER: str = "bright_blue"
CLR_PROGRESS: str = "cyan"

# Shared theme aliases (imported by aesthetic.py, iqa.py)
BORDER_STYLE: str = CLR_BORDER
PROGRESS_STYLE: str = CLR_PROGRESS
KEEPER_STYLE: str = "green"
REJECT_STYLE: str = "red"
UNCERTAIN_STYLE: str = "yellow"
DIM_STYLE: str = "dim"
CLR_KEEPER: str = "green"
CLR_REJECT: str = "red"
CLR_UNCERTAIN: str = "yellow"
CLR_DUPLICATE: str = "dim"
CLR_ACTIVE: str = "bright_white"
CLR_DIM: str = "dim"
CLR_HEADER: str = "bold bright_white"
CLR_CHECK: str = "bright_green"

# ---------------------------------------------------------------------------
# Decision markers
# ---------------------------------------------------------------------------

MARK_KEEPER: str = f"[{CLR_KEEPER}]\u25cf[/]"
MARK_REJECT: str = f"[{CLR_REJECT}]\u25cb[/]"
MARK_UNCERTAIN: str = f"[{CLR_UNCERTAIN}]\u25d0[/]"
MARK_DUPLICATE: str = f"[{CLR_DUPLICATE}]\u25cc[/]"

CLR_SELECT: str = "bright_magenta"
MARK_SELECT: str = f"[{CLR_SELECT}]\u2605[/]"

LABEL_TO_MARK: dict[str, str] = {
    "keeper": MARK_KEEPER,
    "rejected": MARK_REJECT,
    "uncertain": MARK_UNCERTAIN,
    "duplicate": MARK_DUPLICATE,
    "select": MARK_SELECT,
    "KEEPER": MARK_KEEPER,
    "REJECT": MARK_REJECT,
    "AMBIGUOUS": MARK_UNCERTAIN,
}

# ---------------------------------------------------------------------------
# Stage spinners
# ---------------------------------------------------------------------------

SPINNER_S1: list[str] = ["\u25d0", "\u25d3", "\u25d1", "\u25d2"]
SPINNER_S2: list[str] = [
    "\u280b", "\u2819", "\u2839", "\u2838",
    "\u283c", "\u2834", "\u2826", "\u2827",
]
SPINNER_S3: list[str] = [
    "\u28fe", "\u28fd", "\u28fb", "\u28bf",
    "\u28bf", "\u289f", "\u28af", "\u28f7",
]

# ---------------------------------------------------------------------------
# Sparkline and gauge constants
# ---------------------------------------------------------------------------

SPARK_CHARS: str = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
GAUGE_WIDTH: int = 10
GAUGE_FILLED: str = "\u2588"
GAUGE_EMPTY: str = "\u2591"
GAUGE_SCAN_MS_PER_DIM: int = 400
GAUGE_FILL_MS_PER_BAR: int = 100
GAUGE_HOLD_MS: int = 300
REFRESH_PER_SECOND: int = 10
BAR_WIDTH: int = 39
RESULT_BAR_WIDTH: int = 28
SPARKLINE_WIDTH: int = 20
TUI_HANDOFF_SLEEP_SECONDS: float = 1.0

# ---------------------------------------------------------------------------
# Stage 3 dimension names
# ---------------------------------------------------------------------------

S3_DIMENSIONS: list[str] = [
    "sharpness", "exposure", "composition", "expression", "confidence",
]
S3_CONFIDENCE_INDEX: int = 4

def _format_bytes(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if num_bytes < BYTES_PER_KB:
        return f"{num_bytes} B"
    if num_bytes < BYTES_PER_MB:
        return f"{num_bytes / BYTES_PER_KB:.0f} KB"
    if num_bytes < BYTES_PER_GB:
        return f"{num_bytes / BYTES_PER_MB:.1f} MB"
    return f"{num_bytes / BYTES_PER_GB:.2f} GB"


# ---------------------------------------------------------------------------
# One-shot panel printers (show_*) live in a sibling module. They are
# re-exported at the bottom of this file for backward compatibility so
# existing callers of ``cull.dashboard`` keep working unchanged.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Film strip builder
# ---------------------------------------------------------------------------


def build_film_strip(decisions: list[str]) -> str:
    """Map each label to its marker character and join."""
    return "".join(LABEL_TO_MARK.get(d, MARK_UNCERTAIN) for d in decisions)


# ---------------------------------------------------------------------------
# Internal state models for live dashboard
# ---------------------------------------------------------------------------


class _S1Tallies(BaseModel):
    """Stage 1 running tallies."""

    sharp: int = 0
    soft: int = 0
    bokeh: int = 0
    good_light: int = 0
    clip: int = 0
    cast: int = 0
    bursts: int = 0
    dupes: int = 0


class _S2Tallies(BaseModel):
    """Stage 2 running tallies."""

    keepers: int = 0
    ambiguous: int = 0
    rejects: int = 0


class _ActiveLine(BaseModel):
    """Current photo being processed."""

    filename: str = ""
    detail: str = ""


class _AnalysisState(BaseModel):
    """State for Stage 3 gauge animation."""

    path: str = ""
    is_scanning: bool = False
    start_time: float = 0.0
    scores: dict[str, float] = Field(default_factory=dict)
    decision: str = ""
    fill_start: float = 0.0
    is_filled: bool = False


class _Stage2UpdateInput(BaseModel):
    """Input bundle for Stage 2 update."""

    class Config:
        arbitrary_types_allowed = True

    path: Path
    fusion: Any
    routing: str


class _S2ReducerTallies(BaseModel):
    """Stage 2 reducer (cross-photo shoot stats) running tallies."""

    palette_outliers: int = 0
    exposure_outliers: int = 0
    exif_outliers: int = 0
    scene_starts: int = 0


class _Stage2ReducerUpdateInput(BaseModel):
    """Input bundle for one reducer-pass update — flagged counts for the running photo."""

    is_palette_outlier: bool = False
    is_exposure_outlier: bool = False
    is_exif_outlier: bool = False
    is_scene_start: bool = False


class _S4Tallies(BaseModel):
    """Stage 4 running tallies."""

    clusters_found: int = 0
    target: int = 0
    selected: int = 0
    vlm_tiebreaks: int = 0
    skipped: int = 0
    current_compare: tuple[str, str] | None = None
    current_cluster_id: int | None = None


class _S4UpdateInput(BaseModel):
    """Input bundle for Stage 4 tally update."""

    clusters_found: int = 0
    selected: int = 0
    vlm_tiebreaks: int = 0
    skipped: int = 0
    current_compare: tuple[str, str] | None = None
    current_cluster_id: int | None = None


class _StageState(BaseModel):
    """Per-stage progress state."""

    total: int = 0
    done: int = 0
    start_time: float = 0.0
    elapsed: float = 0.0
    is_complete: bool = False
    summary_line: str = ""


# ---------------------------------------------------------------------------
# Formatting helpers for live dashboard
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.0f}s"
    minutes = int(seconds // SECONDS_PER_MINUTE)
    secs = int(seconds % SECONDS_PER_MINUTE)
    return f"{minutes}m{secs:02d}s"


def _extract_s3_scores(result: Stage3Result) -> dict[str, float]:
    """Extract dimension scores from a Stage3Result."""
    return {
        "sharpness": result.sharpness or 0.0,
        "exposure": result.exposure or 0.0,
        "composition": result.composition or 0.0,
        "expression": result.expression or 0.0,
        "confidence": result.confidence,
    }


def _s3_decision_label(result: Stage3Result) -> str:
    """Map Stage3Result to a decision label string."""
    if result.is_keeper is True:
        return "keeper"
    if result.is_keeper is False:
        return "rejected"
    return "uncertain"


# ---------------------------------------------------------------------------
# DashboardLaunchInfo bundle
# ---------------------------------------------------------------------------


class DashboardLaunchInfo(BaseModel):
    """Parameters for constructing a Dashboard."""

    source_path: str
    photo_count: int
    preset: str
    file_size_gb: float


# ---------------------------------------------------------------------------
# Stage 4 tracker (extracted from Dashboard to reduce its public-method count)
# ---------------------------------------------------------------------------


class _Stage4Tracker:
    """Owns every Stage 4 state field and transition for Dashboard."""

    def __init__(self) -> None:
        """Initialize Stage 4 main and sub-phase state."""
        self._s4 = _StageState()
        self._s4_peak = _StageState()
        self._s4_diversity = _StageState()
        self._s4_tournament = _StageState()
        self._s4_narrative = _StageState()
        self._s4_tallies = _S4Tallies()

    def _start(self, target: int) -> None:
        """Begin Stage 4 curator tracking."""
        self._s4 = _StageState(
            total=target, start_time=time.monotonic(),
        )
        self._s4_tallies = _S4Tallies(target=target)

    def _update(self, tally_update: _S4UpdateInput) -> None:
        """Record a Stage 4 tally update."""
        self._s4.done = tally_update.selected
        self._s4.elapsed = time.monotonic() - self._s4.start_time
        self._s4_tallies = _S4Tallies(
            clusters_found=tally_update.clusters_found,
            target=self._s4_tallies.target,
            selected=tally_update.selected,
            vlm_tiebreaks=tally_update.vlm_tiebreaks,
            skipped=tally_update.skipped,
            current_compare=tally_update.current_compare,
            current_cluster_id=tally_update.current_cluster_id,
        )

    def _complete(self, elapsed: float) -> None:
        """Collapse Stage 4 to a summary line."""
        t = self._s4_tallies
        line = (
            f"  [{CLR_CHECK}]\u2713[/] Stage 4  "
            f"[{CLR_SELECT}]{t.selected} selected[/] \u00b7 "
            f"[{CLR_DIM}]{t.vlm_tiebreaks} tiebreaks[/]"
            f"  [{CLR_DIM}]{_fmt_time(elapsed)}[/]"
        )
        self._s4.is_complete = True
        self._s4.summary_line = line

    def _start_peak(self, total: int) -> None:
        """Begin Stage 4 peak sub-phase tracking."""
        _begin_sub_stage(self._s4_peak, total)

    def _update_peak(self, done: int) -> None:
        """Record peak sub-phase progress."""
        _advance_sub_stage(self._s4_peak, done)

    def _complete_peak(self, elapsed: float) -> None:
        """Collapse Stage 4 peak sub-phase to a summary line."""
        _finish_sub_stage(self._s4_peak, _SubStageSummary(label="peak", elapsed=elapsed))

    def _start_diversity(self, total: int) -> None:
        """Begin Stage 4 diversity sub-phase tracking."""
        _begin_sub_stage(self._s4_diversity, total)

    def _update_diversity(self, done: int) -> None:
        """Record diversity sub-phase progress."""
        _advance_sub_stage(self._s4_diversity, done)

    def _complete_diversity(self, elapsed: float) -> None:
        """Collapse Stage 4 diversity sub-phase to a summary line."""
        _finish_sub_stage(self._s4_diversity, _SubStageSummary(label="diversity", elapsed=elapsed))

    def _start_tournament(self, total: int) -> None:
        """Begin Stage 4 tournament sub-phase tracking."""
        _begin_sub_stage(self._s4_tournament, total)

    def _update_tournament(self, done: int) -> None:
        """Record tournament sub-phase progress."""
        _advance_sub_stage(self._s4_tournament, done)

    def _complete_tournament(self, elapsed: float) -> None:
        """Collapse Stage 4 tournament sub-phase to a summary line."""
        _finish_sub_stage(self._s4_tournament, _SubStageSummary(label="tournament", elapsed=elapsed))

    def _start_narrative(self, total: int) -> None:
        """Begin Stage 4 narrative sub-phase tracking."""
        _begin_sub_stage(self._s4_narrative, total)

    def _update_narrative(self, done: int) -> None:
        """Record narrative sub-phase progress."""
        _advance_sub_stage(self._s4_narrative, done)

    def _complete_narrative(self, elapsed: float) -> None:
        """Collapse Stage 4 narrative sub-phase to a summary line."""
        _finish_sub_stage(self._s4_narrative, _SubStageSummary(label="narrative", elapsed=elapsed))


class _Stage3Tracker:
    """Owns every Stage 3 state field and transition for Dashboard."""

    def __init__(self) -> None:
        """Initialize Stage 3 main state, analysis state, and model label."""
        self._s3 = _StageState()
        self._analysis = _AnalysisState()
        self._s3_model: str = ""

    def _start(self, total: int, model_name: str) -> None:
        """Begin Stage 3 tracking."""
        self._s3 = _StageState(
            total=total, start_time=time.monotonic(),
        )
        self._s3_model = model_name

    def _start_analysis(self, path: Path) -> None:
        """Begin scanning animation for one photo."""
        self._analysis = _AnalysisState(
            path=path.name,
            is_scanning=True,
            start_time=time.monotonic(),
        )

    def _complete_analysis(self, path: Path, result: Stage3Result) -> None:
        """Fill gauges with scores and update progress."""
        self._s3.done += 1
        self._s3.elapsed = time.monotonic() - self._s3.start_time
        scores = _extract_s3_scores(result)
        decision = _s3_decision_label(result)
        self._analysis = _AnalysisState(
            path=path.name,
            is_scanning=False,
            scores=scores,
            decision=decision,
            fill_start=time.monotonic(),
        )

    def _clear_analysis(self) -> None:
        """Reset analysis state to idle after gauge hold."""
        self._analysis = _AnalysisState()

    def _complete(self, elapsed: float) -> None:
        """Collapse Stage 3 to a summary line."""
        line = (
            f"  [{CLR_CHECK}]\u2713[/] Stage 3  "
            f"[{CLR_DIM}]{self._s3.done} analyzed[/]"
            f"  [{CLR_DIM}]{_fmt_time(elapsed)}[/]"
        )
        self._s3.is_complete = True
        self._s3.summary_line = line


class _Stage1Tracker:
    """Owns every Stage 1 state field and transition for Dashboard."""

    def __init__(self) -> None:
        """Initialize Stage 1 main state and tallies."""
        self._s1 = _StageState()
        self._s1_tallies = _S1Tallies()

    def _start(self, total: int) -> None:
        """Begin Stage 1 tracking."""
        self._s1 = _StageState(
            total=total, start_time=time.monotonic(),
        )

    def _update(self, result: Stage1Result) -> None:
        """Record one Stage 1 result."""
        self._s1.done += 1
        self._s1.elapsed = time.monotonic() - self._s1.start_time
        self._update_tallies(result)

    def _complete(self, elapsed: float) -> None:
        """Collapse Stage 1 to a summary line."""
        t = self._s1_tallies
        passed = t.sharp + t.bokeh - t.dupes
        line = (
            f"  [{CLR_CHECK}]\u2713[/] Stage 1  "
            f"[{CLR_KEEPER}]{passed} pass[/] \u00b7 "
            f"[{CLR_REJECT}]{t.soft} reject[/] \u00b7 "
            f"[{CLR_DUPLICATE}]{t.dupes} dup[/]"
            f"  [{CLR_DIM}]{_fmt_time(elapsed)}[/]"
        )
        self._s1.is_complete = True
        self._s1.summary_line = line

    def _update_tallies(self, result: Stage1Result) -> None:
        """Update Stage 1 running tallies from a result."""
        if result.blur.is_bokeh:
            self._s1_tallies.bokeh += 1
        elif result.is_pass:
            self._s1_tallies.sharp += 1
        else:
            self._s1_tallies.soft += 1
        _update_s1_light(self._s1_tallies, result)


class _Stage2Tracker:
    """Owns every Stage 2 state field and transition for Dashboard."""

    def __init__(self) -> None:
        """Initialize Stage 2 main, reducer, tallies, and score state."""
        self._s2 = _StageState()
        self._s2_reducer = _StageState()
        self._s2_tallies = _S2Tallies()
        self._s2_reducer_tallies = _S2ReducerTallies()
        self._s2_scores: list[float] = []

    def _start(self, total: int) -> None:
        """Begin Stage 2 tracking."""
        self._s2 = _StageState(
            total=total, start_time=time.monotonic(),
        )

    def _update(self, update_in: _Stage2UpdateInput) -> None:
        """Record one Stage 2 result."""
        self._s2.done += 1
        self._s2.elapsed = time.monotonic() - self._s2.start_time
        self._update_tallies(update_in.routing)
        self._s2_scores.append(update_in.fusion.stage2.composite)

    def _complete(self, elapsed: float) -> None:
        """Collapse Stage 2 to a summary line."""
        t = self._s2_tallies
        line = (
            f"  [{CLR_CHECK}]\u2713[/] Stage 2  "
            f"[{CLR_KEEPER}]{t.keepers} keeper[/] \u00b7 "
            f"[{CLR_UNCERTAIN}]{t.ambiguous} ambiguous[/] \u00b7 "
            f"[{CLR_REJECT}]{t.rejects} reject[/]"
            f"  [{CLR_DIM}]{_fmt_time(elapsed)}[/]"
        )
        self._s2.is_complete = True
        self._s2.summary_line = line

    def _start_reducer(self, total: int) -> None:
        """Begin Stage 2 reducer tracking (cross-photo shoot stats pass)."""
        self._s2_reducer = _StageState(
            total=total, start_time=time.monotonic(),
        )

    def _update_reducer(self, update_in: _Stage2ReducerUpdateInput) -> None:
        """Record one reducer outlier patch."""
        self._s2_reducer.done += 1
        self._s2_reducer.elapsed = time.monotonic() - self._s2_reducer.start_time
        self._s2_reducer_tallies = _update_reducer_tallies(
            self._s2_reducer_tallies, update_in
        )

    def _complete_reducer(self, elapsed: float) -> None:
        """Collapse Stage 2 reducer to a summary line."""
        t = self._s2_reducer_tallies
        line = (
            f"  [{CLR_CHECK}]\u2713[/] Stage 2 reducer  "
            f"[{CLR_DIM}]{t.palette_outliers} palette[/] \u00b7 "
            f"[{CLR_DIM}]{t.exposure_outliers} exposure[/] \u00b7 "
            f"[{CLR_DIM}]{t.exif_outliers} exif[/] \u00b7 "
            f"[{CLR_DIM}]{t.scene_starts} scene-start[/]"
            f"  [{CLR_DIM}]{_fmt_time(elapsed)}[/]"
        )
        self._s2_reducer.is_complete = True
        self._s2_reducer.summary_line = line

    def _update_tallies(self, routing: str) -> None:
        """Update Stage 2 running tallies from routing label."""
        if routing == "KEEPER":
            self._s2_tallies.keepers += 1
        elif routing == "AMBIGUOUS":
            self._s2_tallies.ambiguous += 1
        else:
            self._s2_tallies.rejects += 1


# ---------------------------------------------------------------------------
# Render helpers live in a sibling module to keep this file focused. Import
# here (after all constants/state models are defined) to avoid the circular
# load order — dashboard_render.py re-imports those names from this module.
# ---------------------------------------------------------------------------

from cull.dashboard_render import (  # noqa: E402
    _append_completed,
    _render_header,
    _render_results_panel,
    _render_s1,
    _render_s2,
    _render_s3,
    _render_s4,
    _render_scan,
)


# ---------------------------------------------------------------------------
# Live Dashboard class
# ---------------------------------------------------------------------------


class Dashboard:
    """Live terminal dashboard for the cull pipeline."""

    def _init_stage_states(self) -> None:
        """Initialize stage state objects."""
        self._s1_tracker = _Stage1Tracker()
        self._s2_tracker = _Stage2Tracker()
        self._s3_tracker = _Stage3Tracker()
        self._s4_tracker = _Stage4Tracker()

    @property
    def _s1(self) -> _StageState:
        """Property pass-through so file-local renderers keep working."""
        return self._s1_tracker._s1

    @property
    def _s1_tallies(self) -> _S1Tallies:
        """Property pass-through for Stage 1 tallies."""
        return self._s1_tracker._s1_tallies

    @property
    def _s4(self) -> _StageState:
        """Property pass-through so file-local renderers keep working."""
        return self._s4_tracker._s4

    @property
    def _s4_peak(self) -> _StageState:
        """Property pass-through for peak sub-stage state."""
        return self._s4_tracker._s4_peak

    @property
    def _s4_diversity(self) -> _StageState:
        """Property pass-through for diversity sub-stage state."""
        return self._s4_tracker._s4_diversity

    @property
    def _s4_tournament(self) -> _StageState:
        """Property pass-through for tournament sub-stage state."""
        return self._s4_tracker._s4_tournament

    @property
    def _s4_narrative(self) -> _StageState:
        """Property pass-through for narrative sub-stage state."""
        return self._s4_tracker._s4_narrative

    @property
    def _s4_tallies(self) -> _S4Tallies:
        """Property pass-through for Stage 4 tallies."""
        return self._s4_tracker._s4_tallies

    @property
    def _s2(self) -> _StageState:
        """Property pass-through so file-local renderers keep working."""
        return self._s2_tracker._s2

    @property
    def _s2_reducer(self) -> _StageState:
        """Property pass-through for Stage 2 reducer state."""
        return self._s2_tracker._s2_reducer

    @property
    def _s2_tallies(self) -> _S2Tallies:
        """Property pass-through for Stage 2 tallies."""
        return self._s2_tracker._s2_tallies

    @property
    def _s2_reducer_tallies(self) -> _S2ReducerTallies:
        """Property pass-through for Stage 2 reducer tallies."""
        return self._s2_tracker._s2_reducer_tallies

    @property
    def _s2_scores(self) -> list[float]:
        """Property pass-through for Stage 2 score history."""
        return self._s2_tracker._s2_scores

    @property
    def _s3(self) -> _StageState:
        """Property pass-through so file-local renderers keep working."""
        return self._s3_tracker._s3

    @property
    def _s3_model(self) -> str:
        """Property pass-through for Stage 3 model label."""
        return self._s3_tracker._s3_model

    @property
    def _analysis(self) -> _AnalysisState:
        """Property pass-through for Stage 3 per-photo analysis state."""
        return self._s3_tracker._analysis

    def __init__(self, launch_info: DashboardLaunchInfo) -> None:
        """Initialize dashboard state and rich Live display."""
        self._source = launch_info.source_path
        self._photo_count = launch_info.photo_count
        self._preset = launch_info.preset
        self._file_size_gb = launch_info.file_size_gb
        self._scan_active: bool = False
        self._scan_count: int = 0
        self._scan_bytes: int = 0
        self._console = Console()
        self._live: Live | None = None
        self._is_stopped: bool = False
        self._film: list[str] = []
        self._init_stage_states()
        self._active = _ActiveLine()

    def __enter__(self) -> Dashboard:
        """Suppress loggers, start live display with a pinned stdout file."""
        import sys  # noqa: PLC0415

        self._suppress_library_logs()
        # Pin Rich to the REAL sys.stdout captured at dashboard creation.
        # contextlib.redirect_stdout in _silence_stdio() must not affect Rich.
        self._console = Console(file=sys.stdout, force_terminal=True)
        self._live = Live(
            get_renderable=self._render,
            console=self._console,
            refresh_per_second=REFRESH_PER_SECOND,
            transient=False,
            vertical_overflow="crop",
            auto_refresh=True,
        )
        self._live.__enter__()
        return self


    def _suppress_library_logs(self) -> None:
        """Nuke every existing logger and disable all handlers globally."""
        logging.disable(logging.CRITICAL)
        # Remove ALL handlers from the root and every existing logger.
        for name in list(logging.root.manager.loggerDict.keys()):
            log = logging.getLogger(name)
            log.handlers.clear()
            log.propagate = False
            log.setLevel(logging.CRITICAL)
        logging.root.handlers.clear()
        logging.root.addHandler(logging.NullHandler())
        warnings.filterwarnings("ignore")

    def __exit__(self, *args: object) -> None:
        """Stop the live display."""
        if self._live and not self._is_stopped:
            self._is_stopped = True
            self._live.__exit__(*args)

    # --- Stage 1 (delegates to _Stage1Tracker) ---------------------------

    def start_stage1(self, total: int) -> None:
        """Begin Stage 1 tracking."""
        self._s1_tracker._start(total)
        self._refresh()

    def update_stage1(self, path: Path, result: Stage1Result) -> None:
        """Record one Stage 1 result and refresh display."""
        self._s1_tracker._update(result)
        self._active = _ActiveLine(
            filename=path.name,
            detail=_fmt_s1_detail(result),
        )
        label = "rejected" if not result.is_pass else "keeper"
        self._film.append(label)
        self._refresh()

    def complete_stage1(self, elapsed: float) -> None:
        """Collapse Stage 1 to a summary line."""
        self._s1_tracker._complete(elapsed)
        self._active = _ActiveLine()
        self._refresh()

    # --- Stage 2 (delegates to _Stage2Tracker) ---------------------------

    def start_stage2(self, total: int) -> None:
        """Begin Stage 2 tracking."""
        self._s2_tracker._start(total)
        self._refresh()

    def update_stage2(self, update_in: _Stage2UpdateInput) -> None:
        """Record one Stage 2 result and refresh display."""
        self._s2_tracker._update(update_in)
        self._active = _ActiveLine(
            filename=update_in.path.name,
            detail=_fmt_s2_detail(update_in.fusion),
        )
        self._film.append(update_in.routing)
        self._refresh()

    def complete_stage2(self, elapsed: float) -> None:
        """Collapse Stage 2 to a summary line."""
        self._s2_tracker._complete(elapsed)
        self._active = _ActiveLine()
        self._refresh()

    def start_stage2_loading(self) -> None:
        """Show a loading-models active line while Stage 2 models initialise."""
        self._active = _make_loading_models_line()
        self._refresh()

    def clear_stage2_loading(self) -> None:
        """Remove the loading-models active line once models are ready."""
        self._active = _ActiveLine()
        self._refresh()

    # --- Stage 2 reducer (delegates to _Stage2Tracker) -------------------

    def start_stage2_reducer(self, total: int) -> None:
        """Begin Stage 2 reducer tracking (cross-photo shoot stats pass)."""
        self._s2_tracker._start_reducer(total)
        self._refresh()

    def update_stage2_reducer(self, update_in: _Stage2ReducerUpdateInput) -> None:
        """Record one reducer outlier patch and refresh display."""
        self._s2_tracker._update_reducer(update_in)
        self._refresh()

    def complete_stage2_reducer(self, elapsed: float) -> None:
        """Collapse Stage 2 reducer to a summary line."""
        self._s2_tracker._complete_reducer(elapsed)
        self._refresh()

    # --- Stage 3 (delegates to _Stage3Tracker) ---------------------------

    def start_stage3(self, total: int, model_name: str) -> None:
        """Begin Stage 3 tracking."""
        self._s3_tracker._start(total, model_name)
        self._refresh()

    def start_analysis(self, path: Path) -> None:
        """Begin scanning animation for one photo."""
        self._s3_tracker._start_analysis(path)
        self._refresh()

    def complete_analysis(self, path: Path, result: Stage3Result) -> None:
        """Fill gauges with scores and update progress."""
        self._s3_tracker._complete_analysis(path, result)
        self._film.append(self._s3_tracker._analysis.decision)
        self._refresh()
        time.sleep(GAUGE_HOLD_MS / 1000)
        self._s3_tracker._clear_analysis()
        self._refresh()

    def complete_stage3(self, elapsed: float) -> None:
        """Collapse Stage 3 to a summary line."""
        self._s3_tracker._complete(elapsed)
        self._active = _ActiveLine()
        self._refresh()

    # --- Stage 4 (delegates to _Stage4Tracker) ---------------------------

    def start_stage4(self, target: int) -> None:
        """Begin Stage 4 curator tracking."""
        self._s4_tracker._start(target)
        self._refresh()

    def update_stage4(self, tally_update: _S4UpdateInput) -> None:
        """Record a Stage 4 tally update and refresh display."""
        self._s4_tracker._update(tally_update)
        self._refresh()

    def complete_stage4(self, elapsed: float) -> None:
        """Collapse Stage 4 to a summary line."""
        self._s4_tracker._complete(elapsed)
        self._refresh()

    def start_stage4_peak(self, total: int) -> None:
        """Begin Stage 4 peak sub-phase tracking."""
        self._s4_tracker._start_peak(total)
        self._refresh()

    def update_stage4_peak(self, done: int) -> None:
        """Record peak sub-phase progress and refresh display."""
        self._s4_tracker._update_peak(done)
        self._refresh()

    def complete_stage4_peak(self, elapsed: float) -> None:
        """Collapse Stage 4 peak sub-phase to a summary line."""
        self._s4_tracker._complete_peak(elapsed)
        self._refresh()

    def start_stage4_diversity(self, total: int) -> None:
        """Begin Stage 4 diversity sub-phase tracking."""
        self._s4_tracker._start_diversity(total)
        self._refresh()

    def update_stage4_diversity(self, done: int) -> None:
        """Record diversity sub-phase progress and refresh display."""
        self._s4_tracker._update_diversity(done)
        self._refresh()

    def complete_stage4_diversity(self, elapsed: float) -> None:
        """Collapse Stage 4 diversity sub-phase to a summary line."""
        self._s4_tracker._complete_diversity(elapsed)
        self._refresh()

    def start_stage4_tournament(self, total: int) -> None:
        """Begin Stage 4 tournament sub-phase tracking."""
        self._s4_tracker._start_tournament(total)
        self._refresh()

    def update_stage4_tournament(self, done: int) -> None:
        """Record tournament sub-phase progress and refresh display."""
        self._s4_tracker._update_tournament(done)
        self._refresh()

    def complete_stage4_tournament(self, elapsed: float) -> None:
        """Collapse Stage 4 tournament sub-phase to a summary line."""
        self._s4_tracker._complete_tournament(elapsed)
        self._refresh()

    def start_stage4_narrative(self, total: int) -> None:
        """Begin Stage 4 narrative sub-phase tracking."""
        self._s4_tracker._start_narrative(total)
        self._refresh()

    def update_stage4_narrative(self, done: int) -> None:
        """Record narrative sub-phase progress and refresh display."""
        self._s4_tracker._update_narrative(done)
        self._refresh()

    def complete_stage4_narrative(self, elapsed: float) -> None:
        """Collapse Stage 4 narrative sub-phase to a summary line."""
        self._s4_tracker._complete_narrative(elapsed)
        self._refresh()

    # --- Results ---------------------------------------------------------

    def show_results(self, session_result: Any) -> None:
        """Render the final results card after stopping Live."""
        if self._live and not self._is_stopped:
            self._is_stopped = True
            self._live.__exit__(None, None, None)
            self._live = None
        self._console.print(_render_results_panel(session_result, self._film))

    # --- Public state mutators -------------------------------------------

    def refresh(self) -> None:
        """Trigger a display refresh (delegates to the internal no-op)."""
        self._refresh()

    def set_photo_count(self, count: int) -> None:
        """Set the total photo count for the session."""
        self._photo_count = count

    def set_burst_count(self, count: int) -> None:
        """Set the Stage 1 burst-loser tally."""
        self._s1_tracker._s1_tallies.bursts = count

    def set_dupe_count(self, count: int) -> None:
        """Set the Stage 1 duplicate tally."""
        self._s1_tracker._s1_tallies.dupes = count

    def start_scanning(self) -> None:
        """Show the scanning-duplicates indicator in the active line."""
        self._active = _ActiveLine(filename="scanning duplicates...", detail="")
        self._refresh()

    def stop_scanning(self) -> None:
        """Clear the scanning indicator from the active line."""
        self._active = _ActiveLine()
        self._refresh()

    def update_scan_progress(self, count: int, total_bytes: int) -> None:
        """Update the file-scan running counters."""
        self._scan_count = count
        self._scan_bytes = total_bytes

    def begin_scan(self) -> None:
        """Mark the file-scan phase as active."""
        self._scan_active = True

    def end_scan(self) -> None:
        """Mark the file-scan phase as complete."""
        self._scan_active = False

    # --- Private rendering -----------------------------------------------

    def _refresh(self) -> None:
        """No-op — auto-refresh handles display updates at REFRESH_PER_SECOND."""
        pass

    def _render(self) -> Group:
        """Build the complete live dashboard layout."""
        parts: list[Any] = [_render_header(self)]
        _append_completed(
            parts, self._s1, self._s2, self._s2_reducer, self._s3, self._s4,
            self._s4_peak, self._s4_diversity, self._s4_tournament, self._s4_narrative,
        )
        if self._scan_active and not self._s1.total:
            parts.append(_render_scan(self))
        self._append_active(parts)
        if self._film:
            parts.append(Text(""))
            parts.append(Text.from_markup(
                f"  {build_film_strip(self._film)}"
            ))
        return Group(*parts)

    def _should_show_scan(self) -> bool:
        """Show scan info during active scan or while Stage 1 is running."""
        if self._scan_active:
            return True
        if self._s1.total > 0 and not self._s1.is_complete:
            return True
        return False

    def _append_active(self, parts: list[Any]) -> None:
        """Add the currently active stage panel."""
        if _is_active(self._s1):
            parts.append(_render_s1(self))
        elif _is_active(self._s2):
            parts.append(_render_s2(self))
        elif _is_active(self._s3):
            parts.append(_render_s3(self))
        elif _is_active(self._s4):
            parts.append(_render_s4(self))

# ---------------------------------------------------------------------------
# Dashboard rendering helpers (module-level, max 20 lines each)
# ---------------------------------------------------------------------------


def _is_active(stage: _StageState) -> bool:
    """Check if a stage is currently active."""
    return stage.total > 0 and not stage.is_complete


def _update_reducer_tallies(
    tallies: _S2ReducerTallies, update_in: _Stage2ReducerUpdateInput
) -> _S2ReducerTallies:
    """Return a new tally object with the flagged counters incremented."""
    return _S2ReducerTallies(
        palette_outliers=tallies.palette_outliers + (1 if update_in.is_palette_outlier else 0),
        exposure_outliers=tallies.exposure_outliers + (1 if update_in.is_exposure_outlier else 0),
        exif_outliers=tallies.exif_outliers + (1 if update_in.is_exif_outlier else 0),
        scene_starts=tallies.scene_starts + (1 if update_in.is_scene_start else 0),
    )


def _begin_sub_stage(stage: _StageState, total: int) -> None:
    """Initialise a sub-stage progress tracker in place."""
    stage.total = total
    stage.done = 0
    stage.start_time = time.monotonic()
    stage.elapsed = 0.0
    stage.is_complete = False
    stage.summary_line = ""


def _advance_sub_stage(stage: _StageState, done: int) -> None:
    """Record progress for a sub-stage in place."""
    stage.done = done
    if stage.start_time > 0:
        stage.elapsed = time.monotonic() - stage.start_time


class _SubStageSummary(BaseModel):
    """Input bundle for finishing a sub-stage line."""

    label: str
    elapsed: float


def _finish_sub_stage(stage: _StageState, summary: _SubStageSummary) -> None:
    """Collapse a sub-stage to its summary line in place."""
    stage.is_complete = True
    stage.elapsed = summary.elapsed
    stage.summary_line = (
        f"    [{CLR_CHECK}]\u2713[/] Stage 4 {summary.label}  "
        f"[{CLR_DIM}]{_fmt_time(summary.elapsed)}[/]"
    )


def _update_s1_light(tallies: _S1Tallies, result: Stage1Result) -> None:
    """Update light tallies. Dupes are set separately after detection."""
    exp = result.exposure
    if exp.has_highlight_clip or exp.has_shadow_clip:
        tallies.clip += 1
    elif exp.has_color_cast:
        tallies.cast += 1
    else:
        tallies.good_light += 1


def _fmt_s1_detail(result: Stage1Result) -> str:
    """Format Stage 1 detail metrics for current photo line."""
    b = result.blur
    e = result.exposure
    return (
        f"[{CLR_DIM}]ten:{b.tenengrad:.2f}  "
        f"dr:{e.dr_score:.2f}  "
        f"mid:{e.midtone_pct:.2f}[/]"
    )


def _fmt_s2_detail(fusion: Any) -> str:
    """Format Stage 2 detail metrics for current photo line."""
    s2 = fusion.stage2
    return (
        f"[{CLR_DIM}]TOPIQ {s2.topiq:.2f}  "
        f"Aes {s2.laion_aesthetic:.2f}  "
        f"CLIP {s2.clipiqa:.2f}  "
        f"\u2192 {s2.composite:.3f}[/]"
    )


def _make_loading_models_line() -> _ActiveLine:
    """Return an active line that signals Stage 2 model initialisation."""
    return _ActiveLine(filename="loading Stage 2 models\u2026", detail="")


# ---------------------------------------------------------------------------
# Backward-compat re-exports for the one-shot panel printers. These live in
# ``cull.dashboard_show`` now — the imports below keep ``from cull.dashboard
# import show_results_card`` (and friends) working unchanged. Place at the
# bottom of the module so ``dashboard_show`` can safely pull constants from
# this module during its own import.
# ---------------------------------------------------------------------------

from cull.dashboard_show import (  # noqa: E402
    DiskDisplayInfo,
    DryRunSummary,
    ResultsSummary,
    TuiHandoffCtx,
    show_disk_selection,
    show_dry_run_results,
    show_general_error,
    show_move_complete,
    show_report_writing,
    show_results_card,
    show_tui_handoff,
    show_vlm_load_error,
)
