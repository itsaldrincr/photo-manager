"""Rich renderable builders for the live cull dashboard.

Extracted from ``cull.dashboard`` to keep the main module focused on stage
trackers and the public Dashboard class. All functions here are pure: they
read state from a ``Dashboard`` instance (or state models) and return Rich
renderables (``Text``, ``Panel``) without mutating anything.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from rich.panel import Panel
from rich.text import Text

from cull.dashboard import (
    BAR_WIDTH,
    CLR_ACTIVE,
    CLR_BORDER,
    CLR_CHECK,
    CLR_DIM,
    CLR_DUPLICATE,
    CLR_HEADER,
    CLR_KEEPER,
    CLR_PROGRESS,
    CLR_REJECT,
    CLR_UNCERTAIN,
    GAUGE_EMPTY,
    GAUGE_FILL_MS_PER_BAR,
    GAUGE_FILLED,
    GAUGE_SCAN_MS_PER_DIM,
    GAUGE_WIDTH,
    LABEL_TO_MARK,
    MARK_DUPLICATE,
    MARK_KEEPER,
    MARK_REJECT,
    MARK_SELECT,
    MARK_UNCERTAIN,
    MS_PER_SECOND,
    PERCENT_SCALE,
    REFRESH_PER_SECOND,
    RESULT_BAR_WIDTH,
    S3_CONFIDENCE_INDEX,
    S3_DIMENSIONS,
    SPARK_CHARS,
    SPARKLINE_WIDTH,
    SPINNER_S1,
    SPINNER_S2,
    SPINNER_S3,
    _AnalysisState,
    _format_bytes,
    _fmt_time,
    _S1Tallies,
    _S4Tallies,
    _StageState,
    build_film_strip,
)

if TYPE_CHECKING:
    from cull.dashboard import Dashboard


# ---------------------------------------------------------------------------
# Parameter bundles
# ---------------------------------------------------------------------------


class _EtaInput(BaseModel):
    """Input bundle for _fmt_eta."""

    done: int
    total: int
    elapsed: float


class _BarInput(BaseModel):
    """Input bundle for _build_result_bar."""

    count: int
    total: int
    color: str


# ---------------------------------------------------------------------------
# Leaf formatters
# ---------------------------------------------------------------------------


def _fmt_eta(eta_in: _EtaInput) -> str:
    """Estimate remaining time from progress so far."""
    if eta_in.done <= 0 or eta_in.total <= 0:
        return ""
    rate = eta_in.elapsed / eta_in.done
    remaining = rate * (eta_in.total - eta_in.done)
    return f"~{_fmt_time(remaining)}"


def _build_progress_bar(done: int, total: int) -> str:
    """Build a progress bar string with count and percentage."""
    if total <= 0:
        return ""
    frac = done / total
    filled = int(frac * BAR_WIDTH)
    bar = "\u2501" * filled + " " * (BAR_WIDTH - filled)
    pct = int(frac * PERCENT_SCALE)
    return f"[{CLR_PROGRESS}]{bar}[/]  {done}/{total}  {pct}%"


def _build_gauge_bar(value: float, color: str) -> str:
    """Build a single gauge bar like: filled + empty + score."""
    filled = min(int(value * GAUGE_WIDTH), GAUGE_WIDTH)
    empty = GAUGE_WIDTH - filled
    bar = f"[{color}]{GAUGE_FILLED * filled}[/]"
    rest = f"[bright_black]{GAUGE_EMPTY * empty}[/]"
    return f"{bar}{rest}  [{CLR_ACTIVE}]{value:.2f}[/]"


def _spinner_frame(frames: list[str], elapsed: float) -> str:
    """Pick a spinner frame based on elapsed time."""
    idx = int(elapsed * REFRESH_PER_SECOND) % len(frames)
    return frames[idx]


def _build_result_bar(bar_in: _BarInput) -> str:
    """Build a horizontal bar for the results card."""
    if bar_in.total <= 0:
        return ""
    frac = bar_in.count / bar_in.total
    filled = max(int(frac * RESULT_BAR_WIDTH), 0)
    empty = RESULT_BAR_WIDTH - filled
    bar = f"[{bar_in.color}]{GAUGE_FILLED * filled}[/]"
    rest = f"[bright_black]{GAUGE_EMPTY * empty}[/]"
    pct = int(frac * PERCENT_SCALE)
    return f"{bar}{rest}  {pct}%"


def _decision_color(decision: str) -> str:
    """Map a decision label to its border/fill color."""
    colors: dict[str, str] = {
        "keeper": CLR_KEEPER,
        "rejected": CLR_REJECT,
        "uncertain": CLR_UNCERTAIN,
    }
    return colors.get(decision, CLR_BORDER)


def build_sparkline(scores: list[float], width: int = SPARKLINE_WIDTH) -> str:
    """Bucket scores into bins and map each to a spark character."""
    if not scores:
        return ""
    max_idx = len(SPARK_CHARS) - 1
    bin_size = max(len(scores) // width, 1)
    bins: list[float] = []
    for i in range(0, len(scores), bin_size):
        chunk = scores[i : i + bin_size]
        bins.append(sum(chunk) / len(chunk))
        if len(bins) >= width:
            break
    return "".join(
        SPARK_CHARS[min(int(v * max_idx), max_idx)] for v in bins
    )


def _compute_median(scores: list[float]) -> float:
    """Compute median of a list of floats."""
    if not scores:
        return 0.0
    sorted_scores = sorted(scores)
    mid = len(sorted_scores) // 2
    if len(sorted_scores) % 2 == 0:
        return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
    return sorted_scores[mid]


# ---------------------------------------------------------------------------
# Header / scan / completed-stage renderers
# ---------------------------------------------------------------------------


def _render_scan(db: Dashboard) -> Text:
    """Render the live JPEG scanning state."""
    size_str = _format_bytes(db._scan_bytes)
    if db._scan_active:
        spinner = _spinner_frame(SPINNER_S2, time.monotonic())
        line = (
            f"\n  [{CLR_PROGRESS}]{spinner}[/] Scanning for JPEGs  "
            f"[{CLR_ACTIVE}]{db._scan_count}[/] files  "
            f"[{CLR_DIM}]{size_str}[/]"
        )
    else:
        line = (
            f"\n  [{CLR_CHECK}]\u2713[/] Scanned  "
            f"[{CLR_ACTIVE}]{db._scan_count}[/] files  "
            f"[{CLR_DIM}]{size_str}[/]"
        )
    return Text.from_markup(line)


def _render_header(db: Dashboard) -> Panel:
    """Build the launch header panel."""
    body = Text.from_markup(
        f"  [{CLR_HEADER}]\u25c9 cull[/]\n"
        f"  [{CLR_DIM}]{db._source}[/] \u00b7 "
        f"{db._photo_count:,} photos \u00b7 "
        f"{db._file_size_gb:.1f} GB"
    )
    return Panel(
        body,
        border_style=CLR_BORDER,
        subtitle=f"[bright_cyan]{db._preset}[/]",
        subtitle_align="right",
    )


def _append_completed(
    parts: list[Any], *stages: _StageState,
) -> None:
    """Add summary lines for completed stages."""
    for stage in stages:
        if stage.is_complete:
            parts.append(Text.from_markup(stage.summary_line))


# ---------------------------------------------------------------------------
# Stage 1 renderer
# ---------------------------------------------------------------------------


def _render_s1(db: Dashboard) -> Text:
    """Render Stage 1 active panel content."""
    s = db._s1
    elapsed = time.monotonic() - s.start_time
    spinner = _spinner_frame(SPINNER_S1, elapsed)
    eta = _fmt_eta(_EtaInput(done=s.done, total=s.total, elapsed=s.elapsed))
    t = db._s1_tallies
    size_str = _format_bytes(db._scan_bytes)
    lines = [
        f"\n  [{CLR_PROGRESS}]{spinner}[/] Stage 1  "
        f"Classical Filters  "
        f"[{CLR_DIM}]({db._scan_count} files \u00b7 {size_str})[/]  "
        f"[{CLR_DIM}]{_fmt_time(s.elapsed)} \u00b7 {eta}[/]",
        f"  {_build_progress_bar(s.done, s.total)}",
        "",
        _fmt_s1_tallies(t),
        "",
    ]
    if db._active.filename:
        lines.append(
            f"  [{CLR_ACTIVE}]\u25b8 {db._active.filename}[/]"
            f"  {db._active.detail}"
        )
    return Text.from_markup("\n".join(lines))


def _fmt_s1_tallies(t: _S1Tallies) -> str:
    """Format Stage 1 tally boxes as a single markup line."""
    total = t.sharp + t.soft + t.bokeh
    focus = (
        f"  [{CLR_KEEPER}]\u25cf[/] sharp {t.sharp:>3}  "
        f"[{CLR_REJECT}]\u25cb[/] soft {t.soft:>4}  "
        f"[{CLR_PROGRESS}]\u25d0[/] bokeh {t.bokeh:>3}  "
        f"[{CLR_DIM}]({total})[/]"
    )
    light = (
        f"  [{CLR_KEEPER}]\u25cf[/] ok {t.good_light:>6}  "
        f"[{CLR_REJECT}]\u25cb[/] clip {t.clip:>4}  "
        f"[{CLR_REJECT}]\u25cb[/] cast {t.cast:>4}"
    )
    groups = (
        f"  [{CLR_DIM}]\u25cc[/] bursts {t.bursts:>3}  "
        f"[{CLR_DUPLICATE}]\u25cc[/] dupes {t.dupes:>3}"
    )
    return f"{focus}\n{light}\n{groups}"


# ---------------------------------------------------------------------------
# Stage 2 renderer
# ---------------------------------------------------------------------------


def _render_s2(db: Dashboard) -> Text:
    """Render Stage 2 active panel content."""
    s = db._s2
    elapsed = time.monotonic() - s.start_time
    spinner = _spinner_frame(SPINNER_S2, elapsed)
    eta = _fmt_eta(_EtaInput(done=s.done, total=s.total, elapsed=s.elapsed))
    spark = build_sparkline(db._s2_scores)
    median = _compute_median(db._s2_scores)
    t = db._s2_tallies
    lines = [
        f"\n  [{CLR_PROGRESS}]{spinner}[/] Stage 2  "
        f"IQA + Aesthetic Scoring"
        f"  [{CLR_DIM}]{_fmt_time(s.elapsed)} \u00b7 {eta}[/]",
        f"  {_build_progress_bar(s.done, s.total)}",
        "",
        f"  scores  [{CLR_PROGRESS}]{spark}[/]"
        f"  median {median:.2f}",
        "",
        f"     {MARK_KEEPER} keeper {t.keepers}"
        f"    {MARK_UNCERTAIN} ambiguous {t.ambiguous}"
        f"    {MARK_REJECT} reject {t.rejects}",
    ]
    if db._active.filename:
        lines.append(
            f"\n  [{CLR_ACTIVE}]\u25b8 {db._active.filename}[/]"
            f"  {db._active.detail}"
        )
    return Text.from_markup("\n".join(lines))


# ---------------------------------------------------------------------------
# Stage 3 renderer
# ---------------------------------------------------------------------------


def _render_s3(db: Dashboard) -> Text:
    """Render Stage 3 active panel content."""
    s = db._s3
    elapsed = time.monotonic() - s.start_time
    spinner = _spinner_frame(SPINNER_S3, elapsed)
    eta = _fmt_eta(_EtaInput(done=s.done, total=s.total, elapsed=s.elapsed))
    lines = [
        f"\n  [{CLR_PROGRESS}]{spinner}[/] Stage 3  "
        f"VLM Tiebreaker ({db._s3_model})"
        f"  [{CLR_DIM}]{_fmt_time(s.elapsed)} \u00b7 {eta}[/]",
        f"  {_build_progress_bar(s.done, s.total)}",
    ]
    gauge = _render_gauge(db._analysis)
    if gauge:
        lines.append("")
        lines.append(gauge)
    return Text.from_markup("\n".join(lines))


def _render_gauge(a: _AnalysisState) -> str:
    """Build the Stage 3 analysis gauge panel."""
    if not a.path:
        return ""
    if a.is_scanning:
        return _render_scanning(a)
    if a.scores:
        return _render_filled(a)
    return ""


def _render_scanning(a: _AnalysisState) -> str:
    """Build the scanning animation gauge."""
    elapsed_ms = (time.monotonic() - a.start_time) * MS_PER_SECOND
    active_idx = int(elapsed_ms / GAUGE_SCAN_MS_PER_DIM) % len(S3_DIMENSIONS)
    spinner = _spinner_frame(SPINNER_S3, time.monotonic() - a.start_time)
    empty_bar = f"[bright_black]{GAUGE_EMPTY * GAUGE_WIDTH}[/]"
    lines: list[str] = []
    for i, dim in enumerate(S3_DIMENSIONS):
        prefix = f"  {spinner} " if i == active_idx else "    "
        suffix = f"  [{CLR_DIM}]scanning...[/]" if i == active_idx else ""
        if i == S3_CONFIDENCE_INDEX and i > 0:
            lines.append("")
        lines.append(f"{prefix}{dim:<12} {empty_bar}{suffix}")
    body = "\n".join(lines)
    return (
        f"  [{CLR_BORDER}]\u256d\u2500 analyzing {a.path} "
        f"\u2500\u256e[/]\n{body}\n"
        f"  [{CLR_BORDER}]\u2570\u2500\u2500\u2500\u256f[/]"
    )


def _render_filled(a: _AnalysisState) -> str:
    """Build the filled gauge with scores, cascading one gauge per GAUGE_FILL_MS_PER_BAR."""
    color = _decision_color(a.decision)
    mark = LABEL_TO_MARK.get(a.decision, MARK_UNCERTAIN)
    elapsed_ms = (time.monotonic() - a.fill_start) * MS_PER_SECOND
    visible_count = int(elapsed_ms / GAUGE_FILL_MS_PER_BAR)
    empty_bar = f"[bright_black]{GAUGE_EMPTY * GAUGE_WIDTH}[/]"
    lines: list[str] = []
    for i, dim in enumerate(S3_DIMENSIONS):
        if i == S3_CONFIDENCE_INDEX and i > 0:
            lines.append("")
        if i < visible_count:
            value = a.scores.get(dim, 0.0)
            bar = _build_gauge_bar(value, color)
        else:
            bar = empty_bar
        lines.append(f"    {dim:<12} {bar}")
    body = "\n".join(lines)
    return (
        f"  [{color}]\u256d\u2500 {a.path} "
        f"\u2500\u2500 {mark} {a.decision} \u2500\u256e[/]\n"
        f"{body}\n"
        f"  [{color}]\u2570\u2500\u2500\u2500\u256f[/]"
    )


# ---------------------------------------------------------------------------
# Stage 4 renderer
# ---------------------------------------------------------------------------


def _render_s4(db: Dashboard) -> Text:
    """Render Stage 4 active panel content."""
    s = db._s4
    elapsed = time.monotonic() - s.start_time
    spinner = _spinner_frame(SPINNER_S2, elapsed)
    eta = _fmt_eta(_EtaInput(done=s.done, total=s.total, elapsed=s.elapsed))
    t = db._s4_tallies
    lines = [
        f"  [{CLR_PROGRESS}]{spinner}[/] Stage 4  "
        f"Curator"
        f"  [{CLR_DIM}]{_fmt_time(s.elapsed)} \u00b7 {eta}[/]",
        f"  {_build_progress_bar(s.done, s.total)}",
        "",
        f"  clusters found: {t.clusters_found}"
        f"   target: {t.target}",
        f"  {MARK_SELECT} selected {t.selected}"
        f"    {MARK_UNCERTAIN} vlm tiebreak {t.vlm_tiebreaks}"
        f"    {MARK_REJECT} skipped {t.skipped}",
    ]
    lines.append(_fmt_s4_compare(t))
    return Text.from_markup("\n".join(lines))


def _fmt_s4_compare(t: _S4Tallies) -> str:
    """Format the current-compare line for Stage 4."""
    if not t.current_compare:
        return ""
    left, right = t.current_compare
    cluster_part = (
        f"  [{CLR_DIM}](cluster #{t.current_cluster_id})[/]"
        if t.current_cluster_id is not None
        else ""
    )
    return (
        f"\n  [{CLR_ACTIVE}]\u25b8 comparing "
        f"{left} vs {right}[/]{cluster_part}"
    )


# ---------------------------------------------------------------------------
# Results panel renderer
# ---------------------------------------------------------------------------


def _build_results_lines(sr: Any, film: list[str]) -> list[str]:
    """Build the lines for results panel."""
    s = sr.summary
    total = s.keepers + s.rejected + s.duplicates + s.uncertain
    total_secs = sr.timing.total_seconds or 1.0
    rate = total / total_secs if total_secs > 0 else 0.0
    return [
        "",
        f"  {MARK_KEEPER} keepers    {s.keepers:>4}  "
        f"{_build_result_bar(_BarInput(count=s.keepers, total=total, color=CLR_KEEPER))}",
        f"  {MARK_REJECT} rejected   {s.rejected:>4}  "
        f"{_build_result_bar(_BarInput(count=s.rejected, total=total, color=CLR_REJECT))}",
        f"  {MARK_DUPLICATE} duplicates {s.duplicates:>4}  "
        f"{_build_result_bar(_BarInput(count=s.duplicates, total=total, color=CLR_DUPLICATE))}",
        f"  {MARK_UNCERTAIN} uncertain  {s.uncertain:>4}  "
        f"{_build_result_bar(_BarInput(count=s.uncertain, total=total, color=CLR_UNCERTAIN))}",
        "",
        f"  [{CLR_DIM}]{_fmt_time(total_secs)} total"
        f" \u00b7 {rate:.1f} photos/sec[/]",
        "",
        f"  {build_film_strip(film)}",
        "",
    ]


def _render_results_panel(sr: Any, film: list[str]) -> Panel:
    """Render the final results card panel."""
    lines = _build_results_lines(sr, film)
    body = Text.from_markup("\n".join(lines))
    return Panel(
        body,
        title=f"[{CLR_HEADER}]Results[/]",
        border_style=CLR_BORDER,
    )
