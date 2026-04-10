"""Session-level report card — keep rate, EXIF patterns, advice heuristics.

Uses exifread (listed in pyproject.toml) for EXIF extraction.
No LLM calls — all advice is hardcoded heuristics.
"""

from __future__ import annotations

import logging
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import exifread
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cull.config import REPORT_CARD_EXIF_SAMPLE_SIZE
from cull.models import (
    ExifPattern,
    ExifPatterns,
    PhotoDecision,
    PortraitStats,
    RejectBreakdown,
    ReportCard,
    SessionTiming2,
    Stage2Result,
)
from cull.pipeline import SessionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants — no magic numbers in module code
# ---------------------------------------------------------------------------

SHUTTER_MOTION_RISK_THRESHOLD: float = 1.0 / 125.0  # 1/125 s
MOTION_RISK_ADVICE_THRESHOLD: float = 0.30
EXPOSURE_ADVICE_THRESHOLD: float = 0.20
LOW_KEEP_RATE_THRESHOLD: float = 0.10
_EXIF_HALF_DIVISOR: int = 2
EXIF_HALF_SAMPLE: int = REPORT_CARD_EXIF_SAMPLE_SIZE // _EXIF_HALF_DIVISOR
PANEL_BORDER_STYLE: str = "bright_blue"
SESSION_REPORT_FILENAME: str = "session_report.json"
_REJECT_CATEGORIES: tuple[str, ...] = ("blur", "exposure", "noise", "burst", "duplicate", "vlm")

# Decision labels
_KEEPER_LABELS: frozenset[str] = frozenset({"keeper", "select"})
_REJECT_LABELS: frozenset[str] = frozenset({"rejected", "duplicate"})


# ---------------------------------------------------------------------------
# Pydantic bundles for multi-param helpers
# ---------------------------------------------------------------------------


class _ExifSampleInput(BaseModel):
    """Input bundle for _sample_exif."""

    decisions: list[PhotoDecision]
    count: int


class _AdviceInput(BaseModel):
    """Input bundle for _generate_advice."""

    breakdown: RejectBreakdown
    exif: ExifPatterns
    keep_rate: float


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------


def _load_session(source: Path) -> SessionResult:
    """Load a SessionResult from <source>/session_report.json."""
    report_path = source / SESSION_REPORT_FILENAME
    if not report_path.exists():
        raise FileNotFoundError(
            f"Session report not found at {report_path}. "
            "Run the cull pipeline first to generate it."
        )
    return SessionResult.model_validate_json(report_path.read_text())


# ---------------------------------------------------------------------------
# Keep / reject counts
# ---------------------------------------------------------------------------


def _compute_keep_rate(session: SessionResult) -> tuple[float, int, int]:
    """Return (keep_rate, keep_count, reject_count) from session decisions."""
    keep_count = sum(1 for d in session.decisions if d.decision in _KEEPER_LABELS)
    reject_count = sum(1 for d in session.decisions if d.decision in _REJECT_LABELS)
    total = keep_count + reject_count
    keep_rate = keep_count / total if total > 0 else 0.0
    return keep_rate, keep_count, reject_count


# ---------------------------------------------------------------------------
# Reject breakdown
# ---------------------------------------------------------------------------


def _is_exposure_reject(decision: PhotoDecision) -> bool:
    """Return True if the photo was rejected for exposure reasons."""
    s1 = decision.stage1
    if s1 is None or s1.is_pass:
        return False
    exp = s1.exposure
    return exp.has_highlight_clip or exp.has_shadow_clip or exp.has_color_cast or exp.has_low_dr


def _classify_reject(decision: PhotoDecision) -> str:
    """Return a reject category string for one rejected decision."""
    if decision.decision == "duplicate":
        return "duplicate"
    s1 = decision.stage1
    if s1 is not None:
        if s1.reject_reason == "blur":
            return "blur"
        if s1.reject_reason == "noise":
            return "noise"
        if s1.burst is not None:
            return "burst"
        if _is_exposure_reject(decision):
            return "exposure"
    s3 = decision.stage3
    if s3 is not None and s3.is_keeper is False:
        return "vlm"
    return "blur"


def _compute_reject_breakdown(session: SessionResult) -> RejectBreakdown:
    """Build RejectBreakdown from session decisions, normalized to 1.0."""
    rejects = [d for d in session.decisions if d.decision in _REJECT_LABELS]
    total = len(rejects)
    if total == 0:
        return RejectBreakdown()
    counts: dict[str, int] = {k: 0 for k in _REJECT_CATEGORIES}
    for decision in rejects:
        counts[_classify_reject(decision)] += 1
    return RejectBreakdown(
        blur_pct=counts["blur"] / total,
        exposure_pct=counts["exposure"] / total,
        noise_pct=counts["noise"] / total,
        burst_pct=counts["burst"] / total,
        duplicate_pct=counts["duplicate"] / total,
        vlm_pct=counts["vlm"] / total,
    )


# ---------------------------------------------------------------------------
# EXIF extraction and pattern building
# ---------------------------------------------------------------------------


def _read_exif_tags(photo_path: Path) -> dict[str, Any]:
    """Read EXIF tags from one photo file using exifread."""
    with photo_path.open("rb") as fh:
        tags = exifread.process_file(fh, stop_tag="GPS GPSAltitude", details=False)
    return tags


def _format_aperture(tags: dict[str, Any]) -> str | None:
    """Extract and format aperture as 'f/2.8' string."""
    tag = tags.get("EXIF FNumber")
    if tag is None:
        return None
    ratio = tag.values[0]
    value = float(ratio.num) / float(ratio.den) if ratio.den else None
    return f"f/{value:.1f}" if value else None


def _format_shutter(tags: dict[str, Any]) -> str | None:
    """Extract and format shutter speed as '1/250s' string."""
    tag = tags.get("EXIF ExposureTime")
    if tag is None:
        return None
    ratio = tag.values[0]
    if ratio.den == 1:
        return f"{ratio.num}s"
    return f"{ratio.num}/{ratio.den}s"


def _shutter_to_seconds(shutter_str: str | None) -> float | None:
    """Convert a shutter string like '1/250s' to seconds as a float."""
    if shutter_str is None:
        return None
    cleaned = shutter_str.rstrip("s")
    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) == 2 and float(parts[1]) != 0:
            return float(parts[0]) / float(parts[1])
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_exif_dict(photo_path: Path) -> dict[str, Any] | None:
    """Extract EXIF data from one photo; return None on any failure."""
    try:
        tags = _read_exif_tags(photo_path)
        aperture = _format_aperture(tags)
        shutter = _format_shutter(tags)
        iso_tag = tags.get("EXIF ISOSpeedRatings")
        lens_tag = tags.get("EXIF LensModel")
        fl_tag = tags.get("EXIF FocalLength")
        iso = int(str(iso_tag)) if iso_tag else None
        lens = str(lens_tag) if lens_tag else None
        fl_ratio = fl_tag.values[0] if fl_tag else None
        focal_length = float(fl_ratio.num) / float(fl_ratio.den) if fl_ratio and fl_ratio.den else None
        return {"aperture": aperture, "shutter": shutter, "iso": iso, "lens": lens, "focal_length_mm": focal_length}
    except Exception as exc:
        logger.debug("EXIF read failed for %s: %s", photo_path, exc)
        return None


def _sample_exif(sample_input: _ExifSampleInput) -> list[dict[str, Any]]:
    """Sample up to count photos from decisions and read their EXIF."""
    population = [d.photo.path for d in sample_input.decisions]
    chosen = random.sample(population, min(sample_input.count, len(population)))
    results: list[dict[str, Any]] = []
    for path in chosen:
        exif = _extract_exif_dict(path)
        if exif is not None:
            results.append(exif)
    return results


def _modal_string(values: list[str]) -> str | None:
    """Return the most common string value, or None if empty."""
    if not values:
        return None
    return max(set(values), key=values.count)


def _build_exif_pattern(samples: list[dict[str, Any]]) -> ExifPattern:
    """Compute typical EXIF settings from a list of sample dicts."""
    apertures = [s["aperture"] for s in samples if s.get("aperture")]
    shutters = [s["shutter"] for s in samples if s.get("shutter")]
    isos = [s["iso"] for s in samples if s.get("iso")]
    lenses = [s["lens"] for s in samples if s.get("lens")]
    focal_lengths = [s["focal_length_mm"] for s in samples if s.get("focal_length_mm")]
    return ExifPattern(
        aperture=_modal_string(apertures),
        shutter=_modal_string(shutters),
        iso=int(statistics.median(isos)) if isos else None,
        lens=_modal_string(lenses),
        focal_length_mm=statistics.median(focal_lengths) if focal_lengths else None,
    )


def _compute_motion_risk(reject_samples: list[dict[str, Any]]) -> float:
    """Return fraction of reject samples with shutter below motion-risk threshold."""
    if not reject_samples:
        return 0.0
    below_threshold = sum(
        1 for s in reject_samples
        if _shutter_to_seconds(s.get("shutter")) is not None
        and _shutter_to_seconds(s.get("shutter")) > SHUTTER_MOTION_RISK_THRESHOLD
    )
    return below_threshold / len(reject_samples)


def _compute_exif_patterns(session: SessionResult) -> ExifPatterns:
    """Sample EXIF from keepers and rejects; compute patterns and motion risk."""
    keeper_decisions = [d for d in session.decisions if d.decision in _KEEPER_LABELS]
    reject_decisions = [d for d in session.decisions if d.decision in _REJECT_LABELS]
    keeper_samples = _sample_exif(_ExifSampleInput(decisions=keeper_decisions, count=EXIF_HALF_SAMPLE))
    reject_samples = _sample_exif(_ExifSampleInput(decisions=reject_decisions, count=EXIF_HALF_SAMPLE))
    return ExifPatterns(
        keepers_typical=_build_exif_pattern(keeper_samples),
        rejects_typical=_build_exif_pattern(reject_samples),
        motion_risk_pct=_compute_motion_risk(reject_samples),
    )


# ---------------------------------------------------------------------------
# Portrait stats
# ---------------------------------------------------------------------------


def _collect_portrait_scores(stage2: Stage2Result | None) -> tuple[float | None, bool]:
    """Return (eye_sharpness_avg, is_eyes_closed) from a Stage2Result."""
    if stage2 is None or stage2.portrait is None:
        return None, False
    p = stage2.portrait
    sharpness_vals = [v for v in (p.eye_sharpness_left, p.eye_sharpness_right) if v is not None]
    avg_sharpness = statistics.mean(sharpness_vals) if sharpness_vals else None
    return avg_sharpness, p.is_eyes_closed


def _compute_portrait_stats(session: SessionResult) -> PortraitStats | None:
    """Compute portrait aggregate stats; return None if no portrait data exists."""
    keeper_decisions = [d for d in session.decisions if d.decision in _KEEPER_LABELS]
    portrait_data = [_collect_portrait_scores(d.stage2) for d in keeper_decisions]
    sharpness_vals = [s for s, _ in portrait_data if s is not None]
    if not sharpness_vals:
        return None
    closed_count = sum(1 for _, closed in portrait_data if closed)
    closed_rate = closed_count / len(portrait_data) if portrait_data else 0.0
    return PortraitStats(
        unique_subjects=len(keeper_decisions),
        eye_sharpness_median=statistics.median(sharpness_vals),
        eyes_closed_rate=closed_rate,
    )


# ---------------------------------------------------------------------------
# Session timing
# ---------------------------------------------------------------------------


def _get_capture_time(decision: PhotoDecision) -> datetime | None:
    """Return the capture datetime from EXIF or fall back to fs_mtime."""
    if decision.photo.exif_datetime is not None:
        return decision.photo.exif_datetime
    mtime = decision.photo.fs_mtime
    if mtime is not None:
        return datetime.fromtimestamp(mtime, tz=timezone.utc)
    return None


def _compute_timing(session: SessionResult) -> SessionTiming2:
    """Build SessionTiming2 from decision timestamps."""
    times = [_get_capture_time(d) for d in session.decisions]
    valid_times = [t for t in times if t is not None]
    if not valid_times:
        return SessionTiming2()
    first = min(valid_times)
    last = max(valid_times)
    duration = (last - first).total_seconds()
    photo_count = len(session.decisions)
    photos_per_minute = (photo_count / duration) * 60.0 if duration > 0 else 0.0
    return SessionTiming2(
        duration_seconds=duration,
        photos_per_minute=photos_per_minute,
        first_capture=first,
        last_capture=last,
    )


# ---------------------------------------------------------------------------
# Advice heuristics
# ---------------------------------------------------------------------------


def _motion_advice(motion_pct: float) -> str | None:
    """Return shutter speed advice if motion risk is above threshold."""
    if motion_pct <= MOTION_RISK_ADVICE_THRESHOLD:
        return None
    pct_str = f"{motion_pct:.0%}"
    return (
        f"Consider raising minimum shutter speed to 1/125s or faster "
        f"— {pct_str} of rejects were below it."
    )


def _exposure_advice(exp_pct: float) -> str | None:
    """Return exposure advice if exposure reject rate is above threshold."""
    if exp_pct <= EXPOSURE_ADVICE_THRESHOLD:
        return None
    pct_str = f"{exp_pct:.0%}"
    return f"Exposure issues drove {pct_str} of rejects — consider EVF review or auto-bracketing."


def _generate_advice(advice_input: _AdviceInput) -> list[str]:
    """Generate hardcoded heuristic advice from breakdown, exif, and keep rate."""
    advice: list[str] = []
    motion = _motion_advice(advice_input.exif.motion_risk_pct)
    if motion:
        advice.append(motion)
    exposure = _exposure_advice(advice_input.breakdown.exposure_pct)
    if exposure:
        advice.append(exposure)
    if advice_input.keep_rate < LOW_KEEP_RATE_THRESHOLD:
        advice.append("Keep rate is unusually low — verify your culling thresholds match this preset.")
    if not advice:
        advice.append("No actionable patterns detected this session.")
    return advice


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class _CardMetrics(BaseModel):
    """Intermediate bundle holding computed metrics for ReportCard assembly."""

    keep_rate: float
    keep_count: int
    reject_count: int
    breakdown: RejectBreakdown
    exif_patterns: ExifPatterns
    portrait_stats: PortraitStats | None
    timing: SessionTiming2
    advice: list[str]


def _compute_metrics(session: SessionResult) -> _CardMetrics:
    """Compute all report card metrics from a session."""
    keep_rate, keep_count, reject_count = _compute_keep_rate(session)
    breakdown = _compute_reject_breakdown(session)
    exif_patterns = _compute_exif_patterns(session)
    return _CardMetrics(
        keep_rate=keep_rate,
        keep_count=keep_count,
        reject_count=reject_count,
        breakdown=breakdown,
        exif_patterns=exif_patterns,
        portrait_stats=_compute_portrait_stats(session),
        timing=_compute_timing(session),
        advice=_generate_advice(_AdviceInput(breakdown=breakdown, exif=exif_patterns, keep_rate=keep_rate)),
    )


def build_report_card(source: Path) -> ReportCard:
    """Load session, compute all metrics, and return a populated ReportCard."""
    session = _load_session(source)
    metrics = _compute_metrics(session)
    return ReportCard(
        source_path=str(source),
        keep_rate=metrics.keep_rate,
        keep_count=metrics.keep_count,
        reject_count=metrics.reject_count,
        breakdown=metrics.breakdown,
        exif_patterns=metrics.exif_patterns,
        portrait_stats=metrics.portrait_stats,
        timing=metrics.timing,
        advice=metrics.advice,
    )


def _render_header_panel(card: ReportCard) -> Panel:
    """Build the header Rich Panel with source and keep rate."""
    keep_pct = f"{card.keep_rate:.1%}"
    content = Text()
    content.append(f"Source: {card.source_path}\n")
    content.append(f"Keep rate: {keep_pct}  |  Keepers: {card.keep_count}  |  Rejects: {card.reject_count}")
    return Panel(content, title="Session Report Card", border_style=PANEL_BORDER_STYLE)


def _render_breakdown_panel(card: ReportCard) -> Panel:
    """Build the reject breakdown Rich Panel."""
    bd = card.breakdown
    content = Text()
    content.append(f"Blur:       {bd.blur_pct:.1%}\n")
    content.append(f"Exposure:   {bd.exposure_pct:.1%}\n")
    content.append(f"Noise:      {bd.noise_pct:.1%}\n")
    content.append(f"Burst:      {bd.burst_pct:.1%}\n")
    content.append(f"Duplicate:  {bd.duplicate_pct:.1%}\n")
    content.append(f"VLM:        {bd.vlm_pct:.1%}")
    return Panel(content, title="Reject Breakdown", border_style=PANEL_BORDER_STYLE)


def _render_exif_panel(card: ReportCard) -> Panel:
    """Build the EXIF patterns Rich Panel."""
    kt = card.exif_patterns.keepers_typical
    rt = card.exif_patterns.rejects_typical
    content = Text()
    content.append("Keepers typical:\n")
    content.append(f"  Aperture={kt.aperture}  Shutter={kt.shutter}  ISO={kt.iso}  FL={kt.focal_length_mm}mm\n")
    content.append("Rejects typical:\n")
    content.append(f"  Aperture={rt.aperture}  Shutter={rt.shutter}  ISO={rt.iso}  FL={rt.focal_length_mm}mm\n")
    content.append(f"Motion risk: {card.exif_patterns.motion_risk_pct:.1%}")
    return Panel(content, title="EXIF Patterns", border_style=PANEL_BORDER_STYLE)


def _render_portrait_panel(stats: PortraitStats) -> Panel:
    """Build the portrait stats Rich Panel."""
    content = Text()
    content.append(f"Unique subjects: {stats.unique_subjects}\n")
    content.append(f"Eye sharpness median: {stats.eye_sharpness_median:.3f}\n")
    content.append(f"Eyes-closed rate: {stats.eyes_closed_rate:.1%}")
    return Panel(content, title="Portrait Stats", border_style=PANEL_BORDER_STYLE)


def _render_timing_panel(card: ReportCard) -> Panel:
    """Build the timing Rich Panel."""
    t = card.timing
    content = Text()
    content.append(f"Duration: {t.duration_seconds:.0f}s\n")
    content.append(f"Photos/min: {t.photos_per_minute:.1f}\n")
    if t.first_capture:
        content.append(f"First capture: {t.first_capture.isoformat()}\n")
    if t.last_capture:
        content.append(f"Last capture: {t.last_capture.isoformat()}")
    return Panel(content, title="Session Timing", border_style=PANEL_BORDER_STYLE)


def _render_advice_panel(card: ReportCard) -> Panel:
    """Build the advice Rich Panel with bullet list."""
    content = Text()
    for line in card.advice:
        content.append(f"• {line}\n")
    return Panel(content, title="Advice", border_style=PANEL_BORDER_STYLE)


def render_report_card(card: ReportCard) -> None:
    """Display a report card using Rich Panels to the console."""
    console = Console()
    console.print(_render_header_panel(card))
    console.print(_render_breakdown_panel(card))
    console.print(_render_exif_panel(card))
    if card.portrait_stats is not None:
        console.print(_render_portrait_panel(card.portrait_stats))
    console.print(_render_timing_panel(card))
    console.print(_render_advice_panel(card))
