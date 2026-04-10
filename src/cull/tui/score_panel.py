"""Score detail panel showing Stage 1/2/3 breakdown for a photo."""

from __future__ import annotations

import logging

from pydantic import BaseModel
from textual.widgets import Static

from cull.models import (
    PhotoDecision,
    ShootStatsScore,
    Stage1Result,
    Stage2Result,
    Stage3Result,
)

logger = logging.getLogger(__name__)

PANEL_TITLE: str = "Score Detail"
PASS_MARK: str = "PASS"
FAIL_MARK: str = "FAIL"
EM_DASH: str = "\u2014"
ANOMALY_THRESHOLD: float = 0.5


class _BlurLine(BaseModel):
    """Formatted blur metric display values."""

    tenengrad: str
    fft_ratio: str
    blur_tier: str
    subject_line: str


class _ExposureLine(BaseModel):
    """Formatted exposure metric display values."""

    dr_score: str
    clipping: str
    color_cast: str


def _format_tilt(tilt_degrees: float | None) -> str:
    """Format tilt angle for display, returning em dash when absent."""
    if tilt_degrees is None:
        return EM_DASH
    return f"{tilt_degrees:.1f}\u00b0"


def _format_taste(probability: float | None) -> str:
    """Format taste probability for display, returning em dash when absent."""
    if probability is None:
        return EM_DASH
    return f"{probability:.3f}"


def _format_anomaly_flags(stats: ShootStatsScore | None) -> str:
    """Summarise shoot anomaly scores as a compact flag string."""
    if stats is None:
        return EM_DASH
    flags: list[str] = []
    if stats.palette_outlier_score > ANOMALY_THRESHOLD:
        flags.append("palette")
    if stats.exposure_drift_score > ANOMALY_THRESHOLD:
        flags.append("exposure")
    if stats.exif_anomaly_score > ANOMALY_THRESHOLD:
        flags.append("exif")
    return ", ".join(flags) if flags else "none"


def _format_blur(s1: Stage1Result) -> _BlurLine:
    """Format blur scores into display strings."""
    subject = ""
    if s1.blur.subject_sharpness is not None:
        bg = s1.blur.background_sharpness or 0.0
        bokeh = "  BOKEH" if s1.blur.is_bokeh else ""
        subject = f"Subject sharp: {s1.blur.subject_sharpness:.2f}  Background: {bg:.2f}{bokeh}"
    return _BlurLine(
        tenengrad=f"Tenengrad:     {s1.blur.tenengrad:.2f}",
        fft_ratio=f"FFT ratio:     {s1.blur.fft_ratio:.2f}",
        blur_tier=f"Blur tier:     {s1.blur.blur_tier}",
        subject_line=subject,
    )


def _format_exposure(s1: Stage1Result) -> _ExposureLine:
    """Format exposure scores into display strings."""
    return _ExposureLine(
        dr_score=f"Exposure DR:   {s1.exposure.dr_score:.2f}",
        clipping=f"Clip H: {s1.exposure.clipping_highlight:.1%}  S: {s1.exposure.clipping_shadow:.1%}",
        color_cast=f"Color cast:    {s1.exposure.color_cast_score:.1f}",
    )


def _render_stage1(s1: Stage1Result) -> list[str]:
    """Render Stage 1 section lines."""
    blur = _format_blur(s1)
    exp = _format_exposure(s1)
    tilt_degrees = s1.geometry.tilt_degrees if s1.geometry else None
    tilt = _format_tilt(tilt_degrees)
    lines = [
        "Stage 1 (Classical)",
        f"  {blur.tenengrad}",
        f"  {blur.fft_ratio}",
        f"  {blur.blur_tier}",
        f"  Tilt (deg):    {tilt}",
    ]
    if blur.subject_line:
        lines.append(f"  {blur.subject_line}")
    lines.extend([f"  {exp.dr_score}  {exp.clipping}", f"  {exp.color_cast}"])
    return lines


def _render_stage2(s2: Stage2Result) -> list[str]:
    """Render Stage 2 section lines."""
    composition_val = f"{s2.composition.composite:.3f}" if s2.composition else EM_DASH
    subj_sharp = f"{s2.subject_blur.tenengrad:.2f}" if s2.subject_blur else EM_DASH
    taste_p = _format_taste(s2.taste.probability if s2.taste else None)
    anomaly = _format_anomaly_flags(s2.shoot_stats)
    return [
        "Stage 2 (IQA)",
        f"  TOPIQ:              {s2.topiq:.2f}",
        f"  LAION Aesth:        {s2.laion_aesthetic:.2f}",
        f"  CLIP-IQA+:          {s2.clipiqa:.2f}",
        f"  Composite:          {s2.composite:.3f}",
        f"  Preset:             {s2.preset_used}",
        f"  Composition:        {composition_val}",
        f"  Subject sharpness:  {subj_sharp}",
        f"  Taste p:            {taste_p}",
        f"  Shoot anomaly:      {anomaly}",
    ]


def _render_stage3(s3: Stage3Result) -> list[str]:
    """Render Stage 3 section lines."""
    keeper_str = str(s3.is_keeper).lower() if s3.is_keeper is not None else "n/a"
    flags_str = ", ".join(s3.flags) if s3.flags else "none"
    lines = [
        f"Stage 3 (VLM -- {s3.model_used})",
        f"  keeper: {keeper_str}   confidence: {s3.confidence:.2f}",
    ]
    _append_vlm_scores(s3, lines)
    lines.append(f"  flags: [{flags_str}]")
    return lines


def _append_vlm_scores(s3: Stage3Result, lines: list[str]) -> None:
    """Append sharpness/exposure/composition/expression scores if present."""
    parts: list[str] = []
    if s3.sharpness is not None:
        parts.append(f"sharp: {s3.sharpness:.1f}")
    if s3.exposure is not None:
        parts.append(f"exp: {s3.exposure:.1f}")
    if s3.composition is not None:
        parts.append(f"comp: {s3.composition:.1f}")
    if s3.expression is not None:
        parts.append(f"expr: {s3.expression:.1f}")
    if parts:
        lines.append(f"  {' | '.join(parts)}")


def render_score_text(decision: PhotoDecision) -> str:
    """Build full score panel text from a PhotoDecision."""
    lines: list[str] = [f"--- {PANEL_TITLE} ---", ""]
    if decision.stage1:
        lines.extend(_render_stage1(decision.stage1))
        lines.append("")
    if decision.stage2:
        lines.extend(_render_stage2(decision.stage2))
        lines.append("")
    if decision.stage3:
        lines.extend(_render_stage3(decision.stage3))
    return "\n".join(lines)


class ScorePanel(Static):
    """Toggleable score detail widget for the TUI."""

    DEFAULT_CSS = """
    ScorePanel {
        height: auto;
        width: 1fr;
        border: solid green;
        display: none;
    }
    ScorePanel.visible {
        display: block;
    }
    """

    def show_scores(self, decision: PhotoDecision) -> None:
        """Update panel content with the given decision's scores."""
        self.update(render_score_text(decision))

    def toggle_visible(self) -> None:
        """Toggle the panel visibility."""
        self.toggle_class("visible")
