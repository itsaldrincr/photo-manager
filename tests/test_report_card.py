"""Tests for cull.report_card — build_report_card and helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cull.models import (
    BlurScores,
    ExposureScores,
    PhotoDecision,
    PhotoMeta,
    Stage1Result,
    Stage2Result,
    Stage3Result,
    PortraitScores,
)
from cull.pipeline import SessionResult, SessionSummary, SessionTiming
from cull.report_card import (
    _compute_keep_rate,
    _compute_reject_breakdown,
    _generate_advice,
    _compute_portrait_stats,
    _compute_timing,
    _load_session,
    build_report_card,
    _AdviceInput,
    _ExifSampleInput,
    _sample_exif,
)
from cull.models import ExifPatterns, RejectBreakdown, ExifPattern

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

_BLUR_TIER_DEFAULT: int = 1
_TENENGRAD_DEFAULT: float = 0.75
_FFT_RATIO_DEFAULT: float = 0.60
_DR_SCORE_DEFAULT: float = 0.80
_NOISE_SCORE_DEFAULT: float = 0.10
_COMPOSITE_DEFAULT: float = 0.65
_HIGH_MOTION_RISK_PCT: float = 0.40
_LOW_MOTION_RISK_PCT: float = 0.10
_HIGH_EXPOSURE_PCT: float = 0.30
_LOW_KEEP_RATE: float = 0.05
_NORMAL_KEEP_RATE: float = 0.70

# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _make_blur_scores() -> BlurScores:
    """Build default BlurScores for tests."""
    return BlurScores(
        tenengrad=_TENENGRAD_DEFAULT,
        fft_ratio=_FFT_RATIO_DEFAULT,
        blur_tier=_BLUR_TIER_DEFAULT,
    )


def _make_exposure_scores(*, exposure_fail: bool = False) -> ExposureScores:
    """Build ExposureScores; set clipping flags if exposure_fail=True."""
    return ExposureScores(
        dr_score=_DR_SCORE_DEFAULT,
        clipping_highlight=0.01,
        clipping_shadow=0.05,
        midtone_pct=0.60,
        color_cast_score=0.02,
        has_highlight_clip=exposure_fail,
    )


def _make_stage1(path: Path, *, reject_reason: str | None = None, exposure_fail: bool = False) -> Stage1Result:
    """Build Stage1Result with optional reject reason."""
    is_pass = reject_reason is None and not exposure_fail
    return Stage1Result(
        photo_path=path,
        blur=_make_blur_scores(),
        exposure=_make_exposure_scores(exposure_fail=exposure_fail),
        noise_score=_NOISE_SCORE_DEFAULT,
        is_pass=is_pass,
        reject_reason=reject_reason,
    )


def _make_photo_meta(path: Path, *, exif_datetime: datetime | None = None) -> PhotoMeta:
    """Build PhotoMeta for test use."""
    return PhotoMeta(path=path, filename=path.name, exif_datetime=exif_datetime)


def _make_decision(
    path: Path,
    decision: str,
    *,
    reject_reason: str | None = None,
    exposure_fail: bool = False,
    exif_datetime: datetime | None = None,
    portrait_closed: bool = False,
) -> PhotoDecision:
    """Build a PhotoDecision with full stage data."""
    portrait = PortraitScores(
        eye_sharpness_left=0.8,
        eye_sharpness_right=0.7,
        is_eyes_closed=portrait_closed,
    )
    stage2 = Stage2Result(
        photo_path=path,
        topiq=0.70,
        laion_aesthetic=0.65,
        clipiqa=0.60,
        composite=_COMPOSITE_DEFAULT,
        portrait=portrait,
    )
    return PhotoDecision(
        photo=_make_photo_meta(path, exif_datetime=exif_datetime),
        decision=decision,
        stage1=_make_stage1(path, reject_reason=reject_reason, exposure_fail=exposure_fail),
        stage2=stage2,
        stage_reached=2,
    )


def _make_session(decisions: list[PhotoDecision]) -> SessionResult:
    """Build a SessionResult from a list of PhotoDecision objects."""
    return SessionResult(
        source_path="/tmp/test_source",
        model="test-model",
        preset="general",
        is_portrait=False,
        total_photos=len(decisions),
        stages_run=[1, 2],
        summary=SessionSummary(),
        timing=SessionTiming(),
        decisions=decisions,
    )


def _write_session_json(session: SessionResult, source: Path) -> None:
    """Write a SessionResult to <source>/session_report.json."""
    source.mkdir(parents=True, exist_ok=True)
    report_path = source / "session_report.json"
    report_path.write_text(session.model_dump_json())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_keep_rate_calculation(tmp_path: Path) -> None:
    """7 keepers + 3 rejects should produce keep_rate=0.7."""
    decisions = []
    for i in range(7):
        decisions.append(_make_decision(tmp_path / f"k{i}.jpg", "keeper"))
    for i in range(3):
        decisions.append(_make_decision(tmp_path / f"r{i}.jpg", "rejected", reject_reason="blur"))
    session = _make_session(decisions)

    keep_rate, keep_count, reject_count = _compute_keep_rate(session)

    assert keep_count == 7
    assert reject_count == 3
    assert abs(keep_rate - 0.7) < 1e-9


def test_reject_breakdown_sums_to_one(tmp_path: Path) -> None:
    """Mixed reject reasons must produce percentages summing to 1.0."""
    decisions = [
        _make_decision(tmp_path / "blur.jpg", "rejected", reject_reason="blur"),
        _make_decision(tmp_path / "noise.jpg", "rejected", reject_reason="noise"),
        _make_decision(tmp_path / "exp.jpg", "rejected", exposure_fail=True),
        _make_decision(tmp_path / "dup.jpg", "duplicate"),
    ]
    session = _make_session(decisions)

    breakdown = _compute_reject_breakdown(session)

    total = (
        breakdown.blur_pct
        + breakdown.exposure_pct
        + breakdown.noise_pct
        + breakdown.burst_pct
        + breakdown.duplicate_pct
        + breakdown.vlm_pct
    )
    assert abs(total - 1.0) < 1e-9


def test_motion_risk_advice_triggers_above_threshold(tmp_path: Path) -> None:
    """Motion risk pct > 0.30 should trigger shutter speed advice."""
    exif = ExifPatterns(
        keepers_typical=ExifPattern(shutter="1/250s"),
        rejects_typical=ExifPattern(shutter="1/60s"),
        motion_risk_pct=_HIGH_MOTION_RISK_PCT,
    )
    breakdown = RejectBreakdown(exposure_pct=0.05)
    advice_input = _AdviceInput(
        breakdown=breakdown,
        exif=exif,
        keep_rate=_NORMAL_KEEP_RATE,
    )

    advice = _generate_advice(advice_input)

    assert any("1/125s" in line for line in advice)


def test_low_keep_rate_advice(tmp_path: Path) -> None:
    """Keep rate below 0.10 triggers the threshold advice."""
    exif = ExifPatterns(motion_risk_pct=_LOW_MOTION_RISK_PCT)
    breakdown = RejectBreakdown(exposure_pct=0.05)
    advice_input = _AdviceInput(
        breakdown=breakdown,
        exif=exif,
        keep_rate=_LOW_KEEP_RATE,
    )

    advice = _generate_advice(advice_input)

    assert any("unusually low" in line for line in advice)


def test_no_rules_match_returns_default_advice(tmp_path: Path) -> None:
    """When no heuristic fires, default advice is returned."""
    exif = ExifPatterns(motion_risk_pct=_LOW_MOTION_RISK_PCT)
    breakdown = RejectBreakdown(exposure_pct=0.05)
    advice_input = _AdviceInput(
        breakdown=breakdown,
        exif=exif,
        keep_rate=_NORMAL_KEEP_RATE,
    )

    advice = _generate_advice(advice_input)

    assert len(advice) == 1
    assert "No actionable" in advice[0]


def test_portrait_stats_omitted_without_portrait_data(tmp_path: Path) -> None:
    """If no Stage2 portrait data is present, portrait_stats should be None."""
    decisions = []
    for i in range(5):
        path = tmp_path / f"p{i}.jpg"
        stage2 = Stage2Result(
            photo_path=path,
            topiq=0.70,
            laion_aesthetic=0.65,
            clipiqa=0.60,
            composite=_COMPOSITE_DEFAULT,
            portrait=None,
        )
        photo_dec = PhotoDecision(
            photo=_make_photo_meta(path),
            decision="keeper",
            stage1=_make_stage1(path),
            stage2=stage2,
        )
        decisions.append(photo_dec)
    session = _make_session(decisions)

    stats = _compute_portrait_stats(session)

    assert stats is None


def test_session_timing_uses_exif_datetime(tmp_path: Path) -> None:
    """Timing computation uses exif_datetime when available."""
    t1 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 1, 10, 5, 0, tzinfo=timezone.utc)
    decisions = [
        _make_decision(tmp_path / "a.jpg", "keeper", exif_datetime=t1),
        _make_decision(tmp_path / "b.jpg", "keeper", exif_datetime=t2),
    ]
    session = _make_session(decisions)

    timing = _compute_timing(session)

    assert timing.first_capture == t1
    assert timing.last_capture == t2
    assert abs(timing.duration_seconds - 300.0) < 1e-6


def test_load_session_missing_file_raises(tmp_path: Path) -> None:
    """_load_session raises FileNotFoundError with a clear message if file absent."""
    missing_source = tmp_path / "no_session_here"
    missing_source.mkdir()

    with pytest.raises(FileNotFoundError, match="session_report.json"):
        _load_session(missing_source)


def test_build_report_card_end_to_end(tmp_path: Path) -> None:
    """build_report_card returns a populated ReportCard for a synthetic session."""
    source = tmp_path / "session_source"
    decisions = []
    for i in range(7):
        decisions.append(_make_decision(source / f"k{i}.jpg", "keeper"))
    for i in range(3):
        decisions.append(_make_decision(source / f"r{i}.jpg", "rejected", reject_reason="blur"))
    session = _make_session(decisions)
    _write_session_json(session, source)

    card = build_report_card(source)

    assert card.keep_count == 7
    assert card.reject_count == 3
    assert abs(card.keep_rate - 0.7) < 1e-9
    assert card.source_path == str(source)
    assert len(card.advice) >= 1
    assert card.breakdown is not None
    assert card.timing is not None


def test_sample_exif_caps_at_count(tmp_path: Path) -> None:
    """_sample_exif reads at most 'count' photos even with a larger input list."""
    decisions = [
        _make_decision(tmp_path / f"img{i}.jpg", "keeper")
        for i in range(20)
    ]
    sample_input = _ExifSampleInput(decisions=decisions, count=5)

    results = _sample_exif(sample_input)

    assert len(results) <= 5
