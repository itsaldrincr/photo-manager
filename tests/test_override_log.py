"""Tests for cull.override_log — JSONL override log read/write/build."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import cull.config as cfg
from cull.models import (
    BlurScores,
    BurstInfo,
    ExposureScores,
    PhotoDecision,
    PhotoMeta,
    Stage1Result,
    Stage2Result,
    Stage3Result,
    OverrideEntry,
)
from cull.override_log import build_override_entry, load_overrides, log_override, OverrideContext

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BLUR_TIER_DEFAULT: int = 1
_TENENGRAD_DEFAULT: float = 0.75
_FFT_RATIO_DEFAULT: float = 0.60
_DR_SCORE_DEFAULT: float = 0.80
_COMPOSITE_DEFAULT: float = 0.65
_NOISE_SCORE_DEFAULT: float = 0.10
_THREAD_COUNT: int = 2
_ENTRIES_PER_THREAD: int = 10
_TOTAL_ENTRIES: int = _THREAD_COUNT * _ENTRIES_PER_THREAD

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_photo_meta(path: Path) -> PhotoMeta:
    """Build a minimal PhotoMeta for test use."""
    return PhotoMeta(path=path, filename=path.name)


def _make_blur_scores() -> BlurScores:
    """Build a BlurScores instance with default test values."""
    return BlurScores(
        tenengrad=_TENENGRAD_DEFAULT,
        fft_ratio=_FFT_RATIO_DEFAULT,
        blur_tier=_BLUR_TIER_DEFAULT,
    )


def _make_exposure_scores() -> ExposureScores:
    """Build an ExposureScores instance with default test values."""
    return ExposureScores(
        dr_score=_DR_SCORE_DEFAULT,
        clipping_highlight=0.01,
        clipping_shadow=0.05,
        midtone_pct=0.60,
        color_cast_score=0.02,
    )


def _make_stage1(path: Path) -> Stage1Result:
    """Build a Stage1Result with blur + exposure + noise."""
    return Stage1Result(
        photo_path=path,
        blur=_make_blur_scores(),
        exposure=_make_exposure_scores(),
        noise_score=_NOISE_SCORE_DEFAULT,
    )


def _make_stage2(path: Path) -> Stage2Result:
    """Build a Stage2Result with default composite score."""
    return Stage2Result(
        photo_path=path,
        topiq=0.70,
        laion_aesthetic=0.65,
        clipiqa=0.60,
        composite=_COMPOSITE_DEFAULT,
    )


def _make_stage3(path: Path) -> Stage3Result:
    """Build a Stage3Result with basic keeper signal."""
    return Stage3Result(
        photo_path=path,
        sharpness=0.80,
        exposure=0.75,
        composition=0.70,
        is_keeper=True,
        confidence=0.85,
        model_used="test-model",
    )


def _make_decision(tmp_path: Path) -> PhotoDecision:
    """Build a synthetic PhotoDecision with Stage 1, 2, and 3 data."""
    photo_path = tmp_path / "test.jpg"
    return PhotoDecision(
        photo=_make_photo_meta(photo_path),
        decision="keeper",
        stage1=_make_stage1(photo_path),
        stage2=_make_stage2(photo_path),
        stage3=_make_stage3(photo_path),
        stage_reached=3,
    )


def _make_entry(tmp_path: Path) -> OverrideEntry:
    """Build a synthetic OverrideEntry for test use."""
    decision = _make_decision(tmp_path)
    ctx = OverrideContext(new_label="rejected", session_source="/session/path", origin="tui")
    return build_override_entry(decision, ctx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_log_creates_parent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """log_override creates the parent directory if it does not exist."""
    log_dir = tmp_path / ".cull"
    log_path = log_dir / "overrides.jsonl"
    monkeypatch.setattr(cfg, "OVERRIDE_LOG_DIR", log_dir)
    monkeypatch.setattr(cfg, "OVERRIDE_LOG_PATH", log_path)

    import cull.override_log as ol
    monkeypatch.setattr(ol, "OVERRIDE_LOG_DIR", log_dir)
    monkeypatch.setattr(ol, "OVERRIDE_LOG_PATH", log_path)

    assert not log_dir.exists()
    log_override(_make_entry(tmp_path))
    assert log_dir.exists()
    assert log_path.exists()


def test_log_appends_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """log_override appends entries in order; each line is valid JSON."""
    log_dir = tmp_path / ".cull"
    log_path = log_dir / "overrides.jsonl"
    import cull.override_log as ol
    monkeypatch.setattr(ol, "OVERRIDE_LOG_DIR", log_dir)
    monkeypatch.setattr(ol, "OVERRIDE_LOG_PATH", log_path)

    entries = [_make_entry(tmp_path) for _ in range(3)]
    for entry in entries:
        log_override(entry)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        parsed = json.loads(line)
        assert "photo_path" in parsed
        assert "user_decision" in parsed


def test_load_skips_corrupted_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """load_overrides skips malformed lines and logs a warning."""
    log_dir = tmp_path / ".cull"
    log_path = log_dir / "overrides.jsonl"
    import cull.override_log as ol
    monkeypatch.setattr(ol, "OVERRIDE_LOG_PATH", log_path)
    monkeypatch.setattr(ol, "OVERRIDE_LOG_DIR", log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)
    entry = _make_entry(tmp_path)
    good_line = json.dumps(entry.model_dump(mode="json")) + "\n"
    bad_line = "not valid json {{{\n"

    log_path.write_text(good_line + bad_line + good_line, encoding="utf-8")

    with caplog.at_level("WARNING"):
        results = load_overrides()

    assert len(results) == 2
    assert any("malformed" in r.message.lower() for r in caplog.records)


def test_load_empty_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_overrides returns [] when log file is missing."""
    log_path = tmp_path / ".cull" / "overrides.jsonl"
    import cull.override_log as ol
    monkeypatch.setattr(ol, "OVERRIDE_LOG_PATH", log_path)
    monkeypatch.setattr(ol, "OVERRIDE_LOG_DIR", tmp_path / ".cull")

    assert not log_path.exists()
    result = load_overrides()
    assert result == []


def test_build_override_entry_flattens_stages(tmp_path: Path) -> None:
    """build_override_entry flattens Stage1 blur/exposure/noise into stage1_scores."""
    decision = _make_decision(tmp_path)
    ctx = OverrideContext(new_label="rejected", session_source="/session/path", origin="tui")
    entry = build_override_entry(decision, ctx)

    scores = entry.stage1_scores
    assert "tenengrad" in scores
    assert "fft_ratio" in scores
    assert "dr_score" in scores
    assert "noise_score" in scores
    assert abs(scores["tenengrad"] - _TENENGRAD_DEFAULT) < 1e-6
    assert abs(scores["fft_ratio"] - _FFT_RATIO_DEFAULT) < 1e-6
    assert abs(scores["dr_score"] - _DR_SCORE_DEFAULT) < 1e-6
    assert abs(scores["noise_score"] - _NOISE_SCORE_DEFAULT) < 1e-6
    assert abs(entry.stage2_composite - _COMPOSITE_DEFAULT) < 1e-6
    assert entry.stage3_result is not None
    assert entry.stage3_result["is_keeper"] is True


def test_concurrent_appends_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent threads write entries without data loss (flock serializes appends)."""
    log_dir = tmp_path / ".cull"
    log_path = log_dir / "overrides.jsonl"
    import cull.override_log as ol
    monkeypatch.setattr(ol, "OVERRIDE_LOG_DIR", log_dir)
    monkeypatch.setattr(ol, "OVERRIDE_LOG_PATH", log_path)

    log_dir.mkdir(parents=True, exist_ok=True)

    def _write_entries() -> None:
        for _ in range(_ENTRIES_PER_THREAD):
            log_override(_make_entry(tmp_path))

    threads = [threading.Thread(target=_write_entries) for _ in range(_THREAD_COUNT)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == _TOTAL_ENTRIES
    for line in lines:
        parsed = json.loads(line)
        assert "photo_path" in parsed
