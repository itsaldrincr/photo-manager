"""Tests for src/cull/router.py — file routing and move execution."""

from __future__ import annotations

from pathlib import Path

from cull.config import CullConfig
from cull.models import (
    BlurScores,
    DecisionLabel,
    ExposureScores,
    PhotoDecision,
    PhotoMeta,
    Stage1Result,
)
from cull.router import execute_moves, process_single_move


def _make_decision(source: Path, decision: DecisionLabel = "uncertain") -> PhotoDecision:
    """Build a synthetic PhotoDecision pointing at source."""
    meta = PhotoMeta(path=source, filename=source.name)
    blur = BlurScores(tenengrad=0.8, fft_ratio=0.9, blur_tier=1)
    exposure = ExposureScores(
        dr_score=0.7,
        clipping_highlight=0.01,
        clipping_shadow=0.01,
        midtone_pct=0.5,
        color_cast_score=0.1,
    )
    stage1 = Stage1Result(photo_path=source, blur=blur, exposure=exposure, noise_score=0.1)
    return PhotoDecision(photo=meta, decision=decision, stage1=stage1)


def test_process_single_move_moves_uncertain_photo(tmp_path: Path) -> None:
    """A non-keeper photo is moved into its _review subdir."""
    source = tmp_path / "IMG_0001.jpg"
    source.write_bytes(b"fake-jpeg")
    decision = _make_decision(source, "uncertain")
    config = CullConfig(is_sidecars=False)

    entry = process_single_move(decision, config)

    assert entry is not None
    assert entry.is_success
    assert not source.exists()
    assert entry.destination.exists()


# ---------------------------------------------------------------------------
# B6 regression: sidecar move-report folding, not silently dropped.
# ---------------------------------------------------------------------------


def test_execute_moves_folds_sidecar_entry_into_report(tmp_path: Path) -> None:
    """A sidecar written alongside a moved photo is folded into the MoveReport, not dropped."""
    source = tmp_path / "IMG_0002.jpg"
    source.write_bytes(b"fake-jpeg")
    decision = _make_decision(source, "select")
    config = CullConfig(is_sidecars=True)

    report = execute_moves([decision], config)

    assert report.moved == 2  # photo + sidecar
    assert report.errors == 0
    assert len(report.entries) == 2
    destinations = {entry.destination.name for entry in report.entries}
    assert destinations == {"IMG_0002.jpg", "IMG_0002.xmp"}


def test_execute_moves_skips_sidecar_fold_when_no_sidecar_written(tmp_path: Path) -> None:
    """A moved photo with no sidecar produces exactly one report entry."""
    source = tmp_path / "IMG_0003.jpg"
    source.write_bytes(b"fake-jpeg")
    decision = _make_decision(source, "rejected")
    config = CullConfig(is_sidecars=True)

    report = execute_moves([decision], config)

    assert report.moved == 1
    assert len(report.entries) == 1
