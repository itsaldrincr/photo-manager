"""Moment stacks after Stage 2: representative choice, curation pool, report compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cull.cli_review import _load_session_from_report
from cull.config import CullConfig
from cull.models import BlurScores, BurstInfo, ExposureScores, Stage1Result, Stage2Result
from cull.pipeline import (
    SessionResult,
    _build_all_decisions,
    _DecisionCtx,
    _S4RunInput,
    _Stage1Output,
    _Stage2Output,
    _StageRunCtx,
    _StagesResult,
)
from cull._pipeline.stack_resolution import resolve_moment_stacks
from cull._pipeline.stage4_curator import _collect_candidate_paths
from cull.stage2.fusion import FusionResult
from cull.tui.app import _filter_queue, _find_burst_decisions

SHARP_DULL = "sharp_dull.jpg"
SOFT_BEST = "soft_best.jpg"
MIDDLE = "middle.jpg"
LONE = "lone.jpg"
TENENGRAD = {SHARP_DULL: 900.0, SOFT_BEST: 100.0, MIDDLE: 400.0, LONE: 300.0}
COMPOSITE = {SHARP_DULL: 0.40, SOFT_BEST: 0.90, MIDDLE: 0.60, LONE: 0.70}


def _s1(path: Path) -> Stage1Result:
    """Return a passing Stage1Result with the photo's Tenengrad score."""
    return Stage1Result(
        photo_path=path,
        blur=BlurScores(tenengrad=TENENGRAD[path.name], fft_ratio=0.5, blur_tier=1),
        exposure=ExposureScores(
            dr_score=0.5, clipping_highlight=0.0, clipping_shadow=0.0,
            midtone_pct=0.5, color_cast_score=0.0,
        ),
        noise_score=0.0,
    )


def _fusion(path: Path) -> FusionResult:
    """Return an AMBIGUOUS FusionResult carrying the photo's composite."""
    score = COMPOSITE[path.name]
    stage2 = Stage2Result(photo_path=path, topiq=score, laion_aesthetic=score, clipiqa=score, composite=score)
    return FusionResult(stage2=stage2, routing="AMBIGUOUS")


def _outputs(tmp_path: Path) -> tuple[list[Path], _Stage1Output, _Stage2Output]:
    """Build Stage 1/2 outputs for one three-photo stack plus a singleton."""
    paths = [tmp_path / name for name in (SHARP_DULL, SOFT_BEST, MIDDLE, LONE)]
    for path in paths:
        path.write_bytes(b"")
    stack = sorted(str(p) for p in paths[:3])
    s1_out = _Stage1Output(results={str(p): _s1(p) for p in paths}, survivors=paths, stacks=[stack])
    for member in stack:
        s1_out.results[member].burst = BurstInfo(group_id=0, rank=0, group_size=3)
    s2_out = _Stage2Output(results={str(p): _fusion(p) for p in paths}, ambiguous=list(paths))
    return paths, s1_out, s2_out


def test_representative_is_highest_composite_not_sharpest(tmp_path: Path) -> None:
    """The stack keeps the best Stage 2 composite even though it is the softest frame."""
    _, s1_out, s2_out = _outputs(tmp_path)
    resolve_moment_stacks(s1_out, s2_out)
    assert s1_out.duplicate_paths == {str(tmp_path / SHARP_DULL), str(tmp_path / MIDDLE)}
    assert s1_out.results[str(tmp_path / SOFT_BEST)].burst.is_burst_winner is True
    assert s1_out.results[str(tmp_path / SHARP_DULL)].burst.rank == 2
    assert s1_out.results[str(tmp_path / SHARP_DULL)].is_duplicate is True


def test_composite_tie_breaks_on_sharpness(tmp_path: Path) -> None:
    """Equal composites fall back to Stage 1 Tenengrad."""
    _, s1_out, s2_out = _outputs(tmp_path)
    for fusion in s2_out.results.values():
        fusion.stage2.composite = 0.5
    resolve_moment_stacks(s1_out, s2_out)
    assert str(tmp_path / SHARP_DULL) not in s1_out.duplicate_paths


def test_non_representatives_skip_stage3_but_keep_stage2(tmp_path: Path) -> None:
    """Non-representatives leave the Stage 3 queue, are labelled duplicate, and keep Stage 2 scores."""
    paths, s1_out, s2_out = _outputs(tmp_path)
    resolve_moment_stacks(s1_out, s2_out)
    assert sorted(s2_out.ambiguous) == sorted([tmp_path / SOFT_BEST, tmp_path / LONE])
    decisions = _build_all_decisions(_DecisionCtx(paths=paths, s1_out=s1_out, s2_out=s2_out))
    by_name = {d.photo.filename: d for d in decisions}
    assert by_name[SHARP_DULL].decision == "duplicate"
    assert by_name[SHARP_DULL].stage2 is not None
    assert by_name[SOFT_BEST].decision == "uncertain"


def test_curator_candidates_hold_one_photo_per_stack(tmp_path: Path) -> None:
    """Curation sees the representative and the singleton, never two stack members."""
    paths, s1_out, s2_out = _outputs(tmp_path)
    resolve_moment_stacks(s1_out, s2_out)
    decisions = _build_all_decisions(_DecisionCtx(paths=paths, s1_out=s1_out, s2_out=s2_out))
    for decision in decisions:
        decision.decision = "keeper"
    ctx = _StageRunCtx(config=CullConfig(), paths=paths, dashboard=MagicMock())
    s4_in = _S4RunInput(stages=_StagesResult(s1_out=s1_out, s2_out=s2_out), decisions=decisions, ctx=ctx)
    assert sorted(_collect_candidate_paths(s4_in)) == sorted([tmp_path / SOFT_BEST, tmp_path / LONE])


def _session(tmp_path: Path) -> SessionResult:
    """Build a SessionResult after stack resolution."""
    paths, s1_out, s2_out = _outputs(tmp_path)
    resolve_moment_stacks(s1_out, s2_out)
    decisions = _build_all_decisions(_DecisionCtx(paths=paths, s1_out=s1_out, s2_out=s2_out))
    return SessionResult(source_path=str(tmp_path), decisions=decisions)


def test_report_round_trips_and_tui_sees_stack(tmp_path: Path) -> None:
    """A new report loads, and the TUI queues and burst view show every stack member."""
    report = tmp_path / "session_report.json"
    report.write_text(_session(tmp_path).model_dump_json(), encoding="utf-8")
    loaded = _load_session_from_report(report)
    assert len(_filter_queue(loaded.decisions, "duplicate")) == 2
    assert len(_find_burst_decisions(loaded, 0)) == 3


def test_old_report_without_stack_fields_still_loads(tmp_path: Path) -> None:
    """A report written before moment stacks (no burst info, legacy labels) parses."""
    data = json.loads(_session(tmp_path).model_dump_json())
    for decision in data["decisions"]:
        decision["stage1"].pop("burst")
        decision["stage1"].pop("is_duplicate")
    data["decisions"][0]["decision"] = "rejected"
    report = tmp_path / "session_report.json"
    report.write_text(json.dumps(data), encoding="utf-8")
    loaded = _load_session_from_report(report)
    assert loaded.decisions[0].stage1.burst is None
    assert _find_burst_decisions(loaded, 0) == []


@pytest.mark.parametrize("has_stage2", [False])
def test_without_stage2_sharpest_member_represents(tmp_path: Path, has_stage2: bool) -> None:
    """When Stage 2 did not run, Tenengrad alone picks the representative."""
    _, s1_out, _ = _outputs(tmp_path)
    resolve_moment_stacks(s1_out, None)
    assert str(tmp_path / SHARP_DULL) not in s1_out.duplicate_paths
