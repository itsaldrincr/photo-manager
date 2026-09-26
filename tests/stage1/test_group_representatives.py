"""Stage 1 duplicate + burst resolution: every group keeps its sharpest member."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from cull._pipeline import stage1_runner
from cull._pipeline.stage1_runner import (
    _preflight_dupes_into_output,
    _Stage1LoopInput,
    _Stage1Output,
    _Stage1WorkCtx,
)
from cull.config import CullConfig
from cull.models import BlurScores, ExposureScores, Stage1Result
from cull.stage1.burst import BurstResult
from cull.stage1.duplicate import DuplicateGroup, DuplicateResult

PHOTO_A = Path("/shoot/A.jpg")
PHOTO_B = Path("/shoot/B.jpg")
PHOTO_C = Path("/shoot/C.jpg")
PHOTO_LONE = Path("/shoot/Z.jpg")

TENENGRAD: dict[Path, float] = {PHOTO_A: 100.0, PHOTO_B: 200.0, PHOTO_C: 900.0, PHOTO_LONE: 50.0}


def _s1_result(path: Path) -> Stage1Result:
    """Return a passing Stage1Result carrying the path's Tenengrad score."""
    return Stage1Result(
        photo_path=path,
        blur=BlurScores(tenengrad=TENENGRAD[path], fft_ratio=0.5, blur_tier=1),
        exposure=ExposureScores(
            dr_score=0.5, clipping_highlight=0.0, clipping_shadow=0.0,
            midtone_pct=0.5, color_cast_score=0.0,
        ),
        noise_score=0.0,
    )


def _output_after_loop(paths: list[Path]) -> _Stage1Output:
    """Return a Stage 1 output as the per-image loop would leave it: all pass."""
    output = _Stage1Output()
    output.results = {str(p): _s1_result(p) for p in paths}
    output.survivors = list(paths)
    return output


def _resolve(output: _Stage1Output, paths: list[Path]) -> list[Path]:
    """Run preflight-merge and group resolution; return the final survivors."""
    loop_in = _Stage1LoopInput(paths=paths, config=CullConfig())
    loop_out = _output_after_loop(paths)
    output.results.update(loop_out.results)
    output.survivors.extend(loop_out.survivors)
    stage1_runner._resolve_groups(loop_in, output)
    return output.survivors


def _preflight(dup_groups: list[list[Path]], paths: list[Path]) -> _Stage1Output:
    """Run the duplicate preflight with find_duplicates mocked to dup_groups."""
    output = _Stage1Output()
    loop_in = _Stage1LoopInput(paths=paths, config=CullConfig())
    dup_result = DuplicateResult(duplicate_groups=[DuplicateGroup(paths=g) for g in dup_groups])
    with patch.object(stage1_runner, "find_duplicates", return_value=dup_result):
        _preflight_dupes_into_output(_Stage1WorkCtx(loop_in=loop_in, output=output, dashboard=MagicMock()))
    return output


def _burst(groups: list[list[Path]]) -> BurstResult:
    """Return a BurstResult whose winners are each group's sharpest member."""
    winners = [max(g, key=TENENGRAD.__getitem__) for g in groups]
    losers = [p for g in groups for p in g if p not in winners]
    return BurstResult(groups=groups, winners=winners, losers=losers)


def test_dup_and_burst_group_keeps_only_the_sharpest() -> None:
    """A,B,C near-identical burst with C sharpest leaves exactly C."""
    paths = [PHOTO_A, PHOTO_B, PHOTO_C, PHOTO_LONE]
    output = _preflight([[PHOTO_A, PHOTO_B, PHOTO_C]], paths)
    with patch.object(stage1_runner, "detect_bursts", return_value=_burst([[PHOTO_A, PHOTO_B, PHOTO_C]])):
        survivors = _resolve(output, paths)
    assert sorted(survivors) == [PHOTO_C, PHOTO_LONE]
    assert output.results[str(PHOTO_C)].is_duplicate is False
    assert output.results[str(PHOTO_A)].is_duplicate is True


def test_dup_group_without_burst_keeps_sharpest_not_first_name() -> None:
    """A duplicate group with no burst keeps the highest-Tenengrad member."""
    paths = [PHOTO_A, PHOTO_B, PHOTO_C]
    output = _preflight([[PHOTO_A, PHOTO_B, PHOTO_C]], paths)
    with patch.object(stage1_runner, "detect_bursts", return_value=_burst([])):
        survivors = _resolve(output, paths)
    assert survivors == [PHOTO_C]


def test_dup_and_burst_chained_groups_keep_exactly_one() -> None:
    """Dup {A,B} and burst {B,C} connect through B, so only C survives."""
    paths = [PHOTO_A, PHOTO_B, PHOTO_C]
    output = _preflight([[PHOTO_A, PHOTO_B]], paths)
    with patch.object(stage1_runner, "detect_bursts", return_value=_burst([[PHOTO_B, PHOTO_C]])):
        survivors = _resolve(output, paths)
    assert survivors == [PHOTO_C]
    assert str(PHOTO_A) in output.duplicate_paths
    assert str(PHOTO_B) in output.duplicate_paths
