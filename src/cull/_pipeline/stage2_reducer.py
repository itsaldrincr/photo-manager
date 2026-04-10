"""Stage 2 reducer — cross-photo shoot-stats pass and fusion repatching."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from cull.dashboard import Dashboard, _Stage2ReducerUpdateInput
from cull.stage2 import shoot_stats as shoot_stats_module
from cull.stage2.fusion import ReducerPatchInput, patch_reducer_scores
from cull.stage2.shoot_stats import ShootStatsInput

from cull._pipeline.stage1_runner import _Stage1Output

if TYPE_CHECKING:
    from cull.pipeline import _Stage2Output, _StageRunCtx


class _Stage2ReducerOutput(BaseModel):
    """Aggregated outputs of the Stage 2 reducer pass."""

    reducer_scores: dict[str, Any] = Field(default_factory=dict)
    patched_count: int = 0


class _S2ReducerRunInput(BaseModel):
    """Input bundle for Stage 2 reducer execution."""

    model_config = {"arbitrary_types_allowed": True}

    s2_out: "_Stage2Output"
    s1_out: _Stage1Output
    ctx: "_StageRunCtx"


REDUCER_OUTLIER_THRESHOLD: float = 0.5
REDUCER_SCENE_BONUS_THRESHOLD: float = 0.5


def _build_reducer_input(run_in: _S2ReducerRunInput) -> ShootStatsInput:
    """Assemble a ShootStatsInput from the current Stage 2 + Stage 1 results."""
    stage2_results = [
        fusion.stage2 for fusion in run_in.s2_out.results.values()
    ]
    stage1_results = [
        run_in.s1_out.results[str(fusion.stage2.photo_path)]
        for fusion in run_in.s2_out.results.values()
        if str(fusion.stage2.photo_path) in run_in.s1_out.results
    ]
    return ShootStatsInput(
        stage2_results=stage2_results, stage1_results=stage1_results,
    )


def _emit_reducer_updates(
    reducer_scores: dict[str, Any], dashboard: Dashboard
) -> None:
    """Push one dashboard update per scored photo with the flagged outliers."""
    for score in reducer_scores.values():
        update = _Stage2ReducerUpdateInput(
            is_palette_outlier=score.palette_outlier_score >= REDUCER_OUTLIER_THRESHOLD,
            is_exposure_outlier=score.exposure_drift_score >= REDUCER_OUTLIER_THRESHOLD,
            is_exif_outlier=score.exif_anomaly_score >= REDUCER_OUTLIER_THRESHOLD,
            is_scene_start=score.scene_start_bonus >= REDUCER_SCENE_BONUS_THRESHOLD,
        )
        dashboard.update_stage2_reducer(update)


def _run_s2_reducer(run_in: _S2ReducerRunInput) -> _Stage2ReducerOutput:
    """Execute the cross-photo Stage 2 reducer pass and patch fusion in place."""
    if not run_in.s2_out.results:
        return _Stage2ReducerOutput()
    t0 = time.monotonic()
    run_in.ctx.dashboard.start_stage2_reducer(len(run_in.s2_out.results))
    reducer_scores = shoot_stats_module.compute(_build_reducer_input(run_in))
    patch_in = ReducerPatchInput(
        fusion_results=run_in.s2_out.results,
        reducer_scores=reducer_scores,
        config=run_in.ctx.config,
    )
    patch_reducer_scores(patch_in)
    _reroute_after_patch(run_in.s2_out)
    _emit_reducer_updates(reducer_scores, run_in.ctx.dashboard)
    elapsed = time.monotonic() - t0
    run_in.ctx.dashboard.complete_stage2_reducer(elapsed)
    return _Stage2ReducerOutput(reducer_scores=reducer_scores, patched_count=len(reducer_scores))


def _reroute_after_patch(s2_out: "_Stage2Output") -> None:
    """Rebuild keepers / ambiguous / rejects from the (now patched) fusion routings."""
    from cull._pipeline.stage2_runner import _classify_s2_routing  # noqa: PLC0415

    s2_out.keepers = []
    s2_out.ambiguous = []
    s2_out.rejects = []
    for fusion in s2_out.results.values():
        _classify_s2_routing(fusion, s2_out)
