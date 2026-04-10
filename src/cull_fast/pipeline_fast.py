"""Fast Stage 2 substitution.

Mirrors `cull.pipeline._run_s2` shape exactly so it is a drop-in replacement
at the wrapper level. Reuses `_load_tensor_batch`, `_apply_exposure_to_scores`,
`_classify_s2_routing`, `_run_portrait_if_needed` from `cull.pipeline` per
spec §5B.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from cull.config import STAGE2_BATCH_SIZE
from cull.dashboard import _Stage2UpdateInput
from cull.pipeline import (
    _S2RunInput,
    _Stage2LoopInput,
    _Stage2Output,
    _Stage2ScoreInput,
    _apply_exposure_to_scores,
    _classify_s2_routing,
    _load_tensor_batch,
    _run_portrait_if_needed,
    _unload_stage2_models,
)
from cull.stage2.fusion import FusionResult, IqaScores
from cull.stage2.iqa import select_device
from cull._pipeline.stage2_scoring import (
    _apply_geometry_to_scores,
    _apply_stage1_blur_context_to_scores,
)
from cull_fast.fusion_fast import (
    _FastComputeInput,
    _FusionFastInput,
    _compute_composite_fast,
    _rescale_preset_weights,
    build_iqa_from_musiq,
)
from cull_fast.musiq import (
    MusiQScorePair,
    _MusiQBatchRequest,
    score_musiq_batch,
    unload_musiq,
)

logger = logging.getLogger(__name__)

IQA_EXPOSURE_DEFAULT: float = 0.0


class _FastBatchInput(BaseModel):
    """Input bundle for one fast Stage 2 batch."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_paths: list[Path]
    loop_in: _Stage2LoopInput
    device: str


class _FastBatchOutcome(BaseModel):
    """Outputs of one fast Stage 2 batch: (path, fusion) pairs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pairs: list[tuple[Path, FusionResult]]


class _FastEmitInput(BaseModel):
    """Input bundle for emitting one fast batch outcome to output and dashboard."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    outcome: _FastBatchOutcome
    loop_in: _Stage2LoopInput
    output: _Stage2Output
    device: str


class _FastRunState(BaseModel):
    """Mutable state carried through the fast batch loop."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    loop_in: _Stage2LoopInput
    output: _Stage2Output
    device: str


def _build_iqa_list(
    musiq_pairs: list[MusiQScorePair],
) -> list[IqaScores]:
    """Convert MUSIQ score pairs into IqaScores list with default exposure."""
    return [
        build_iqa_from_musiq(
            _FusionFastInput(musiq_scores=pair, exposure_score=IQA_EXPOSURE_DEFAULT)
        )
        for pair in musiq_pairs
    ]


def _score_musiq_for_chunk(
    batch_in: _FastBatchInput,
) -> list[MusiQScorePair]:
    """Load tensors for chunk and score them through MUSIQ batch."""
    tensor_batch, _pil = _load_tensor_batch(batch_in.chunk_paths)
    request = _MusiQBatchRequest(
        tensor_batch=tensor_batch,
        photo_paths=batch_in.chunk_paths,
        device=batch_in.device,
    )
    return score_musiq_batch(request)


def _fuse_iqa_pairs(
    iqa_list: list[IqaScores], batch_in: _FastBatchInput
) -> list[tuple[Path, FusionResult]]:
    """Fuse IqaScores list into (path, FusionResult) pairs using preset weights."""
    preset = batch_in.loop_in.config.preset
    weights = _rescale_preset_weights(preset)
    return [
        (
            batch_in.chunk_paths[idx],
            _compute_composite_fast(
                _FastComputeInput(scores=iqa, weights=weights, preset=preset)
            ),
        )
        for idx, iqa in enumerate(iqa_list)
    ]


def _score_one_fast_batch(batch_in: _FastBatchInput) -> _FastBatchOutcome:
    """Score one chunk of paths through MUSIQ + fast fusion; returns outcome."""
    musiq_pairs = _score_musiq_for_chunk(batch_in)
    if len(musiq_pairs) != len(batch_in.chunk_paths):
        raise RuntimeError(
            f"MUSIQ scorer returned {len(musiq_pairs)} results for "
            f"{len(batch_in.chunk_paths)} input paths"
        )
    iqa_list = _build_iqa_list(musiq_pairs)
    if len(iqa_list) != len(batch_in.chunk_paths):
        raise RuntimeError(
            f"IqaScores builder returned {len(iqa_list)} results for "
            f"{len(batch_in.chunk_paths)} input paths"
        )
    _apply_exposure_to_scores(iqa_list, batch_in.loop_in.s1_results)
    _apply_stage1_blur_context_to_scores(iqa_list, batch_in.loop_in.s1_results)
    _apply_geometry_to_scores(iqa_list, batch_in.loop_in.s1_results)
    pairs = _fuse_iqa_pairs(iqa_list, batch_in)
    return _FastBatchOutcome(pairs=pairs)


def _build_score_input(
    path: Path, state: _FastRunState
) -> _Stage2ScoreInput:
    """Build a _Stage2ScoreInput for one path from current run state."""
    return _Stage2ScoreInput(
        path=path,
        tensor=None,
        config=state.loop_in.config,
        device=state.device,
        s1_result=state.loop_in.s1_results.get(str(path)),
    )


def _emit_one_fast_pair(
    pair: tuple[Path, FusionResult], state: _FastRunState
) -> None:
    """Route one (path, fusion) pair into output and update dashboard."""
    path, fusion = pair
    state.output.results[str(path)] = fusion
    _classify_s2_routing(fusion, state.output)
    _run_portrait_if_needed(_build_score_input(path, state), state.output)


def _emit_fast_outcome(
    emit_in: _FastEmitInput, dashboard: object
) -> None:
    """Emit every pair in a fast outcome to output and dashboard."""
    state = _FastRunState(
        loop_in=emit_in.loop_in, output=emit_in.output, device=emit_in.device
    )
    for pair in emit_in.outcome.pairs:
        _emit_one_fast_pair(pair, state)
        dashboard.update_stage2(  # type: ignore[attr-defined]
            _Stage2UpdateInput(path=pair[0], fusion=pair[1], routing=pair[1].routing)
        )


def _init_fast_output() -> _Stage2Output:
    """Build an empty _Stage2Output bundle."""
    return _Stage2Output(
        results={}, portraits={}, ambiguous=[], keepers=[], rejects=[]
    )


def _build_fast_loop_input(run_in: _S2RunInput) -> _Stage2LoopInput:
    """Build the Stage 2 loop input bundle from the run input."""
    return _Stage2LoopInput(
        survivors=run_in.s1_out.survivors,
        config=run_in.ctx.config,
        s1_results=run_in.s1_out.results,
        source_path=run_in.ctx.source_path,
    )


class _FastLoopCtx(BaseModel):
    """Bundle carried through the fast batch iteration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    loop_in: _Stage2LoopInput
    output: _Stage2Output
    device: str
    dashboard: object


def _process_fast_chunk(
    chunk_paths: list[Path], ctx: _FastLoopCtx
) -> None:
    """Score one chunk and emit its outcome to output and dashboard."""
    batch_in = _FastBatchInput(
        chunk_paths=chunk_paths, loop_in=ctx.loop_in, device=ctx.device
    )
    outcome = _score_one_fast_batch(batch_in)
    emit_in = _FastEmitInput(
        outcome=outcome,
        loop_in=ctx.loop_in,
        output=ctx.output,
        device=ctx.device,
    )
    _emit_fast_outcome(emit_in, ctx.dashboard)


def _iterate_fast_batches(ctx: _FastLoopCtx) -> None:
    """Iterate survivors in STAGE2_BATCH_SIZE chunks and process each."""
    survivors = ctx.loop_in.survivors
    for i in range(0, len(survivors), STAGE2_BATCH_SIZE):
        chunk = survivors[i: i + STAGE2_BATCH_SIZE]
        _process_fast_chunk(chunk, ctx)


def _run_s2_fast(run_in: _S2RunInput) -> _Stage2Output:
    """Execute fast Stage 2 (MUSIQ-only) and record timing."""
    t0 = time.monotonic()
    loop_in = _build_fast_loop_input(run_in)
    output = _init_fast_output()
    dashboard = run_in.ctx.dashboard
    dashboard.start_stage2(len(loop_in.survivors))
    ctx = _FastLoopCtx(
        loop_in=loop_in, output=output,
        device=select_device(), dashboard=dashboard,
    )
    _iterate_fast_batches(ctx)
    run_in.ctx.timings.stage2 = time.monotonic() - t0
    dashboard.complete_stage2(run_in.ctx.timings.stage2)
    return output


def _unload_stage2_models_fast() -> None:
    """Free fast-mode Stage 2 models (MUSIQ) and any legacy stage2 leftovers."""
    unload_musiq()
    _unload_stage2_models()
