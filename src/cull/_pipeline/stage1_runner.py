"""Stage 1 runner — per-photo assessment, preflight duplicates, burst detection."""

from __future__ import annotations

import logging
import time
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cull.config import CullConfig, STAGE1_WORKER_COUNT
from cull.dashboard import Dashboard
from cull.models import ExposureScores, Stage1Result
from cull.stage1 import duplicate as duplicate_module
from cull.stage1.blur import BlurResult
from cull.stage1.burst import _BurstInput, detect_bursts
from cull.stage1.duplicate import find_duplicates
from cull.stage1.exposure import ExposureResult
from cull.stage1.geometry import GeometryResult
from cull.stage1.worker import Stage1WorkerResult, assess_one

logger = logging.getLogger(__name__)


class _Stage1Ctx(BaseModel):
    """Per-photo Stage 1 assessment results."""

    blur: BlurResult
    exposure: ExposureResult
    geometry: GeometryResult
    noise_score: float
    is_noisy: bool


class _Stage1Output(BaseModel):
    """Aggregated Stage 1 results and routing sets."""

    results: dict[str, Stage1Result] = Field(default_factory=dict)
    survivors: list[Path] = Field(default_factory=list)
    rejected: list[Path] = Field(default_factory=list)
    duplicate_paths: set[str] = Field(default_factory=set)
    burst_losers: set[str] = Field(default_factory=set)
    failed_paths: list[Path] = Field(default_factory=list)
    encodings: dict[str, Any] = Field(default_factory=dict)


class _Stage1LoopInput(BaseModel):
    """Input bundle for Stage 1 per-image loop."""

    paths: list[Path]
    config: CullConfig
    source_path: Path | None = None


class _WorkerOutcome(BaseModel):
    """Result or failure from one pool worker invocation."""

    model_config = {"arbitrary_types_allowed": True}

    image_path: Path
    result: Stage1WorkerResult | None = None
    error: str | None = None


class _DrainCtx(BaseModel):
    """Bundle for draining pool results on the main thread."""

    model_config = {"arbitrary_types_allowed": True}

    output: _Stage1Output
    dashboard: object


def _safe_worker(image_path: Path, config: CullConfig) -> _WorkerOutcome:
    """Invoke assess_one and wrap any exception as a failed outcome.

    Errors are logged so the failure reason is visible; we return an outcome
    object rather than re-raising so one bad photo cannot crash the whole
    parallel Stage 1 pool.
    """
    try:
        return _WorkerOutcome(image_path=image_path, result=assess_one(image_path, config))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stage 1 worker failed for %s: %s", image_path, exc)
        return _WorkerOutcome(image_path=image_path, error=str(exc))


def _worker_result_to_ctx(w: Stage1WorkerResult) -> _Stage1Ctx:
    """Convert Stage1WorkerResult to _Stage1Ctx for result assembly."""
    return _Stage1Ctx(
        blur=w.blur,
        exposure=w.exposure,
        geometry=w.geometry,
        noise_score=w.noise.noise_score,
        is_noisy=w.noise.is_noisy,
    )


def _handle_worker_result(outcome: _WorkerOutcome, drain_ctx: _DrainCtx) -> None:
    """Process one pool outcome: log failures or update output and dashboard."""
    if outcome.error is not None:
        logger.warning("Stage 1 failed for %s: %s", outcome.image_path, outcome.error)
        drain_ctx.output.failed_paths.append(outcome.image_path)
        return
    ctx = _worker_result_to_ctx(outcome.result)
    result = _build_stage1_result(outcome.image_path, ctx)
    drain_ctx.output.results[str(outcome.image_path)] = result
    _classify_s1_result(result, drain_ctx.output)
    drain_ctx.dashboard.update_stage1(outcome.image_path, result)


def _build_exposure_scores(exp: ExposureResult) -> ExposureScores:
    """Convert ExposureResult into the shared ExposureScores model."""
    return ExposureScores(
        dr_score=exp.dynamic_range,
        clipping_highlight=exp.clipping.highlight_pct,
        clipping_shadow=exp.clipping.shadow_pct,
        midtone_pct=exp.midtone_pct,
        color_cast_score=exp.color_cast.cast_score,
        has_highlight_clip=exp.has_highlight_clip,
        has_shadow_clip=exp.has_shadow_clip,
        has_color_cast=exp.has_color_cast,
        has_low_dr=exp.has_low_dr,
    )


def _build_stage1_result(path: Path, ctx: _Stage1Ctx) -> Stage1Result:
    """Assemble a Stage1Result from individual assessments."""
    is_reject = ctx.blur.is_blurry or ctx.is_noisy
    reason = "blur" if ctx.blur.is_blurry else ("noise" if ctx.is_noisy else None)
    return Stage1Result(
        photo_path=path,
        blur=ctx.blur.scores,
        exposure=_build_exposure_scores(ctx.exposure),
        noise_score=ctx.noise_score,
        is_pass=not is_reject,
        reject_reason=reason,
        geometry=ctx.geometry.scores,
    )


def _run_stage1_loop(loop_in: _Stage1LoopInput, dashboard: Dashboard) -> _Stage1Output:
    """Run per-image Stage 1 assessments via spawn Pool with dashboard updates."""
    output = _Stage1Output()
    dashboard.start_stage1(len(loop_in.paths))
    worker_fn = partial(_safe_worker, config=loop_in.config)
    drain_ctx = _DrainCtx(output=output, dashboard=dashboard)
    mp_ctx = get_context("spawn")
    with mp_ctx.Pool(processes=STAGE1_WORKER_COUNT) as pool:
        for outcome in pool.imap_unordered(worker_fn, loop_in.paths):
            _handle_worker_result(outcome, drain_ctx)
    return output


def _classify_s1_result(result: Stage1Result, output: _Stage1Output) -> None:
    """Route a Stage 1 result into survivors or rejected lists."""
    if result.is_pass:
        output.survivors.append(result.photo_path)
    else:
        output.rejected.append(result.photo_path)


def _stamp_duplicate_flags(output: _Stage1Output) -> None:
    """Set is_duplicate=True on Stage1Result objects for all duplicate paths."""
    for key in output.duplicate_paths:
        result = output.results.get(key)
        if result is not None:
            result.is_duplicate = True


def _filter_survivors(output: _Stage1Output) -> list[Path]:
    """Remove burst losers and duplicates from the survivor list."""
    return [
        p for p in output.survivors
        if str(p) not in output.burst_losers
        and str(p) not in output.duplicate_paths
    ]


def _run_s1(ctx: Any) -> _Stage1Output:
    """Execute Stage 1 and record timing."""
    t0 = time.monotonic()
    loop_in = _Stage1LoopInput(paths=ctx.paths, config=ctx.config, source_path=ctx.source_path)
    ctx.dashboard.start_scanning()
    s1_out = _Stage1Output()
    _preflight_dupes_into_output(_Stage1WorkCtx(loop_in=loop_in, output=s1_out, dashboard=ctx.dashboard))
    _run_stage1_loop_into(_Stage1WorkCtx(loop_in=loop_in, output=s1_out, dashboard=ctx.dashboard))
    _apply_burst_only(loop_in, s1_out)
    _stamp_duplicate_flags(s1_out)
    s1_out.survivors = _filter_survivors(s1_out)
    ctx.dashboard.set_burst_count(len(s1_out.burst_losers))
    ctx.dashboard.refresh()
    _unload_imagededup_cnn()
    elapsed = time.monotonic() - t0
    ctx.timings.stage1 = elapsed
    ctx.dashboard.complete_stage1(elapsed)
    return s1_out


def _preflight_dupes_into_output(ctx: _Stage1WorkCtx) -> None:
    """Detect duplicates up-front so the live counter shows real values."""
    if not ctx.loop_in.paths:
        return
    image_dir = ctx.loop_in.source_path if ctx.loop_in.source_path is not None else ctx.loop_in.paths[0].parent
    dup_result = find_duplicates(image_dir)
    ctx.output.encodings = dup_result.encodings
    for group in dup_result.duplicate_groups:
        for dup_path in group.paths[1:]:
            ctx.output.duplicate_paths.add(str(dup_path))
    ctx.dashboard.set_dupe_count(len(ctx.output.duplicate_paths))
    ctx.dashboard.refresh()


def _run_stage1_loop_into(ctx: _Stage1WorkCtx) -> None:
    """Run the per-image loop, merging results into an existing output bundle."""
    loop_out = _run_stage1_loop(ctx.loop_in, ctx.dashboard)
    ctx.output.results.update(loop_out.results)
    ctx.output.survivors.extend(loop_out.survivors)
    ctx.output.rejected.extend(loop_out.rejected)
    ctx.output.failed_paths.extend(loop_out.failed_paths)


def _apply_burst_only(loop_in: _Stage1LoopInput, output: _Stage1Output) -> None:
    """Run burst detection only — duplicates already detected in preflight."""
    if not output.survivors:
        return
    blur_scores = {
        str(p): output.results[str(p)].blur.tenengrad
        for p in output.survivors
        if str(p) in output.results
    }
    burst_result = detect_bursts(_BurstInput(
        image_paths=output.survivors,
        config=loop_in.config,
        blur_scores=blur_scores,
    ))
    for loser in burst_result.losers:
        output.burst_losers.add(str(loser))


class _Stage1WorkCtx(BaseModel):
    """Bundle for Stage 1 work helpers (preflight + per-image loop)."""

    model_config = {"arbitrary_types_allowed": True}

    loop_in: _Stage1LoopInput
    output: _Stage1Output
    dashboard: object


def _unload_imagededup_cnn() -> None:
    """Free Stage 1 imagededup MobileNetV3 cache before Stage 2 loads its models."""
    duplicate_module._unload_cnn()
