"""Fast-mode pipeline orchestrator.

Mirrors `cull.pipeline._run_pipeline_impl` shape but substitutes `_run_s2_fast`
for Stage 2. Stage 3 / Stage 4 share a single in-process VLM session via the
`vlm_session(...)` context manager that wraps stage execution.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict

from cull.config import STAGE_IQA, STAGE_VLM
from cull.dashboard import Dashboard
from cull.models import Stage3Result
from cull.pipeline import (
    _PipelineRunInput,
    _RunState,
    _S2RunInput,
    _S3RunInput,
    _Stage2Output,
    _StageRunCtx,
    _StagesResult,
    _finalize_run,
    _make_dashboard,
    _resolve_vlm_session_scope,
    _run_s1,
    _run_s3,
    _scan_with_dashboard,
    SessionResult,
)
from cull.vlm_registry import resolve_alias
from cull_fast.pipeline_fast import _run_s2_fast, _unload_stage2_models_fast

__all__ = ["run_fast_pipeline"]


class _FastS3BuildInput(BaseModel):
    """Bundle of ctx + stage outputs for building the fast Stage 3 input."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ctx: _StageRunCtx
    s1_out: object
    s2_out: _Stage2Output


def run_fast_pipeline(run_in: _PipelineRunInput) -> SessionResult:
    """Run the fast-mode pipeline and return the SessionResult."""
    with _make_dashboard(run_in) as dashboard:
        result = _run_fast_with_session(run_in, dashboard)
        dashboard.show_results(result)
        return result


def _run_fast_with_session(
    run_in: _PipelineRunInput, dashboard: Dashboard
) -> SessionResult:
    """Open the VLM session scope and execute the fast run + finalize inside it."""
    ctx = _build_fast_ctx(run_in, dashboard)
    with _resolve_vlm_session_scope(run_in.config) as session:
        ctx.vlm_session = session
        state = _execute_run_fast(ctx)
        return _finalize_run(state, run_in)


def _build_fast_ctx(
    run_in: _PipelineRunInput, dashboard: Dashboard
) -> _StageRunCtx:
    """Scan source, update dashboard, and build a fresh fast-mode ctx."""
    paths = _scan_with_dashboard(run_in.source_path, dashboard)
    dashboard.set_photo_count(len(paths))
    return _StageRunCtx(
        config=run_in.config, paths=paths,
        source_path=run_in.source_path, dashboard=dashboard,
    )


def _execute_run_fast(ctx: _StageRunCtx) -> _RunState:
    """Run fast stage execution against an existing ctx; return run state."""
    t_start = time.monotonic()
    stages = _execute_stages_inline_fast(ctx)
    return _RunState(
        ctx=ctx, stages=stages, paths=ctx.paths,
        total_seconds=time.monotonic() - t_start,
    )


def _execute_stages_inline_fast(ctx: _StageRunCtx) -> _StagesResult:
    """Run fast pipeline stages without opening a new dashboard context."""
    s2_out: _Stage2Output | None = None
    s3_results: dict[str, Stage3Result] = {}
    s1_out = _run_s1(ctx)
    if STAGE_IQA in ctx.config.stages:
        s2_out = _run_s2_fast(_S2RunInput(s1_out=s1_out, ctx=ctx))
        _unload_stage2_models_fast()
    if STAGE_VLM in ctx.config.stages and s2_out:
        build_in = _FastS3BuildInput(ctx=ctx, s1_out=s1_out, s2_out=s2_out)
        s3_results = _run_s3(_build_fast_s3_input(build_in))
    return _StagesResult(s1_out=s1_out, s2_out=s2_out, s3_results=s3_results)


def _build_fast_s3_input(build_in: _FastS3BuildInput) -> _S3RunInput:
    """Build the _S3RunInput bundle for fast-mode Stage 3 dispatch."""
    entry = resolve_alias(build_in.ctx.config.model)
    return _S3RunInput(
        s2_out=build_in.s2_out, s1_out=build_in.s1_out,
        ctx=build_in.ctx, model_name=entry.display_name,
    )
