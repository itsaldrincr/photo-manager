"""Pipeline orchestrator — runs Stage 1, Stage 2, Stage 3 in sequence."""

from __future__ import annotations

import contextlib
import gc
import logging
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cull.config import (
    CullConfig,
    JPEG_EXTENSIONS,
    STAGE_IQA,
    STAGE_VLM,
)
from cull.vlm_session import vlm_session
from cull.dashboard import (
    Dashboard,
    DashboardLaunchInfo,
)
from cull.models import (
    CurationResult,
    PhotoDecision,
    Stage3Result,
)
from cull.stage1 import duplicate as duplicate_module
from cull import clip_loader
from cull.stage2.aesthetic import unload_predictor
from cull.stage2.composition import unload_topiq_iaa
from cull.stage2.iqa import unload_metrics
from cull.stage2.portrait import unload_face_landmarker

from cull._pipeline.stage1_runner import _run_s1, _Stage1Output
from cull._pipeline.stage2_runner import (
    _run_s2,
    _S2RunInput,
    _Stage2Output,
)
from cull._pipeline.stage2_reducer import (
    _run_s2_reducer,
    _S2ReducerRunInput,
)
from cull._pipeline.stage2_scoring import _SearchCache
from cull._pipeline.stage3_runner import (
    _run_s3_if_configured,
    _S3MaybeRunInput,
)
from cull._pipeline.stage4_curator import (
    _run_s4,
    _S4RunInput,
)
from cull._pipeline.decision_assembly import (
    _build_all_decisions,
    _build_summary,
    _DecisionCtx,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session result models
# ---------------------------------------------------------------------------


class SessionTiming(BaseModel):
    """Wall-clock timing per pipeline stage."""

    stage1_seconds: float = 0.0
    stage2_seconds: float = 0.0
    stage3_seconds: float = 0.0
    total_seconds: float = 0.0


class SessionSummary(BaseModel):
    """Decision counts across all photos."""

    keepers: int = 0
    rejected: int = 0
    duplicates: int = 0
    uncertain: int = 0
    selected: int = 0


class SessionResult(BaseModel):
    """Full pipeline run output — decisions, timing, and summary."""

    run_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    source_path: str = ""
    model: str = ""
    preset: str = "general"
    is_portrait: bool = False
    total_photos: int = 0
    stages_run: list[int] = Field(default_factory=list)
    summary: SessionSummary = Field(default_factory=SessionSummary)
    timing: SessionTiming = Field(default_factory=SessionTiming)
    decisions: list[PhotoDecision] = Field(default_factory=list)
    curation: CurationResult | None = None


# ---------------------------------------------------------------------------
# Internal parameter bundles (max 2 params rule)
# ---------------------------------------------------------------------------


class _StageTimings(BaseModel):
    """Mutable timing collector for pipeline stages."""

    stage1: float = 0.0
    stage2: float = 0.0
    stage3: float = 0.0


class _PipelineCtx(BaseModel):
    """Top-level context carrying config and paths through the pipeline."""

    config: CullConfig
    paths: list[Path]


class _SessionInput(BaseModel):
    """Input bundle for assembling the final SessionResult."""

    config: CullConfig
    source_path: str
    total_photos: int
    decisions: list[PhotoDecision]
    total_seconds: float
    curation: CurationResult | None = None


# ---------------------------------------------------------------------------
# Stage execution wrappers
# ---------------------------------------------------------------------------


class _StageRunCtx(BaseModel):
    """Shared context for running pipeline stages."""

    model_config = {"arbitrary_types_allowed": True}

    config: CullConfig
    paths: list[Path]
    source_path: Path | None = None
    timings: _StageTimings = Field(default_factory=_StageTimings)
    dashboard: Any = None  # Dashboard instance
    vlm_session: Any = None  # VlmSession — Any avoids import cycle


# `_S2RunInput` uses a forward reference to `_StageRunCtx`, and
# `_S2ReducerRunInput` uses forward references to `_Stage2Output` and
# `_StageRunCtx`. Rebuild both once the referenced classes are defined in
# this module's namespace.
_S2RunInput.model_rebuild(
    _types_namespace={"_StageRunCtx": _StageRunCtx}
)
_S2ReducerRunInput.model_rebuild(
    _types_namespace={
        "_Stage2Output": _Stage2Output,
        "_StageRunCtx": _StageRunCtx,
    }
)


class _StagesResult(BaseModel):
    """Collected outputs from all pipeline stages."""

    model_config = {"arbitrary_types_allowed": True}

    s1_out: _Stage1Output
    s2_out: _Stage2Output | None = None
    s3_results: dict[str, Stage3Result] = Field(default_factory=dict)
    search_cache: _SearchCache | None = None


def _unload_stage2_models() -> None:
    """Free Stage 2 neural network models before Stage 3."""
    unload_metrics()
    unload_predictor()
    unload_face_landmarker()
    unload_topiq_iaa()
    clip_loader.unload()
    gc.collect()


def _unload_imagededup_cnn() -> None:
    """Free Stage 1 imagededup MobileNetV3 cache before Stage 2 loads its models."""
    duplicate_module._unload_cnn()


def _assemble_session(session_in: _SessionInput, ctx: _StageRunCtx) -> SessionResult:
    """Build final SessionResult from pipeline outputs."""
    return SessionResult(
        source_path=session_in.source_path,
        model=session_in.config.model,
        preset=session_in.config.preset,
        is_portrait=session_in.config.is_portrait,
        total_photos=session_in.total_photos,
        stages_run=session_in.config.stages,
        summary=_build_summary(session_in.decisions),
        timing=SessionTiming(
            stage1_seconds=ctx.timings.stage1,
            stage2_seconds=ctx.timings.stage2,
            stage3_seconds=ctx.timings.stage3,
            total_seconds=session_in.total_seconds,
        ),
        decisions=session_in.decisions,
        curation=session_in.curation,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class _PipelineRunInput(BaseModel):
    """Input bundle for run_pipeline."""

    config: CullConfig
    source_path: Path
    file_size_gb: float = 0.0


def run_pipeline(run_in: _PipelineRunInput) -> SessionResult:
    """Orchestrate Stage 1, 2, 3 and return SessionResult."""
    return _run_pipeline_impl(run_in)


class _RunState(BaseModel):
    """Mutable state captured during a pipeline run, used by finalize."""

    model_config = {"arbitrary_types_allowed": True}

    ctx: _StageRunCtx
    stages: _StagesResult
    paths: list[Path]
    total_seconds: float


def _make_dashboard(run_in: _PipelineRunInput) -> Dashboard:
    """Create a Dashboard instance for the given run input."""
    launch_info = DashboardLaunchInfo(
        source_path=str(run_in.source_path),
        photo_count=0,
        preset=run_in.config.preset,
        file_size_gb=run_in.file_size_gb,
    )
    return Dashboard(launch_info)


def _build_run_ctx(
    run_in: _PipelineRunInput, dashboard: Dashboard
) -> _StageRunCtx:
    """Scan source, update dashboard photo count, and build a fresh ctx."""
    paths = _scan_with_dashboard(run_in.source_path, dashboard)
    dashboard.set_photo_count(len(paths))
    return _StageRunCtx(
        config=run_in.config, paths=paths,
        source_path=run_in.source_path, dashboard=dashboard,
    )


def _execute_run(ctx: _StageRunCtx) -> _RunState:
    """Run pipeline stages against an existing ctx and return run state."""
    t_start = time.monotonic()
    stages = _execute_stages_inline(ctx)
    return _RunState(
        ctx=ctx, stages=stages, paths=ctx.paths,
        total_seconds=time.monotonic() - t_start,
    )


def _finalize_run(state: _RunState, run_in: _PipelineRunInput) -> SessionResult:
    """Build decisions, run Stage 4, and assemble the SessionResult."""
    dec_ctx = _DecisionCtx(
        paths=state.paths, s1_out=state.stages.s1_out,
        s2_out=state.stages.s2_out, s3_results=state.stages.s3_results,
    )
    decisions = _build_all_decisions(dec_ctx)
    curation = _run_s4(_S4RunInput(stages=state.stages, decisions=decisions, ctx=state.ctx))
    session_in = _SessionInput(
        config=run_in.config, source_path=str(run_in.source_path),
        total_photos=len(state.paths), decisions=decisions,
        total_seconds=state.total_seconds, curation=curation,
    )
    return _assemble_session(session_in, state.ctx)


def _resolve_vlm_session_scope(
    config: CullConfig,
) -> AbstractContextManager:
    """Return vlm_session CM if Stage 3/4 will run, else nullcontext."""
    needs_vlm = STAGE_VLM in config.stages or config.curate_target is not None
    if needs_vlm:
        return vlm_session(config.model)
    return contextlib.nullcontext()


def _run_with_session(
    run_in: _PipelineRunInput, dashboard: Any
) -> SessionResult:
    """Open the VLM session scope and execute the run + finalize inside it."""
    ctx = _build_run_ctx(run_in, dashboard)
    with _resolve_vlm_session_scope(run_in.config) as session:
        ctx.vlm_session = session
        state = _execute_run(ctx)
        return _finalize_run(state, run_in)


def _run_pipeline_impl(run_in: _PipelineRunInput) -> SessionResult:
    """Implementation of the pipeline execution."""
    with _make_dashboard(run_in) as dashboard:
        result = _run_with_session(run_in, dashboard)
        dashboard.show_results(result)
        return result


def _execute_stages_inline(ctx: _StageRunCtx) -> _StagesResult:
    """Run stages without opening a new dashboard context."""
    s2_out: _Stage2Output | None = None
    s3_results: dict[str, Stage3Result] = {}
    s1_out = _run_s1(ctx)
    if STAGE_IQA in ctx.config.stages:
        s2_out = _run_s2(_S2RunInput(s1_out=s1_out, ctx=ctx))
        _run_s2_reducer(_S2ReducerRunInput(s2_out=s2_out, s1_out=s1_out, ctx=ctx))
        _unload_stage2_models()
        s3_results = _run_s3_if_configured(
            _S3MaybeRunInput(ctx=ctx, s2_out=s2_out, s1_out=s1_out),
        )
    search_cache = s2_out.search_cache if s2_out is not None else None
    return _StagesResult(
        s1_out=s1_out, s2_out=s2_out,
        s3_results=s3_results, search_cache=search_cache,
    )


def _scan_with_dashboard(source_path: Path, dashboard: Dashboard) -> list[Path]:
    """Scan for JPEGs while updating the dashboard's scan state."""
    paths: list[Path] = []
    total_bytes = 0
    dashboard.begin_scan()
    for p in source_path.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in JPEG_EXTENSIONS:
            continue
        paths.append(p)
        total_bytes += p.stat().st_size
        dashboard.update_scan_progress(len(paths), total_bytes)
    dashboard.end_scan()
    return sorted(paths)
