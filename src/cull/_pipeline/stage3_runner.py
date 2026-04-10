"""Stage 3 runner — VLM tiebreaker loop and frozen cull_fast entry points."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from cull.config import (
    CullConfig,
    DASHBOARD_REFRESH_SLEEP_SECONDS,
    STAGE_VLM,
)
from cull.dashboard import Dashboard
from cull.models import Stage1Result, Stage3Result
from cull.stage3.prompt import PromptContext
from cull.stage3.vlm_scoring import VlmRequest, VlmScoreCallInput, score_photo
from cull.vlm_registry import resolve_alias

if TYPE_CHECKING:
    from cull.pipeline import _Stage1Output, _Stage2Output, _StageRunCtx


class _Stage3LoopInput(BaseModel):
    """Input bundle for Stage 3 VLM loop."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ambiguous: list[Path]
    config: CullConfig
    model_name: str = ""
    s1_results: dict[str, Stage1Result] = Field(default_factory=dict)
    s2_results: dict[str, Any] = Field(default_factory=dict)
    session: Any = None  # VlmSession — Any avoids import cycle


class _PromptContextInput(BaseModel):
    """Input bundle for prompt context assembly."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    s1: Stage1Result | None = None
    s2_fusion: Any = None
    config: CullConfig


def _build_stage1_prompt_context(s1: Stage1Result | None) -> dict[str, bool]:
    """Extract Stage 1 boolean prompt flags from the Stage 1 result."""
    if s1 is None:
        return {
            "motion_blur_detected": False,
            "is_bokeh": False,
            "has_highlight_clip": False,
            "has_shadow_clip": False,
            "has_color_cast": False,
        }
    return {
        "motion_blur_detected": s1.blur.is_motion_blur,
        "is_bokeh": s1.blur.is_bokeh,
        "has_highlight_clip": s1.exposure.has_highlight_clip,
        "has_shadow_clip": s1.exposure.has_shadow_clip,
        "has_color_cast": s1.exposure.has_color_cast,
    }


def _build_stage2_prompt_context(stage2: Any) -> dict[str, Any]:
    """Extract Stage 2 prompt context values from the fusion result."""
    if stage2 is None:
        return {
            "stage2_composite": None,
            "composition_score": None,
            "dominant_emotion": None,
            "has_face": False,
            "eyes_closed": False,
            "face_occluded": False,
        }
    portrait = stage2.portrait
    composition = stage2.composition
    return {
        "stage2_composite": stage2.composite,
        "composition_score": composition.composite if composition is not None else None,
        "dominant_emotion": portrait.dominant_emotion if portrait is not None else None,
        "has_face": portrait is not None,
        "eyes_closed": portrait.is_eyes_closed if portrait is not None else False,
        "face_occluded": portrait.is_face_occluded if portrait is not None else False,
    }


def _build_prompt_context(prompt_in: _PromptContextInput) -> PromptContext:
    """Build VLM PromptContext from Stage 1 and Stage 2 signals."""
    stage2 = prompt_in.s2_fusion.stage2 if prompt_in.s2_fusion is not None else None
    stage1_context = _build_stage1_prompt_context(prompt_in.s1)
    stage2_context = _build_stage2_prompt_context(stage2)
    return PromptContext(
        preset=prompt_in.config.preset,
        **stage1_context,
        **stage2_context,
    )


def _score_single_s3(path: Path, loop_in: _Stage3LoopInput) -> Stage3Result:
    """Score one ambiguous photo via VLM."""
    s1 = loop_in.s1_results.get(str(path))
    s2_fusion = loop_in.s2_results.get(str(path))
    context = _build_prompt_context(
        _PromptContextInput(s1=s1, s2_fusion=s2_fusion, config=loop_in.config)
    )
    request = VlmRequest(image_path=path, context=context, model=loop_in.model_name)
    call_in = VlmScoreCallInput(request=request, session=loop_in.session)
    return score_photo(call_in)


def _run_stage3_loop(loop_in: _Stage3LoopInput, dashboard: Dashboard) -> dict[str, Stage3Result]:
    """Run VLM tiebreaker on ambiguous photos with dashboard."""
    dashboard.start_stage3(len(loop_in.ambiguous), loop_in.model_name)
    results: dict[str, Stage3Result] = {}
    # 1 worker is intentional: we run the future on a separate thread so the
    # main thread can poll dashboard refresh via _wait_with_refresh, not for
    # parallelism. Stage 3 photos are processed strictly serially.
    with ThreadPoolExecutor(max_workers=1) as pool:
        for path in loop_in.ambiguous:
            dashboard.start_analysis(path)
            future: Future = pool.submit(_score_single_s3, path, loop_in)
            _wait_with_refresh(future, dashboard)
            result = future.result()
            results[str(path)] = result
            dashboard.complete_analysis(path, result)
    return results


def _wait_with_refresh(future: object, dashboard: Dashboard) -> None:
    """Poll future while refreshing the dashboard for animation."""
    while not future.done():
        dashboard.refresh()
        time.sleep(DASHBOARD_REFRESH_SLEEP_SECONDS)


class _S3RunInput(BaseModel):
    """Input bundle for Stage 3 execution."""

    model_config = {"arbitrary_types_allowed": True}

    s2_out: Any  # _Stage2Output — Any avoids circular import with cull.pipeline
    s1_out: Any  # _Stage1Output — Any avoids circular import with cull.pipeline
    ctx: Any  # _StageRunCtx — Any avoids circular import with cull.pipeline
    model_name: str


def _run_s3(run_in: _S3RunInput) -> dict[str, Stage3Result]:
    """Execute Stage 3 and record timing."""
    t0 = time.monotonic()
    loop_in = _Stage3LoopInput(
        ambiguous=run_in.s2_out.ambiguous,
        config=run_in.ctx.config,
        model_name=run_in.model_name,
        s1_results=run_in.s1_out.results,
        s2_results=run_in.s2_out.results,
        session=run_in.ctx.vlm_session,
    )
    s3_results = _run_stage3_loop(loop_in, run_in.ctx.dashboard)
    elapsed = time.monotonic() - t0
    run_in.ctx.timings.stage3 = elapsed
    run_in.ctx.dashboard.complete_stage3(elapsed)
    return s3_results


class _S3MaybeRunInput(BaseModel):
    """Input bundle for the optional Stage 3 dispatch helper."""

    model_config = {"arbitrary_types_allowed": True}

    ctx: Any  # _StageRunCtx — Any avoids circular import with cull.pipeline
    s2_out: Any  # _Stage2Output — Any avoids circular import with cull.pipeline
    s1_out: Any  # _Stage1Output — Any avoids circular import with cull.pipeline


def _run_s3_if_configured(run_in: _S3MaybeRunInput) -> dict[str, Stage3Result]:
    """Run Stage 3 VLM if configured; return empty dict otherwise."""
    if STAGE_VLM not in run_in.ctx.config.stages:
        return {}
    entry = resolve_alias(run_in.ctx.config.model)
    s3_in = _S3RunInput(
        s2_out=run_in.s2_out, s1_out=run_in.s1_out,
        ctx=run_in.ctx, model_name=entry.display_name,
    )
    return _run_s3(s3_in)
