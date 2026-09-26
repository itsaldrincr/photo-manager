"""Stage 4 curator wiring — assembles CuratorInput and runs curation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from cull.models import CurationResult, PhotoDecision
from cull.stage2.portrait import PortraitResult
from cull.stage4.curator import CuratorInput, curate

from cull._pipeline.event_routing import EventRouteInput, collect_event_scores

if TYPE_CHECKING:
    from cull.pipeline import _StagesResult, _StageRunCtx

logger = logging.getLogger(__name__)


class _S4RunInput(BaseModel):
    """Input bundle for _run_s4."""

    model_config = {"arbitrary_types_allowed": True}

    stages: Any  # _StagesResult — Any avoids circular import with cull.pipeline
    decisions: list[PhotoDecision] = Field(default_factory=list)
    ctx: Any  # _StageRunCtx — Any avoids circular import with cull.pipeline


# Stage 4 draws from every photo that passed Stage 1 (blur/exposure/noise), is
# a moment-stack representative or a singleton, and was scored by Stage 2, not
# only from Stage 2/3 "keeper" routing. The 0.94/0.85 routing cutoffs are calibrated for
# auto-keep precision, not for building a curation pool: on a 214-photo
# wedding card they left 3 keepers, so `--curate 80` could pick at most 3.
_CANDIDATE_EXCLUDED_LABELS: frozenset[str] = frozenset({"duplicate"})


def _is_stack_non_representative(decision: PhotoDecision) -> bool:
    """Return True if the photo is in a moment stack but not its representative."""
    burst = decision.stage1.burst if decision.stage1 is not None else None
    return burst is not None and not burst.is_burst_winner


def _collect_candidate_paths(s4_in: _S4RunInput) -> list[Path]:
    """Return curation candidates: stack representatives and singletons with a Stage 2 score."""
    scored = _collect_composite_scores(s4_in.stages)
    return [
        d.photo.path
        for d in s4_in.decisions
        if d.decision not in _CANDIDATE_EXCLUDED_LABELS
        and str(d.photo.path) in scored
        and not _is_stack_non_representative(d)
    ]


def _collect_composite_scores(stages: Any) -> dict[str, float]:
    """Extract Stage 2 composite scores keyed by str(path).

    Note: Returns 0.0 for photos with no Stage 2 result (sentinel value).
    Photos missing Stage 2 results should be skipped during curation.
    """
    if stages.s2_out is None:
        return {}
    scores = {}
    for key, fusion in stages.s2_out.results.items():
        if fusion.stage2 is not None:
            scores[key] = fusion.stage2.composite
        else:
            logger.warning("Photo %s has no Stage 2 result; assigning sentinel 0.0", key)
            scores[key] = 0.0
    return scores


def _collect_portraits(stages: Any) -> dict[str, PortraitResult]:
    """Return Stage 2 portrait results, or empty dict when Stage 2 skipped."""
    if stages.s2_out is None:
        return {}
    return dict(stages.s2_out.portraits)


def _collect_event_scores(s4_in: _S4RunInput) -> dict[str, float]:
    """Return event scores for the event preset, else an empty dict."""
    if not s4_in.ctx.config.is_event:
        return {}
    stages = s4_in.stages
    return collect_event_scores(EventRouteInput(
        s1_out=stages.s1_out, s2_out=stages.s2_out, s3_results=stages.s3_results,
    ))


def _build_curator_input(s4_in: _S4RunInput) -> CuratorInput:
    """Assemble CuratorInput from stages, decisions, and run context."""
    cache = s4_in.stages.search_cache
    return CuratorInput(
        keepers=_collect_candidate_paths(s4_in),
        encodings=s4_in.stages.s1_out.encodings,
        composite_scores=_collect_composite_scores(s4_in.stages),
        config=s4_in.ctx.config,
        dashboard=s4_in.ctx.dashboard,
        s1_results=s4_in.stages.s1_out.results,
        portraits=_collect_portraits(s4_in.stages),
        search_embeddings=cache.embeddings if cache else None,
        search_path_to_row=cache.path_to_row if cache else None,
        vlm_session=s4_in.ctx.vlm_session,
        event_scores=_collect_event_scores(s4_in),
    )


def _mark_selected(decisions: list[PhotoDecision], selected: set[str]) -> None:
    """Flip decision label to 'select' for any photo whose path is in selected."""
    for decision in decisions:
        if str(decision.photo.path) in selected:
            decision.decision = "select"


def _run_s4(s4_in: _S4RunInput) -> CurationResult | None:
    """Execute Stage 4 curation if --curate target was provided."""
    if s4_in.ctx.config.curate_target is None:
        return None
    t0 = time.monotonic()
    curator_input = _build_curator_input(s4_in)
    if not curator_input.keepers:
        return None
    s4_in.ctx.dashboard.start_stage4(target=s4_in.ctx.config.curate_target)
    result = curate(curator_input)
    selected_paths = {str(sel.path) for sel in result.selected}
    _mark_selected(s4_in.decisions, selected_paths)
    elapsed = time.monotonic() - t0
    s4_in.ctx.timings.stage4 = elapsed
    s4_in.ctx.dashboard.complete_stage4(elapsed)
    return result
