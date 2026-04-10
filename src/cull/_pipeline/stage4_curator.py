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

if TYPE_CHECKING:
    from cull.pipeline import _StagesResult, _StageRunCtx

logger = logging.getLogger(__name__)


class _S4RunInput(BaseModel):
    """Input bundle for _run_s4."""

    model_config = {"arbitrary_types_allowed": True}

    stages: Any  # _StagesResult — Any avoids circular import with cull.pipeline
    decisions: list[PhotoDecision] = Field(default_factory=list)
    ctx: Any  # _StageRunCtx — Any avoids circular import with cull.pipeline


def _collect_keeper_paths(s4_in: _S4RunInput) -> list[Path]:
    """Return list of photo paths whose decision is currently 'keeper'."""
    return [d.photo.path for d in s4_in.decisions if d.decision == "keeper"]


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


def _build_curator_input(s4_in: _S4RunInput) -> CuratorInput:
    """Assemble CuratorInput from stages, decisions, and run context."""
    cache = s4_in.stages.search_cache
    return CuratorInput(
        keepers=_collect_keeper_paths(s4_in),
        encodings=s4_in.stages.s1_out.encodings,
        composite_scores=_collect_composite_scores(s4_in.stages),
        config=s4_in.ctx.config,
        dashboard=s4_in.ctx.dashboard,
        s1_results=s4_in.stages.s1_out.results,
        portraits=_collect_portraits(s4_in.stages),
        search_embeddings=cache.embeddings if cache else None,
        search_path_to_row=cache.path_to_row if cache else None,
        vlm_session=s4_in.ctx.vlm_session,
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
    curator_input = _build_curator_input(s4_in)
    if not curator_input.keepers:
        return None
    s4_in.ctx.dashboard.start_stage4(target=s4_in.ctx.config.curate_target)
    t0 = time.monotonic()
    result = curate(curator_input)
    elapsed = time.monotonic() - t0
    selected_paths = {str(sel.path) for sel in result.selected}
    _mark_selected(s4_in.decisions, selected_paths)
    s4_in.ctx.dashboard.complete_stage4(elapsed)
    return result
