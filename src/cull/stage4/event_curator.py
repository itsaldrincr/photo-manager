"""Event preset curation: rank representatives by event_score, diversify with MMR.

No VLM tournament and no narrative-flow swap: pairwise VLM comparisons of burst
frames on the singles-mixer shoot were near chance and flipped with image
order (order consistency 5-50%).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cull.config import CLUSTER_THRESHOLD, EVENT_MMR_LAMBDA, EVENT_PRESET
from cull.models import CurationResult, CuratorSelection
from cull.stage4.diversity import MmrContext, MmrInput, select as diversity_select

if TYPE_CHECKING:
    from cull.stage4.curator import CuratorInput


def rank_by_event_score(curator_input: CuratorInput) -> list[Path]:
    """Return candidates best first by event_score, ties broken on path."""
    scores = curator_input.event_scores
    return sorted(curator_input.keepers, key=lambda p: (-scores.get(str(p), 0.0), str(p)))


def _select(ranked: list[Path], curator_input: CuratorInput) -> list[Path]:
    """Return N candidates by MMR over CLIP rows, or the top N without a search cache."""
    target = curator_input.config.curate_target or len(ranked)
    embeddings = curator_input.search_embeddings
    path_to_row = curator_input.search_path_to_row
    if embeddings is None or path_to_row is None:
        return ranked[:target]
    context = MmrContext(
        embeddings=np.asarray(embeddings), path_to_row=path_to_row,
        lambda_quality=EVENT_MMR_LAMBDA, target_count=target,
    )
    return diversity_select(MmrInput(candidates=ranked, scores=curator_input.event_scores), context)


def _selection(path: Path, curator_input: CuratorInput) -> CuratorSelection:
    """Return the CuratorSelection record for one picked representative."""
    key = str(path)
    return CuratorSelection(
        path=path, cluster_id=0, cluster_size=1,
        composite=curator_input.composite_scores.get(key, 0.0),
        is_vlm_winner=False,
        reason=f"event score {curator_input.event_scores.get(key, 0.0):.3f}",
    )


def curate_event(curator_input: CuratorInput) -> CurationResult:
    """Pick N representatives by event_score with MMR diversity."""
    t0 = time.perf_counter()
    chosen = _select(rank_by_event_score(curator_input), curator_input)
    selections = [_selection(path, curator_input) for path in chosen]
    return CurationResult(
        is_enabled=True,
        target_count=curator_input.config.curate_target or len(selections),
        actual_count=len(selections),
        cluster_count=0,
        vlm_tiebreakers=0,
        threshold_used=CLUSTER_THRESHOLD[EVENT_PRESET],
        elapsed_seconds=time.perf_counter() - t0,
        selected=selections,
    )
