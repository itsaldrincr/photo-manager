"""Stage 4 narrative-flow regulariser: shot-type variety across selected photos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from cull.config import (
    CullConfig,
    NARRATIVE_VARIETY_MIN,
    SHOT_CLOSE_UP_RATIO,
    SHOT_MEDIUM_RATIO,
    SHOT_SALIENCY_WIDE_RATIO,
)
from cull.models import CuratorSelection
from cull.saliency import SaliencyRequest, SaliencyResult, compute_saliency
from cull.stage2.portrait import PortraitResult, assess_portrait

logger = logging.getLogger(__name__)

ShotType = Literal["close", "medium", "wide"]

_DEFAULT_CONFIG = CullConfig()


class NarrativeFlowInput(BaseModel):
    """Input for the narrative-flow variety check."""

    selections: list[CuratorSelection]
    candidates: dict[str, Path]


class _SwapContext(BaseModel):
    """Internal bundle for a propose-swap operation."""

    flow_input: NarrativeFlowInput
    shots: list[ShotType]
    current_score: float


class _CandidateSearch(BaseModel):
    """Internal bundle for searching candidates by shot type."""

    candidates: dict[str, Path]
    current_paths: set[str]
    target_type: ShotType


def check(flow_input: NarrativeFlowInput) -> tuple[list[CuratorSelection], float]:
    """Return (possibly-swapped selections, narrative_flow_score)."""
    shots = [_shot_type_for(sel) for sel in flow_input.selections]
    score = _variety_score(shots)
    if score >= NARRATIVE_VARIETY_MIN:
        return flow_input.selections, score
    ctx = _SwapContext(flow_input=flow_input, shots=shots, current_score=score)
    return _propose_swap(ctx)


def _shot_type_for(selection: CuratorSelection) -> ShotType:
    """Classify a selection as close/medium/wide using face bbox or saliency."""
    portrait = _get_portrait(selection.path)
    if portrait.has_face and portrait.face_bbox is not None:
        return _classify_by_face(selection.path, portrait.face_bbox)
    return _classify_by_saliency(selection.path)


def _get_portrait(image_path: Path) -> PortraitResult:
    """Return PortraitResult for a given path; log on error."""
    try:
        return assess_portrait(image_path, _DEFAULT_CONFIG)
    except Exception as exc:
        logger.warning("Portrait assessment failed for %s: %s", image_path, exc)
        return PortraitResult(face_count=0)


def _classify_by_face(
    image_path: Path, face_bbox: tuple[int, int, int, int]
) -> ShotType:
    """Classify shot type from face-bbox-to-frame area ratio."""
    from PIL import Image

    try:
        with Image.open(image_path) as img:
            frame_area = img.width * img.height
    except Exception as exc:
        logger.warning("Cannot open image %s: %s", image_path, exc)
        return "medium"
    x0, y0, x1, y1 = face_bbox
    face_area = max(0, x1 - x0) * max(0, y1 - y0)
    ratio = face_area / frame_area if frame_area > 0 else 0.0
    return _ratio_to_shot(ratio)


def _ratio_to_shot(ratio: float) -> ShotType:
    """Map a face-area-to-frame ratio to a ShotType."""
    if ratio >= SHOT_CLOSE_UP_RATIO:
        return "close"
    if ratio >= SHOT_MEDIUM_RATIO:
        return "medium"
    return "wide"


def _classify_by_saliency(image_path: Path) -> ShotType:
    """Classify shot type from saliency bbox extent."""
    try:
        result: SaliencyResult = compute_saliency(SaliencyRequest(image_path=image_path))
    except Exception as exc:
        logger.warning("Saliency failed for %s: %s", image_path, exc)
        return "medium"
    x0, y0, x1, y1 = result.bbox
    extent = (x1 - x0) * (y1 - y0)
    if extent >= SHOT_SALIENCY_WIDE_RATIO:
        return "wide"
    return "close"


def _variety_score(shots: list[ShotType]) -> float:
    """Return shot-type variety as a 0–1 score."""
    if not shots:
        return 0.0
    total = len(shots)
    counts = {t: shots.count(t) for t in ("close", "medium", "wide")}
    unique_types = sum(1 for c in counts.values() if c > 0)
    dominant_fraction = max(counts.values()) / total
    return (unique_types / 3.0 + (1.0 - dominant_fraction)) / 2.0


def _build_search(ctx: _SwapContext, needed: ShotType) -> _CandidateSearch:
    """Build a _CandidateSearch from swap context and needed shot type."""
    current_paths = {str(sel.path) for sel in ctx.flow_input.selections}
    return _CandidateSearch(
        candidates=ctx.flow_input.candidates,
        current_paths=current_paths,
        target_type=needed,
    )


def _apply_swap(
    ctx: _SwapContext, candidate: CuratorSelection
) -> tuple[list[CuratorSelection], float]:
    """Swap dominant selection for candidate; return best result."""
    swap_idx = _worst_index(ctx.shots)
    new_selections = list(ctx.flow_input.selections)
    new_selections[swap_idx] = candidate
    new_shots = [_shot_type_for(sel) for sel in new_selections]
    new_score = _variety_score(new_shots)
    if new_score > ctx.current_score:
        logger.info("Narrative swap: variety %.2f → %.2f", ctx.current_score, new_score)
        return new_selections, new_score
    return ctx.flow_input.selections, ctx.current_score


def _propose_swap(ctx: _SwapContext) -> tuple[list[CuratorSelection], float]:
    """Try swapping a dominant shot with a candidate of a missing type."""
    needed = _needed_shot_type(ctx.shots)
    if needed is None:
        return ctx.flow_input.selections, ctx.current_score
    candidate = _find_candidate(_build_search(ctx, needed))
    if candidate is None:
        return ctx.flow_input.selections, ctx.current_score
    return _apply_swap(ctx, candidate)


def _needed_shot_type(shots: list[ShotType]) -> ShotType | None:
    """Return the first shot type absent from the current selections."""
    for t in ("close", "medium", "wide"):
        if t not in shots:
            return t  # type: ignore[return-value]
    return None


def _find_candidate(search: _CandidateSearch) -> CuratorSelection | None:
    """Return first candidate matching target_type not already selected."""
    for key, path in search.candidates.items():
        if key in search.current_paths or str(path) in search.current_paths:
            continue
        dummy = CuratorSelection(
            path=path, cluster_id=0, cluster_size=1, composite=0.0, is_vlm_winner=False
        )
        if _shot_type_for(dummy) == search.target_type:
            return dummy
    return None


def _worst_index(shots: list[ShotType]) -> int:
    """Return index of the dominant shot type's last occurrence."""
    counts = {t: shots.count(t) for t in ("close", "medium", "wide")}
    dominant = max(counts, key=lambda t: counts[t])
    for i in range(len(shots) - 1, -1, -1):
        if shots[i] == dominant:
            return i
    return len(shots) - 1
