"""MMR-based diversity selection over CLIP embeddings.

Pure-numpy implementation — no torch, no CLIP instantiation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cull.config import DPP_QUALITY_TRADEOFF

logger = logging.getLogger(__name__)


class MmrInput(BaseModel):
    """Candidates and their quality scores for MMR selection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidates: list[Path]
    scores: dict[str, float]


@dataclass
class MmrContext:
    """Embeddings matrix and selection parameters for MMR."""

    embeddings: np.ndarray
    path_to_row: dict[str, int]
    lambda_quality: float = DPP_QUALITY_TRADEOFF
    target_count: int = 10


class _MmrScoreInput(BaseModel):
    """Bundle for computing one MMR step score."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    relevance: np.ndarray
    penalty: np.ndarray
    lambda_q: float


class _LoopState(BaseModel):
    """All mutable state for one MMR loop iteration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rows: np.ndarray
    relevance: np.ndarray
    lambda_q: float
    selected: list[int] = Field(default_factory=list)
    remaining: list[int] = Field(default_factory=list)


def _max_relevance_index(score_input: _MmrScoreInput) -> int:
    """Return the index maximising the MMR objective."""
    mmr = score_input.lambda_q * score_input.relevance
    mmr -= (1.0 - score_input.lambda_q) * score_input.penalty
    return int(np.argmax(mmr))


def _diversity_penalty(
    candidate_rows: np.ndarray,
    selected_rows: np.ndarray,
) -> np.ndarray:
    """Return max cosine similarity of each candidate to the selected set."""
    if selected_rows.shape[0] == 0:
        return np.zeros(candidate_rows.shape[0])
    sims = candidate_rows @ selected_rows.T
    return sims.max(axis=1)


def _pick_next(loop: _LoopState) -> int:
    """Return local index into remaining of the next best candidate."""
    rem_rows = loop.rows[loop.remaining]
    no_selected = not loop.selected
    sel_rows = (
        np.empty((0, loop.rows.shape[1]), dtype=np.float32)
        if no_selected
        else loop.rows[loop.selected]
    )
    penalty = _diversity_penalty(rem_rows, sel_rows)
    score_input = _MmrScoreInput(
        relevance=loop.relevance[loop.remaining],
        penalty=penalty,
        lambda_q=loop.lambda_q,
    )
    return _max_relevance_index(score_input)


def _gather_rows(candidates: list[Path], context: MmrContext) -> np.ndarray:
    """Return embedding matrix rows for candidates; zero-row on cache miss."""
    dim = context.embeddings.shape[1]
    rows = np.zeros((len(candidates), dim), dtype=np.float32)
    for i, path in enumerate(candidates):
        row_idx = context.path_to_row.get(str(path))
        if row_idx is not None:
            rows[i] = context.embeddings[row_idx]
    return rows


def _build_relevance(candidates: list[Path], scores: dict[str, float]) -> np.ndarray:
    """Return quality-score vector for candidates; default 0.0 on miss."""
    return np.array([scores.get(str(p), 0.0) for p in candidates], dtype=np.float32)


def select(mmr_input: MmrInput, context: MmrContext) -> list[Path]:
    """Return up to target_count paths chosen by MMR diversity."""
    candidates = mmr_input.candidates
    if not candidates:
        return []
    rows = _gather_rows(candidates, context)
    relevance = _build_relevance(candidates, mmr_input.scores)
    loop = _LoopState(
        rows=rows,
        relevance=relevance,
        lambda_q=context.lambda_quality,
        selected=[],
        remaining=list(range(len(candidates))),
    )
    while len(loop.selected) < context.target_count and loop.remaining:
        local_idx = _pick_next(loop)
        loop.selected.append(loop.remaining.pop(local_idx))
    logger.debug("MMR selected %d of %d candidates", len(loop.selected), len(candidates))
    return [candidates[i] for i in loop.selected]
