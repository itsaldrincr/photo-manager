"""Stage 4 single-elimination tournament bracket using VLM pairwise comparisons."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from cull.config import CullConfig
from cull.models import Stage1Result
from cull.stage3.prompt import PromptContext
from cull.stage4.vlm_tiebreak import (
    CuratorTiebreakCallInput,
    CuratorTiebreakInput,
    compare_photos,
)

logger = logging.getLogger(__name__)

MIN_CANDIDATES: int = 2
MATCH_PAIR_SIZE: int = 2  # Photos per match in a single-elim bracket.


class TournamentInput(BaseModel):
    """Input bundle for running a single-elimination tournament."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidates: list[Path]
    config: CullConfig
    session: Any = None  # VlmSession — Any avoids import cycle


class TournamentContext(BaseModel):
    """Lookup tables the tournament draws on for seeding and match context."""

    s1_results: dict[str, Stage1Result]
    composite_scores: dict[str, float]


class _MatchParams(BaseModel):
    """Internal bundle for a single pairwise match."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    photo_a: Path
    photo_b: Path
    ctx: TournamentContext
    model: str
    session: Any = None  # VlmSession — Any avoids import cycle


def _seed_bracket(candidates: list[Path], scores: dict[str, float]) -> list[Path]:
    """Return candidates sorted descending by composite score (highest seed first)."""
    return sorted(candidates, key=lambda p: scores.get(str(p), 0.0), reverse=True)


def _build_prompt_context(path: Path, s1: dict[str, Stage1Result]) -> PromptContext:
    """Build PromptContext from Stage1Result for a given path, or return defaults."""
    result = s1.get(str(path))
    if result is None:
        return PromptContext()
    return PromptContext(
        motion_blur_detected=result.blur.is_motion_blur,
        # Stage 1 has no eye-state signal; Stage 2 portrait mode owns is_eyes_closed.
        eyes_closed=False,
        has_highlight_clip=result.exposure.has_highlight_clip,
        has_shadow_clip=result.exposure.has_shadow_clip,
        has_color_cast=result.exposure.has_color_cast,
    )


def _play_match(params: _MatchParams) -> Path:
    """Run one VLM pairwise comparison and return the winner path."""
    context_a = _build_prompt_context(params.photo_a, params.ctx.s1_results)
    context_b = _build_prompt_context(params.photo_b, params.ctx.s1_results)
    tiebreak_input = CuratorTiebreakInput(
        photo_a=params.photo_a,
        photo_b=params.photo_b,
        context=context_a,
        context_b=context_b,
        model=params.model,
    )
    call_in = CuratorTiebreakCallInput(
        tiebreak_input=tiebreak_input, session=params.session
    )
    result = compare_photos(call_in)
    logger.debug("Match %s vs %s → winner=%s", params.photo_a.name, params.photo_b.name, result.winner.name)
    return result.winner


def _run_round(bracket: list[Path], match_cfg: _MatchParams) -> tuple[list[Path], list[Path]]:
    """Play one full round: pair up bracket entries, return (winners, losers)."""
    winners: list[Path] = []
    losers: list[Path] = []
    for i in range(0, len(bracket), MATCH_PAIR_SIZE):
        if i + 1 >= len(bracket):
            winners.append(bracket[i])
        else:
            pair = match_cfg.model_copy(update={"photo_a": bracket[i], "photo_b": bracket[i + 1]})
            winner = _play_match(pair)
            loser = bracket[i + 1] if winner == bracket[i] else bracket[i]
            winners.append(winner)
            losers.append(loser)
    return winners, losers


def _build_match_cfg(
    inp: TournamentInput, context: TournamentContext
) -> _MatchParams:
    """Build the seed _MatchParams used as the per-round template."""
    return _MatchParams(
        photo_a=inp.candidates[0], photo_b=inp.candidates[0], ctx=context,
        model=inp.config.model, session=inp.session,
    )


def run(inp: TournamentInput, context: TournamentContext) -> list[Path]:
    """Seed candidates by composite score and run single-elim until one remains.

    Returns candidates in elimination order (winner first, then losers by round).
    """
    if len(inp.candidates) < MIN_CANDIDATES:
        return list(inp.candidates)
    bracket = _seed_bracket(inp.candidates, context.composite_scores)
    total_rounds = math.ceil(math.log2(len(bracket)))
    logger.info("Tournament: %d candidates, %d rounds", len(bracket), total_rounds)
    match_cfg = _build_match_cfg(inp, context)
    elimination_order: list[Path] = []
    for round_num in range(total_rounds):
        logger.debug("Round %d/%d: %d remaining", round_num + 1, total_rounds, len(bracket))
        bracket, losers = _run_round(bracket, match_cfg)
        elimination_order.extend(losers)
    elimination_order.insert(0, bracket[0])
    return elimination_order
