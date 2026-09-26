"""Moment-stack resolution — choose each stack's representative after Stage 2."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cull.stage1.representatives import rank_stack

from cull._pipeline.stage1_runner import _Stage1Output

if TYPE_CHECKING:
    from cull._pipeline.stage2_runner import _Stage2Output


def _composite(key: str, s2_out: "_Stage2Output | None") -> float:
    """Return the post-reducer Stage 2 composite, or -inf when unscored."""
    if s2_out is None:
        return float("-inf")
    fusion = s2_out.results.get(key)
    if fusion is None or fusion.stage2 is None:
        return float("-inf")
    return fusion.stage2.composite


def _member_scores(s1_out: _Stage1Output, s2_out: "_Stage2Output | None") -> dict[str, tuple[float, float]]:
    """Return (Stage 2 composite, Stage 1 Tenengrad) for every stack member."""
    return {
        member: (_composite(member, s2_out), s1_out.results[member].blur.tenengrad)
        for stack in s1_out.stacks
        for member in stack
    }


def _stamp_ranking(ranked: list[str], s1_out: _Stage1Output) -> None:
    """Write final rank and representative flag onto each member's BurstInfo."""
    for rank, member in enumerate(ranked):
        result = s1_out.results[member]
        if result.burst is None:
            continue
        result.burst.rank = rank
        result.burst.is_burst_winner = rank == 0
        result.is_duplicate = rank > 0


def _drop_from_routing(s2_out: "_Stage2Output | None", removed: set[str]) -> None:
    """Remove non-representatives from Stage 2 routing lists so Stage 3 skips them."""
    if s2_out is None:
        return
    s2_out.keepers = [p for p in s2_out.keepers if str(p) not in removed]
    s2_out.ambiguous = [p for p in s2_out.ambiguous if str(p) not in removed]
    s2_out.rejects = [p for p in s2_out.rejects if str(p) not in removed]


def _apply_ranking(s1_out: _Stage1Output, scores: dict[str, tuple[float, float]]) -> set[str]:
    """Rank every stack by scores, stamp the result, and return the non-representatives."""
    duplicates: set[str] = set()
    for stack in s1_out.stacks:
        ranked = rank_stack(stack, scores)
        _stamp_ranking(ranked, s1_out)
        duplicates.update(ranked[1:])
    s1_out.duplicate_paths = duplicates
    return duplicates


def resolve_moment_stacks(s1_out: _Stage1Output, s2_out: "_Stage2Output | None") -> None:
    """Pick each stack's highest-composite member; mark the rest as duplicates.

    Tenengrad breaks composite ties, and decides alone when Stage 2 did not run.
    """
    duplicates = _apply_ranking(s1_out, _member_scores(s1_out, s2_out))
    _drop_from_routing(s2_out, duplicates)


def resolve_event_stacks(s1_out: _Stage1Output, event_scores: dict[str, float]) -> None:
    """Pick each stack's highest event_score member; Tenengrad breaks ties."""
    scores = {
        member: (event_scores.get(member, float("-inf")), s1_out.results[member].blur.tenengrad)
        for stack in s1_out.stacks
        for member in stack
    }
    _apply_ranking(s1_out, scores)
