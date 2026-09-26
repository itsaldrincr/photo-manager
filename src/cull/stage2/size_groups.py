"""Group same-size images so a batched metric never stacks mixed shapes.

Upright decoding gives portrait frames a (W, H) swapped against landscape
ones, and torch.cat cannot stack them into one (N, C, H, W) batch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

Item = TypeVar("Item")
Result = TypeVar("Result")


def group_indices_by_size(sizes: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Return input indices grouped by equal size, groups in first-seen order."""
    groups: dict[tuple[int, int], list[int]] = {}
    for index, size in enumerate(sizes):
        groups.setdefault(size, []).append(index)
    return list(groups.values())


def score_in_size_groups(
    items: Sequence[tuple[tuple[int, int], Item]],
    score_group: Callable[[list[Item]], list[Result]],
) -> list[Result]:
    """Score (size, item) pairs one same-size group at a time; return scores in input order."""
    scores: list[Result | None] = [None] * len(items)
    for indices in group_indices_by_size([size for size, _ in items]):
        group_scores = score_group([items[i][1] for i in indices])
        for index, score in zip(indices, group_scores):
            scores[index] = score
    return scores  # type: ignore[return-value]  every index is filled above
