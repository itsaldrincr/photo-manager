"""Moment stacks: connected duplicate/burst groups and their ranking."""

from __future__ import annotations

from datetime import datetime

from cull.config import DUPLICATE_TIME_WINDOW_SECONDS


def is_within_moment_window(first: datetime, second: datetime) -> bool:
    """Return True if two capture times are close enough to share a moment."""
    return abs((first - second).total_seconds()) <= DUPLICATE_TIME_WINDOW_SECONDS


def _find_root(parent: dict[str, str], node: str) -> str:
    """Return the union-find root of node, compressing the path on the way."""
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != root:
        parent[node], node = root, parent[node]
    return root


def connected_groups(groups: list[list[str]]) -> list[list[str]]:
    """Merge groups that share any member into connected components (size >= 2)."""
    parent: dict[str, str] = {}
    for group in groups:
        for member in group:
            parent.setdefault(member, member)
        for member in group[1:]:
            parent[_find_root(parent, member)] = _find_root(parent, group[0])
    components: dict[str, list[str]] = {}
    for member in parent:
        components.setdefault(_find_root(parent, member), []).append(member)
    return sorted(sorted(c) for c in components.values() if len(c) > 1)


def rank_stack(stack: list[str], scores: dict[str, tuple[float, ...]]) -> list[str]:
    """Return stack members best first by score tuple, ties broken on path name."""
    lowest: tuple[float, ...] = (float("-inf"),)
    return sorted(stack, key=lambda m: (tuple(-s for s in scores.get(m, lowest)), m))
