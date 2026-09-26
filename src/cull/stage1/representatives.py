"""Pick one representative per connected duplicate/burst group."""

from __future__ import annotations


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
    return [sorted(c) for c in components.values() if len(c) > 1]


def select_group_losers(groups: list[list[str]], scores: dict[str, float]) -> set[str]:
    """Return every member except the highest-scoring one of each connected group.

    Ties break on the path name so the result is reproducible.
    """
    losers: set[str] = set()
    for component in connected_groups(groups):
        winner = min(component, key=lambda m: (-scores.get(m, 0.0), m))
        losers.update(m for m in component if m != winner)
    return losers

