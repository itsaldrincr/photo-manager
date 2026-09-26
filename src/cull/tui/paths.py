"""Resolve where each photo lives on disk now, caching answers to keep stats off keypresses."""

from __future__ import annotations

from pathlib import Path

from cull.config import CullConfig
from cull.models import PhotoDecision

_RECOVERY_DIRS: tuple[tuple[str, ...], ...] = (
    (),
    ("_curated", "_selects"),
    ("_review", "_uncertain"),
    ("_review", "_rejected"),
    ("_review", "_duplicates"),
)


def _recover_missing_path(decision: PhotoDecision) -> Path | None:
    """Search known review/curation folders for a moved photo by filename."""
    source = decision.photo.path
    for parts in _RECOVERY_DIRS:
        candidate = source.parent.joinpath(*parts, source.name)
        if candidate.exists():
            return candidate
    return None


def resolve_on_disk(decision: PhotoDecision, config: CullConfig) -> Path:
    """Return the photo's current path: destination, source, routed, or recovered."""
    from cull.router import route_photo  # noqa: PLC0415

    if decision.destination is not None and decision.destination.exists():
        return decision.destination
    if decision.photo.path.exists():
        return decision.photo.path
    routed = route_photo(decision, config)
    if routed.exists():
        return routed
    recovered = _recover_missing_path(decision)
    return recovered if recovered is not None else decision.photo.path


class PathCache:
    """Cache of resolved photo paths keyed by the original source path.

    Files do not move while the TUI runs (moves happen on save, then the app
    exits), so a resolved path stays valid for the whole session.
    """

    def __init__(self, config: CullConfig) -> None:
        self._config = config
        self._paths: dict[str, Path] = {}

    def resolve(self, decision: PhotoDecision) -> Path:
        """Return the cached path, resolving (with stat calls) on first use only."""
        key = str(decision.photo.path)
        cached = self._paths.get(key)
        if cached is not None:
            return cached
        resolved = resolve_on_disk(decision, self._config)
        self._paths[key] = resolved
        return resolved

    def cached(self, decision: PhotoDecision) -> Path | None:
        """Return the cached path without touching the disk."""
        return self._paths.get(str(decision.photo.path))

    def merge(self, resolved: dict[str, Path]) -> None:
        """Adopt paths resolved in the background."""
        self._paths.update(resolved)


def resolve_all(decisions: list[PhotoDecision], config: CullConfig) -> dict[str, Path]:
    """Worker: resolve every decision's path (pure; returns a new mapping)."""
    return {str(d.photo.path): resolve_on_disk(d, config) for d in decisions}
