# stdlib
import logging
import os
import re
from pathlib import Path

# third-party
from pydantic import BaseModel

# local
from cull.config import VLM_ALIASES, VLM_DEFAULT_ALIAS, VLM_MODELS_ROOT

logger = logging.getLogger(__name__)

VLM_CONFIG_FILENAME: str = "config.json"
VLM_VISION_KEY: str = "vision_config"
VLM_WEIGHTS_FILENAME: str = "model.safetensors"
VLM_WEIGHTS_INDEX_FILENAME: str = "model.safetensors.index.json"
_SLUG_PATTERN: re.Pattern[str] = re.compile(r"[^a-z0-9]+")


class VLMEntry(BaseModel):
    """One discovered VLM on disk."""

    alias: str
    directory: Path
    display_name: str


class VLMRegistry(BaseModel):
    """Cache of discovered VLMs and alias to entry index."""

    entries: list[VLMEntry]
    by_alias: dict[str, VLMEntry]


class VLMResolutionError(Exception):
    """Raised when an alias does not resolve to a discoverable VLM."""


def discover_vlms(root: Path = VLM_MODELS_ROOT) -> VLMRegistry:
    """Scan `root` and return a VLMRegistry of valid VLM dirs."""
    if not root.is_dir():
        return VLMRegistry(entries=[], by_alias={})
    candidates = [d for d in root.iterdir() if d.is_dir() and _has_vision_config(d)]
    by_alias: dict[str, VLMEntry] = dict(_apply_override_table(candidates))
    claimed_dirs: set[Path] = {entry.directory for entry in by_alias.values()}
    unclaimed = [c for c in candidates if c not in claimed_dirs]
    slug_entries = _build_slug_entries(unclaimed, set(by_alias.keys()))
    for entry in slug_entries:
        by_alias[entry.alias] = entry
    return VLMRegistry(entries=list(by_alias.values()), by_alias=by_alias)


def resolve_alias(alias: str, registry: VLMRegistry | None = None) -> VLMEntry:
    """Return the VLMEntry for `alias` or raise VLMResolutionError."""
    reg = registry if registry is not None else discover_vlms()
    entry = reg.by_alias.get(alias)
    if entry is None:
        raise VLMResolutionError(
            f"VLM alias '{alias}' not found in registry. "
            f"Known aliases: {list(reg.by_alias.keys())}"
        )
    return entry


def run_vlm_preflight(alias: str = VLM_DEFAULT_ALIAS) -> None:
    """Validate that `alias` resolves to a structurally valid VLM on disk.

    Raises VLMResolutionError on failure. Never calls mlx_vlm.load().
    """
    registry = discover_vlms()
    entry = resolve_alias(alias, registry)
    _validate_directory_structure(entry.directory)


# ---- private helpers ----


def _has_vision_config(candidate: Path) -> bool:
    """Return True iff candidate/config.json exists and has 'vision_config' key."""
    config_path = candidate / VLM_CONFIG_FILENAME
    if not config_path.is_file():
        return False
    import json

    try:
        data = json.loads(config_path.read_text())
        return VLM_VISION_KEY in data
    except (json.JSONDecodeError, OSError):
        return False


def _auto_slug(display_name: str) -> str:
    """Return a lowercase alias with non-alphanumerics collapsed to hyphens."""
    return _SLUG_PATTERN.sub("-", display_name.lower()).strip("-")


def _apply_override_table(dirs: list[Path]) -> dict[str, VLMEntry]:
    """Match VLM_ALIASES substrings against discovered directory names."""
    result: dict[str, VLMEntry] = {}
    for alias, substring in VLM_ALIASES.items():
        matched = next((d for d in dirs if substring in d.name), None)
        if matched is not None:
            result[alias] = VLMEntry(
                alias=alias,
                directory=matched,
                display_name=matched.name,
            )
    return result


def _build_slug_entries(
    candidates: list[Path], taken: set[str]
) -> list[VLMEntry]:
    """Return auto-slug VLMEntry list for dirs not already in `taken`."""
    result: list[VLMEntry] = []
    seen: set[str] = set(taken)
    for candidate in candidates:
        slug = _auto_slug(candidate.name)
        if slug in seen:
            logger.warning(
                "VLM alias collision: '%s' already taken; skipping '%s'",
                slug,
                candidate.name,
            )
            continue
        seen.add(slug)
        result.append(VLMEntry(alias=slug, directory=candidate, display_name=candidate.name))
    return result


def _validate_directory_structure(directory: Path) -> None:
    """Raise VLMResolutionError if the dir is missing config or weights."""
    config_path = directory / VLM_CONFIG_FILENAME
    if not config_path.is_file():
        raise VLMResolutionError(f"Missing {VLM_CONFIG_FILENAME} in {directory}")
    if not _has_vision_config(directory):
        raise VLMResolutionError(
            f"config.json in {directory} lacks '{VLM_VISION_KEY}' key"
        )
    has_weights = (directory / VLM_WEIGHTS_FILENAME).is_file()
    has_index = (directory / VLM_WEIGHTS_INDEX_FILENAME).is_file()
    if not has_weights and not has_index:
        raise VLMResolutionError(
            f"No weights found in {directory}: "
            f"expected '{VLM_WEIGHTS_FILENAME}' or '{VLM_WEIGHTS_INDEX_FILENAME}'"
        )
