"""File routing — compute destinations and execute moves."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from cull.config import CullConfig
from cull.models import PhotoDecision
from cull.sidecar import SidecarWriteInput, write_for_decision

logger = logging.getLogger(__name__)

REVIEW_DIR: str = "_review"
REJECTED_SUBDIR: str = "_rejected"
DUPLICATES_SUBDIR: str = "_duplicates"
UNCERTAIN_SUBDIR: str = "_uncertain"
CURATED_DIR: str = "_curated"
SELECTS_SUBDIR: str = "_selects"
SIDECAR_DECISIONS: frozenset[str] = frozenset({"keeper", "select"})
SIDECAR_SUFFIX: str = ".xmp"


# ---------------------------------------------------------------------------
# MoveReport model
# ---------------------------------------------------------------------------


class MoveEntry(BaseModel):
    """Record of a single file move operation."""

    source: Path
    destination: Path
    is_success: bool = True
    error: str | None = None


class MoveReport(BaseModel):
    """Summary of all file move operations."""

    total: int = 0
    moved: int = 0
    skipped: int = 0
    errors: int = 0
    is_dry_run: bool = False
    entries: list[MoveEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

_LABEL_TO_SUBDIR: dict[str, str] = {
    "rejected": REJECTED_SUBDIR,
    "duplicate": DUPLICATES_SUBDIR,
    "uncertain": UNCERTAIN_SUBDIR,
}


def _review_subdir(decision: PhotoDecision) -> str | None:
    """Return the review subfolder name, or None for keepers."""
    return _LABEL_TO_SUBDIR.get(decision.decision)


def _select_destination(source: Path) -> Path:
    """Return the _curated/_selects destination path for a select decision."""
    return source.parent / CURATED_DIR / SELECTS_SUBDIR / source.name


def route_photo(decision: PhotoDecision, config: CullConfig) -> Path:
    """Return destination path based on decision label."""
    source = decision.photo.path
    if decision.decision == "select":
        return _select_destination(source)
    subdir = _review_subdir(decision)
    if subdir is None:
        return source
    review_root = source.parent / REVIEW_DIR / subdir
    return review_root / source.name


# ---------------------------------------------------------------------------
# Move execution
# ---------------------------------------------------------------------------


def _ensure_parent(destination: Path) -> None:
    """Create parent directory for destination if it does not exist."""
    destination.parent.mkdir(parents=True, exist_ok=True)


def _move_file(source: Path, destination: Path) -> MoveEntry:
    """Move a single file, choosing os.rename or shutil.move."""
    try:
        _ensure_parent(destination)
        os.rename(str(source), str(destination))
        return MoveEntry(source=source, destination=destination)
    except OSError:
        logger.debug("os.rename failed, falling back to shutil.move")
        try:
            shutil.move(str(source), str(destination))
            return MoveEntry(source=source, destination=destination)
        except OSError as exc:
            logger.warning("shutil.move failed: %s", exc)
            return MoveEntry(source=source, destination=destination, is_success=False, error=str(exc))


def _record_dry_run(source: Path, destination: Path) -> MoveEntry:
    """Log a dry-run move without touching the filesystem."""
    logger.info("DRY RUN: %s -> %s", source, destination)
    return MoveEntry(source=source, destination=destination)


class _MoveCtx(BaseModel):
    """Input context for processing a batch of moves."""

    decisions: list[PhotoDecision]
    config: CullConfig


def _is_sidecar_decision(decision: PhotoDecision) -> bool:
    """Return True only for keeper/select decisions eligible for sidecars."""
    return decision.decision in SIDECAR_DECISIONS


def _write_sidecar_if_enabled(decision: PhotoDecision, config: CullConfig) -> None:
    """Write XMP sidecar before move when sidecars enabled and decision qualifies."""
    if not config.is_sidecars or not _is_sidecar_decision(decision):
        return
    write_for_decision(SidecarWriteInput(decision=decision, config=config))


def _move_sidecar_alongside(source: Path, destination: Path) -> None:
    """Detect a .xmp next to the source and move it next to the destination."""
    sidecar_src = source.with_suffix(SIDECAR_SUFFIX)
    if not sidecar_src.exists():
        return
    sidecar_dest = destination.with_suffix(SIDECAR_SUFFIX)
    _move_file(sidecar_src, sidecar_dest)


def _current_source(decision: PhotoDecision) -> Path:
    """Return the photo's current on-disk source, preferring a prior destination."""
    if decision.destination is not None and decision.destination.exists():
        return decision.destination
    return decision.photo.path


def process_single_move(decision: PhotoDecision, config: CullConfig) -> MoveEntry | None:
    """Route and move (or dry-run) a single photo; return None when no move needed."""
    destination = route_photo(decision, config)
    source = _current_source(decision)
    if destination == source:
        _write_sidecar_if_enabled(decision, config)
        return None
    _write_sidecar_if_enabled(decision, config)
    if config.is_dry_run:
        return _record_dry_run(source, destination)
    entry = _move_file(source, destination)
    _move_sidecar_alongside(source, destination)
    if entry.is_success:
        decision.destination = entry.destination
    return entry


def execute_moves(decisions: list[PhotoDecision], config: CullConfig) -> MoveReport:
    """Execute file moves for all decisions, respecting dry_run."""
    report = MoveReport(total=len(decisions), is_dry_run=config.is_dry_run)
    for decision in decisions:
        entry = process_single_move(decision, config)
        if entry is None:
            report.skipped += 1
            continue
        report.entries.append(entry)
        if entry.is_success:
            report.moved += 1
        else:
            report.errors += 1
    return report
