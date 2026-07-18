"""Append-only JSONL override log for user culling decision corrections.

POSIX-only: uses fcntl.flock for thread-safe append.
On non-POSIX systems, the lock is a no-op and concurrent writes may interleave.
"""

from __future__ import annotations

import json
import logging

try:
    import fcntl
    _HAVE_FCNTL = True
except ImportError:
    _HAVE_FCNTL = False

from pydantic import BaseModel

from cull.config import OVERRIDE_LOG_DIR, OVERRIDE_LOG_PATH
from cull.models import DecisionLabel, OverrideEntry, PhotoDecision
from cull.taste_features import TasteFeatureInputs, build_taste_feature_row, flatten_stage1_scores

logger = logging.getLogger(__name__)


class OverrideContext(BaseModel):
    """Bundle for building an OverrideEntry from a PhotoDecision."""

    new_label: DecisionLabel
    session_source: str
    origin: str


def _ensure_log_dir() -> None:
    """Create OVERRIDE_LOG_DIR if it does not already exist."""
    OVERRIDE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _extract_stage1_scores(decision: PhotoDecision) -> dict[str, float]:
    """Flatten a decision's Stage1Result into a flat scalar dict."""
    return flatten_stage1_scores(decision.stage1)


def _extract_geometry_pair(decision: PhotoDecision) -> tuple[float | None, float | None]:
    """Pull (tilt, keystone) from Stage1Result.geometry, defaulting to (None, None)."""
    if decision.stage1 is None or decision.stage1.geometry is None:
        return (None, None)
    geometry = decision.stage1.geometry
    return (float(geometry.tilt_degrees), float(geometry.keystone_degrees))


def _extract_stage3_dict(decision: PhotoDecision) -> dict | None:
    """Serialize Stage3Result to dict, or return None."""
    if decision.stage3 is None:
        return None
    return decision.stage3.model_dump(mode="json")


def _stage2_extension_fields(decision: PhotoDecision) -> dict[str, object]:
    """Pluck the stage2 extension scores off a decision into a kwargs dict."""
    if decision.stage2 is None:
        return {}
    stage2 = decision.stage2
    return {
        "stage2_composition": stage2.composition,
        "stage2_taste": stage2.taste,
        "stage2_subject_blur": stage2.subject_blur,
        "stage2_shoot_outliers": stage2.shoot_stats,
    }


def _canonical_feature_row(decision: PhotoDecision, stage1_scores: dict[str, float]) -> list[float] | None:
    """Build the forward-compat canonical taste row for this decision, or None if no Stage 2 yet.

    Captured at write time so future feature-space changes can retrain
    directly from the log without recomputing from raw stage data.
    """
    if decision.stage2 is None:
        return None
    stage2 = decision.stage2
    composition = stage2.composition
    subject_blur = stage2.subject_blur
    inputs = TasteFeatureInputs(
        stage1_scores=stage1_scores,
        stage2_composite=stage2.composite,
        composition_composite=composition.composite if composition else None,
        thirds_alignment=composition.thirds_alignment if composition else None,
        negative_space_balance=composition.negative_space_balance if composition else None,
        subject_blur_tenengrad=subject_blur.tenengrad if subject_blur else None,
    )
    return build_taste_feature_row(inputs).tolist()


def build_override_entry(decision: PhotoDecision, ctx: OverrideContext) -> OverrideEntry:
    """Build an OverrideEntry from a PhotoDecision with flattened stage data."""
    stage2_composite = decision.stage2.composite if decision.stage2 is not None else None
    tilt, keystone = _extract_geometry_pair(decision)
    stage1_scores = _extract_stage1_scores(decision)
    return OverrideEntry(
        photo_path=str(decision.photo.path),
        filename=decision.photo.filename,
        original_decision=decision.decision,
        user_decision=ctx.new_label,
        stage1_scores=stage1_scores,
        stage2_composite=stage2_composite,
        stage3_result=_extract_stage3_dict(decision),
        session_source=ctx.session_source,
        override_origin=ctx.origin,
        tilt_degrees=tilt,
        keystone_degrees=keystone,
        feature_row=_canonical_feature_row(decision, stage1_scores),
        **_stage2_extension_fields(decision),
    )


def _write_entry(entry: OverrideEntry) -> None:
    """Open log file in append mode, flock it, write one JSONL line."""
    line = json.dumps(entry.model_dump(mode="json")) + "\n"
    with open(OVERRIDE_LOG_PATH, "a", encoding="utf-8") as fh:
        if _HAVE_FCNTL:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(line)
        finally:
            if _HAVE_FCNTL:
                fcntl.flock(fh, fcntl.LOCK_UN)


def log_override(entry: OverrideEntry) -> None:
    """Append one OverrideEntry as a JSONL line; creates parent dir if missing."""
    try:
        _ensure_log_dir()
        _write_entry(entry)
    except OSError as exc:
        logger.warning("Failed to write override log: %s", exc)


def _parse_line(line: str) -> OverrideEntry | None:
    """Parse one JSONL line into an OverrideEntry, returning None on failure."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
        return OverrideEntry.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Skipping malformed override log line: %s", exc)
        return None


def load_overrides() -> list[OverrideEntry]:
    """Read all OverrideEntry records from OVERRIDE_LOG_PATH; returns [] if missing."""
    if not OVERRIDE_LOG_PATH.exists():
        return []
    try:
        with open(OVERRIDE_LOG_PATH, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as exc:
        logger.warning("Failed to read override log: %s", exc)
        return []
    results: list[OverrideEntry] = []
    for line in lines:
        entry = _parse_line(line)
        if entry is not None:
            results.append(entry)
    return results
