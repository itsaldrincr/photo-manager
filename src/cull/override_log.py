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

logger = logging.getLogger(__name__)

_STAGE1_BLUR_KEYS = ("tenengrad", "fft_ratio")
_STAGE1_EXPOSURE_KEYS = ("dr_score", "clipping_highlight", "clipping_shadow", "midtone_pct", "color_cast_score")


class OverrideContext(BaseModel):
    """Bundle for building an OverrideEntry from a PhotoDecision."""

    new_label: DecisionLabel
    session_source: str
    origin: str


def _ensure_log_dir() -> None:
    """Create OVERRIDE_LOG_DIR if it does not already exist."""
    OVERRIDE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _extract_stage1_scores(decision: PhotoDecision) -> dict[str, float]:
    """Flatten Stage1Result blur + exposure + noise + geometry into a flat dict."""
    if decision.stage1 is None:
        return {}
    scores: dict[str, float] = {}
    blur = decision.stage1.blur
    for key in _STAGE1_BLUR_KEYS:
        value = getattr(blur, key, None)
        if value is not None:
            scores[key] = float(value)
    exposure = decision.stage1.exposure
    for key in _STAGE1_EXPOSURE_KEYS:
        value = getattr(exposure, key, None)
        if value is not None:
            scores[key] = float(value)
    scores["noise_score"] = float(decision.stage1.noise_score)
    geometry = decision.stage1.geometry
    if geometry is not None:
        scores["tilt_degrees"] = float(geometry.tilt_degrees)
        scores["keystone_degrees"] = float(geometry.keystone_degrees)
    return scores


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


def build_override_entry(decision: PhotoDecision, ctx: OverrideContext) -> OverrideEntry:
    """Build an OverrideEntry from a PhotoDecision with flattened stage data."""
    stage2_composite = decision.stage2.composite if decision.stage2 is not None else None
    tilt, keystone = _extract_geometry_pair(decision)
    return OverrideEntry(
        photo_path=str(decision.photo.path),
        filename=decision.photo.filename,
        original_decision=decision.decision,
        user_decision=ctx.new_label,
        stage1_scores=_extract_stage1_scores(decision),
        stage2_composite=stage2_composite,
        stage3_result=_extract_stage3_dict(decision),
        session_source=ctx.session_source,
        override_origin=ctx.origin,
        tilt_degrees=tilt,
        keystone_degrees=keystone,
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
