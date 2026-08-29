"""Session report writer — serialises SessionResult to JSON on disk."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from cull.config import CullConfig
from cull.pipeline import SessionResult

logger = logging.getLogger(__name__)

REPORT_FILENAME: str = "session_report.json"
TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"


def _archive_path(base_path: Path) -> Path:
    """Return a timestamp-suffixed sibling path for an existing report."""
    stem = base_path.stem
    suffix = base_path.suffix
    timestamp = datetime.now(tz=timezone.utc).strftime(TIMESTAMP_FORMAT)
    return base_path.with_name(f"{stem}_{timestamp}{suffix}")


def _archive_existing(base_path: Path) -> Path | None:
    """Rename an existing report aside so the new run's report becomes canonical.

    `--review` always reads REPORT_FILENAME. Writing a new run's report under a
    timestamped name instead left a stale session_report.json in place, and the
    next `--review` save replayed the stale decisions over the new layout
    (2026-08-30: 80 curated selects moved back into _review/).
    """
    if not base_path.exists():
        return None
    archived = _archive_path(base_path)
    os.replace(base_path, archived)
    logger.info("Previous report archived to %s", archived)
    return archived


def _serialise(session_result: SessionResult) -> str:
    """Serialise SessionResult to a JSON string."""
    return session_result.model_dump_json(indent=2)


def _atomic_write_text(target: Path, content: str) -> None:
    """Write content to target via a sibling temp file + atomic os.replace."""
    tmp_path = target.with_name(target.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, target)


def write_report(session_result: SessionResult, overwrite: bool = False) -> Path:
    """Write session_report.json to the source path.

    The new report is always written to REPORT_FILENAME, because that is the
    file `--review` reads. When ``overwrite`` is False, an existing report is
    first renamed to a timestamp-suffixed sibling so it is preserved.
    Review-mode saves pass ``overwrite=True`` and replace the file in place.
    """
    source_dir = Path(session_result.source_path)
    base_path = source_dir / REPORT_FILENAME
    if not overwrite:
        _archive_existing(base_path)
    target = base_path
    content = _serialise(session_result)
    _atomic_write_text(target, content)
    logger.info("Report written to %s", target)
    return target
