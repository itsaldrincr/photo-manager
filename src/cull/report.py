"""Session report writer — serialises SessionResult to JSON on disk."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from cull.config import CullConfig
from cull.pipeline import SessionResult

logger = logging.getLogger(__name__)

REPORT_FILENAME: str = "session_report.json"
TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"


def _unique_path(base_path: Path) -> Path:
    """Return base_path if it does not exist; otherwise append a timestamp suffix."""
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    timestamp = datetime.now(tz=timezone.utc).strftime(TIMESTAMP_FORMAT)
    return base_path.with_name(f"{stem}_{timestamp}{suffix}")


def _serialise(session_result: SessionResult) -> str:
    """Serialise SessionResult to a JSON string."""
    return session_result.model_dump_json(indent=2)


def write_report(session_result: SessionResult, overwrite: bool = False) -> Path:
    """Write session_report.json to the source path.

    When ``overwrite`` is False, preserve any existing report by writing a
    timestamp-suffixed sibling. Review-mode saves should pass ``overwrite=True``
    so the next `--review` session reloads the updated decisions.
    """
    source_dir = Path(session_result.source_path)
    base_path = source_dir / REPORT_FILENAME
    target = base_path if overwrite else _unique_path(base_path)
    content = _serialise(session_result)
    target.write_text(content, encoding="utf-8")
    logger.info("Report written to %s", target)
    return target
