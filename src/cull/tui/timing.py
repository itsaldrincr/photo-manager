"""Debug log and keypress-to-image-placed latency, enabled by CULL_TUI_DEBUG."""

from __future__ import annotations

import os
import time
from pathlib import Path

DEBUG_ENV_VAR: str = "CULL_TUI_DEBUG"
DEBUG_LOG_PATH: Path = Path.home() / ".cache" / "cull" / "tui_debug.log"


def is_enabled() -> bool:
    """Return True when the debug log is switched on."""
    return bool(os.environ.get(DEBUG_ENV_VAR))


def debug_log(message: str) -> bool:
    """Append a timestamped line to the debug log; return False if it could not be written."""
    if not is_enabled():
        return False
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.time():.3f} {message}\n")
    except OSError:
        return False
    return True


class LatencyTimer:
    """Measures keypress to image-painted latency across two UI events."""

    def __init__(self) -> None:
        self._started: float | None = None
        self.samples_ms: list[float] = []

    def start(self) -> None:
        """Mark the moment a navigation or decision key was handled."""
        if is_enabled():
            self._started = time.perf_counter()

    def finish(self, label: str) -> None:
        """Record latency since the last start, once the new image has been painted."""
        if self._started is None:
            return
        elapsed_ms = (time.perf_counter() - self._started) * 1000
        self._started = None
        self.samples_ms.append(elapsed_ms)
        debug_log(f"latency key->placed {elapsed_ms:.1f}ms {label}")
