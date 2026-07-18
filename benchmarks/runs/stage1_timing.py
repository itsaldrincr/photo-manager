"""Time a full Stage 1 pass over a directory (wall-clock gate for burst ranking).

Runs the real Stage 1 (_run_s1: per-photo loop + preflight duplicates + burst
detection + winner selection) over a photo directory and prints elapsed seconds.
Used to compare blur-only vs face-aware winner selection on a fixed smoke
corpus: the face-aware regression must stay < 25% (burst_selection gate).

Usage:
    python3 benchmarks/runs/stage1_timing.py <photo_dir>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any


class _NoopDashboard:
    """Permissive no-op dashboard: every Stage 1 hook call is absorbed."""

    def __getattr__(self, _name: str) -> Any:
        return lambda *a, **k: None


def _time_stage1(photo_dir: Path) -> float:
    """Run a full Stage 1 pass over the directory; return elapsed seconds."""
    from cull.config import CullConfig  # noqa: PLC0415
    from cull._pipeline.stage1_runner import _run_s1  # noqa: PLC0415
    from cull._pipeline.orchestrator import _StageRunCtx  # noqa: PLC0415

    paths = sorted(set(photo_dir.glob("*.JPG")) | set(photo_dir.glob("*.jpg")))
    config = CullConfig(preset="wedding", is_portrait=True, is_dry_run=True, stages=[1])
    ctx = _StageRunCtx(config=config, paths=paths, source_path=photo_dir, dashboard=_NoopDashboard())
    start = time.monotonic()
    _run_s1(ctx)
    return time.monotonic() - start


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage1_timing.py <photo_dir>")
    elapsed = _time_stage1(Path(sys.argv[1]))
    print(f"stage1_elapsed_seconds={elapsed:.3f}")  # noqa: T201 — CLI result surface


if __name__ == "__main__":
    main()
