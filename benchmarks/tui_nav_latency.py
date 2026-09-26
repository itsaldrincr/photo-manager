"""Scripted Pilot run of the review TUI: keypress -> image-placed latency.

Usage:
    PYTHONPATH=src python3.13 benchmarks/tui_nav_latency.py SESSION_DIR [--steps 29] [--cadence 0.3]

SESSION_DIR holds session_report.json and its photos. The photos are
symlinked into a temp copy (so autosave never touches SESSION_DIR) and every
decision is set to "uncertain" so all photos sit in one queue.

Runs twice against one fresh preview cache: "cold" (empty cache, previews
render while you browse) and "reopen" (cache filled by the first run). The
terminal is headless, so the numbers are UI-side: key handled -> frame with
the new placeholder cells painted. median_bytes counts image escapes per
keypress; frame_bytes is one full repaint (Textual's text, including the
placeholder cells).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

os.environ["CULL_TUI_DEBUG"] = "1"

from cull.config import CullConfig  # noqa: E402
from cull.pipeline import SessionResult  # noqa: E402
from cull.tui import images, kitty, previews  # noqa: E402
from cull.tui.app import AppInput, CullApp  # noqa: E402
from cull.tui.photo_view import PhotoView  # noqa: E402

TERMINAL_SIZE: tuple[int, int] = (200, 60)
# A 13 pt font on a Retina display in Ghostty reports about 16x34 px cells.
DEFAULT_CELL: kitty.CellSize = kitty.CellSize(width=16.0, height=34.0)
FIRST_IMAGE_TIMEOUT_SECONDS: float = 60.0
STEP_TIMEOUT_SECONDS: float = 10.0


class CopyPlan:
    """Maps paths under the real session folder to the same place in the copy."""

    def __init__(self, source: Path, dest: Path) -> None:
        self.source = source
        self.dest = dest

    def relink(self, path_str: str) -> str:
        """Return where ``path_str`` lives in the copy."""
        return str(self.dest / Path(path_str).relative_to(self.source))

    def link(self, original: str | None) -> None:
        """Symlink one existing photo into the copy."""
        if not original or not Path(original).exists():
            return
        target = Path(self.relink(original))
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(original)


def build_copy(plan: CopyPlan) -> Path:
    """Symlink every photo into the copy and write an all-uncertain report there."""
    report = json.loads((plan.source / "session_report.json").read_text(encoding="utf-8"))
    for decision in report["decisions"]:
        plan.link(decision["photo"]["path"])
        plan.link(decision.get("destination"))
        decision["photo"]["path"] = plan.relink(decision["photo"]["path"])
        if decision.get("destination"):
            decision["destination"] = plan.relink(decision["destination"])
        decision["decision"] = "uncertain"
    report["source_path"] = str(plan.dest)
    (plan.dest / "session_report.json").write_text(json.dumps(report), encoding="utf-8")
    return plan.dest


class TerminalBytes:
    """Counts bytes the app queues for the terminal."""

    def __init__(self) -> None:
        self.total = 0

    def __call__(self, _app: object, data: str) -> None:
        self.total += len(data.encode("utf-8"))


def full_frame_bytes(app: CullApp) -> int:
    """Return the size of one full repaint, placeholder cells included."""
    update = app.screen._compositor.render_full_update()
    return len(update.render_segments(app.console).encode("utf-8"))


async def _wait_for(condition: object, timeout: float) -> bool:
    """Poll ``condition()`` until true or timeout."""
    deadline = time.perf_counter() + timeout
    while not condition():  # type: ignore[operator]
        if time.perf_counter() > deadline:
            return False
        await asyncio.sleep(0.002)
    return True


async def run_once(session_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    """Open the app, browse ``args.steps`` photos, return summary numbers."""
    session = SessionResult.model_validate_json((session_dir / "session_report.json").read_text(encoding="utf-8"))
    app = CullApp(AppInput(session=session, config=CullConfig()))
    counter = TerminalBytes()
    images.terminal_write = counter
    started = time.perf_counter()
    async with app.run_test(size=TERMINAL_SIZE) as pilot:
        view = app.query_one(PhotoView)
        await _wait_for(lambda: view.shown is not None, FIRST_IMAGE_TIMEOUT_SECONDS)
        first_image_ms = (time.perf_counter() - started) * 1000
        await asyncio.sleep(args.settle)
        step_bytes: list[int] = []
        for _ in range(args.steps):
            before_samples, before_bytes = len(app.latency.samples_ms), counter.total
            await pilot.press("right")
            await _wait_for(lambda: len(app.latency.samples_ms) > before_samples, STEP_TIMEOUT_SECONDS)
            step_bytes.append(counter.total - before_bytes)
            await asyncio.sleep(args.cadence)
        frame_bytes = full_frame_bytes(app)
    samples = sorted(app.latency.samples_ms)
    return {
        "first_image_ms": first_image_ms,
        "n": len(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": samples[int(len(samples) * 0.9)],
        "max_ms": samples[-1],
        "median_bytes": statistics.median(step_bytes),
        "frame_bytes": frame_bytes,
        "samples": [round(x) for x in app.latency.samples_ms],
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--steps", type=int, default=29)
    parser.add_argument("--cadence", type=float, default=0.3, help="seconds between keypresses")
    parser.add_argument("--settle", type=float, default=2.0, help="seconds after the first image")
    return parser.parse_args()


def main() -> None:
    """Run cold then reopen against one fresh cache and print both."""
    args = _parse_args()
    kitty.query_cell_size = lambda: DEFAULT_CELL
    with tempfile.TemporaryDirectory(prefix="cull-bench-") as scratch:
        previews.PREVIEW_CACHE_DIR = Path(scratch) / "previews"
        copy = build_copy(CopyPlan(args.session_dir.resolve(), Path(scratch) / "session"))
        for label in ("cold", "reopen"):
            (copy / ".cull_tui_state.json").unlink(missing_ok=True)
            result = asyncio.run(run_once(copy, args))
            print(label, " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in result.items()))


if __name__ == "__main__":
    main()
