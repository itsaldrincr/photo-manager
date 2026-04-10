"""Shared helpers for golden-baseline tests.

Used by test_perf_p1_batch_golden, test_perf_p3_pool_golden,
and test_perf_p4lite_clip_golden.
"""

from __future__ import annotations

from pathlib import Path

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

MANIFEST_FILENAME: str = "manifest.json"
FIRST_N_DIVERGENCES: int = 10
SCORE_TOLERANCE: float = 1e-5


@dataclasses.dataclass
class Divergence:
    """A single per-field score divergence between baseline and current run."""

    filename: str
    field: str
    expected: float
    current: float
    abs_diff: float


def compute_corpus_fingerprint(manifest_path: Path) -> str:
    """Compute sha256 fingerprint of sorted name/sha256 pairs in a manifest."""
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    lines = [
        f"{entry['name']}\t{entry['sha256']}"
        for entry in manifest["files"]
        if entry["name"] != MANIFEST_FILENAME
    ]
    lines.sort()
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return digest


def baseline_filename(kind: str, corpus_dir: Path) -> str:
    """Return the baseline JSON filename for a given kind and corpus dir."""
    return f"{kind}_baseline_{corpus_dir.name}.json"


def _get_file_category(manifest: dict, filename: str) -> str:
    """Look up the category for a file in the manifest."""
    for entry in manifest["files"]:
        if entry["name"] == filename:
            return entry.get("category", "unknown")
    return "unknown"


def group_divergences_by_category(divergences: list, manifest: dict) -> dict[str, list]:
    """Bucket divergences by the category field from the manifest."""
    grouped: dict[str, list] = {}
    for div in divergences:
        category = _get_file_category(manifest, div.filename)
        grouped.setdefault(category, []).append(div)
    return grouped


def _format_category_line(category: str, divs: list, total: int) -> str:
    """Format a single category summary line."""
    count = len(divs)
    if count == 0:
        return f"  {category}:    0/{total} files diverged"
    max_diff = max(d.abs_diff for d in divs)
    return f"  {category}:    {count}/{total} files diverged (max abs_diff: {max_diff:.4g})"


def _format_first_divergences(divergences: list) -> str:
    """Format the first N divergences as a detail block."""
    lines = [f"\nFirst {FIRST_N_DIVERGENCES} divergences:"]
    for div in divergences[:FIRST_N_DIVERGENCES]:
        lines.append(
            f"  {div.filename}  .{div.field}"
            f"    baseline={div.expected:.6g}"
            f"  current={div.current:.6g}"
            f"  diff={div.abs_diff:.2e}"
        )
    return "\n".join(lines)


def format_divergence_report(grouped: dict, total_by_category: dict) -> str:
    """Format a divergence report grouped by category with per-file details."""
    header_lines = ["Stage 2 baseline drift:"]
    all_divs: list = []
    for category, total in total_by_category.items():
        divs = grouped.get(category, [])
        header_lines.append(_format_category_line(category, divs, total))
        all_divs.extend(divs)
    report = "\n".join(header_lines)
    if all_divs:
        report += _format_first_divergences(all_divs)
    return report


# ---------------------------------------------------------------------------
# Shared batched-path scorer
# ---------------------------------------------------------------------------


@dataclass
class _FakeS2State:
    """Minimal stand-in for Dashboard._s2 used by _MinimalDashboard."""

    done: int = 0


@dataclass
class _MinimalDashboard:
    """Minimal dashboard stub satisfying _run_stage2_loop's surface."""

    _s2: _FakeS2State = field(default_factory=_FakeS2State)
    stage2_calls: list[tuple[Path, Any]] = field(default_factory=list)

    def start_stage2(self, total: int) -> None:
        """No-op — counter reset handled by _FakeS2State default."""

    def start_stage2_loading(self) -> None:
        """No-op — prewarm progress indicator."""

    def clear_stage2_loading(self) -> None:
        """No-op — prewarm progress indicator."""

    def update_stage2(self, update_in: Any) -> None:
        """Record call and increment done counter."""
        self._s2.done += 1
        self.stage2_calls.append((update_in.path, update_in.fusion))

    def complete_stage2(self, elapsed: float) -> None:
        """No-op."""


def _extract_scores_from_calls(
    calls: list[tuple[Path, Any]],
) -> dict[str, dict[str, float]]:
    """Convert dashboard call list to {filename: {metric: score}} mapping."""
    result: dict[str, dict[str, float]] = {}
    for path, fusion in calls:
        result[path.name] = {
            "topiq": fusion.stage2.topiq,
            "laion_aesthetic": fusion.stage2.laion_aesthetic,
            "clipiqa": fusion.stage2.clipiqa,
        }
    return result


def score_corpus_via_batched_path(
    photo_paths: list[Path],
) -> dict[str, dict[str, float]]:
    """Score photos via the batched Stage 2 pipeline path.

    REAL inference — loads actual ML models. Do NOT call from mock-only pytest tests.
    Returns {filename: {"topiq": float, "laion_aesthetic": float, "clipiqa": float}}.
    Device selection is handled internally by _run_stage2_loop.
    """
    from cull.config import CullConfig  # noqa: PLC0415
    from cull.pipeline import _Stage2LoopInput, _run_stage2_loop  # noqa: PLC0415

    config = CullConfig(is_portrait=False)
    loop_in = _Stage2LoopInput(survivors=photo_paths, config=config)
    dashboard = _MinimalDashboard()
    _run_stage2_loop(loop_in, dashboard)
    return _extract_scores_from_calls(dashboard.stage2_calls)
