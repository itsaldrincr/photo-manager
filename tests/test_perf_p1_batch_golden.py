"""Golden-baseline regression test for Stage 2 P1 scoring.

Verifies that the Stage 2 scorers (TOPIQ, CLIP-IQA+, LAION Aesthetics) produce
scores matching the committed baseline within SCORE_TOLERANCE for the test corpus.

Requires:
    tests/fixtures/p1_baseline_<corpus_name>.json  — produced by _capture_p1_baseline.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests._golden_helpers import (
    MANIFEST_FILENAME,
    SCORE_TOLERANCE,
    Divergence,
    baseline_filename,
    compute_corpus_fingerprint,
    format_divergence_report,
    group_divergences_by_category,
    score_corpus_via_batched_path,
)

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"
ABS_TOL: float = SCORE_TOLERANCE
SCORE_FIELDS: tuple[str, ...] = ("topiq", "laion_aesthetic", "clipiqa")


def _load_baseline(baseline_path: Path, corpus_path: Path) -> dict:
    """Load baseline JSON, failing loudly if it does not exist."""
    if not baseline_path.exists():
        pytest.fail(
            f"No baseline for corpus {corpus_path.name!r} at {baseline_path}.\n"
            f"Generate via: python tests/_capture_p1_baseline.py --corpus {corpus_path}"
        )
    with baseline_path.open() as fh:
        return json.load(fh)


def _check_fingerprint(baseline: dict, corpus_path: Path) -> None:
    """Fail loudly if the live corpus fingerprint differs from the baseline."""
    stored = baseline["corpus_fingerprint"]
    recomputed = compute_corpus_fingerprint(corpus_path / MANIFEST_FILENAME)
    if stored == recomputed:
        return
    pytest.fail(
        f"Baseline/corpus fingerprint drift detected for corpus {corpus_path.name!r}\n"
        f"  stored:     {stored}\n"
        f"  recomputed: {recomputed}\n"
        f"The manifest was regenerated without updating the baseline, or the corpus files "
        f"were edited without updating the manifest. Regenerate the baseline via:\n"
        f"  python tests/_capture_p1_baseline.py --corpus {corpus_path}\n"
        f"Do NOT modify the baseline's fingerprint to make the test pass."
    )


def _collect_divergences(baseline: dict, current: dict) -> list[Divergence]:
    """Compare baseline scores against current run scores; return all divergences."""
    divergences: list[Divergence] = []
    for name, expected_scores in baseline["scores"].items():
        current_scores = current.get(name, {})
        for field in SCORE_FIELDS:
            expected = expected_scores[field]
            actual = current_scores.get(field, float("nan"))
            if not math.isclose(expected, actual, rel_tol=0, abs_tol=ABS_TOL):
                divergences.append(
                    Divergence(
                        filename=name,
                        field=field,
                        expected=expected,
                        current=actual,
                        abs_diff=abs(actual - expected),
                    )
                )
    return divergences


def _build_total_by_category(manifest: dict) -> dict[str, int]:
    """Count total files per category from the manifest."""
    totals: dict[str, int] = {}
    for entry in manifest["files"]:
        if entry["name"] == MANIFEST_FILENAME:
            continue
        cat = entry.get("category", "unknown")
        totals[cat] = totals.get(cat, 0) + 1
    return totals


def _get_photo_paths(corpus_path: Path, baseline: dict) -> list[Path]:
    """Return sorted list of photo paths from baseline score keys."""
    return sorted([corpus_path / name for name in baseline["scores"]])


def test_stage2_matches_baseline(corpus_path: Path, corpus_manifest: dict) -> None:
    """Verify Stage 2 scores match the committed baseline for the corpus."""
    baseline_path = FIXTURES_DIR / baseline_filename("p1", corpus_path)
    baseline = _load_baseline(baseline_path, corpus_path)
    _check_fingerprint(baseline, corpus_path)
    photo_paths = _get_photo_paths(corpus_path, baseline)
    current = score_corpus_via_batched_path(photo_paths)
    divergences = _collect_divergences(baseline, current)
    if not divergences:
        return
    grouped = group_divergences_by_category(divergences, corpus_manifest)
    total_by_category = _build_total_by_category(corpus_manifest)
    report = format_divergence_report(grouped, total_by_category)
    pytest.fail(report)
