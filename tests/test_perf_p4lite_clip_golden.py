"""Golden-baseline and mocked plumbing tests for P4-lite aesthetic scoring.

test_aesthetic_matches_baseline — real-model regression gate. Requires a committed
    baseline at tests/fixtures/p4lite_baseline_<corpus_name>.json with real scores.
    Skipped automatically if the baseline has not yet been populated by the operator.
test_aesthetic_scorer_plumbing — mocked plumbing test (no real models). Safe to run
    via automated pytest at all times (~1 second).

Populate the baseline by running:
    python tests/_capture_p4lite_baseline.py --corpus <path>
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from PIL import Image

import cull.stage2.aesthetic as _aesthetic_mod
from tests._golden_helpers import (
    MANIFEST_FILENAME,
    Divergence,
    baseline_filename,
    compute_corpus_fingerprint,
    format_divergence_report,
    group_divergences_by_category,
)
from tests._mock_scorers import _int_to_mock_score, _pil_fingerprint

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"
ABS_TOL: float = 1e-9
SCORE_FIELD: str = "aesthetic"
SUBSET_SIZE: int = 16


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_baseline(baseline_path: Path, corpus_path: Path) -> dict:
    """Load baseline JSON, failing loudly if it does not exist."""
    if not baseline_path.exists():
        pytest.fail(
            f"No baseline for corpus {corpus_path.name!r} at {baseline_path}.\n"
            f"Generate via: python tests/_capture_p4lite_baseline.py"
            f" --corpus {corpus_path}"
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
        f"Baseline/corpus fingerprint drift for corpus {corpus_path.name!r}\n"
        f"  stored:     {stored}\n"
        f"  recomputed: {recomputed}\n"
        f"Regenerate via: python tests/_capture_p4lite_baseline.py"
        f" --corpus {corpus_path}"
    )


def _is_baseline_populated(baseline: dict) -> bool:
    """Return True if all baseline scores are non-null floats."""
    return all(
        v.get(SCORE_FIELD) is not None
        for v in baseline["scores"].values()
    )


def _score_all_photos(corpus_path: Path, names: list[str]) -> dict[str, float]:
    """Score all photos via real aesthetic scorer; return {name: score}."""
    result: dict[str, float] = {}
    for name in names:
        pil_image = Image.open(corpus_path / name)
        scores = _aesthetic_mod.score_aesthetic_batch([pil_image])
        result[name] = scores[0]
    return result


def _collect_divergences(baseline: dict, current: dict) -> list[Divergence]:
    """Compare baseline scores against current scores; return all divergences."""
    divergences: list[Divergence] = []
    for name, expected_scores in baseline["scores"].items():
        actual = current.get(name, float("nan"))
        expected = float(expected_scores[SCORE_FIELD])
        if not math.isclose(expected, actual, rel_tol=0, abs_tol=ABS_TOL):
            divergences.append(
                Divergence(
                    filename=name,
                    field=SCORE_FIELD,
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


# ---------------------------------------------------------------------------
# Real-model regression gate
# ---------------------------------------------------------------------------


def test_aesthetic_matches_baseline(
    corpus_path: Path, corpus_manifest: dict
) -> None:
    """Verify aesthetic scores match the committed p4lite baseline for the corpus."""
    baseline_path = FIXTURES_DIR / baseline_filename("p4lite", corpus_path)
    baseline = _load_baseline(baseline_path, corpus_path)
    _check_fingerprint(baseline, corpus_path)
    if not _is_baseline_populated(baseline):
        pytest.skip(
            f"Baseline scores are not yet populated for corpus {corpus_path.name!r}.\n"
            f"Run: python tests/_capture_p4lite_baseline.py --corpus {corpus_path}"
        )
    names = list(baseline["scores"].keys())
    current = _score_all_photos(corpus_path, names)
    divergences = _collect_divergences(baseline, current)
    if not divergences:
        return
    grouped = group_divergences_by_category(divergences, corpus_manifest)
    total_by_category = _build_total_by_category(corpus_manifest)
    report = format_divergence_report(grouped, total_by_category)
    pytest.fail(report)


# ---------------------------------------------------------------------------
# Mocked plumbing test — safe for automated pytest, no real model loads
# ---------------------------------------------------------------------------


def _pick_photos(corpus_manifest: dict) -> list[str]:
    """Return a deterministic 16-photo subset from manifest (sorted by name)."""
    entries = [
        e["name"]
        for e in corpus_manifest["files"]
        if e["name"] != MANIFEST_FILENAME
    ]
    return sorted(entries)[:SUBSET_SIZE]


def _expected_mock_score(pil_image: object) -> float:
    """Compute the deterministic mock aesthetic score for one PIL image."""
    return _int_to_mock_score(_pil_fingerprint(pil_image), "aesthetic")


def test_aesthetic_scorer_plumbing(
    corpus_path: Path, corpus_manifest: dict, mock_scorers: None
) -> None:
    """Verify aesthetic scorer plumbing: order, count, and mock fingerprint match."""
    photo_names = _pick_photos(corpus_manifest)
    assert len(photo_names) == SUBSET_SIZE, (
        f"Expected {SUBSET_SIZE} photos, got {len(photo_names)}"
    )
    pil_images = [Image.open(corpus_path / name) for name in photo_names]
    scores = _aesthetic_mod.score_aesthetic_batch(pil_images)
    assert len(scores) == SUBSET_SIZE, (
        f"Score count mismatch: expected {SUBSET_SIZE}, got {len(scores)}"
    )
    for name, pil_img, score in zip(photo_names, pil_images, scores):
        expected = _expected_mock_score(pil_img)
        assert score == expected, (
            f"{name}: mock score mismatch: got {score}, expected {expected}"
        )
