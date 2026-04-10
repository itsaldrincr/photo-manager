"""Mocked plumbing test for Stage 1 scorer chain.

Verifies that each photo is processed exactly once, results come back
in the correct order, and per-path mock fingerprints are consistent.
No real models are loaded — all scorers are replaced by mock_scorers fixture.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

import cull.stage1.blur as _blur_mod
import cull.stage1.exposure as _exposure_mod
import cull.stage1.noise as _noise_mod

from cull.config import CullConfig
from cull.stage1.blur import BlurResult
from cull.stage1.exposure import ExposureResult
from cull.stage1.noise import NoiseResult
from tests._mock_scorers import (
    mock_assess_blur,
    mock_assess_exposure,
    mock_assess_noise,
)

logger = logging.getLogger(__name__)

SUBSET_SIZE: int = 16
MANIFEST_FILENAME: str = "manifest.json"


class _FingerprintCheckInput(BaseModel):
    """Input for fingerprint check validation."""

    corpus_path: Path
    photo_names: list[str]
    results: list[tuple[BlurResult, ExposureResult, NoiseResult]]

    class Config:
        """Allow Path and custom types in Pydantic."""

        arbitrary_types_allowed = True


def _pick_photos(corpus_manifest: dict) -> list[str]:
    """Return a deterministic 16-photo subset from manifest (sorted by name)."""
    entries = [
        e["name"]
        for e in corpus_manifest["files"]
        if e["name"] != MANIFEST_FILENAME
    ]
    return sorted(entries)[:SUBSET_SIZE]


def _score_photo(corpus_path: Path, name: str) -> tuple[BlurResult, ExposureResult, NoiseResult]:
    """Run all three Stage 1 scorers on one photo via module attributes."""
    path = corpus_path / name
    config = CullConfig(is_portrait=False)
    blur = _blur_mod.assess_blur(path, config)
    exposure = _exposure_mod.assess_exposure(path)
    noise = _noise_mod.assess_noise(path)
    return blur, exposure, noise


def _score_corpus_subset(
    corpus_path: Path, photo_names: list[str]
) -> list[tuple[BlurResult, ExposureResult, NoiseResult]]:
    """Score all photos sequentially; return results in input order."""
    return [_score_photo(corpus_path, name) for name in photo_names]


def _expected_result(
    corpus_path: Path, name: str
) -> tuple[BlurResult, ExposureResult, NoiseResult]:
    """Compute expected mock output for one photo (independent call)."""
    path = corpus_path / name
    config = CullConfig(is_portrait=False)
    return (
        mock_assess_blur(path, config),
        mock_assess_exposure(path),
        mock_assess_noise(path),
    )


def _assert_all_distinct(results: list[tuple]) -> None:
    """Assert all noise_scores are distinct — catches alignment/swap bugs."""
    scores = [r[2].noise_score for r in results]
    assert len(scores) == len(set(scores)), (
        f"Duplicate noise_score values detected — alignment bug: {scores}"
    )


def _assert_fingerprint_match(check_input: _FingerprintCheckInput) -> None:
    """Assert pipeline results match independently-computed mock results."""
    for name, (blur, exposure, noise) in zip(check_input.photo_names, check_input.results):
        exp_blur, exp_exposure, exp_noise = _expected_result(check_input.corpus_path, name)
        assert blur == exp_blur, f"{name} blur mismatch: {blur!r} != {exp_blur!r}"
        assert exposure == exp_exposure, f"{name} exposure mismatch"
        assert noise == exp_noise, f"{name} noise mismatch"


def test_stage1_scorer_plumbing(
    corpus_path: Path, corpus_manifest: dict, mock_scorers: None
) -> None:
    """Verify Stage 1 plumbing: order, completeness, and mock fingerprint match."""
    photo_names = _pick_photos(corpus_manifest)
    assert len(photo_names) == SUBSET_SIZE, (
        f"Expected {SUBSET_SIZE} photos, got {len(photo_names)}"
    )

    results = _score_corpus_subset(corpus_path, photo_names)

    assert len(results) == SUBSET_SIZE, (
        f"Result count mismatch: expected {SUBSET_SIZE}, got {len(results)}"
    )

    _assert_all_distinct(results)
    check_input = _FingerprintCheckInput(
        corpus_path=corpus_path,
        photo_names=photo_names,
        results=results,
    )
    _assert_fingerprint_match(check_input)
