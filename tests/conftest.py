"""Shared pytest fixtures for golden-baseline tests.

Exposes the --corpus flag and corpus_path / corpus_manifest fixtures
used by test_perf_p1_batch_golden, test_perf_p3_pool_golden,
and test_perf_p4lite_clip_golden.

Also exposes mock_scorers fixture for plumbing tests that must not load real ML models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cull.config as _config
from cull.config import CullConfig
from cull.stage1.worker import Stage1WorkerResult
from tests._mock_scorers import (
    mock_assess_blur,
    mock_assess_exposure,
    mock_assess_geometry,
    mock_assess_noise,
    mock_score_aesthetic,
    mock_score_aesthetic_batch,
    mock_score_clipiqa_batch,
    mock_score_topiq_batch,
)

MANIFEST_FILENAME: str = "manifest.json"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --corpus PATH option.

    Absolute path to a fixture corpus. Defaults to cull.config.PERF_CORPUS_PATH.
    The corpus directory's basename is used to key the baseline file
    (p1_baseline_<basename>.json, etc.).
    """
    parser.addoption(
        "--corpus",
        type=str,
        default=None,
        help="Absolute path to a fixture corpus directory.",
    )


def _resolve_corpus_path(request: pytest.FixtureRequest) -> Path:
    """Return corpus path from CLI option or config default."""
    raw = request.config.getoption("--corpus")
    if raw is not None:
        return Path(raw)
    return _config.PERF_CORPUS_PATH


def _assert_corpus_valid(corpus_path: Path) -> None:
    """Fail loudly if corpus directory or manifest.json is missing."""
    if not corpus_path.exists():
        pytest.fail(
            f"Corpus directory not found: {corpus_path}\n"
            f"Pass --corpus <path> or set PERF_CORPUS_PATH env var."
        )
    manifest = corpus_path / MANIFEST_FILENAME
    if not manifest.exists():
        pytest.fail(
            f"manifest.json not found in corpus: {corpus_path}\n"
            f"The corpus directory must contain a manifest.json file."
        )


@pytest.fixture(scope="session")
def corpus_path(request: pytest.FixtureRequest) -> Path:
    """Session-scoped corpus directory path, validated on access."""
    path = _resolve_corpus_path(request)
    _assert_corpus_valid(path)
    return path


@pytest.fixture(scope="session")
def corpus_manifest(corpus_path: Path) -> dict:
    """Session-scoped manifest.json loaded as a dict."""
    manifest_path = corpus_path / MANIFEST_FILENAME
    with manifest_path.open() as fh:
        return json.load(fh)


def _mock_assess_one(image_path: Path, config: CullConfig) -> Stage1WorkerResult:
    """Build a Stage1WorkerResult from the four deterministic mocks (no real decode)."""
    return Stage1WorkerResult(
        image_path=image_path,
        blur=mock_assess_blur(image_path, config),
        exposure=mock_assess_exposure(image_path),
        noise=mock_assess_noise(image_path),
        geometry=mock_assess_geometry(image_path),
    )


def _patch_stage1_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch Stage 1 assess functions in source modules and the worker's entry point."""
    monkeypatch.setattr("cull.stage1.blur.assess_blur", mock_assess_blur)
    monkeypatch.setattr("cull.stage1.exposure.assess_exposure", mock_assess_exposure)
    monkeypatch.setattr("cull.stage1.noise.assess_noise", mock_assess_noise)
    monkeypatch.setattr("cull.stage1.geometry.assess_geometry", mock_assess_geometry)
    monkeypatch.setattr("cull.stage1.worker.assess_one", _mock_assess_one)


@pytest.fixture
def mock_scorers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch all scorer entry points with deterministic mock implementations."""
    monkeypatch.setattr("cull.stage2.iqa.score_topiq_batch", mock_score_topiq_batch)
    monkeypatch.setattr("cull.stage2.iqa.score_clipiqa_batch", mock_score_clipiqa_batch)
    monkeypatch.setattr("cull.stage2.aesthetic.score_aesthetic_batch", mock_score_aesthetic_batch)
    monkeypatch.setattr("cull.stage2.aesthetic.score_aesthetic", mock_score_aesthetic)
    monkeypatch.setattr("cull.pipeline.score_topiq_batch", mock_score_topiq_batch)
    monkeypatch.setattr("cull.pipeline.score_clipiqa_batch", mock_score_clipiqa_batch)
    monkeypatch.setattr("cull.pipeline.score_aesthetic_batch", mock_score_aesthetic_batch)
    _patch_stage1_workers(monkeypatch)
