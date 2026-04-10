"""Mocked plumbing test: Stage1WorkerResult survives spawn-Pool pickling and IPC.

Verifies that sequential and multiprocessing.Pool(spawn) paths produce identical
Stage1WorkerResult values — proving pickling, IPC, and reassembly correctness.
No real cv2/blur_detector models are loaded; all scorer calls go through mocks.
"""

from __future__ import annotations

import functools
import multiprocessing
from pathlib import Path

from cull.config import CullConfig
from cull.stage1.worker import Stage1WorkerResult
from tests._mock_scorers import (
    mock_assess_blur,
    mock_assess_exposure,
    mock_assess_geometry,
    mock_assess_noise,
)

POOL_SUBSET_SIZE: int = 20
POOL_WORKER_COUNT: int = 2
MANIFEST_FILENAME: str = "manifest.json"


def _pick_pool_photos(corpus_manifest: dict) -> list[str]:
    """Return a deterministic 20-photo subset from manifest sorted by name."""
    entries = [
        e["name"]
        for e in corpus_manifest["files"]
        if e["name"] != MANIFEST_FILENAME
    ]
    return sorted(entries)[:POOL_SUBSET_SIZE]


def _build_result(image_path: Path, config: CullConfig) -> Stage1WorkerResult:
    """Build Stage1WorkerResult via mock scorers — safe to call in spawned workers."""
    blur = mock_assess_blur(image_path, config)
    exposure = mock_assess_exposure(image_path)
    noise = mock_assess_noise(image_path)
    geometry = mock_assess_geometry(image_path)
    return Stage1WorkerResult(
        image_path=image_path,
        blur=blur,
        exposure=exposure,
        noise=noise,
        geometry=geometry,
    )


def _run_sequential(paths: list[Path], config: CullConfig) -> list[Stage1WorkerResult]:
    """Run _build_result sequentially and return results in input order."""
    return [_build_result(p, config) for p in paths]


def _run_pool(paths: list[Path], config: CullConfig) -> list[Stage1WorkerResult]:
    """Run _build_result via spawn-context Pool and return results sorted by path."""
    ctx = multiprocessing.get_context("spawn")
    worker_fn = functools.partial(_build_result, config=config)
    with ctx.Pool(processes=POOL_WORKER_COUNT) as pool:
        results = list(pool.imap_unordered(worker_fn, paths))
    return sorted(results, key=lambda r: r.image_path)


def _sorted_results(results: list[Stage1WorkerResult]) -> list[Stage1WorkerResult]:
    """Return results sorted by image_path for deterministic comparison."""
    return sorted(results, key=lambda r: r.image_path)


def test_pool_deep_equality(
    corpus_path: Path, corpus_manifest: dict, mock_scorers: None
) -> None:
    """Sequential and Pool paths produce identical Stage1WorkerResult per photo."""
    photo_names = _pick_pool_photos(corpus_manifest)
    assert len(photo_names) == POOL_SUBSET_SIZE, (
        f"Expected {POOL_SUBSET_SIZE} photos, got {len(photo_names)}"
    )
    paths = [corpus_path / name for name in photo_names]
    config = CullConfig()

    sequential = _sorted_results(_run_sequential(paths, config))
    pool_results = _run_pool(paths, config)

    assert len(pool_results) == POOL_SUBSET_SIZE, (
        f"Pool result count mismatch: expected {POOL_SUBSET_SIZE}, got {len(pool_results)}"
    )
    for seq_r, pool_r in zip(sequential, pool_results):
        assert seq_r == pool_r, (
            f"Result mismatch for {seq_r.image_path.name}:\n"
            f"  sequential: {seq_r!r}\n"
            f"  pool:       {pool_r!r}"
        )
