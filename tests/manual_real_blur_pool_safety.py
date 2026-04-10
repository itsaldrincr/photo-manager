"""RUN MANUALLY ONLY. Loads real cv2 + blur_detector for 20 photos.

CPU-bound, low panic risk, but per project policy do not run via pytest.
User runs `python tests/manual_real_blur_pool_safety.py` when they want a
real-pool sanity check. Pass --corpus <path> to override the default corpus.

Usage:
    python tests/manual_real_blur_pool_safety.py
    python tests/manual_real_blur_pool_safety.py --corpus /path/to/corpus
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import multiprocessing
import sys
from pathlib import Path

import cull.config as _config
from cull.config import CullConfig
from cull.stage1.worker import Stage1WorkerResult, assess_one

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SUBSET_SIZE: int = 20
POOL_WORKER_COUNT: int = 2
MANIFEST_FILENAME: str = "manifest.json"


def _resolve_corpus(raw: str | None) -> Path:
    """Return corpus path from CLI arg or config default."""
    if raw is not None:
        return Path(raw)
    return _config.PERF_CORPUS_PATH


def _pick_photos(corpus_path: Path) -> list[Path]:
    """Return sorted 20-photo subset from corpus directory."""
    manifest = json.loads((corpus_path / MANIFEST_FILENAME).read_text())
    names = sorted(
        e["name"] for e in manifest["files"] if e["name"] != MANIFEST_FILENAME
    )
    return [corpus_path / name for name in names[:SUBSET_SIZE]]


def _run_sequential(paths: list[Path], config: CullConfig) -> list[Stage1WorkerResult]:
    """Run assess_one sequentially and return results in input order."""
    return [assess_one(p, config) for p in paths]


def _run_pool(paths: list[Path], config: CullConfig) -> list[Stage1WorkerResult]:
    """Run assess_one via spawn-context Pool; return sorted by path."""
    ctx = multiprocessing.get_context("spawn")
    worker_fn = functools.partial(assess_one, config=config)
    with ctx.Pool(processes=POOL_WORKER_COUNT) as pool:
        results = list(pool.imap_unordered(worker_fn, paths))
    return sorted(results, key=lambda r: r.image_path)


def _compare(sequential: list[Stage1WorkerResult], pool_results: list[Stage1WorkerResult]) -> list[str]:
    """Return list of diff strings for any mismatches."""
    diffs: list[str] = []
    for seq_r, pool_r in zip(sequential, pool_results):
        if seq_r != pool_r:
            diffs.append(f"MISMATCH {seq_r.image_path.name}: seq={seq_r!r} pool={pool_r!r}")
    return diffs


def _parse_corpus_arg() -> Path:
    """Parse --corpus CLI argument and return resolved corpus path."""
    parser = argparse.ArgumentParser(description="Real blur pool safety check.")
    parser.add_argument("--corpus", default=None, help="Path to corpus directory.")
    args = parser.parse_args()
    return _resolve_corpus(args.corpus)


def _report_diffs(diffs: list[str]) -> None:
    """Log all mismatch diffs and exit with failure code."""
    logger.warning("FAIL: %d mismatch(es)", len(diffs))
    for diff in diffs:
        logger.warning("  %s", diff)
    sys.exit(1)


def main() -> None:
    """Run real-model pool safety check and log PASS or FAIL."""
    corpus_path = _parse_corpus_arg()
    if not corpus_path.exists():
        logger.error("FAIL: corpus not found: %s", corpus_path)
        sys.exit(1)
    paths = _pick_photos(corpus_path)
    config = CullConfig()
    sequential = sorted(_run_sequential(paths, config), key=lambda r: r.image_path)
    pool_results = _run_pool(paths, config)
    diffs = _compare(sequential, pool_results)
    if diffs:
        _report_diffs(diffs)
    logger.info("PASS: %d photos — sequential == pool", len(sequential))


if __name__ == "__main__":
    main()
