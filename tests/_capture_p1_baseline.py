"""CLI script to capture Stage 2 P1 golden baseline scores.

Usage:
    python tests/_capture_p1_baseline.py [--corpus PATH]

Writes tests/fixtures/p1_baseline_<corpus_basename>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cull.config as _config
from tests._golden_helpers import (
    MANIFEST_FILENAME,
    baseline_filename,
    compute_corpus_fingerprint,
    score_corpus_via_batched_path,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Capture Stage 2 P1 baseline.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_config.PERF_CORPUS_PATH,
        help="Path to corpus directory.",
    )
    return parser.parse_args()


def _get_git_sha() -> str | None:
    """Return current git HEAD sha or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        logger.warning("Could not resolve git HEAD sha: %s", exc)
        return None


def _load_photo_paths(corpus_dir: Path) -> list[Path]:
    """Return sorted list of non-manifest photo paths from the corpus."""
    manifest_path = corpus_dir / MANIFEST_FILENAME
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    entries = [e for e in manifest["files"] if e["name"] != MANIFEST_FILENAME]
    return [corpus_dir / e["name"] for e in sorted(entries, key=lambda e: e["name"])]


def _collect_scores(corpus_dir: Path) -> dict[str, dict]:
    """Score all corpus photos via the batched pipeline path."""
    photo_paths = _load_photo_paths(corpus_dir)
    logger.info("Scoring %d photos via batched path", len(photo_paths))
    return score_corpus_via_batched_path(photo_paths)


def _write_baseline(corpus_dir: Path, scores: dict) -> Path:
    """Write baseline JSON to fixtures dir and return the output path."""
    fingerprint = compute_corpus_fingerprint(corpus_dir / MANIFEST_FILENAME)
    git_sha = _get_git_sha()
    baseline = {
        "corpus_name": corpus_dir.name,
        "corpus_fingerprint": fingerprint,
        "corpus_path": str(corpus_dir),
        "baseline_created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_git_sha": git_sha,
        "scores": scores,
    }
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURES_DIR / baseline_filename("p1", corpus_dir)
    with out_path.open("w") as fh:
        json.dump(baseline, fh, indent=2)
    logger.info("Baseline written to %s", out_path)
    return out_path


def main() -> None:
    """Entry point: score all corpus photos via batched path, write baseline."""
    args = _parse_args()
    corpus_dir: Path = args.corpus
    logger.info("Capturing baseline for corpus: %s", corpus_dir)
    scores = _collect_scores(corpus_dir)
    _write_baseline(corpus_dir, scores)


if __name__ == "__main__":
    main()
