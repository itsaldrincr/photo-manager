"""CLI script to capture P4-lite aesthetic golden baseline scores.

Usage:
    python tests/_capture_p4lite_baseline.py [--corpus PATH]

Writes tests/fixtures/p4lite_baseline_<corpus_basename>.json.
Operator-discretion only — do NOT run via automated pytest.
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
from cull.stage2.aesthetic import score_aesthetic_batch
from tests._golden_helpers import (
    MANIFEST_FILENAME,
    baseline_filename,
    compute_corpus_fingerprint,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Capture P4-lite aesthetic baseline.")
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


def _score_one_photo(path: Path) -> float:
    """Load a PIL image and return its aesthetic score via score_aesthetic_batch."""
    from PIL import Image  # noqa: PLC0415

    pil_image = Image.open(path)
    scores = score_aesthetic_batch([pil_image])
    return scores[0]


def _collect_scores(corpus_dir: Path) -> dict[str, dict[str, float]]:
    """Score all corpus photos and return {filename: {"aesthetic": score}} map."""
    photo_paths = _load_photo_paths(corpus_dir)
    logger.info("Scoring %d photos via score_aesthetic_batch", len(photo_paths))
    return {
        path.name: {"aesthetic": _score_one_photo(path)}
        for path in photo_paths
    }


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
    out_path = FIXTURES_DIR / baseline_filename("p4lite", corpus_dir)
    with out_path.open("w") as fh:
        json.dump(baseline, fh, indent=2)
    logger.info("Baseline written to %s", out_path)
    return out_path


def main() -> None:
    """Entry point: score all corpus photos via aesthetic batch, write baseline."""
    args = _parse_args()
    corpus_dir: Path = args.corpus
    logger.info("Capturing P4-lite baseline for corpus: %s", corpus_dir)
    scores = _collect_scores(corpus_dir)
    _write_baseline(corpus_dir, scores)


if __name__ == "__main__":
    main()
