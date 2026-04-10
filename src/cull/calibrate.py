"""Corpus calibration runner — bakes Stage 2 + aesthetic golden baselines.

Operator-discretion utility called via `cull --calibrate <corpus_dir>`. Reads
the corpus manifest, scores every photo with the real ML pipeline, and writes
two baseline JSONs to `tests/fixtures/p*_baseline_<corpus_name>.json` so future
non-regression tests have a reference point.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

MANIFEST_FILENAME: str = "manifest.json"
BASELINE_DIR_PARTS: tuple[str, str] = ("tests", "fixtures")
P1_BASELINE_KIND: str = "p1"
P4LITE_BASELINE_KIND: str = "p4lite"
JSON_INDENT: int = 2


class CalibrationRequest(BaseModel):
    """Input for a calibration run."""

    corpus_dir: Path


class CalibrationResult(BaseModel):
    """Output of a calibration run with paths to the freshly-baked baselines."""

    corpus_name: str
    photo_count: int
    p1_baseline_path: Path
    p4lite_baseline_path: Path
    duration_seconds: float


class CalibrationError(Exception):
    """Raised when the corpus dir or manifest is missing or malformed."""


def _resolve_fixtures_dir() -> Path:
    """Return absolute path to tests/fixtures relative to the cull package."""
    root = Path(__file__).resolve().parent.parent.parent
    return root.joinpath(*BASELINE_DIR_PARTS)


def _validate_corpus(corpus_dir: Path) -> Path:
    """Verify corpus dir + manifest.json exist and return manifest path."""
    if not corpus_dir.is_dir():
        raise CalibrationError(f"corpus dir not found: {corpus_dir}")
    manifest_path = corpus_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise CalibrationError(f"manifest.json missing in {corpus_dir}")
    return manifest_path


def _load_photo_paths(manifest_path: Path) -> list[Path]:
    """Return sorted list of photo paths from a corpus manifest."""
    manifest = json.loads(manifest_path.read_text())
    entries = [e for e in manifest["files"] if e["name"] != MANIFEST_FILENAME]
    sorted_entries = sorted(entries, key=lambda e: e["name"])
    corpus_dir = manifest_path.parent
    return [corpus_dir / e["name"] for e in sorted_entries]


def _get_git_sha() -> str | None:
    """Return current git HEAD sha or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Could not resolve git HEAD sha: %s", exc)
        return None


def _ensure_tests_importable() -> None:
    """Append project root to sys.path so `tests._golden_helpers` resolves."""
    import sys  # noqa: PLC0415

    project_root = Path(__file__).resolve().parent.parent.parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _score_p1(photo_paths: list[Path]) -> dict[str, dict[str, float]]:
    """Score photos via the batched Stage 2 pipeline path; returns per-file metrics."""
    _ensure_tests_importable()
    from tests._golden_helpers import score_corpus_via_batched_path  # noqa: PLC0415

    return score_corpus_via_batched_path(photo_paths)


def _score_one_aesthetic(path: Path) -> float:
    """Run score_aesthetic_batch on a single PIL image and return its score."""
    from PIL import Image  # noqa: PLC0415
    from cull.stage2.aesthetic import score_aesthetic_batch  # noqa: PLC0415

    pil_image = Image.open(path)
    return score_aesthetic_batch([pil_image])[0]


def _score_p4lite(photo_paths: list[Path]) -> dict[str, dict[str, float]]:
    """Score photos via score_aesthetic_batch and return {filename: {aesthetic}} map."""
    return {p.name: {"aesthetic": _score_one_aesthetic(p)} for p in photo_paths}


class _BaselineWriteInput(BaseModel):
    """Bundle of fields for writing one baseline JSON file."""

    kind: str
    corpus_dir: Path
    scores: dict[str, dict[str, float]]


def _baseline_path(kind: str, corpus_name: str) -> Path:
    """Return absolute path for a baseline JSON of a given kind + corpus."""
    return _resolve_fixtures_dir() / f"{kind}_baseline_{corpus_name}.json"


def _build_baseline_payload(write_in: _BaselineWriteInput) -> dict:
    """Assemble the baseline JSON payload with metadata + scores."""
    return {
        "corpus_name": write_in.corpus_dir.name,
        "corpus_path": str(write_in.corpus_dir),
        "baseline_created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_git_sha": _get_git_sha(),
        "scores": write_in.scores,
    }


def _write_baseline(write_in: _BaselineWriteInput) -> Path:
    """Write a baseline JSON to tests/fixtures and return its absolute path."""
    out_path = _baseline_path(write_in.kind, write_in.corpus_dir.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_baseline_payload(write_in)
    out_path.write_text(json.dumps(payload, indent=JSON_INDENT))
    logger.info("Baseline written: %s", out_path)
    return out_path


def run_calibration(request: CalibrationRequest) -> CalibrationResult:
    """Bake p1 + p4lite baselines for the given corpus and return paths + counts."""
    start = time.monotonic()
    manifest_path = _validate_corpus(request.corpus_dir)
    photo_paths = _load_photo_paths(manifest_path)
    logger.info("Calibrating %d photos in %s", len(photo_paths), request.corpus_dir)
    p1_scores = _score_p1(photo_paths)
    p4lite_scores = _score_p4lite(photo_paths)
    p1_path = _write_baseline(_BaselineWriteInput(
        kind=P1_BASELINE_KIND, corpus_dir=request.corpus_dir, scores=p1_scores,
    ))
    p4lite_path = _write_baseline(_BaselineWriteInput(
        kind=P4LITE_BASELINE_KIND, corpus_dir=request.corpus_dir, scores=p4lite_scores,
    ))
    return CalibrationResult(
        corpus_name=request.corpus_dir.name, photo_count=len(photo_paths),
        p1_baseline_path=p1_path, p4lite_baseline_path=p4lite_path,
        duration_seconds=time.monotonic() - start,
    )
