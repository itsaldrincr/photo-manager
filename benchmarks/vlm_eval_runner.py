"""Subprocess runner: score one eval set with one VLM, emit JSON to stdout.

Runs in its own process so model memory is fully returned to the OS on exit
(OOM safety on low-RAM hosts). Invoked by benchmarks/harness.py.

Usage:
    python3 benchmarks/vlm_eval_runner.py <model_alias> <eval_dir> <out_json>
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from pydantic import BaseModel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

JPEG_SUFFIXES: tuple[str, str] = (".jpg", ".jpeg")


class PhotoVerdict(BaseModel):
    """One photo's VLM judgment plus timing."""

    name: str
    is_keeper: bool | None
    confidence: float
    latency_seconds: float
    is_parse_error: bool


class RunnerResult(BaseModel):
    """Full eval-run output for one model."""

    model_alias: str
    verdicts: list[PhotoVerdict]
    load_seconds: float
    peak_metal_gb: float


class RunnerArgs(BaseModel):
    """Parsed CLI arguments."""

    model_alias: str
    eval_dir: Path
    out_path: Path


def _parse_args(argv: list[str]) -> RunnerArgs:
    """Validate and bundle the three positional arguments."""
    if len(argv) != 4:
        raise SystemExit("usage: vlm_eval_runner.py <model_alias> <eval_dir> <out_json>")
    return RunnerArgs(
        model_alias=argv[1], eval_dir=Path(argv[2]), out_path=Path(argv[3])
    )


def _list_eval_photos(eval_dir: Path) -> list[Path]:
    """Return sorted JPEG paths directly inside the eval dir."""
    photos = [
        p for p in sorted(eval_dir.iterdir())
        if p.is_file() and p.suffix.lower() in JPEG_SUFFIXES
    ]
    if not photos:
        raise SystemExit(f"no JPEGs found in {eval_dir}")
    return photos


def _score_one(photo: Path, session: object) -> PhotoVerdict:
    """Score a single photo through the production stage-3 path."""
    from cull.stage3.prompt import PromptContext  # noqa: PLC0415
    from cull.stage3.vlm_scoring import (  # noqa: PLC0415
        VlmRequest,
        VlmScoreCallInput,
        score_photo,
    )

    started = time.monotonic()
    request = VlmRequest(image_path=photo, context=PromptContext())
    result = score_photo(VlmScoreCallInput(request=request, session=session))
    return PhotoVerdict(
        name=photo.name,
        is_keeper=result.is_keeper,
        confidence=result.confidence,
        latency_seconds=time.monotonic() - started,
        is_parse_error=result.is_parse_error,
    )


def _peak_metal_gb() -> float:
    """Return peak Metal memory in GB, 0.0 if MLX is unavailable."""
    try:
        import mlx.core as mx  # noqa: PLC0415

        return round(mx.get_peak_memory() / 1e9, 2)
    except ImportError:
        logger.warning("mlx.core unavailable; peak memory unknown")
        return 0.0


def _run_eval(args: RunnerArgs) -> RunnerResult:
    """Load the model, score every eval photo, collect metrics."""
    from cull.vlm_session import vlm_session  # noqa: PLC0415

    photos = _list_eval_photos(args.eval_dir)
    load_started = time.monotonic()
    with vlm_session(args.model_alias) as session:
        load_seconds = time.monotonic() - load_started
        verdicts = [_score_one(photo, session) for photo in photos]
    return RunnerResult(
        model_alias=args.model_alias,
        verdicts=verdicts,
        load_seconds=round(load_seconds, 2),
        peak_metal_gb=_peak_metal_gb(),
    )


def main() -> None:
    """Entry point: run eval and write JSON to the output path."""
    args = _parse_args(sys.argv)
    result = _run_eval(args)
    args.out_path.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
