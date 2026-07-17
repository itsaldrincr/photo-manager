"""Orchestrator: benchmark MediaPipe blendshapes vs DeepFace emotion for stage-2 portrait.

Runs each model in its own subprocess (one heavy model resident at a time),
merges the two stage outputs, computes agreement on the decision-relevant
signal (expression bucket forwarded into cull's stage3/4 VLM prompts), and
writes a durable swap/no-swap report.

Usage:
    python3 benchmarks/portrait_eval.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from portrait_eval_models import (
    DeepfaceStageOutput,
    ExpressionBucket,
    MediapipeStageOutput,
    SWAP_AGREEMENT_MIN,
    SWAP_RSS_SAVINGS_MIN_MB,
    SWAP_SPEEDUP_MIN,
    list_image_paths,
)

BENCH_DIR: Path = Path(__file__).resolve().parent
EVAL_SET_DIR: Path = BENCH_DIR / "eval_set"
EXTRA_PORTRAIT_DIR: Path = Path("/Users/alrelador/Desktop/cull-test")
RUNS_DIR: Path = BENCH_DIR / "runs"
REPORT_PATH: Path = RUNS_DIR / "portrait_eval_report.md"
RUNNER_PATH: Path = BENCH_DIR / "portrait_eval_runner.py"

EXTRA_PORTRAIT_MAX: int = 30
RUNNER_TIMEOUT_SECONDS: int = 1800


class MergedReading(BaseModel):
    """One photo's paired MediaPipe and DeepFace readings."""

    name: str
    mediapipe_bucket: ExpressionBucket
    deepface_bucket: ExpressionBucket
    deepface_label: str
    mediapipe_latency_seconds: float
    deepface_latency_seconds: float
    ear_eyes_closed: bool | None
    blendshape_eyes_closed: bool | None
    agree: bool


class EvalReport(BaseModel):
    """Full comparison summary for the swap/no-swap decision."""

    photo_count_total: int
    face_photo_count: int
    agreement_rate: float
    eyes_closed_cross_check_rate: float
    mediapipe_timing: dict
    deepface_timing: dict
    mediapipe_mean_latency_seconds: float
    deepface_mean_latency_seconds: float
    speedup_ratio: float
    disagreements: list[MergedReading]
    recommendation: str
    reasoning: list[str]


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gather_candidate_photos() -> list[Path]:
    """Return the eval_set plus up to EXTRA_PORTRAIT_MAX non-duplicate extra photos."""
    eval_photos = list_image_paths(EVAL_SET_DIR)
    known_names = {p.name for p in eval_photos}
    extra_all = [p for p in list_image_paths(EXTRA_PORTRAIT_DIR) if p.name not in known_names]
    return eval_photos + extra_all[:EXTRA_PORTRAIT_MAX]


class _StageInvocation(BaseModel):
    """Inputs for one subprocess stage run."""

    stage: str
    input_path: Path
    output_path: Path


def _run_stage(invocation: _StageInvocation) -> None:
    """Invoke portrait_eval_runner.py for one stage in a fresh subprocess."""
    argv = [
        sys.executable, str(RUNNER_PATH), invocation.stage,
        str(invocation.input_path), str(invocation.output_path),
    ]
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=RUNNER_TIMEOUT_SECONDS,
        check=False, cwd=BENCH_DIR,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{invocation.stage} stage failed: {completed.stderr.strip()[-2000:]}")


def _merge_readings(mp_out: MediapipeStageOutput, df_out: DeepfaceStageOutput) -> list[MergedReading]:
    """Pair MediaPipe and DeepFace readings by filename for face-bearing photos."""
    mp_by_name = {r.name: r for r in mp_out.readings if r.has_face}
    df_by_name = {r.name: r for r in df_out.readings}
    return [_merge_one(mp_by_name[name], df_by_name[name]) for name in sorted(mp_by_name) if name in df_by_name]


def _merge_one(mp_reading: object, df_reading: object) -> MergedReading:
    """Build one MergedReading from a matched MediaPipe/DeepFace pair."""
    return MergedReading(
        name=mp_reading.name,  # type: ignore[attr-defined]
        mediapipe_bucket=mp_reading.expression_bucket,  # type: ignore[attr-defined]
        deepface_bucket=df_reading.expression_bucket,  # type: ignore[attr-defined]
        deepface_label=df_reading.dominant_emotion,  # type: ignore[attr-defined]
        mediapipe_latency_seconds=mp_reading.latency_seconds,  # type: ignore[attr-defined]
        deepface_latency_seconds=df_reading.latency_seconds,  # type: ignore[attr-defined]
        ear_eyes_closed=mp_reading.ear_eyes_closed,  # type: ignore[attr-defined]
        blendshape_eyes_closed=mp_reading.blendshape_eyes_closed,  # type: ignore[attr-defined]
        agree=mp_reading.expression_bucket == df_reading.expression_bucket,  # type: ignore[attr-defined]
    )


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""
    return round(sum(values) / len(values), 4) if values else 0.0


def _recommend(report_inputs: "_RecommendationInputs") -> tuple[str, list[str]]:
    """Apply the swap/no-swap rubric and return (recommendation, reasoning)."""
    reasons = [
        f"agreement {report_inputs.agreement_rate:.1%} (threshold {SWAP_AGREEMENT_MIN:.0%})",
        f"speedup {report_inputs.speedup_ratio:.2f}x (threshold {SWAP_SPEEDUP_MIN:.2f}x)",
        f"RSS savings {report_inputs.rss_savings_mb:.1f} MB (threshold {SWAP_RSS_SAVINGS_MIN_MB:.0f} MB)",
    ]
    agreement_ok = report_inputs.agreement_rate >= SWAP_AGREEMENT_MIN
    lighter_faster = (
        report_inputs.speedup_ratio >= SWAP_SPEEDUP_MIN
        and report_inputs.rss_savings_mb >= SWAP_RSS_SAVINGS_MIN_MB
    )
    if agreement_ok and lighter_faster:
        return "SWAP", reasons
    if agreement_ok:
        return "HYBRID", reasons + ["agreement is high but speed/RSS savings are marginal"]
    return "KEEP", reasons + ["agreement below threshold; blendshapes do not reproduce DeepFace's signal"]


class _RecommendationInputs(BaseModel):
    """Inputs to the swap/no-swap rubric."""

    agreement_rate: float
    speedup_ratio: float
    rss_savings_mb: float


def _build_report(stage_outputs: "_StageOutputs") -> EvalReport:
    """Reduce both stage outputs into the final EvalReport."""
    mp_out, df_out = stage_outputs.mediapipe, stage_outputs.deepface
    merged = _merge_readings(mp_out, df_out)
    agreement_rate = _mean([float(m.agree) for m in merged])
    cross_check = _mean([
        float(m.ear_eyes_closed == m.blendshape_eyes_closed)
        for m in merged if m.ear_eyes_closed is not None
    ])
    mp_mean = _mean([m.mediapipe_latency_seconds for m in merged])
    df_mean = _mean([m.deepface_latency_seconds for m in merged])
    speedup = round(df_mean / mp_mean, 2) if mp_mean else 0.0
    rss_savings = df_out.timing.rss_delta_mb - mp_out.timing.rss_delta_mb
    recommendation, reasoning = _recommend(_RecommendationInputs(
        agreement_rate=agreement_rate, speedup_ratio=speedup, rss_savings_mb=rss_savings,
    ))
    return EvalReport(
        photo_count_total=stage_outputs.photo_count_total,
        face_photo_count=len(merged),
        agreement_rate=agreement_rate,
        eyes_closed_cross_check_rate=cross_check,
        mediapipe_timing=mp_out.timing.model_dump(),
        deepface_timing=df_out.timing.model_dump(),
        mediapipe_mean_latency_seconds=mp_mean,
        deepface_mean_latency_seconds=df_mean,
        speedup_ratio=speedup,
        disagreements=[m for m in merged if not m.agree],
        recommendation=recommendation,
        reasoning=reasoning,
    )


class _StageOutputs(BaseModel):
    """Bundle of both stage results plus the total candidate photo count."""

    mediapipe: MediapipeStageOutput
    deepface: DeepfaceStageOutput
    photo_count_total: int


def _write_report_markdown(report: EvalReport) -> None:
    """Write the durable, human-readable swap/no-swap report."""
    RUNS_DIR.mkdir(exist_ok=True)
    lines = _report_header_lines(report) + _report_disagreement_lines(report)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def _report_header_lines(report: EvalReport) -> list[str]:
    """Return the summary section of the report as markdown lines."""
    mp_t, df_t = report.mediapipe_timing, report.deepface_timing
    return [
        "# Portrait Eval: MediaPipe Blendshapes vs DeepFace Emotion",
        "",
        f"Generated {_utc_now()}",
        "",
        f"- Candidate photos: {report.photo_count_total}",
        f"- Face-bearing photos evaluated: {report.face_photo_count}",
        f"- **Agreement rate (decision-relevant expression bucket): {report.agreement_rate:.1%}**",
        f"- EAR-vs-blendshape eyes-closed cross-check: {report.eyes_closed_cross_check_rate:.1%}",
        f"- MediaPipe (blendshapes) mean per-photo latency: {report.mediapipe_mean_latency_seconds*1000:.2f} ms",
        f"- DeepFace mean per-photo latency: {report.deepface_mean_latency_seconds:.3f} s",
        f"- Speedup (DeepFace / MediaPipe): {report.speedup_ratio:.1f}x",
        f"- MediaPipe load: {mp_t['load_seconds']}s, RSS delta {mp_t['rss_delta_mb']} MB",
        f"- DeepFace load: {df_t['load_seconds']}s, RSS delta {df_t['rss_delta_mb']} MB",
        "",
        f"## Recommendation: {report.recommendation}",
        "",
        *[f"- {r}" for r in report.reasoning],
        "",
        "## Disagreement cases",
        "",
        "| photo | mediapipe bucket | deepface bucket | deepface label |",
        "|---|---|---|---|",
    ]


def _report_disagreement_lines(report: EvalReport) -> list[str]:
    """Return the disagreement table rows as markdown lines."""
    if not report.disagreements:
        return ["| — none — | | | |"]
    return [
        f"| {d.name} | {d.mediapipe_bucket} | {d.deepface_bucket} | {d.deepface_label} |"
        for d in report.disagreements
    ]


def _print_summary(report: EvalReport) -> None:
    """Print the headline numbers to stdout."""
    print(json.dumps({
        "photo_count_total": report.photo_count_total,
        "face_photo_count": report.face_photo_count,
        "agreement_rate": report.agreement_rate,
        "eyes_closed_cross_check_rate": report.eyes_closed_cross_check_rate,
        "mediapipe_mean_latency_seconds": report.mediapipe_mean_latency_seconds,
        "deepface_mean_latency_seconds": report.deepface_mean_latency_seconds,
        "speedup_ratio": report.speedup_ratio,
        "mediapipe_timing": report.mediapipe_timing,
        "deepface_timing": report.deepface_timing,
        "recommendation": report.recommendation,
        "reasoning": report.reasoning,
    }, indent=2))


def main() -> None:
    """Gather photos, run both stages, merge results, write and print the report."""
    RUNS_DIR.mkdir(exist_ok=True)
    photos = _gather_candidate_photos()
    manifest_path = RUNS_DIR / "portrait_eval_manifest.json"
    mp_out_path = RUNS_DIR / "portrait_eval_mediapipe.json"
    df_out_path = RUNS_DIR / "portrait_eval_deepface.json"
    manifest_path.write_text(json.dumps([str(p) for p in photos]))
    _run_stage(_StageInvocation(stage="mediapipe", input_path=manifest_path, output_path=mp_out_path))
    _run_stage(_StageInvocation(stage="deepface", input_path=mp_out_path, output_path=df_out_path))
    stage_outputs = _StageOutputs(
        mediapipe=MediapipeStageOutput.model_validate_json(mp_out_path.read_text()),
        deepface=DeepfaceStageOutput.model_validate_json(df_out_path.read_text()),
        photo_count_total=len(photos),
    )
    report = _build_report(stage_outputs)
    _write_report_markdown(report)
    _print_summary(report)


if __name__ == "__main__":
    main()
