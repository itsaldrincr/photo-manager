"""A/B harness: evaluate VLM candidates against ground truth, gate promotions.

Each candidate runs in a fresh subprocess (model memory returned to the OS on
exit — required on low-RAM hosts). Results are logged durably to
benchmarks/runs/log.jsonl and benchmarks/LOG.md for later human review.

Usage:
    python3 benchmarks/harness.py eval <model_alias> [--python <interpreter>]
    python3 benchmarks/harness.py gate <candidate_alias>
    python3 benchmarks/harness.py report
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

BENCH_DIR: Path = Path(__file__).resolve().parent
EVAL_DIR: Path = BENCH_DIR / "eval_set"
GROUND_TRUTH_PATH: Path = BENCH_DIR / "ground_truth.json"
RUNS_DIR: Path = BENCH_DIR / "runs"
LOG_JSONL: Path = RUNS_DIR / "log.jsonl"
LOG_MD: Path = BENCH_DIR / "LOG.md"
CHAMPION_PATH: Path = BENCH_DIR / "champion.json"
RUNNER_PATH: Path = BENCH_DIR / "vlm_eval_runner.py"

RUNNER_TIMEOUT_SECONDS: int = 3600
PEAK_METAL_LIMIT_GB: float = 12.0
MIN_ACCURACY_DELTA: float = 0.025
MIN_SPEEDUP_RATIO: float = 1.10
MAX_PARSE_ERROR_RATE: float = 0.05


class EvalMetrics(BaseModel):
    """Aggregate metrics for one model's eval run."""

    model_alias: str
    accuracy: float
    keep_recall: float
    reject_recall: float
    parse_error_rate: float
    mean_latency_seconds: float
    load_seconds: float
    peak_metal_gb: float
    photo_count: int
    timestamp: str


class GateVerdict(BaseModel):
    """Promotion decision for a candidate vs the champion."""

    candidate: str
    champion: str
    promote: bool
    reasons: list[str]


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_ground_truth() -> dict[str, str]:
    """Return {photo_name: 'keep'|'reject'} from the labels file."""
    data = json.loads(GROUND_TRUTH_PATH.read_text())
    return data["labels"]


class _AgreementCounts(BaseModel):
    """Confusion counts for keep/reject agreement."""

    keep_hit: int = 0
    keep_total: int = 0
    reject_hit: int = 0
    reject_total: int = 0


class _LabeledVerdict(BaseModel):
    """One photo's ground-truth label paired with the model's prediction."""

    label: str
    predicted_keep: bool


def _count_agreement(verdicts: list[dict]) -> _AgreementCounts:
    """Compare runner verdicts to ground truth labels."""
    labels = _load_ground_truth()
    counts = _AgreementCounts()
    for verdict in verdicts:
        label = labels.get(verdict["name"])
        if label is None or verdict["is_keeper"] is None:
            continue
        pair = _LabeledVerdict(label=label, predicted_keep=verdict["is_keeper"])
        _tally(counts, pair)
    return counts


def _tally(counts: _AgreementCounts, pair: _LabeledVerdict) -> None:
    """Update confusion counts for one photo."""
    if pair.label == "keep":
        counts.keep_total += 1
        counts.keep_hit += int(pair.predicted_keep)
        return
    counts.reject_total += 1
    counts.reject_hit += int(not pair.predicted_keep)


def _safe_ratio(hit: int, total: int) -> float:
    """Return hit/total guarding against empty denominators."""
    return round(hit / total, 4) if total else 0.0


def _metrics_from_runner(raw: dict) -> EvalMetrics:
    """Reduce a runner result JSON into aggregate metrics."""
    verdicts = raw["verdicts"]
    counts = _count_agreement(verdicts)
    latencies = [v["latency_seconds"] for v in verdicts]
    errors = sum(1 for v in verdicts if v["is_parse_error"])
    total_hits = counts.keep_hit + counts.reject_hit
    total_labeled = counts.keep_total + counts.reject_total
    return EvalMetrics(
        model_alias=raw["model_alias"],
        accuracy=_safe_ratio(total_hits, total_labeled),
        keep_recall=_safe_ratio(counts.keep_hit, counts.keep_total),
        reject_recall=_safe_ratio(counts.reject_hit, counts.reject_total),
        parse_error_rate=_safe_ratio(errors, len(verdicts)),
        mean_latency_seconds=round(sum(latencies) / len(latencies), 2),
        load_seconds=raw["load_seconds"],
        peak_metal_gb=raw["peak_metal_gb"],
        photo_count=len(verdicts),
        timestamp=_utc_now(),
    )


class _EvalCommand(BaseModel):
    """Inputs for one subprocess eval run."""

    model_alias: str
    python_bin: str


def _run_subprocess_eval(command: _EvalCommand) -> EvalMetrics:
    """Run the eval runner in a fresh process and aggregate its output."""
    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / f"raw_{command.model_alias}_{_utc_now()}.json"
    argv = [
        command.python_bin, str(RUNNER_PATH),
        command.model_alias, str(EVAL_DIR), str(out_path),
    ]
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=RUNNER_TIMEOUT_SECONDS,
        check=False, cwd=BENCH_DIR.parent,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"runner failed: {completed.stderr.strip()[-500:]}")
    return _metrics_from_runner(json.loads(out_path.read_text()))


def _append_log(metrics: EvalMetrics) -> None:
    """Durably record a run in JSONL and the human-readable log."""
    RUNS_DIR.mkdir(exist_ok=True)
    with LOG_JSONL.open("a") as handle:
        handle.write(metrics.model_dump_json() + "\n")
    line = (
        f"| {metrics.timestamp} | {metrics.model_alias} | {metrics.accuracy:.1%} "
        f"| {metrics.keep_recall:.1%} | {metrics.reject_recall:.1%} "
        f"| {metrics.mean_latency_seconds}s | {metrics.peak_metal_gb} GB "
        f"| {metrics.parse_error_rate:.1%} |\n"
    )
    _ensure_md_header()
    with LOG_MD.open("a") as handle:
        handle.write(line)


def _ensure_md_header() -> None:
    """Create LOG.md with its table header if missing."""
    if LOG_MD.exists():
        return
    LOG_MD.write_text(
        "# A/B Run Log\n\n"
        "| timestamp | model | accuracy | keep recall | reject recall "
        "| mean latency | peak metal | parse errors |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )


def _load_champion() -> EvalMetrics | None:
    """Return the current champion metrics, if recorded."""
    if not CHAMPION_PATH.exists():
        return None
    return EvalMetrics.model_validate_json(CHAMPION_PATH.read_text())


def _latest_metrics_for(alias: str) -> EvalMetrics | None:
    """Return the most recent logged metrics for an alias."""
    if not LOG_JSONL.exists():
        return None
    found: EvalMetrics | None = None
    for line in LOG_JSONL.read_text().splitlines():
        entry = EvalMetrics.model_validate_json(line)
        if entry.model_alias == alias:
            found = entry
    return found


class _Matchup(BaseModel):
    """Candidate and champion metrics under judgment."""

    candidate: EvalMetrics
    champion: EvalMetrics


def _judge(matchup: _Matchup) -> GateVerdict:
    """Apply promotion rules: hard gates first, then improvement test."""
    reasons = _hard_gate_failures(matchup)
    if reasons:
        return GateVerdict(
            candidate=matchup.candidate.model_alias,
            champion=matchup.champion.model_alias,
            promote=False, reasons=reasons,
        )
    return _judge_improvement(matchup)


def _hard_gate_failures(matchup: _Matchup) -> list[str]:
    """Return failure reasons for any violated hard constraint."""
    candidate, champion = matchup.candidate, matchup.champion
    reasons: list[str] = []
    if candidate.peak_metal_gb > PEAK_METAL_LIMIT_GB:
        reasons.append(f"peak metal {candidate.peak_metal_gb} GB exceeds limit")
    if candidate.parse_error_rate > MAX_PARSE_ERROR_RATE:
        reasons.append(f"parse errors {candidate.parse_error_rate:.1%} too high")
    if candidate.accuracy < champion.accuracy - MIN_ACCURACY_DELTA:
        reasons.append(
            f"accuracy regression {candidate.accuracy:.1%} < "
            f"{champion.accuracy:.1%} - {MIN_ACCURACY_DELTA:.1%}"
        )
    return reasons


def _judge_improvement(matchup: _Matchup) -> GateVerdict:
    """Promote only for a meaningful accuracy or speed win."""
    candidate, champion = matchup.candidate, matchup.champion
    better_accuracy = candidate.accuracy >= champion.accuracy + MIN_ACCURACY_DELTA
    speedup = champion.mean_latency_seconds / max(
        candidate.mean_latency_seconds, 0.01
    )
    faster = speedup >= MIN_SPEEDUP_RATIO
    promote = better_accuracy or (faster and candidate.accuracy >= champion.accuracy)
    reasons = [
        f"accuracy {candidate.accuracy:.1%} vs champion {champion.accuracy:.1%}",
        f"latency {candidate.mean_latency_seconds}s vs "
        f"{champion.mean_latency_seconds}s (speedup {speedup:.2f}x)",
    ]
    return GateVerdict(
        candidate=candidate.model_alias, champion=champion.model_alias,
        promote=promote, reasons=reasons,
    )


def _cmd_eval(argv: list[str]) -> None:
    """Run one candidate eval and log it."""
    alias = argv[0]
    python_bin = argv[argv.index("--python") + 1] if "--python" in argv else sys.executable
    metrics = _run_subprocess_eval(
        _EvalCommand(model_alias=alias, python_bin=python_bin)
    )
    _append_log(metrics)
    print(metrics.model_dump_json(indent=2))


def _cmd_gate(argv: list[str]) -> None:
    """Judge the latest run of a candidate against the champion."""
    candidate = _latest_metrics_for(argv[0])
    champion = _load_champion()
    if candidate is None or champion is None:
        raise SystemExit("need both a logged candidate run and a champion.json")
    verdict = _judge(_Matchup(candidate=candidate, champion=champion))
    print(verdict.model_dump_json(indent=2))
    sys.exit(0 if verdict.promote else 1)


def _cmd_report(_: list[str]) -> None:
    """Print the human-readable log."""
    if not LOG_MD.exists():
        raise SystemExit("no runs logged yet")
    print(LOG_MD.read_text())


def main() -> None:
    """Dispatch subcommands: eval, gate, report."""
    commands = {"eval": _cmd_eval, "gate": _cmd_gate, "report": _cmd_report}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit(f"usage: harness.py {{{'|'.join(commands)}}} ...")
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
