"""Subprocess dispatch + stale-runner guard shared by all three test layers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from fair_expr_models import RunnerManifest, RunnerStageOutput

BENCH_DIR: Path = Path(__file__).resolve().parent
RUNNER_PATH: Path = BENCH_DIR / "fair_expr_runner.py"
RUNNER_TIMEOUT_SECONDS: int = 3600
STALE_RUNNER_PATTERN: str = "expression_eval_runner|cull --dry-run"


def assert_no_stale_runners() -> None:
    """Abort loudly if a stale heavy-model runner is still resident."""
    result = subprocess.run(
        ["pgrep", "-f", STALE_RUNNER_PATTERN], capture_output=True, text=True, check=False,
    )
    if result.stdout.strip():
        raise SystemExit(
            "ABORT: stale runner process(es) detected "
            f"(pids: {result.stdout.strip().splitlines()}) — "
            "kill them before running a heavy-model stage on this 18 GB host."
        )


class StageInvocation(BaseModel):
    """One subprocess stage run: which stage, what manifest, where to write output."""

    stage: str
    manifest: RunnerManifest
    output_path: Path


def run_stage(invocation: StageInvocation) -> RunnerStageOutput:
    """Write the manifest, invoke fair_expr_runner.py in a fresh subprocess, read output."""
    assert_no_stale_runners()
    invocation.output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = invocation.output_path.with_suffix(".manifest.json")
    manifest_path.write_text(invocation.manifest.model_dump_json())
    argv = [
        sys.executable, str(RUNNER_PATH), invocation.stage,
        str(manifest_path), str(invocation.output_path),
    ]
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=RUNNER_TIMEOUT_SECONDS, check=False, cwd=BENCH_DIR,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{invocation.stage} stage failed: {completed.stderr.strip()[-3000:]}")
    return RunnerStageOutput.model_validate_json(invocation.output_path.read_text())
