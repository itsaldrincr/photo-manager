"""Baseline: does the current blur-only burst winner match the owner's keeper?

Mirrors production Stage 1 burst grouping (src/cull/_pipeline/stage1_runner.py
_apply_burst_only): run the Stage 1 per-photo loop over a shoot to get blur
tenengrad + non-rejected survivors, then detect_bursts over those survivors so
burst groups are formed exactly as production forms them. For every burst group
(and, secondarily, every DINOv2/CNN duplicate group) that contains >= 1 implicit
owner keeper (wedding_ground_truth.json), record whether the CURRENT winner —
blur-max for bursts, group representative for duplicates — is an owner keeper.

Metric: winner_matches_owner_rate over groups with >= 1 keeper, per shoot and pooled.

Usage:
    python3 benchmarks/runs/burst_baseline.py <out_json>
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cull.config import CullConfig
from cull._pipeline.stage1_runner import _run_stage1_loop, _Stage1LoopInput
from cull.stage1.burst import _BurstInput, detect_bursts, select_burst_winner, BurstScoringInput
from burst_rank_candidate import select_face_aware_winner
from cull.stage1.duplicate import find_duplicates
from cull.stage2.portrait import unload_face_landmarker

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("burst_baseline")

GROUND_TRUTH_PATH = Path("benchmarks/runs/wedding_ground_truth.json")
SHOOTS: dict[str, Path] = {
    "exn_backup": Path(
        "/Volumes/personal_folder/media/Photos/from-external/1 Photography/"
        "4 Weddings/220924 ExN Dance Rehearsal/Backup"
    ),
    "zd_backup": Path(
        "/Volumes/personal_folder/media/Photos/from-external/1 Photography/"
        "4 Weddings/290325 Zack and Dora's Wedding Rehearsal/Backup"
    ),
}


class _NoopDashboard:
    """Permissive no-op dashboard: every Stage 1 hook call is absorbed."""

    def __getattr__(self, _name: str) -> Any:
        return lambda *a, **k: None


class _MeasureCtx(BaseModel):
    """Shoot-level measurement context: GT prefix + ground-truth map."""

    model_config = {"arbitrary_types_allowed": True}

    prefix: str
    gt: dict


class _GroupSpan(BaseModel):
    """A candidate group, its current-production winner, and optional face winner."""

    model_config = {"arbitrary_types_allowed": True}

    group: list[Path]
    winner: Path
    face_winner: Path | None = None


class _ShootPass(BaseModel):
    """Stage 1 loop output + config for one shoot, for burst measurement."""

    model_config = {"arbitrary_types_allowed": True}

    loop_out: Any
    config: CullConfig


def _load_ground_truth() -> dict[str, dict[str, str]]:
    """Return the wedding implicit keep/reject ground-truth map."""
    return json.loads(GROUND_TRUTH_PATH.read_text())


def _gt_key(prefix: str, path: Path) -> str:
    """Return the ground-truth key for a shoot photo (matches score driver)."""
    return f"{prefix}_{path.name}"


def _run_stage1(day_dir: Path) -> Any:
    """Run the Stage 1 per-photo loop (no burst/dup filtering) over a shoot dir."""
    paths = sorted(set(day_dir.glob("*.JPG")) | set(day_dir.glob("*.jpg")))
    config = CullConfig(preset="wedding", is_portrait=True, is_dry_run=True, stages=[1])
    loop_in = _Stage1LoopInput(paths=paths, config=config, source_path=day_dir)
    logger.info("=== %s: %d photos, running Stage 1 loop ===", day_dir.name, len(paths))
    return _run_stage1_loop(loop_in, _NoopDashboard()), paths, config


def _is_keeper(ctx: _MeasureCtx, path: Path) -> bool:
    """Return True if the photo is an implicit owner keeper."""
    return ctx.gt.get(_gt_key(ctx.prefix, path), {}).get("label") == "keep"


def _group_record(ctx: _MeasureCtx, span: _GroupSpan) -> dict | None:
    """Return a measurement record for a group with >= 1 keeper, else None."""
    keepers = [p for p in span.group if _is_keeper(ctx, p)]
    if not keepers:
        return None
    record = {
        "size": len(span.group),
        "num_keepers": len(keepers),
        "winner": span.winner.name,
        "winner_is_keeper": _is_keeper(ctx, span.winner),
        "members": [p.name for p in span.group],
    }
    if span.face_winner is not None:
        record["face_winner"] = span.face_winner.name
        record["face_winner_is_keeper"] = _is_keeper(ctx, span.face_winner)
    return record


def _burst_span(ctx: _MeasureCtx, scoring: BurstScoringInput) -> _GroupSpan:
    """Build a span with the blur-only winner and the face-aware winner for a group."""
    winner, _ = select_burst_winner(scoring)
    face_winner = None
    if any(_is_keeper(ctx, p) for p in scoring.group):  # only score keeper-bearing groups
        face_winner, _ = select_face_aware_winner(scoring)
    return _GroupSpan(group=scoring.group, winner=winner, face_winner=face_winner)


def _measure_bursts(ctx: _MeasureCtx, shoot: _ShootPass) -> list[dict]:
    """Form burst groups over survivors and record keeper-bearing group outcomes."""
    survivors = shoot.loop_out.survivors
    blur_scores = {
        str(p): shoot.loop_out.results[str(p)].blur.tenengrad
        for p in survivors if str(p) in shoot.loop_out.results
    }
    burst = detect_bursts(_BurstInput(image_paths=survivors, config=shoot.config, blur_scores=blur_scores))
    records = [
        _group_record(ctx, _burst_span(ctx, BurstScoringInput(group=group, blur_scores=blur_scores)))
        for group in burst.groups
    ]
    result = [r for r in records if r is not None]
    unload_face_landmarker()  # release MediaPipe after ranking this shoot's bursts
    return result


def _measure_duplicates(ctx: _MeasureCtx, day_dir: Path) -> list[dict]:
    """Form DINOv2/CNN duplicate groups and record keeper-bearing group outcomes."""
    dup = find_duplicates(day_dir)
    records = []
    for dgroup in dup.duplicate_groups:
        paths = list(dgroup.paths)
        record = _group_record(ctx, _GroupSpan(group=paths, winner=paths[0]))  # rep = paths[0]
        if record is not None:
            records.append(record)
    return records


def _rate(records: list[dict], key: str) -> float | None:
    """Return the share of records whose `key` field is True, or None if empty."""
    scored = [r for r in records if key in r]
    if not scored:
        return None
    return sum(r[key] for r in scored) / len(scored)


def _summarize(records: list[dict]) -> dict:
    """Return blur-only + face-aware rates, group count, and multi-keeper count."""
    return {
        "keeper_bearing_groups": len(records),
        "winner_matches_owner_rate": _rate(records, "winner_is_keeper"),
        "face_aware_matches_owner_rate": _rate(records, "face_winner_is_keeper"),
        "multi_keeper_groups": sum(1 for r in records if r["num_keepers"] > 1),
    }


def main() -> None:
    """Measure burst + duplicate winner/owner agreement across both shoots."""
    out_path = Path(sys.argv[1])
    gt = _load_ground_truth()
    per_shoot: dict[str, Any] = {}
    pooled_burst: list[dict] = []
    pooled_dup: list[dict] = []
    for prefix, day_dir in SHOOTS.items():
        loop_out, _paths, config = _run_stage1(day_dir)
        ctx = _MeasureCtx(prefix=prefix, gt=gt)
        burst_records = _measure_bursts(ctx, _ShootPass(loop_out=loop_out, config=config))
        dup_records = _measure_duplicates(ctx, day_dir)
        pooled_burst.extend(burst_records)
        pooled_dup.extend(dup_records)
        per_shoot[prefix] = {
            "survivors": len(loop_out.survivors),
            "rejected": len(loop_out.rejected),
            "burst": _summarize(burst_records),
            "duplicate": _summarize(dup_records),
            "burst_records": burst_records,
            "duplicate_records": dup_records,
        }
        logger.info("%s burst: %s", prefix, per_shoot[prefix]["burst"])
        logger.info("%s duplicate: %s", prefix, per_shoot[prefix]["duplicate"])
    out = {
        "per_shoot": per_shoot,
        "pooled_burst": _summarize(pooled_burst),
        "pooled_duplicate": _summarize(pooled_dup),
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    logger.info("Pooled burst: %s", out["pooled_burst"])
    logger.info("Pooled duplicate: %s", out["pooled_duplicate"])
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
