"""Score the calibration corpus via the real Stage 1 + Stage 2 (+ reducer) pipeline.

Runs cull's internal stage machinery directly (no CLI, no file moves, no TUI)
so we capture every raw per-metric value fusion.py consumes: topiq,
laion_aesthetic, clipiqa, exposure, composition composite, taste probability,
subject-blur tenengrad, tilt penalty (recomputed from Stage1 geometry),
is_bokeh, and the shoot-level reducer terms (palette_outlier, exposure_drift,
exif_anomaly, scene_start_bonus) plus portrait quality signals.

Each "shoot" (day) is scored in its own pipeline run against its own
directory, matching how cull is actually invoked in production (one source
folder per session) so the reducer's cross-photo shoot stats are computed
against the right population instead of a pooled multi-day corpus.

Usage:
    python3 score_calib_corpus.py <corpus_root_with_day_subdirs> <out_json>
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class _ResultsCtx:
    """Bundle of Stage 1 + Stage 2 result maps keyed by str(photo_path)."""

    s2_results: dict
    s1_results: dict

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


class _StubDashboard:
    """No-op dashboard satisfying every hook stage1/stage2/reducer call."""

    def start_stage1(self, total: int) -> None:
        logger.info("Stage 1 starting: %d photos", total)

    def update_stage1(self, path: Any, result: Any) -> None:
        pass

    def complete_stage1(self, elapsed: float) -> None:
        logger.info("Stage 1 complete in %.1fs", elapsed)

    def start_scanning(self) -> None:
        pass

    def set_burst_count(self, count: int) -> None:
        logger.info("Burst losers: %d", count)

    def set_dupe_count(self, count: int) -> None:
        logger.info("Duplicates: %d", count)

    def refresh(self) -> None:
        pass

    def start_stage2_loading(self) -> None:
        logger.info("Loading Stage 2 models...")

    def clear_stage2_loading(self) -> None:
        logger.info("Stage 2 models loaded")

    def start_stage2(self, total: int) -> None:
        logger.info("Stage 2 starting: %d photos", total)

    def update_stage2(self, update_in: Any) -> None:
        pass

    def complete_stage2(self, elapsed: float) -> None:
        logger.info("Stage 2 complete in %.1fs", elapsed)

    def start_stage2_reducer(self, total: int) -> None:
        logger.info("Stage 2 reducer starting: %d photos", total)

    def update_stage2_reducer(self, update_in: Any) -> None:
        pass

    def complete_stage2_reducer(self, elapsed: float) -> None:
        logger.info("Stage 2 reducer complete in %.1fs", elapsed)


def _tilt_penalty_raw(s1_result: Any) -> float | None:
    """Recompute the raw tilt_penalty term the same way stage2_scoring does."""
    from cull.config import KEYSTONE_PENALTY_DEGREES, TILT_PENALTY_DEGREES

    if s1_result is None or s1_result.geometry is None:
        return None
    tilt_norm = abs(s1_result.geometry.tilt_degrees) / TILT_PENALTY_DEGREES
    keystone_norm = abs(s1_result.geometry.keystone_degrees) / KEYSTONE_PENALTY_DEGREES
    return float(min(1.0, max(tilt_norm, keystone_norm)))


def _row_for_photo(path: Path, results: _ResultsCtx) -> dict[str, Any]:
    """Assemble one per-photo row of every fusion-consumed raw metric."""
    fusion = results.s2_results[str(path)]
    stage2 = fusion.stage2
    s1 = results.s1_results.get(str(path))
    portrait = stage2.portrait
    composition = stage2.composition
    taste = stage2.taste
    subject_blur = stage2.subject_blur
    shoot_stats = stage2.shoot_stats
    return {
        "filename": path.name,
        "topiq": stage2.topiq,
        "laion_aesthetic": stage2.laion_aesthetic,
        "clipiqa": stage2.clipiqa,
        "exposure": s1.exposure.dr_score if s1 else None,
        "composite_current_weights": stage2.composite,
        "routing_current_weights": fusion.routing,
        "composition": composition.composite if composition else None,
        "taste_probability": taste.probability if taste else None,
        "taste_label_count": taste.label_count_at_score if taste else None,
        "taste_weight_applied": taste.weight_applied if taste else None,
        "subject_blur_tenengrad": subject_blur.tenengrad if subject_blur else None,
        "subject_blur_has_subject": subject_blur.has_subject if subject_blur else None,
        "is_bokeh": s1.blur.is_bokeh if s1 else None,
        "tilt_penalty": _tilt_penalty_raw(s1),
        "palette_outlier_score": shoot_stats.palette_outlier_score if shoot_stats else None,
        "exposure_drift_score": shoot_stats.exposure_drift_score if shoot_stats else None,
        "exif_anomaly_score": shoot_stats.exif_anomaly_score if shoot_stats else None,
        "scene_start_bonus": shoot_stats.scene_start_bonus if shoot_stats else None,
        "eye_sharpness_left": portrait.eye_sharpness_left if portrait else None,
        "eye_sharpness_right": portrait.eye_sharpness_right if portrait else None,
        "is_eyes_closed": portrait.is_eyes_closed if portrait else None,
        "is_face_occluded": portrait.is_face_occluded if portrait else None,
        "has_face": bool(portrait) and (
            portrait.eye_sharpness_left is not None or portrait.dominant_emotion is not None
        ),
    }


def _run_pipeline_stages(paths: list[Path]) -> tuple[dict, dict, list[Path], list[Path]]:
    """Run Stage 1 -> 2 (+reducer) with burst/dupe losers re-admitted to Stage 2."""
    from cull.config import CullConfig
    from cull._pipeline.stage1_runner import _run_s1
    from cull._pipeline.stage2_runner import _run_s2, _S2RunInput
    from cull._pipeline.stage2_reducer import _run_s2_reducer, _S2ReducerRunInput
    from cull._pipeline.orchestrator import _StageRunCtx, _unload_stage2_models

    config = CullConfig(preset="wedding", is_portrait=True, is_dry_run=True, stages=[1, 2])
    ctx = _StageRunCtx(config=config, paths=paths, source_path=None, dashboard=_StubDashboard())

    t0 = time.monotonic()
    s1_out = _run_s1(ctx)
    # Weight-fitting variant: burst losers and duplicates are re-admitted to
    # Stage 2 so every implicit-labeled photo gets a composite. Production
    # would auto-drop them before fusion; that gap is analyzed separately
    # from the production-faithful scoring run.
    readmitted = [
        p for p in paths
        if p not in s1_out.survivors
        and str(p) in (s1_out.burst_losers | s1_out.duplicate_paths)
    ]
    s1_out.survivors = s1_out.survivors + readmitted
    logger.info("Re-admitted %d burst/duplicate losers for scoring", len(readmitted))
    logger.info(
        "Stage 1 done: %d survivors / %d rejected / %d burst_losers / %d duplicates / %d failed",
        len(s1_out.survivors), len(s1_out.rejected), len(s1_out.burst_losers),
        len(s1_out.duplicate_paths), len(s1_out.failed_paths),
    )
    s2_out = _run_s2(_S2RunInput(s1_out=s1_out, ctx=ctx))
    _run_s2_reducer(_S2ReducerRunInput(s2_out=s2_out, s1_out=s1_out, ctx=ctx))
    _unload_stage2_models()
    logger.info("Stage 1+2 total: %.1fs", time.monotonic() - t0)

    dropped_survivors = [p for p in paths if p not in s1_out.survivors]
    return s1_out.results, s2_out.results, s1_out.survivors, dropped_survivors


def _score_one_day(day_dir: Path) -> dict[str, Any]:
    """Run the full stage1+2+reducer pipeline for one day's directory."""
    paths = sorted(set(day_dir.glob("*.JPG")) | set(day_dir.glob("*.jpg")))
    logger.info("=== %s: %d photos ===", day_dir.name, len(paths))
    s1_results, s2_results, survivors, dropped = _run_pipeline_stages(paths)
    results_ctx = _ResultsCtx(s2_results=s2_results, s1_results=s1_results)
    rows = {}
    for p in survivors:
        key = f"{day_dir.name}_{p.name}"
        try:
            rows[key] = _row_for_photo(p, results_ctx)
        except KeyError:
            logger.warning("No Stage 2 result for survivor %s", p.name)
    return {
        "day": day_dir.name,
        "total_photos": len(paths),
        "stage1_survivors": len(survivors),
        "stage1_dropped": [f"{day_dir.name}_{p.name}" for p in dropped],
        "rows": rows,
    }


def main() -> None:
    """Score every day subdir under corpus_root; write merged rows to JSON."""
    corpus_root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    day_dirs = sorted(p for p in corpus_root.iterdir() if p.is_dir())
    logger.info("Corpus root: %s (%d day dirs)", corpus_root, len(day_dirs))

    all_rows: dict[str, Any] = {}
    all_dropped: list[str] = []
    total_photos = 0
    total_survivors = 0
    for day_dir in day_dirs:
        day_out = _score_one_day(day_dir)
        all_rows.update(day_out["rows"])
        all_dropped.extend(day_out["stage1_dropped"])
        total_photos += day_out["total_photos"]
        total_survivors += day_out["stage1_survivors"]

    out = {
        "corpus_root": str(corpus_root),
        "total_photos": total_photos,
        "stage1_survivors": total_survivors,
        "stage1_dropped": all_dropped,
        "rows": all_rows,
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    logger.info(
        "Wrote %d scored rows (%d dropped pre-Stage2) across %d days to %s",
        len(all_rows), len(all_dropped), len(day_dirs), out_path,
    )


if __name__ == "__main__":
    main()
