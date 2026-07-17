---
id: task_713
name: revive_shoot_stats_signals
state: DONE
step: 3 of 3
depends: []
checkpoint: 2c2d12
created: 2026-07-18
---

## Program (immutable — set at planning)
1. Add capture_time + ExifSummary fields to Stage1Result; populate them in
   stage1/worker.py (single EXIF read per photo, alongside the existing
   decode) and thread through _pipeline/stage1_runner.py.
2. Add palette_lab to Stage2Result; compute a cheap LAB centroid in
   _pipeline/stage2_scoring.py from the already-decoded pil_1280 batch and
   wire it through stage2_runner.py + stage2/fusion.py (IqaScores field,
   preserved across the reducer patch round-trip).
3. Replace shoot_stats.py's silent getattr fallbacks with reads of the real
   fields, keeping graceful None handling for photos missing EXIF/palette.

## Registers (mutable — agent writes after each step)
Step 1 DONE (checkpoint 2c2d12): models.py — added ExifSummary, Stage1Result.capture_time,
Stage1Result.exif. stage1/worker.py — added _read_exif_summary + parse helpers
(single exifread.process_file per photo), Stage1WorkerResult carries capture_time/exif.
_pipeline/stage1_runner.py — _Stage1Ctx/_worker_result_to_ctx/_build_stage1_result thread
the new fields through.

Step 2 DONE (checkpoint 2c2d12): models.py — added Stage2Result.palette_lab. config.py —
added PALETTE_CENTROID_DOWNSAMPLE_PX in the palette/exposure-drift section.
_pipeline/stage2_scoring.py — _palette_lab_centroid/score_palette_lab_batch/
apply_palette_lab_to_scores (cv2 RGB->LAB mean over a 32x32 downsample of pil_1280).
_pipeline/stage2_runner.py — _score_one_chunk calls apply_palette_lab_to_scores.
stage2/fusion.py — IqaScores.palette_lab field; _build_stage2_result and _stage2_to_iqa
both carry it through so the reducer-patch round-trip in patch_reducer_scores doesn't
drop it. This last file was outside the original ownership list but was a hard
technical requirement — see Notes.

Step 3 DONE (checkpoint 2c2d12): shoot_stats.py — _palette_lab_for_result,
_exif_dict_for_s1, _capture_seconds now read the real fields directly instead of
getattr(..., None) fallbacks. Behaviour on missing data (None) is unchanged
(zeros(3) / empty dict / None respectively).

## Working Memory (scratch values the agent carries forward)
Real-data probe over scratchpad/bench30 (30 real travel photos) confirmed all three
signals now fire non-constant:
  palette_outlier_score: min=0.082 max=1.000 mean=0.788 (30/30 nonzero)
  exif_anomaly_score:    min=0.188 max=0.750 mean=0.381 (30/30 nonzero)
  scene_start_bonus:     28/30 nonzero, 29 distinct scene_ids
Observation (not in scope to fix): SCENE_BOUNDARY_GAP_MULTIPLIER(4.0) x
BURST_GAP_DEFAULT_SECONDS(0.5s) = a 2s scene-gap threshold, which is far smaller than
typical inter-shot cadence in real travel photography (tens of seconds to minutes) —
once real timestamps flow, nearly every non-burst photo now trips a scene boundary.
That's a threshold-tuning question for whoever owns PRESET_QUALITY_POLICY / the
reducer weights, not a plumbing defect.

## Acceptance Criteria
- [x] palette_outlier_score, exif_anomaly_score, scene_start_bonus fire (non-constant,
      non-zero) on real photos with real EXIF, not just on duck-typed mocks.
- [x] Unit tests added for each revived signal with realistic inputs
      (tests/stage1/test_worker_exif.py, tests/stage2/test_palette_lab.py,
      tests/stage2/test_shoot_stats.py real-model wiring tests).
- [x] New fields are additive-only: p1/p4lite golden baselines compare only
      topiq/laion_aesthetic/clipiqa (never composite/shoot_stats), confirmed
      unaffected by inspection.
- [x] `pytest tests -q -k "not golden and not perf"`: 513 passed (baseline 486
      before this task; +19 new tests; the intervening 494 came from another
      agent's concurrent task_712 work, unrelated to this task).

## Transition Rules
- IF current step DONE → increment step, update Registers, continue
- IF all steps DONE → set state: VERIFY, self-check acceptance criteria
- IF verify passes → set state: DONE, update MAP.md flag
- IF verify fails → set state to failed step number, note what failed
