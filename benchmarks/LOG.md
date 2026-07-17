# A/B Run Log

| timestamp | model | accuracy | keep recall | reject recall | mean latency | peak metal | parse errors |
|---|---|---|---|---|---|---|---|
| 2026-07-16T20:42:32+00:00 | qwen3-vl-4b | 62.5% | 100.0% | 25.0% | 6.27s | 6.43 GB | 0.0% |
| 2026-07-16T20:48:13+00:00 | qwen3-vl-8b-instruct-4bit | 62.5% | 100.0% | 25.0% | 7.56s | 7.29 GB | 0.0% |
| 2026-07-16T20:51:21+00:00 | qwen3-vl-4b-instruct-4bit | 57.5% | 100.0% | 15.0% | 4.34s | 4.42 GB | 0.0% |
| 2026-07-16T20:54:02+00:00 | gemma-4-e4b-it-4bit | 55.0% | 90.0% | 20.0% | 3.8s | 6.43 GB | 0.0% |

## Round 1 verdicts (2026-07-17)

All three candidates REJECTED; champion `qwen3-vl-4b` (4B, 8-bit) stands.

- `qwen3-vl-8b-instruct-4bit`: identical accuracy (62.5%), 0.83x speed (slower), +0.9 GB peak → no improvement.
- `qwen3-vl-4b-instruct-4bit`: accuracy 57.5% (hard-gate regression). Confirms the
  owner's prediction that Q4 on small models regresses — reject recall fell 25% → 15%
  at identical keep recall. Q8 stays the floor for the 4B.
- `gemma-4-e4b-it-4bit` (mlx-vlm 0.6.3 venv): fastest (3.8s/photo) but worst accuracy
  (55%, keep recall 90%) → hard-gate regression.

Notes for round 2: all models show heavy keep-bias on context-free prompts
(reject recall ≤ 25%). Production stage 3 supplies PromptContext hints, so a
hint-enriched eval variant would better discriminate; consider adding
stage-2 signals to the eval harness before the next model round.
Rejected model weights kept under models/ for retest; delete to reclaim ~14.6 GB.

## Pipeline optimization gate (2026-07-17, commits 171ed92 + 305d3d3)

30-photo corpus (~/Desktop/cull-test-bench), full pipeline incl. VLM, dry-run:

| metric | baseline (v1.0.2) | optimized | delta |
|---|---|---|---|
| total wall-clock | 7m30s | **5m03s** | **-33%** |
| Stage 1 | 60s | 46s | -23% |
| Stage 2 | 4m25s | 2m51s | -35% |
| Stage 3 per photo | ~7.6s | ~5.9s | -23% |
| peak RSS (whole run) | ~8 GB | 7.25 GB | -9% |

VERDICT: PROMOTE — both optimization commits stay.
Routing deltas (24->25 keepers, 16->14 stage-3 analyzed) are attributable to
the intentional B2 exposure-corruption fix (7 photos repatched by the reducer
had corrupted composites before), not to the decode-once refactor (whose
stage-1 outputs were proven numerically identical by tests/stage1/test_worker.py).

## Gemma-4 E4B quant sweep conclusion (2026-07-17)

- `e4b-it-qat-OptiQ-4bit` (7.0 GB): FAILS TO LOAD in mlx-vlm 0.6.3 — OptiQ embeds
  calibration tensors (input_max/output_min) needing the mlx-optiq runtime.
- `unsloth e4b-it-UD-MLX-4bit` (6.2 GB): FAILS TO LOAD — incompatible weight layout.
- No 6-bit E4B exists on mlx-community; `e4b-it-8bit` (12.7 GB) breaches the
  12 GB Metal gate on this 18 GB host.
- VERDICT: Gemma-4 E4B is not competitive for photo judgment at any quant that
  fits this machine (plain 4-bit scored 55% vs champion 62.5%). Closed until
  hardware or a compatible higher-quality quant changes. Dead downloads removed.
| 2026-07-17T03:38:52+00:00 | gemma-4-e4b-it-6bit | 60.0% | 85.0% | 35.0% | 4.83s | 8.14 GB | 0.0% |

## Gemma-4 E4B Q6 (home-built, 2026-07-17)

No prebuilt 6-bit exists; converted from bf16 via mlx_vlm convert (6.6 GB,
7.09 effective bits/weight). Result: 60.0% accuracy (keep 85% / reject 35%),
4.83s/photo, 8.14 GB peak. Q6 recovers +5pp over Q4 (55%) — quantization
sensitivity confirmed in BOTH families — but still 2.5pp under champion with
worse keep recall. GATE: within accuracy tolerance band but no improvement →
REJECTED. Champion qwen3-vl-4b (8-bit) stands. Gemma line now fully closed:
Q4 rejected, Q6 rejected, Q8 exceeds memory gate, OptiQ/unsloth unloadable.
| 2026-07-17T03:48:01+00:00 | gemma-4-12b-it-4bit | 67.5% | 100.0% | 35.0% | 8.37s | 8.18 GB | 0.0% |
| 2026-07-17T04:00:31+00:00 | gemma-4-12b-it-6bit | 62.5% | 100.0% | 25.0% | 10.33s | 11.09 GB | 0.0% |

## CHAMPION SWAP (2026-07-17)

`gemma-4-12b-it-4bit` PROMOTED over `qwen3-vl-4b` (4B, 8-bit):
- v1 eval set: 67.5% vs 62.5% · holdout v2 (disjoint): 65.0% vs 60.0% —
  +5pp on BOTH independent sets (n=80 total)
- 12B Q6 (home-built) scored 62.5% — worse than Q4; not pursued
- Cost accepted: +2s/photo stage 3 (~+30s per 30-photo run), peak 8.18 GB
- Infra: main env upgraded mlx-vlm 0.3.12 -> 0.6.3 (pinned <0.6.4 due to
  known upstream regressions); zero golden-baseline drift after upgrade;
  full test suite green
- VLM_DEFAULT_ALIAS -> gemma-4-12b; qwen3-vl-4b retained as fallback alias

## DINOv2 dedupe tier — PROMOTED (2026-07-17)

Implemented as a SECOND TIER (CNN retained), not a replacement: Stage 4 curation
clustering consumes `DuplicateResult.encodings` (MobileNetV3) and its per-preset
CLUSTER_THRESHOLD values are calibrated to that embedding distribution — swapping
the backbone outright would have silently invalidated an unmeasured calibration.
DINOv2 pairs merge into CNN groups via connected components; its embeddings are
never exposed downstream.

Measured e2e on the labeled variant corpus (real production find_duplicates):

| variant | CNN only | CNN + DINOv2 |
|---|---|---|
| crop | 10/20 | **20/20** |
| rotation | 19/20 | **20/20** |
| reencode / resize / brightness | 20/20 | 20/20 |

Cost: +29.5 ms/photo (202 ms total, under the 250 ms budget). Suite 475 passed
(465 baseline + 10 new), zero regressions. `cull setup` status=ok (5 manifest
entries). GATE: PROMOTE.

## Current pipeline benchmark (2026-07-17, post-Gemma + post-DINOv2)

30 Japan photos (scratchpad/bench30), full pipeline dry-run, machine quiet:

| stage | time | rate |
|---|---|---|
| Stage 1 (incl. DINOv2 tier) | 54s | 1.8 s/photo |
| Stage 2 | 1m52s | 4.0 s/photo (28 survivors) |
| Stage 3 (Gemma-4-12B) | 2m28s | 7.4 s/photo (20 ambiguous) |
| **total** | **5m19s** | peak RSS 7.23 GB |

CAVEAT: different 30-photo corpus than the 7m30s/5m03s runs (the original
bench set became inaccessible via macOS TCC), so totals are NOT directly
comparable — per-photo rates are. Stage 2 is ~4.0 s/photo vs ~8.8 s/photo at
the v1.0.2 baseline. Stage 3 is 7.4 s/photo on Gemma-4-12B vs 5.9 s/photo on
Qwen-4B — the accuracy swap cost ~1.5 s/photo, as expected.

METHOD NOTE: an earlier run of this benchmark was discarded — it executed
concurrently with the expression eval (DeepFace 6.5 GB + Py-Feat) on an 18 GB
host, driving free memory to ~64 MB and inflating Stage 2 to 7m56s and Stage 3
past 36m. Those numbers measured swap contention, not the pipeline. Always
benchmark serially with no other model process resident.
