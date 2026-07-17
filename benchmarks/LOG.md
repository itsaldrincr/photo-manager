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
