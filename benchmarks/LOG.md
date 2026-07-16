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
