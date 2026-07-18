# Calibration report — routing thresholds + holiday genre weights

## Corpus

- Ground-truth photos (NAS, owner-triaged): 323 (78 keep / 245 reject expected)
- Filename collisions across days: 0 (verified before scoring)
- Matched to a Stage 2 composite: 293 (77 keep / 216 reject)
- Dropped before Stage 2 (Stage 1 blur/noise/burst/duplicate reject): 30
  - Of those, ground-truth KEEPERS dropped by Stage 1 (uncorrectable by Stage 2 threshold/weight tuning): 1 ['day5_keep_DSCF9099.JPG']

Composite-reconstruction sanity check (recomputed vs. real pipeline composite, current holiday weights):
- max abs diff: 0.198031, mean abs diff: 0.0633087 (n=293)

## Routing threshold sweep (current holiday weights)

Current thresholds: AMBIGUOUS_MIN=0.48, KEEPER_MIN=0.72

Best under VLM-share <= 35%: AMBIGUOUS_MIN=0.85, KEEPER_MIN=0.94, error=0.1361, vlm_share=0.3481

Frontier (best error at each VLM-share cap):

| vlm_share_cap | ambiguous_min | keeper_min | error | vlm_share |
|---|---|---|---|---|
| 10% | 0.90 | 0.93 | 0.2180 | 0.0922 |
| 15% | 0.88 | 0.92 | 0.1984 | 0.1399 |
| 20% | 0.87 | 0.93 | 0.1822 | 0.1945 |
| 25% | 0.87 | 0.95 | 0.1718 | 0.2253 |
| 30% | 0.86 | 0.94 | 0.1490 | 0.2901 |
| 35% | 0.85 | 0.94 | 0.1361 | 0.3481 |
| 50% | 0.83 | 0.95 | 0.0800 | 0.4881 |
| 100% | 0.79 | 0.95 | 0.0659 | 0.6894 |

## Logistic regression fit (full data, standardized metrics -> keep/reject)

### Core 4 (topiq / laion_aesthetic / clipiqa / exposure)

| metric | current weight | LR coefficient | fitted weight (floored, normalized) |
|---|---|---|---|
| topiq | 0.300 | +1.0823 | 0.639 |
| laion_aesthetic | 0.325 | +0.6119 | 0.361 |
| clipiqa | 0.225 | -0.4242 | 0.000 |
| exposure | 0.150 | -0.7661 | 0.000 |

### Non-core (composition / taste / penalties / bonus) — reported only, NOT applied per task scope

| metric | current weight | LR coefficient | expected sign | agrees |
|---|---|---|---|---|
| composition | 0.150 | +0.5646 | positive | yes |
| taste_probability | 0.150 | -0.6352 | positive | **NO** |
| scene_start_bonus | 0.040 | +0.1952 | positive | yes |
| tilt_penalty | 0.100 | -0.2402 | negative | yes |
| palette_outlier_score | 0.050 | -0.2549 | negative | yes |
| exposure_drift_score | 0.050 | -0.3578 | negative | yes |
| exif_anomaly_score | 0.030 | -0.3423 | negative | yes |

## 5-fold cross-validated routing accuracy (each config at its own best threshold, VLM-share <= 35%)

| fold | current weights acc | fitted weights acc |
|---|---|---|
| 1 | 0.9231 | 0.8974 |
| 2 | 0.8974 | 0.9111 |
| 3 | 0.9250 | 0.9744 |
| 4 | 0.8537 | 0.8462 |
| 5 | 0.8718 | 0.8947 |
| **mean** | **0.8942** | **0.9048** |

Improvement: +1.06 points (gate requires >= 3.0)

## Gate verdict: FAIL

## Honest limitations

- **Single-genre corpus.** All 323 labeled photos are from one holiday/travel
  trip. Fitted weights apply to the `holiday` preset only. Generalizing them
  to `general` is NOT defensible from this data alone — `general` mixes
  genres this corpus never covers (weddings, wildlife, landscape-only shoots).
- **Single-owner taste.** Labels reflect one person's keep/reject judgment on
  one trip. That is the intended signal (personal taste), not a claim of
  universal photographic quality.
- **Taste-model confound.** The real pipeline's `taste` term is live and at
  full ramp weight (label_count=602 >= TASTE_RAMP_LABELS=20) from
  `~/.cull/taste_profile.joblib`, itself trained on the disagreement-biased
  override log. This is captured faithfully as "current pipeline" behavior
  (per the task's per-metric capture step) but means both the current-weight
  baseline AND the fitted-weight composite include a `taste_probability`
  feature whose own model was trained on biased data. The core-4 weight fit
  in this report does not touch the `taste` weight, but readers should not
  interpret `taste_probability`'s regression coefficient as bias-free signal.
- **Per-day shoot-stats.** Each day was scored as its own pipeline "shoot" run
  (5 separate stage1+2 invocations) so palette/exposure/EXIF reducer stats
  are computed against the correct per-day population, matching real
  production usage (one `cull` invocation per source folder) rather than a
  pooled 5-day corpus that would distort those outlier scores.
- **Stage 1 drop-outs are out of scope for weight/threshold fitting.** Photos
  Stage 1 filters (blur/noise/burst/duplicate) never reach Stage 2 fusion and
  so cannot be fixed by retuning composite weights or routing thresholds;
  they are reported separately, not folded into the CV routing-accuracy
  numbers.
