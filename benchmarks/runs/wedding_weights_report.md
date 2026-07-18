# Wedding-preset weights from implicit Edited/ labels — gate verdict FAIL (weights unchanged)

PROVENANCE: labels are IMPLICIT — "keep" means the owner exported an edited
version of the Backup photo, mined by content-matching renamed Edited
exports back to Backup originals with DINOv2 (crop-robust embeddings,
cosine nearest-neighbor). No human labeled these photos for this task;
every number below inherits that caveat.

## Corpora and matching

Charmaine's Wedding (the handoff's example) has only 35 Backup / 31 Edited
photos — too few negatives — so the two Backup+Edited shoots with real
reject populations were used instead:

| shoot | Backup | Edited | clean NN accepts | burst-sibling cases | uncertain (dropped) |
|---|---|---|---|---|---|
| 220924 ExN Dance Rehearsal | 257 | 87 | 59 | 28 (sim >= 0.89, margin < 0.03) | 5 |
| 290325 Zack & Dora Rehearsal | 107 | 60 | 48 | 0 | 12 |

Edited filenames are sequential renamed exports (no DSCF/HIEP token), so
filename matching is impossible. Match policy: accept when cosine
similarity >= 0.60 AND best-vs-second margin >= 0.03. Burst-sibling cases
(similarity >= 0.85 but margin below 0.03 — visually indistinguishable
burst frames) label the top-1 backup keep and EXCLUDE the runner-up frame
from ground truth (16 exclusions). Uncertain matches are dropped entirely.
Ground truth: 347 photos, 116 keep / 231 reject.

## Headline side-finding: Stage 1 auto-drops most of the owner's selects

Production-faithful scoring (real Stage 1 burst/duplicate filtering, one
run per shoot) kept only 105 of 347 labeled photos; **67 of the 116
implicit keepers were auto-dropped by Stage 1** (62 burst losers + 215
duplicate drops on the ExN shoot alone: dance rehearsals are continuous
burst coverage). Caveat: a "dropped keeper" usually has a surviving
near-identical sibling, so this is not 67 lost moments — but it does mean
burst/duplicate representative selection, not Stage 2 fusion, decides most
keep/reject outcomes on burst-heavy wedding shoots. If wedding-preset
behavior is to be tuned, BURST/dedupe representative choice is the lever
with 3x the leverage of composite weights.

Because of that, the weight fit was run on a second scoring pass that
re-admits burst/duplicate losers to Stage 2 (all 347 labeled photos get
composites; reducer stats per shoot). The production-faithful
survivors-only fit is reported as a secondary check.

## Primary fit (all 347 labeled photos, burst/dupe losers re-admitted)

Composite-reconstruction sanity check (recomputed vs. real pipeline
composite, current wedding weights): max abs diff 0.0966, mean 0.0386
(n=347). Larger than the holiday calibration's 0.198/0.063 max/mean
pattern scaled — the re-admitted rows are exactly the ones production
fusion never sees, so reconstruction there leans on imputed None-handling;
treated as acceptable for a FAIL-gate conclusion, would need tightening
before any PASS could be trusted.

## Routing threshold sweep (current wedding weights)

Current thresholds: AMBIGUOUS_MIN=0.85, KEEPER_MIN=0.94

Best under VLM-share <= 35%: AMBIGUOUS_MIN=0.83, KEEPER_MIN=0.95, error=0.3256, vlm_share=0.2565

Frontier (best error at each VLM-share cap):

| vlm_share_cap | ambiguous_min | keeper_min | error | vlm_share |
|---|---|---|---|---|
| 10% | 0.88 | 0.93 | 0.3333 | 0.0576 |
| 15% | 0.88 | 0.93 | 0.3333 | 0.0576 |
| 20% | 0.84 | 0.95 | 0.3273 | 0.1988 |
| 25% | 0.83 | 0.93 | 0.3257 | 0.2478 |
| 30% | 0.83 | 0.95 | 0.3256 | 0.2565 |
| 35% | 0.83 | 0.95 | 0.3256 | 0.2565 |
| 50% | 0.83 | 0.95 | 0.3256 | 0.2565 |
| 100% | 0.79 | 0.95 | 0.3052 | 0.5562 |

## Logistic regression fit (full data, standardized metrics -> keep/reject)

### Core 4 (topiq / laion_aesthetic / clipiqa / exposure)

| metric | current weight | LR coefficient | fitted weight (floored, normalized) |
|---|---|---|---|
| topiq | 0.250 | +0.1407 | 0.374 |
| laion_aesthetic | 0.400 | +0.2356 | 0.626 |
| clipiqa | 0.250 | -0.2654 | 0.000 |
| exposure | 0.100 | -0.2080 | 0.000 |

### Non-core (composition / taste / penalties / bonus) — reported only, NOT applied per task scope

| metric | current weight | LR coefficient | expected sign | agrees |
|---|---|---|---|---|
| composition | 0.150 | -0.4836 | positive | **NO** |
| taste_probability | 0.150 | -0.3093 | positive | **NO** |
| scene_start_bonus | 0.040 | +0.1296 | positive | yes |
| tilt_penalty | 0.050 | +0.0568 | negative | **NO** |
| palette_outlier_score | 0.050 | +0.0736 | negative | **NO** |
| exposure_drift_score | 0.050 | -0.0691 | negative | yes |
| exif_anomaly_score | 0.030 | -0.0485 | negative | yes |

## 5-fold cross-validated routing accuracy (each config at its own best threshold, VLM-share <= 35%)

| fold | current weights acc | fitted weights acc |
|---|---|---|
| 1 | 0.7593 | 0.7708 |
| 2 | 0.6842 | 0.6852 |
| 3 | 0.6667 | 0.6935 |
| 4 | 0.6889 | 0.6765 |
| 5 | 0.6667 | 0.6667 |
| **mean** | **0.6931** | **0.6985** |

Improvement: +0.54 points (gate requires >= 3.0)

## Gate verdict: FAIL

## Honest limitations

- **Implicit labels, not triage labels.** "Keep" means the owner exported an
  edited version of the photo; "reject" means they did not. Editing selection
  conflates artistic preference, client requests, and near-duplicate choice —
  it is a noisier signal than the explicit keep/reject triage used for the
  holiday calibration.
- **Edited->Backup matching is itself model-based.** Pairs come from DINOv2
  nearest-neighbor cosine matching (edited exports are renamed, so filename
  matching is impossible). Only high-confidence matches (similarity/margin
  gated) are used; unmatched Edited files are excluded and reported.
- **Two-shoot corpus, one photographer, rehearsal-heavy.** Both shoots are
  wedding-rehearsal/dance events by the same owner. Weights fitted here may
  not transfer to ceremony/reception coverage.
- **Single-owner taste + taste-model confound.** As in the holiday
  calibration: the live taste term is trained on the disagreement-biased
  override log and is captured as-is in both baseline and fitted composites.
- **Per-shoot scoring.** Each shoot was scored as its own stage1+2 run so
  reducer stats use the correct population.
- **Stage 1 drop-outs are out of scope for weight fitting** and are reported
  separately; a dropped ground-truth keeper cannot be recovered by weight
  tuning.

## Secondary fit (production-faithful survivors only, n=105)

| config | CV routing accuracy |
|---|---|
| current wedding weights | 0.6506 |
| fitted core-4 weights | 0.6266 |

Improvement: -2.41 points (gate requires >= 3.0) -> **FAIL**

## Verdict

Both fits FAIL the +3.0 CV-point promotion gate (primary
+0.54, secondary
-2.41); current wedding weights
(topiq 0.25 / laion 0.40 / clipiqa 0.25 / exposure 0.10) stay. This
mirrors the holiday attempt (+1.06 FAIL): composite-weight tuning is not
where wedding culling quality lives. The negative composition/taste
coefficients on this corpus additionally suggest the owner's export choice
on rehearsal shoots tracks moment/pose over formal composition — another
reason not to fit weights from implicit labels alone.
