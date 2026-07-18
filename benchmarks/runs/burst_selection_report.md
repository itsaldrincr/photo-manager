# Burst representative selection — face-aware winner gate FAILED

Signal-quality wave, Task A (handover 2026-07-19). Verdict: **FAIL — KEEP
blur-only `select_burst_winner`.** A MediaPipe eyes-open signal blended with
blur made the owner-match rate WORSE, not better; production is unchanged.

## Provenance & method

- Labels are the owner's IMPLICIT keep/reject decisions
  (`benchmarks/runs/wedding_ground_truth.json`, 347 rows, 116 keep / 231
  reject), derived from which frames survived into the owner's `Edited/`
  output. They are a proxy for owner preference, not deliberate per-burst
  adjudication — see caveats.
- Production Stage 1 burst grouping was reproduced exactly
  (`benchmarks/runs/burst_baseline.py`): the per-photo Stage 1 loop over each
  NAS Backup shoot, then `detect_bursts` over the survivors, forming the same
  temporal→dHash burst groups production forms. Both shoots' photos survived
  Stage 1 (ExN 257/257, ZD 106/106 — no blur/noise rejects), so every photo
  entered burst grouping.
- Metric: `winner_matches_owner_rate` — over burst groups containing ≥1
  implicit keeper, the share whose selected winner is an owner keeper.
- Candidate scorer (archived, out of the production tree, at
  `benchmarks/runs/burst_rank_candidate.py`): winner = argmax of
  `0.5·blur_norm + 0.5·eyes_open`, where `blur_norm` is within-group
  max-normalized tenengrad and `eyes_open = min(1, mean_EAR / 0.28)` from
  MediaPipe (neutral 0.5 for members with no detected face). Computed only for
  burst-group members (a bounded set).

## Baseline vs face-aware (burst winner selection)

| shoot | keeper-bearing burst groups | blur-only rate | face-aware rate |
|---|---|---|---|
| ExN (220924) | 25 | 0.560 | 0.480 |
| ZD (290325) | 0 | — | — |
| **pooled** | **25** | **0.560** | **0.480** |

Multi-keeper groups (selection cannot be perfect — owner kept >1 frame): 1 of 25.

**Gate:** requires `winner_matches_owner_rate` to improve by ≥ 10 pp pooled.
Observed change: **−8 pp** (0.560 → 0.480). The wall-clock and suite clauses
were not reached — the rate clause fails outright, and no operating point or
weight recovers it (see below).

## Why it fails (not a scorer bug)

The face-aware scorer changed the winner in only 2 of 25 groups, both from a
correct (owner-keeper) winner to a wrong one — zero groups flipped the other
way. Per-frame diagnostic on the two flips (blur tenengrad + MediaPipe EAR):

| group | frame | role | tenengrad | blur_norm | EAR | eyes_open | blended |
|---|---|---|---|---|---|---|---|
| 1 | DSCF1409 | owner keeper (blur winner) | 4428 | 1.000 | **0.099** | 0.354 | 0.677 |
| 1 | DSCF1410 | face-aware pick | 3888 | 0.878 | 0.293 | 1.000 | **0.939** |
| 2 | DSCF1588 | owner keeper (blur winner) | 701.9 | 1.000 | 0.238 | 0.849 | 0.925 |
| 2 | DSCF1589 | face-aware pick | 695.1 | 0.990 | 0.248 | 0.887 | **0.939** |

- **Group 1:** the owner's keeper is a genuine near-blink (EAR 0.099, well
  below the 0.20 closed-eye threshold). The scorer correctly identified it as a
  blink and picked the open-eyed frame — but the owner kept the blink. On this
  corpus the owner's burst preference does not track open eyes; optimizing for
  eyes-open moves away from it.
- **Group 2:** two near-identical frames (Δtenengrad 1%, ΔEAR 0.01). The flip
  is within metric noise; the implicit label cannot distinguish them.

Because zero groups exist where eyes-open would *recover* an owner keeper, no
reweighting or threshold can lift the rate above baseline — the achievable
ceiling for this signal on this corpus is ≤ baseline.

## Caveats

- **Implicit labels.** `Edited/`-derived keep/reject is a preference proxy, not
  deliberate per-burst adjudication; group 2's noise-margin flip is exactly the
  granularity this label cannot resolve.
- **Small, single-shoot sample.** Only 25 keeper-bearing burst groups, all from
  ExN; ZD produced 0 (its keepers do not co-occur inside multi-frame burst
  groups). The burst-winner lever is therefore both narrow and, here, negative.
- **The larger lever is duplicates, not bursts.** The keeper losses flagged in
  the handover (215 duplicate losers on ExN) come from the DINOv2/CNN duplicate
  bucket, whose representative is currently `paths[0]` (arbitrary order), not a
  quality pick. Measured duplicate `winner_matches_owner_rate`: **0.739 pooled**
  (23 keeper-bearing groups, 9 multi-keeper; ExN 0.765 over 17, ZD 0.667 over
  6). This is where representative choice actually decides keeper survival.

## Recommendation (future work, not implemented)

Do NOT wire eyes-open into burst selection — it is validated-negative here.
The duplicate representative choice (`stage1/duplicate.py`, `paths[0]`) is the
higher-leverage target, but the same eyes-open signal is unlikely to help on
this corpus for the same reason; a different signal (subject sharpness, or an
explicit owner-edit signal) and a proper gate would be needed. Measure before
touching selection.

## Artifacts

- `benchmarks/runs/burst_baseline.py` — production-faithful measurement driver
  (blur-only + face-aware in one pass).
- `benchmarks/runs/burst_rank_candidate.py` — the rejected scorer (reproducible).
- `benchmarks/runs/burst_baseline.json` — full per-group records, both shoots.
- `benchmarks/runs/stage1_timing.py` — wall-clock harness (unused; rate clause
  failed before the wall-clock clause was reached).
