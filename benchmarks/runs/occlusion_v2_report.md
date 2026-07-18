# Occlusion v2 candidate signals — Gemma weak labels

PROVENANCE: labels are MODEL-GENERATED weak labels (labeler: gemma-4-12b,
prompts occlusion_v1 + occlusion_v2, double-pass agreement). NOT human
ground truth. Signals are cheap, computed from the same full-res image +
MediaPipe landmarks the production detector consumes.

- Joined agreed labels with a detected face: 138 (25 occluded, 113 clean)
- Operating points chosen to maximize F1 s.t. clean-FP <= 0.1

## Per-candidate operating points (higher score = more occluded)

| candidate | ROC-AUC | threshold | precision | recall | F1 | clean-FP |
|---|---|---|---|---|---|---|
| blendshape_std | 0.628 | -0.1096 | 0.357 | 0.200 | 0.256 | 0.080 |
| blendshape_mean | 0.592 | -0.06324 | 0.214 | 0.120 | 0.154 | 0.097 |
| texture_ratio | 0.564 | -0.5785 | 0.182 | 0.080 | 0.111 | 0.080 |
| skin_outside_hull_frac | 0.541 | -0 | 0.200 | 0.040 | 0.067 | 0.035 |
| boundary_edge_density | 0.502 | inf | 0.000 | 0.000 | 0.000 | 0.000 |
| ensemble(blendshape_std+blendshape_mean) | 0.602 | 1.234 | 0.333 | 0.200 | 0.250 | 0.088 |

## Verdict: FAIL — best candidate blendshape_std F1 0.256 < 0.55 at clean-FP <= 0.1. No production change; texture-ratio detect_occlusion stays.

## Interpretation

- **Blendshape activation is the least-weak signal but still weak.** Occluded
  faces show somewhat flatter blendshape variance (blendshape_std AUC 0.628),
  consistent with the hypothesis that MediaPipe emits low/flat activations over
  occluded regions — but the separation is far too soft to threshold: at the
  best clean-FP-bounded operating point it recovers only 5 of 25 occluded faces
  (recall 0.20) at precision 0.357.
- **Geometry and edge signals are near-random.** skin_outside_hull_frac (AUC
  0.541) and boundary_edge_density (AUC 0.502) carry essentially no signal. The
  hand-over-face geometry assumption is too narrow — real occluders here are
  varied (drinks, microphones, candles, other people, veils, hair), many are
  not skin, and MediaPipe fits the landmark hull *over* the occluder so the
  "extra skin outside the hull" cue rarely fires. The face-boundary edge band is
  swamped by ordinary facial and background edges.
- **The ensemble does not help.** Averaging the two blendshape signals (the only
  two above AUC 0.59) yields AUC 0.602, F1 0.250 — no lift over blendshape_std
  alone, because they are correlated views of the same activation magnitude.
- **The texture ratio is confirmed a no-op on real occluders** (AUC 0.564, F1
  0.111), reproducing the prior recalibration finding from a fresh signal path.

## Synthetic gate: not reached

The gate's second clause (synthetic-eval F1 within 0.05 of the texture ratio's
0.776) is only worth evaluating for a candidate that clears the real-corpus bar.
None did, so the synthetic set was not regenerated — clearing the real bar is a
precondition, and no candidate came close (best 0.256 vs 0.55 required).

## Caveats

- Labels are Gemma double-pass weak labels (25 occluded / 113 clean of 138) —
  the heavy class imbalance makes precision fragile at any operating point.
- All signals are cheap approximations; a genuinely discriminative occlusion
  detector on this corpus likely needs either the VLM itself (expensive, the
  thing this was meant to avoid) or a trained occlusion head, not a hand-crafted
  geometric/blendshape heuristic.

## Recommendation (future work, not implemented)

Keep the production texture-ratio `detect_occlusion` and its
`PORTRAIT_FACE_OCCLUSION_MIN = 0.32` (already a documented no-op that never
false-flags real portraits — harmless). Do not adopt any candidate here. A real
fix would require a labeled occlusion head or accepting VLM cost; both are out
of scope for a cheap-signal swap.

## Artifacts

- `benchmarks/occlusion_v2_signals.py` — signal extractor (blendshapes-enabled
  FaceLandmarker over the corpus originals).
- `benchmarks/occlusion_v2_eval.py` — AUC + operating-point + gate driver.
- `benchmarks/runs/occlusion_v2_signals.json` — raw per-face signals (156 faces).
- `benchmarks/runs/occlusion_v2_manifest.json`, `occlusion_v2_eval_config.json`
  — inputs.
