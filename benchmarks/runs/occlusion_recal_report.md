# Occlusion threshold recalibration — Gemma weak labels vs synthetic eval

PROVENANCE: labels are MODEL-GENERATED weak labels (labeler: gemma-4-12b,
prompts occlusion_v1 + occlusion_v2, temperature 0.0, double-pass
agreement filter). They are NOT human ground truth; every number below
inherits that caveat. Gemma self-reported confidence is uninformative
(benchmarks/LOG.md 2026-07-18) and was NOT used for gating.

## Interpretation (why the verdict is KEEP despite a "better" F1 existing)

The headline result is negative and important: **the production
texture-variance ratio carries almost no signal on real occluders.**
Weak-labeled-occluded faces span ratios 0.56–0.96 (median 0.74) while clean
faces span 0.43–0.98 (median 0.82) — the distributions overlap almost
completely. At the current threshold 0.32 the detector NEVER fires on any of
the 138 real faces (recall 0, F1 0): in production it is a no-op, exactly as
suspected when it was flagged as a known gap. The mechanically-best sweep
point (0.80, F1 0.356) only "wins" because beating F1=0 is free; it would
false-flag 43% of clean faces, and with the wedding preset's 0.12
face_occlusion_penalty that converts a harmless no-op into an active harm.
The gate therefore includes a clean-false-flag-rate constraint (<= 0.10),
which no candidate threshold satisfies with useful recall.

Why synthetic and real diverge: the synthetic eval's occluders (opaque
rectangles, skin ellipses, blur patches) are locally texture-flat, which is
precisely what the Laplacian-variance ratio measures. Real occluders —
overwhelmingly hands in this corpus (22 of 25 pass-1 occluders), typically
prayer-clasped hands at the Vigil (21 of the 41 vigil faces) — carry their
own edges, knuckle lines, and shadows, so the "occluded" region is no less
textured than eyes or mouth. Detecting real occlusion needs a semantic
signal (e.g. MediaPipe blendshape confidence collapse, face-parsing
coverage, or a VLM check on stage-3-routed portraits), not a texture
statistic. Recorded as future work; NOT implemented here.

## Weak-label corpus

- Face-gated real photos labeled: 160
- Double-pass agreement kept: 138 | disagreed (dropped): 18 | unparsed (dropped): 4
- Disagreement rate (comparable photos): 0.115
- Joined with a production ratio (face detected): 138
- Composition — friends: 48 (3 occluded); vigil: 41 (21 occluded); weddings: 49 (1 occluded)

## Real-corpus sweep (weak labels)

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.06 | 0.000 | 0.000 | 0.000 |
| 0.08 | 0.000 | 0.000 | 0.000 |
| 0.10 | 0.000 | 0.000 | 0.000 |
| 0.12 | 0.000 | 0.000 | 0.000 |
| 0.14 | 0.000 | 0.000 | 0.000 |
| 0.16 | 0.000 | 0.000 | 0.000 |
| 0.18 | 0.000 | 0.000 | 0.000 |
| 0.20 | 0.000 | 0.000 | 0.000 |
| 0.22 | 0.000 | 0.000 | 0.000 |
| 0.24 | 0.000 | 0.000 | 0.000 |
| 0.26 | 0.000 | 0.000 | 0.000 |
| 0.28 | 0.000 | 0.000 | 0.000 |
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.32 | 0.000 | 0.000 | 0.000 |
| 0.34 | 0.000 | 0.000 | 0.000 |
| 0.36 | 0.000 | 0.000 | 0.000 |
| 0.38 | 0.000 | 0.000 | 0.000 |
| 0.40 | 0.000 | 0.000 | 0.000 |
| 0.42 | 0.000 | 0.000 | 0.000 |
| 0.44 | 0.000 | 0.000 | 0.000 |
| 0.46 | 0.000 | 0.000 | 0.000 |
| 0.48 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.000 | 0.000 | 0.000 |
| 0.52 | 0.000 | 0.000 | 0.000 |
| 0.54 | 0.000 | 0.000 | 0.000 |
| 0.56 | 0.125 | 0.040 | 0.061 |
| 0.58 | 0.182 | 0.080 | 0.111 |
| 0.60 | 0.154 | 0.080 | 0.105 |

## Synthetic-occluder sweep (labels from generated eval set)

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.06 | 1.000 | 0.509 | 0.675 |
| 0.08 | 1.000 | 0.528 | 0.691 |
| 0.10 | 1.000 | 0.528 | 0.691 |
| 0.12 | 1.000 | 0.547 | 0.707 |
| 0.14 | 0.967 | 0.547 | 0.699 |
| 0.16 | 0.967 | 0.547 | 0.699 |
| 0.18 | 0.938 | 0.566 | 0.706 |
| 0.20 | 0.938 | 0.566 | 0.706 |
| 0.22 | 0.914 | 0.604 | 0.727 |
| 0.24 | 0.892 | 0.623 | 0.733 |
| 0.26 | 0.895 | 0.642 | 0.747 |
| 0.28 | 0.895 | 0.642 | 0.747 |
| 0.30 | 0.854 | 0.660 | 0.745 |
| 0.32 | 0.844 | 0.717 | 0.776 |
| 0.34 | 0.826 | 0.717 | 0.768 |
| 0.36 | 0.826 | 0.717 | 0.768 |
| 0.38 | 0.830 | 0.736 | 0.780 |
| 0.40 | 0.830 | 0.736 | 0.780 |
| 0.42 | 0.765 | 0.736 | 0.750 |
| 0.44 | 0.765 | 0.736 | 0.750 |
| 0.46 | 0.750 | 0.736 | 0.743 |
| 0.48 | 0.741 | 0.755 | 0.748 |
| 0.50 | 0.714 | 0.755 | 0.734 |
| 0.52 | 0.702 | 0.755 | 0.727 |
| 0.54 | 0.695 | 0.774 | 0.732 |
| 0.56 | 0.683 | 0.774 | 0.726 |
| 0.58 | 0.677 | 0.792 | 0.730 |
| 0.60 | 0.677 | 0.792 | 0.730 |

## Gate

- Current threshold 0.32: real F1 0.000, synthetic F1 0.776
- Best real-corpus threshold 0.80: real F1 0.356, synthetic F1 0.803
- Clean-face false-flag rate at best threshold: 0.434
- Rule: adopt only if real F1 gain >= 0.05 and synthetic F1 drop <= 0.05 and clean false-flag rate <= 0.10

## Verdict: KEEP 0.32
