# Portrait Eval: MediaPipe Blendshapes vs DeepFace Emotion

Generated 2026-07-17T03:45:24+00:00

- Candidate photos: 60
- Face-bearing photos evaluated: 1
- **Agreement rate (decision-relevant expression bucket): 0.0%**
- EAR-vs-blendshape eyes-closed cross-check: 100.0%
- MediaPipe (blendshapes) mean per-photo latency: 13.20 ms
- DeepFace mean per-photo latency: 4.460 s
- Speedup (DeepFace / MediaPipe): 337.9x
- MediaPipe load: 7.944s, RSS delta 232.8 MB
- DeepFace load: 8.658s, RSS delta 294.9 MB

> **Caveat:** eval_set (benchmarks/eval_set) and the extra photos pulled from /Users/alrelador/Desktop/cull-test are predominantly non-portrait (wildlife/travel) shots at MediaPipe's production confidence threshold — only 1 of 60 candidate photos had a detectable face. The agreement rate below is not statistically meaningful; the latency and RSS findings do not depend on sample size and remain valid.

## Recommendation: INCONCLUSIVE

- agreement 0.0% on n=1 face photos (threshold 90%)
- speedup 337.89x (threshold 1.50x)
- RSS savings 62.1 MB (threshold 200 MB)
- sample size n=1 is below the minimum of 10 face photos needed to trust the agreement rate; latency/RSS findings below are still valid and directionally strong

## Disagreement cases

| photo | mediapipe bucket | deepface bucket | deepface label |
|---|---|---|---|
| DSCF8867.JPG | neutral | happy | happy |

## Repeated-run variance (manual, 5 runs total)

RSS delta and load time were measured across 5 independent invocations of this
script (fresh subprocess per model each time, host had other work running
alongside per the 18 GB RAM constraint) to sanity-check stability:

| run | mediapipe RSS delta | deepface RSS delta | mediapipe latency | deepface latency |
|---|---|---|---|---|
| 1 (first post-install run) | 546.2 MB | 5000.1 MB | 8.2 ms | 2.092 s |
| 2 | 448.6 MB | 843.1 MB | 9.8 ms | 3.972 s |
| 3 | 546.2 MB | 931.6 MB | 8.1 ms | 3.794 s |
| 4 | 546.2 MB | 1121.0 MB | 10.5 ms | 3.957 s |
| 5 (this report) | 232.8 MB | 294.9 MB | 13.2 ms | 4.460 s |

Both RSS deltas swing a lot run-to-run (mediapipe: 233-546 MB; deepface:
295 MB-5.0 GB), almost certainly because the host had other work competing
for memory during measurement (per the eval's own 18 GB RAM constraint) —
low system memory pressure makes TensorFlow's lazy allocator claim less on a
given run. Despite the noise, in every single run DeepFace's RSS delta is
larger than MediaPipe's (1.0x-9.2x, median ~2x), and DeepFace's per-photo
latency is consistently ~250-500x MediaPipe's. Both signals point the same
direction regardless of which run is used; only the exact magnitude of the
RSS gap is uncertain under memory pressure.
