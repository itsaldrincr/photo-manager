# Portrait Eval: MediaPipe Blendshapes vs DeepFace Emotion

Generated 2026-07-17T04:01:23+00:00

- Candidate photos: 30
- Face-bearing photos evaluated: 30
- **Agreement rate (decision-relevant expression bucket): 30.0%**
- EAR-vs-blendshape eyes-closed cross-check: 76.7%
- MediaPipe (blendshapes) mean per-photo latency: 8.20 ms
- DeepFace mean per-photo latency: 1.050 s
- Speedup (DeepFace / MediaPipe): 128.1x
- MediaPipe load: 5.781s, RSS delta 546.1 MB
- DeepFace load: 5.111s, RSS delta 6552.9 MB

## Recommendation: KEEP

- agreement 30.0% on n=30 face photos (threshold 90%)
- speedup 128.11x (threshold 1.50x)
- RSS savings 6006.8 MB (threshold 200 MB)
- agreement below threshold; blendshapes do not reproduce DeepFace's signal

## Disagreement cases

| photo | mediapipe bucket | deepface bucket | deepface label |
|---|---|---|---|
| DSCF0166 52 Edited.jpg | neutral | negative | fear |
| DSCF0180.JPG | neutral | negative | fear |
| DSCF0210.JPG | happy | neutral | neutral |
| DSCF0213.JPG | neutral | negative | sad |
| DSCF0226.JPG | happy | negative | sad |
| DSCF0244.JPG | neutral | negative | fear |
| DSCF0261.JPG | neutral | negative | fear |
| DSCF0264.JPG | neutral | negative | angry |
| DSCF0366.JPG | happy | negative | angry |
| DSCF0369.JPG | happy | negative | sad |
| DSCF0370 145 Edited.jpg | happy | negative | sad |
| DSCF0373.JPG | neutral | surprised | surprise |
| DSCF0377.JPG | neutral | happy | happy |
| DSCF0384.JPG | happy | negative | sad |
| DSCF0387.JPG | neutral | negative | sad |
| DSCF0484.JPG | neutral | negative | sad |
| DSCF0494.JPG | neutral | negative | sad |
| DSCF0588 181 Edited.jpg | neutral | negative | fear |
| DSCF1422.JPG | neutral | negative | angry |
| DSCF1490.JPG | happy | negative | fear |
| DSCF1491.JPG | neutral | negative | sad |
