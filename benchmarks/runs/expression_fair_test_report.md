# Fair Three-Layer Test: EmotiEffLib (enet_b0_8_va_mtl, ONNX) vs DeepFace Emotion

Generated 2026-07-17T17:21:28+00:00

## Verdict: REPLACE DeepFace with EmotiEffLib

- Layer A: EmotiEffLib macro-F1 0.578 vs DeepFace 0.256 (needs >= DeepFace-0.03) -> PASS
- Layer B: EmotiEffLib mean flip-rate 0.176 vs DeepFace 0.272 (needs <=) -> PASS
- Layer C: EmotiEffLib happy-rate separation 0.285 vs DeepFace 0.099 (needs >= 0.079, i.e. within 20%) -> PASS

## Layer A — labeled benchmark anchor (RAF-DB)

Dataset: `deanngkl/raf-db-7emotions` (RAF-DB, Li & Deng 2017, HF mirror). Test images use the mirror's preserved train_/test_ RAF-DB filename prefix — this recovers RAF-DB's OFFICIAL test partition, not a self-defined split.

- License: RAF-DB's original license requires a research-use EULA with the authors; this HF reupload carries no explicit license tag. Used here for internal, non-redistributed benchmarking only — a known gray area, flagged not resolved (mirrors this project's existing pattern for non-permissive dependencies).
- Provenance caveat: The 'neutral' class in this mirror is sourced entirely from AffectNet, not RAF-DB (verified: every neutral row's image path starts with 'affectnet.zip', zero train_/test_ RAF-DB rows). All other six classes are genuine RAF-DB test-partition images (path prefix 'test_').

### EmotiEffLib 8-class -> dataset 7-class taxonomy mapping

anger->angry, disgust->disgust, fear->fear, happiness->happy, neutral->neutral, sadness->sad, surprise->surprise, **contempt has no RAF-DB/DeepFace equivalent** (scored as a miss whenever predicted).

| model | n | accuracy | macro-F1 | sub-analysis macro-F1 (sad/fear/neutral/angry) |
|---|---|---|---|---|
| deepface | 420 | 0.283 | 0.256 | 0.317 |
| emotiefflib | 420 | 0.576 | 0.578 | 0.610 |

### Per-class accuracy

| model | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| deepface | 0.167 | 0.000 | 0.250 | 0.633 | 0.383 | 0.367 | 0.183 |
| emotiefflib | 0.333 | 0.350 | 0.700 | 0.867 | 0.450 | 0.850 | 0.483 |

#### deepface full confusion matrix

| true \ pred | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| angry | 10 | 1 | 3 | 6 | 31 | 7 | 2 |
| disgust | 18 | 0 | 7 | 5 | 21 | 8 | 1 |
| fear | 8 | 0 | 15 | 18 | 5 | 9 | 5 |
| happy | 3 | 0 | 2 | 38 | 9 | 8 | 0 |
| neutral | 6 | 0 | 7 | 5 | 23 | 18 | 1 |
| sad | 7 | 1 | 9 | 9 | 12 | 22 | 0 |
| surprise | 7 | 0 | 11 | 10 | 18 | 3 | 11 |

#### deepface sad/fear/neutral/angry confusion matrix

| true \ pred | sad | fear | neutral | angry |
|---|---|---|---|---|
| sad | 22 | 9 | 12 | 7 |
| fear | 9 | 15 | 5 | 8 |
| neutral | 18 | 7 | 23 | 6 |
| angry | 7 | 3 | 31 | 10 |

#### emotiefflib full confusion matrix

| true \ pred | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| angry | 20 | 1 | 3 | 2 | 27 | 7 | 0 |
| disgust | 29 | 21 | 1 | 1 | 2 | 5 | 0 |
| fear | 6 | 0 | 42 | 2 | 1 | 7 | 2 |
| happy | 2 | 2 | 0 | 52 | 0 | 1 | 0 |
| neutral | 8 | 2 | 0 | 4 | 27 | 6 | 9 |
| sad | 5 | 1 | 1 | 1 | 1 | 51 | 0 |
| surprise | 6 | 2 | 18 | 0 | 1 | 4 | 29 |

#### emotiefflib sad/fear/neutral/angry confusion matrix

| true \ pred | sad | fear | neutral | angry |
|---|---|---|---|---|
| sad | 51 | 1 | 1 | 5 |
| fear | 7 | 42 | 1 | 6 |
| neutral | 6 | 0 | 27 | 8 |
| angry | 7 | 3 | 27 | 20 |

## Layer B — robustness, label-free

This measures STABILITY (agreement with the model's own unperturbed prediction), not correctness. A stable model keeps its prediction under a perturbation.

| model | n faces | blur_r2 | brightness_0.6 | brightness_1.4 | downscale_64 | downscale_96 | mean flip-rate |
|---|---|---|---|---|---|---|---|
| deepface | 50 | 0.180 | 0.340 | 0.180 | 0.400 | 0.260 | 0.272 |
| emotiefflib | 50 | 0.220 | 0.080 | 0.100 | 0.240 | 0.240 | 0.176 |

### EmotiEffLib valence/arousal drift (mean |delta|)

| perturbation | valence drift | arousal drift |
|---|---|---|
| blur_r2 | 0.054 | 0.056 |
| brightness_0.6 | 0.045 | 0.036 |
| brightness_1.4 | 0.041 | 0.031 |
| downscale_64 | 0.084 | 0.073 |
| downscale_96 | 0.063 | 0.040 |

## Layer C — context-distribution separation

### Face-gate survivors

| corpus | photos sampled | survived mediapipe gate |
|---|---|---|
| weddings | 80 | 41 |
| vigil | 30 | 30 |
| friends | 40 | 22 |

### Per-corpus distributions

| corpus | model | n | happy-rate | negative-rate | mean valence | std valence | mean arousal | std arousal |
|---|---|---|---|---|---|---|---|---|
| weddings | deepface | 41 | 0.366 | 0.293 | — | — | — | — |
| weddings | emotiefflib | 41 | 0.585 | 0.098 | 0.380 | 0.481 | 0.223 | 0.269 |
| vigil | deepface | 30 | 0.267 | 0.600 | — | — | — | — |
| vigil | emotiefflib | 30 | 0.300 | 0.500 | 0.080 | 0.366 | 0.008 | 0.144 |
| friends | deepface | 22 | 0.273 | 0.318 | — | — | — | — |
| friends | emotiefflib | 22 | 0.455 | 0.136 | 0.243 | 0.394 | 0.079 | 0.128 |

### Weddings-vs-Vigil separation effect size

| model | happy-rate diff (weddings - vigil) | Cohen's d valence |
|---|---|---|
| deepface | 0.099 | — |
| emotiefflib | 0.285 | 0.689 |

