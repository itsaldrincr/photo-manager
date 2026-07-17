# DINOv2-small vs imagededup CNN + dHash — Dedupe Eval

Corpus: 120 photos across 20 groups (20 base + 5 controlled variants each). Natural EXIF bursts (<0.5s apart) found in source corpus: 0.

- Method A1: production `imagededup` CNN (`cull/stage1/duplicate.py`, threshold=0.98).
- Method A2: production dHash gate (`cull/stage1/burst.py`, hamming<=8 -> similarity>=0.875).
- Method B: `facebook/dinov2-small` cosine similarity, best-F1 threshold swept on this corpus.

## Overall pairwise precision / recall / F1

| method | threshold | precision | recall | f1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| imagededup CNN (production) | 0.980 | 1.000 | 0.797 | 0.887 | 239 | 0 | 61 | 6840 |
| dHash gate (production) | 0.875 | 1.000 | 0.680 | 0.810 | 204 | 0 | 96 | 6840 |
| facebook/dinov2-small | 0.800 | 1.000 | 1.000 | 1.000 | 300 | 0 | 0 | 6840 |

## Recall by variant type

| variant | imagededup CNN (production) | dHash gate (production) | facebook/dinov2-small |
|---|---|---|---|
| reencode | 1.000 (20/20) | 1.000 (20/20) | 1.000 (20/20) |
| resize | 1.000 (20/20) | 1.000 (20/20) | 1.000 (20/20) |
| crop | 0.500 (10/20) | 0.050 (1/20) | 1.000 (20/20) |
| brightness | 1.000 (20/20) | 1.000 (20/20) | 1.000 (20/20) |
| rotation | 0.950 (19/20) | 1.000 (20/20) | 1.000 (20/20) |

## Latency and model memory

| method | mean ms/photo | p95 ms/photo | model memory (MB, fp32) |
|---|---|---|---|
| imagededup CNN (production) | 173.13 | 243.10 | 3.71 |
| dHash gate (production) | 86.70 | 165.97 | 0.00 |
| facebook/dinov2-small | 29.47 | 40.63 | 88.23 |

## Where hash fails but DINOv2 succeeds

- DSCF7340/crop: weak_sim=0.750 (miss) vs strong_sim=0.944 (hit)
- DSCF7966/crop: weak_sim=0.844 (miss) vs strong_sim=0.967 (hit)
- DSCF8580/crop: weak_sim=0.672 (miss) vs strong_sim=0.908 (hit)
- DSCF8595/crop: weak_sim=0.719 (miss) vs strong_sim=0.965 (hit)
- DSCF8637/crop: weak_sim=0.703 (miss) vs strong_sim=0.937 (hit)
- DSCF8672/crop: weak_sim=0.781 (miss) vs strong_sim=0.962 (hit)
- DSCF8674/crop: weak_sim=0.781 (miss) vs strong_sim=0.932 (hit)
- DSCF8821/crop: weak_sim=0.750 (miss) vs strong_sim=0.940 (hit)
- DSCF8898/crop: weak_sim=0.859 (miss) vs strong_sim=0.946 (hit)
- DSCF8930/crop: weak_sim=0.766 (miss) vs strong_sim=0.931 (hit)
- DSCF8997/crop: weak_sim=0.609 (miss) vs strong_sim=0.956 (hit)
- DSCF9053/crop: weak_sim=0.703 (miss) vs strong_sim=0.962 (hit)
- DSCF9181/crop: weak_sim=0.781 (miss) vs strong_sim=0.977 (hit)
- DSCF9198/crop: weak_sim=0.719 (miss) vs strong_sim=0.942 (hit)
- DSCF9207/crop: weak_sim=0.734 (miss) vs strong_sim=0.977 (hit)
- DSCF9317/crop: weak_sim=0.719 (miss) vs strong_sim=0.970 (hit)
- DSCF9647/crop: weak_sim=0.609 (miss) vs strong_sim=0.924 (hit)
- DSCF9661/crop: weak_sim=0.844 (miss) vs strong_sim=0.955 (hit)
- DSCF9839/crop: weak_sim=0.766 (miss) vs strong_sim=0.965 (hit)

## Decision: ADD-DINOV2-TIER

- largest DINOv2 recall gain over CNN/dhash on crop+rotation: +0.500 (materiality threshold: +0.150)
- DINOv2 mean latency 29.47ms/photo vs budget 250ms/photo (OK)
- overall recall — CNN 0.797, dhash 0.680, DINOv2 1.000
