# Liturgy V/A window validation — Gemma weak labels

PROVENANCE: MODEL-GENERATED weak labels (gemma-4-12b, expression_v1 +
expression_v2 double-pass agreement). NOT human ground truth.

- Proposed window: valence in [-0.4, 0.2], arousal in [-0.3, 0.3]
- Agreed-distressed faces (mined this wave): 0
- Distressed OUTSIDE window: 0.0% (gate >= 90%)
- Agreed-reverent faces (prior artifacts): 32
- Reverent INSIDE window: 53.1% (gate >= 70%)

## Verdict: STOP (base rate) — only 0 agreed-distressed faces mined (< 20). Window unvalidated; no preset. Distress does not occur at a useful base rate in the mined corpora.

## Mining detail (why zero distressed)

Candidate photos were face-gated with the mandated `fair_expr_runner.py
mediapipe_bbox` (the same portrait-tuned FaceLandmarker the reverence corpus
used, for comparability), then the detected face crops were Gemma
double-pass-labeled with the EXISTING `expression_v1` + `expression_v2` prompts
(not paraphrased anew).

| source (candid, distress-plausible) | photos | faces detected | detection rate |
|---|---|---|---|
| GE2025 rally (14 Events) | 124 | 1 | 0.8% |
| FES event (14 Events) | 40 | 2 | 5.0% |
| 8 Street — Sunday 080924 | 158 | 23 | 14.6% |
| 8 Street — Chinatown (110924) | 223 | 3 | 1.3% |
| 8 Street — broad sample (1000, many days) | 1000 | 6 | 0.6% |
| **total** | **1545** | **35** | **2.3%** |

Gemma double-pass over the 35 detected faces: 30 agreed (5 disagreed, 0 parse
error), distributed **joyful 12, reverent/prayerful 10, neutral 7, unclear 1,
distressed 0**. Not one distressed face.

## Finding: the distress corpus does not exist here (two compounding scarcities)

1. **Distress base rate is ~0 in the available corpora.** The reverence
   prototype already found 3 distressed / 97 vigil+wedding photos; this wave adds
   0 distressed / 35 candid street+event faces. Even street candids with a
   detectable frontal face read as joyful/neutral/reverent — people photographed
   close-up in public are rarely in distress.
2. **The corpora that might contain distress have almost no detectable faces.**
   Candid street/rally photography is wide-shot: the portrait-tuned face-gate
   detects a usable frontal face in only 2.3% of frames. The corpora with high
   face-detection yield (liturgy, weddings — congregation close-ups) are exactly
   the reverent, distress-free ones. The two requirements — distress present AND
   a close frontal face — do not co-occur in this owner's library.

## Secondary finding: the window under-covers reverent faces too

Independently of the distress side, only **53.1% (17/32)** of the agreed-reverent
faces fall inside the proposed window (valence [-0.4, +0.2], arousal [-0.3, +0.3])
— below the 70% reverent-inside bar. The reverence report's "~75%" estimate does
not hold against the actual EmotiEffLib V/A join. So even the reverent-coverage
half of the gate fails; the window is unvalidated on both sides.

## Verdict rationale & recommendation

No `liturgy` preset is wired. The proposal to stop penalizing reverence proxies
inside a V/A window remains unvalidated: there are no distressed faces to prove
the window excludes them, and the window covers only 53% of reverent faces
anyway. This confirms and extends the reverence report's conclusion that the
reverent-vs-distressed question is **unanswerable on this library** — it needs a
purpose-collected distress corpus with close frontal faces (news/documentary
sets, or owner-labeled exceptions), which is future data-collection work, not
something more weak-labeling of the same events can fix.

## Artifacts

- `benchmarks/liturgy_validate.py` — window validation driver (this report).
- `benchmarks/runs/liturgy_expr_v1.json`, `liturgy_expr_v2.json` — Gemma
  double-pass expression labels over the 35 mined faces.
- Candidate/provenance/crop manifests are session-scratch (the mined face crops
  are transient; the labeled outputs above are the durable evidence).
