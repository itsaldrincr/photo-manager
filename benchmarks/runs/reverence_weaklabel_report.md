# Reverence weak-label prototype — EmotiEffLib V/A vs Gemma labels

PROVENANCE: expression labels are MODEL-GENERATED weak labels (labeler:
gemma-4-12b, prompts expression_v1 + expression_v2, double-pass agreement
filter). NOT human ground truth. This is a prototype analysis only — no
production scoring change is implied or implemented.

- Joined faces (agreed label + EmotiEffLib reading): 86
- By context: vigil 51, weddings 35

## Cross-tab

| gemma \ emotieff | Anger | Contempt | Disgust | Fear | Happiness | Neutral | Sadness | Surprise |
|---|---|---|---|---|---|---|---|---|
| reverent/prayerful | 0 | 0 | 2 | 0 | 5 | 4 | 11 | 10 |
| joyful | 0 | 0 | 1 | 0 | 24 | 0 | 1 | 2 |
| neutral | 1 | 2 | 1 | 0 | 1 | 6 | 0 | 1 |
| distressed | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |
| eyes-closed-rest | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| unclear | 1 | 1 | 1 | 0 | 1 | 3 | 1 | 2 |

## Valence/arousal by Gemma label

| gemma label | n | valence mean±sd | arousal mean±sd |
|---|---|---|---|
| reverent/prayerful | 32 | -0.058±0.277 | +0.026±0.253 |
| joyful | 28 | +0.541±0.381 | +0.236±0.154 |
| neutral | 12 | +0.029±0.210 | +0.009±0.163 |
| distressed | 3 | -0.218±0.200 | +0.182±0.506 |
| eyes-closed-rest | 1 | +0.448±0.000 | -0.135±0.000 |
| unclear | 10 | -0.034±0.167 | +0.144±0.223 |

## Contrast: reverent/prayerful vs distressed

- Group sizes: reverent/prayerful 32, distressed 3
- Groups too small for effect sizes (need >= 5 each) — separation question UNANSWERABLE on this corpus.

## Contrast: reverent/prayerful vs joyful

- Group sizes: reverent/prayerful 32, joyful 28
- Valence: Cohen's d -1.80, AUC 0.113
- Arousal: Cohen's d -1.00, AUC 0.241
## Findings

1. **The key question is unanswerable on liturgy/wedding corpora alone.**
   Gemma double-pass found only 3 agreed distressed faces in 97 photos —
   vigils and weddings simply do not produce distressed expressions at a
   useful base rate. Answering reverent-vs-distressed needs a corpus that
   actually contains distress (news/documentary sets, or owner-labeled
   exceptions); collecting one is future work, not something more weak
   labeling of the same events can fix.
2. **Reverent faces are systematically misread by the 7/8-class head.**
   EmotiEffLib maps agreed-reverent faces to Sadness (11/32) and Surprise
   (10/32) far more than Neutral (4/32). A consumer that treats "sad" as a
   negative-quality signal would penalize prayer — the exact liturgy
   failure mode the owner flagged.
3. **Reverent is well-separated from joyful in V/A space** (valence
   Cohen's d -1.80, AUC 0.89 in the joyful direction; arousal d -1.00).
   Reverent clusters near-neutral valence (-0.06±0.28) and near-zero
   arousal (+0.03±0.25) — calm, not negative. So V/A carries usable signal
   even where the discrete label misleads.
4. Double-pass disagreement rate 11.3%, concentrated on the
   reverent↔neutral boundary (5/11) — consistent with those being adjacent
   categories rather than labeling noise.

## Proposal (NOT implemented — owner decision required)

If a `liturgy` preset is ever added, do not add a reverence bonus yet (the
distressed contrast is unvalidated). Instead, make the preset *stop
penalizing* reverence proxies:

- Treat `dominant_emotion in {sad, surprise}` as neutral for any
  penalty/hint logic when `valence` is in [-0.4, +0.2] and `arousal` is in
  [-0.3, +0.3] (the observed reverent cluster, covering ~75% of agreed
  reverent faces).
- Halve `eyes_closed_penalty` when the same V/A window holds — closed eyes
  at a liturgy are frequently prayer, and Gemma's reverent definition
  explicitly includes them.
- Validation gate before wiring: >= 20 owner-labeled distressed liturgy
  faces showing the window excludes them (the 3 weak-labeled distressed
  faces here all have |valence| > 0.2 or arousal > 0.5 — consistent but
  far too few).
