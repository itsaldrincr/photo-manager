---
id: task_720
name: occlusion_weaklabel_recal
state: EXECUTING
step: 2 of 5
depends: []
checkpoint: 7c21e4
created: 2026-07-18
---

## Program (immutable — set at planning)
1. Build ~150-photo face-bearing corpus (Vigil + weddings + friends contexts) pulled from NAS, face-gated with the MediaPipe stage.
2. Gemma double-pass weak labels: strict-JSON `{"face_occluded": bool, "occluder": str|null, "confidence": float}` with two paraphrased prompts; keep agreements only; log disagreement rate. Serial, one model resident.
3. Run production detect_occlusion over corpus; sweep PORTRAIT_FACE_OCCLUSION_MIN; P/R/F1 vs weak labels AND vs regenerated synthetic eval.
4. GATE: adopt new threshold only if real-corpus F1 +>=0.05 and synthetic F1 drop <=0.05. Write benchmarks/runs/occlusion_recal_report.md with provenance ("labeler: gemma-4-12b, prompt vX").
5. If adopted: config + tests updated, non-golden suite green, commit as own unit.

## Registers (mutable — agent writes after each step)
- [ckpt 04b9bd] Step 1 DONE: 700 photos pulled (vigil 270 / weddings 240 / friends 165 / moments 25; NAS ssh for vigil, SMB for the rest — ssh user lacks read perms on from-external). MediaPipe gate → 234 faces; corpus = 160 (vigil 55 / weddings 55 / friends 50, seed 42). Old-session scratchpad survived: synthetic occlusion_eval set (108 imgs + manifest) + score_calib_corpus.py + fit_and_report.py copied over.
- [ckpt 04b9bd] Step 2 IN FLIGHT: committed benchmarks/{weaklabel_models,gemma_weaklabel_runner,occlusion_ratio_runner,occlusion_recal}.py; Gemma occlusion_v1 pass running bg (~3.6 s/crop).

## Working Memory (scratch values the agent carries forward)
- SP=/private/tmp/claude-501/-Users-alrelador/d6882f01-c8b9-42a3-ba55-daa9f4359beb/scratchpad
- Corpus: $SP/occl_corpus.json; manifests: occl_crops_manifest / occl_full_manifest / occl_syn_manifest
- Labels out: $SP/occl_labels_v1.json, occl_labels_v2.json; ratios: occl_ratios_real.json, occl_ratios_syn.json
- Wedding Task C pulls staged at $SP/wedding_c/{exn,zd}_{backup,edited} (257/87, 107/60)
- Wedding preset current weights: topiq .25 / laion .40 / clipiqa .25 / exposure .10; occl penalty 0.12

## Acceptance Criteria
- [ ] benchmarks/runs/occlusion_recal_report.md exists with full tables + gemma provenance + weak-label caveat
- [ ] Raw label JSONs committed under benchmarks/runs/
- [ ] Gate verdict recorded; config changed ONLY on PASS; suite green if changed
- [ ] Disagreement rate logged; no gating on Gemma self-confidence

## Transition Rules
- IF current step DONE → increment step, update Registers, continue
- IF all steps DONE → set state: VERIFY, self-check acceptance criteria
- IF verify passes → set state: DONE, update MAP.md flag
- IF verify fails → set state to failed step number, note what failed
