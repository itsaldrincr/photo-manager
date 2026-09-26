# Tasks: event-shoot cull (overnight 2026-09-27)

Labels, eval scripts, reports: theseus `~/Workspace/scratch/singles-mixer-20260926/`.
Originals: odysseus `~/Desktop/singles-mixer-20260926` (never modify). Runs: odysseus `~/Work/cull-runs/`.

- [x] Merge fix/event-shoot-judgement + feat/moment-stacks + tui/pro-review into integrate/event-cull — photo-manager
- [x] Full test suite green except the 5 known failures, under `capped` — theseus
- [x] Score Qwen3.8-27B bake-off; keep or delete — odysseus (best: rating AUC 0.76, group pick 62.5%, 16 s/photo, 21 GB; kept)
- [ ] Qwen3.8 event rating for a 200-photo labelled sample (80 done + 120 queued) — odysseus
- [x] Fix Stage 2 crash on mixed-orientation batches (3a8eb55) — photo-manager
- [ ] Integration features run, Stages 1-2, all 488 (int-features) — odysseus
- [ ] Merge perf/pipeline-speed when the speed agent reports; check decisions unchanged — photo-manager
- [ ] Event preset: fit people-first composite (valence, arousal, faces, TOPIQ) with moment-grouped CV — photo-manager
- [ ] Iterate on a ~150-photo moment-intact sample until curation metrics beat baseline — odysseus
- [ ] Full 488 run with --curate 100 on the final stack; score it — odysseus
- [ ] Deliver 100 selects as copies to odysseus ~/Desktop/singles-mixer-20260926-selects + contact-sheet page — odysseus
- [ ] Ship: merge integrate/event-cull to main after gates; update odysseus clone; theseus editable install follows main — photo-manager
- [ ] Republish cull-tui page with real-shoot screenshots of every view — theseus
- [x] Fix pyproject dependency conflict (transformers override) so a fresh install resolves — photo-manager
- [ ] Low priority: Laya zero-shot over our metrics vs the fitted classifier; keep only if it wins — theseus
