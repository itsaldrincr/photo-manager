# Tasks: event-shoot cull (overnight 2026-09-27)

Labels, eval scripts, reports: theseus `~/Workspace/scratch/singles-mixer-20260926/`.
Originals: odysseus `~/Desktop/singles-mixer-20260926` (never modify). Runs: odysseus `~/Work/cull-runs/`.

- [x] Merge fix/event-shoot-judgement + feat/moment-stacks + tui/pro-review into integrate/event-cull — photo-manager
- [x] Full test suite green except the 5 known failures, under `capped` — theseus
- [x] Score Qwen3.8-27B bake-off; keep or delete — odysseus (best: rating AUC 0.76, group pick 62.5%, 16 s/photo, 21 GB; kept)
- [x] Qwen3.8 event rating for all 488 (AUC 0.72; 672px: 20% faster, AUC 0.70, rejected) — odysseus
- [x] gemma-4-12b all 488: AUC 0.68; curate 58%/9 bad vs Qwen 63%/6; hybrid no better; skipping low cheap-score photos costs keeps — odysseus
- [x] Fix Stage 2 crash on mixed-orientation batches (3a8eb55) — photo-manager
- [x] Integration features run, Stages 1-2, all 488 (62 min; 0 label keeps lost in S1) — odysseus
- [x] Merge perf/pipeline-speed (Stage 2 181→97 s, 19.8→9.3 GB peak on 40 photos; decisions unchanged) — photo-manager
- [x] Event preset design: VLM rating primary, cheap LR tiebreak; sim curate-100 63% keeps / 6 bad vs old 36% / 28 — photo-manager
- [x] feat/event-judge implemented and merged (683 passed, 3 known failures) — photo-manager
- [x] Event sample run, 114 photos: works end to end; 41 min; curate-25 56% keeps vs 30% base — odysseus
- [ ] Final run 488 --preset event --curate 100 (running) — odysseus
- [x] Iterate via offline curation simulator + 114-photo sample — odysseus
- [ ] Full 488 run with --curate 100 on the final stack; score it — odysseus
- [ ] Deliver 100 selects as copies to odysseus ~/Desktop/singles-mixer-20260926-selects + contact-sheet page — odysseus
- [ ] Ship: merge integrate/event-cull to main after gates; update odysseus clone; theseus editable install follows main — photo-manager
- [ ] Republish cull-tui page with real-shoot screenshots (blocked: theseus screen locked overnight) — theseus
- [x] Fix pyproject dependency conflict (transformers override) so a fresh install resolves — photo-manager
- [x] Laya: dropped without install. Zero-shot base ~0.36 vs 0.32 random (eesel review); fine-tuned it would duplicate the metrics LR, and the VLM rating is what moved curation — theseus
