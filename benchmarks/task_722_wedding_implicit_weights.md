---
id: task_722
name: wedding_implicit_weights
state: PENDING
step: 0 of 4
depends: [721]
checkpoint: 5b4590
created: 2026-07-18
---

## Program (immutable — set at planning)
1. Mine implicit keep labels from the Weddings archive. NOTE (planner survey 2026-07-18): Charmaine's Wedding has only 35 Backup/31 Edited (insufficient negatives); use "220924 ExN Dance Rehearsal" (258 Backup / 87 Edited), optionally "290325 Zack and Dora's Wedding Rehearsal" (107/60) as holdout. Edited files are RENAMED exports (no DSCF token) — match Edited→Backup by embedding nearest-neighbor (reuse repo CNN/DINOv2 machinery), verify match quality by distance distribution.
2. Re-score Backup with the production pipeline per the calibration_report.md approach (per-shoot stage1+2 run, composite reconstruction sanity check).
3. Fit wedding-preset weights (LR on standardized metrics), 5-fold CV routing accuracy vs current weights, gate at +3.0 CV points.
4. Apply only on PASS; else record verdict. Either way commit benchmarks/runs/wedding_weights_report.md + artifacts.

## Registers (mutable — agent writes after each step)
— empty —

## Working Memory (scratch values the agent carries forward)
— empty —

## Acceptance Criteria
- [ ] Edited→Backup matching validated (distances reported, ambiguous matches dropped)
- [ ] benchmarks/runs/wedding_weights_report.md with CV table + gate verdict + implicit-label caveats
- [ ] Weights changed ONLY on +3.0 CV PASS; suite green if changed
- [ ] Stage-1 drop-outs reported separately (calibration pattern)

## Transition Rules
- IF current step DONE → increment step, update Registers, continue
- IF all steps DONE → set state: VERIFY, self-check acceptance criteria
- IF verify passes → set state: DONE, update MAP.md flag
- IF verify fails → set state to failed step number, note what failed
