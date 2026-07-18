---
id: task_721
name: reverence_weaklabel_proto
state: PENDING
step: 0 of 3
depends: [720]
checkpoint: bc07d6
created: 2026-07-18
---

## Program (immutable — set at planning)
1. ~60 Vigil + ~40 wedding face photos; Gemma double-pass closed-vocab labels {reverent/prayerful, joyful, neutral, distressed, surprised, eyes-closed-rest, unclear}; keep agreements only.
2. Cross-tabulate EmotiEffLib (7-class label + valence/arousal) vs Gemma labels; effect sizes for reverent-vs-distressed separation in V/A space.
3. Write benchmarks/runs/reverence_weaklabel_report.md + raw JSONs; PROPOSE (not implement) a liturgy-preset rule if supported. NO production scoring change.

## Registers (mutable — agent writes after each step)
— empty —

## Working Memory (scratch values the agent carries forward)
— empty —

## Acceptance Criteria
- [ ] benchmarks/runs/reverence_weaklabel_report.md + raw JSONs committed
- [ ] Provenance + weak-label caveat present
- [ ] No production scoring/preset change
- [ ] Effect sizes reported for reverent vs distressed in V/A space

## Transition Rules
- IF current step DONE → increment step, update Registers, continue
- IF all steps DONE → set state: VERIFY, self-check acceptance criteria
- IF verify passes → set state: DONE, update MAP.md flag
- IF verify fails → set state to failed step number, note what failed
