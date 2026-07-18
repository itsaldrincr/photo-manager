---
id: task_723
name: closeout
state: PENDING
step: 0 of 5
depends: [720, 721, 722]
checkpoint: 8e8d2a
created: 2026-07-18
---

## Program (immutable — set at planning)
1. Full suite: non-golden (baseline 538) + golden 4 — all green; no leftover model PIDs.
2. CHANGELOG.md entry + version bump if any production change landed.
3. Audit task files, then merge gemma-weak-labeling → main, delete branch.
4. NAS sync via SMB mount (COPYFILE_DISABLE=1, openrsync excludes per handover).
5. Push origin main; verify NAS HEAD == local HEAD == origin/main; clean tree.

## Registers (mutable — agent writes after each step)
— empty —

## Working Memory (scratch values the agent carries forward)
— empty —

## Acceptance Criteria
- [ ] All tests green (non-golden + golden)
- [ ] main == origin/main, branch pruned, clean tree
- [ ] NAS mirror synced and HEAD verified
- [ ] Durable artifacts under benchmarks/runs/ committed

## Transition Rules
- IF current step DONE → increment step, update Registers, continue
- IF all steps DONE → set state: VERIFY, self-check acceptance criteria
- IF verify passes → set state: DONE, update MAP.md flag
- IF verify fails → set state to failed step number, note what failed
