---
id: task_701
name: calibration_progress_bar
state: DONE
step: 3 of 3
depends: []
checkpoint: ddd495
created: 2026-04-27
---

## Program (immutable — set at planning)
1. Write `specs/calibration_progress_bar.md` capturing intent, reporter contract, integration points, and out-of-scope items.
2. Write `tests/test_calibrate_progress.py` with failing tests covering: NullProgress no-op, RichProgress factory selection, run_calibration emits start/advance×N/end events in the correct order. Tests must mock `_score_p1` and `_score_one_aesthetic` so no ML loads.
3. Implement: create `src/cull/calibrate_progress.py` (CalibrationProgress Protocol, NullProgress, RichProgress, calibration_progress context-manager factory). Modify `src/cull/calibrate.py` to take a `progress` param and emit events. Modify `src/cull/cli_subcommands.py:_run_calibrate` to instantiate the reporter (rich when stdout is a TTY, null otherwise).

## Registers (mutable — agent writes after each step)
- step 1 done @ nonce 10574d → wrote `specs/calibration_progress_bar.md` (7 sections: intent, constraints, contract, integration points, UX, tests, out-of-scope). Pinned reporter API and event sequence.
- step 2 done @ nonce 8f9ea6 → wrote `tests/test_calibrate_progress.py` with 4 tests. Confirmed failing (ModuleNotFoundError: cull.calibrate_progress) — failure path is exactly the "no impl yet" condition.
- step 3 done @ nonce 9a058b → created `src/cull/calibrate_progress.py` (PhaseStart, CalibrationProgress Protocol, NullProgress, RichProgress, calibration_progress factory). Modified `src/cull/calibrate.py` (run_calibration now takes `progress` param, _score_p4lite advances per photo, new _score_with_progress + _ScoredCorpus extracted to keep run_calibration ≤20 lines). Modified `src/cull/cli_subcommands.py:_run_calibrate` to wrap run_calibration in `calibration_progress(use_rich=sys.stdout.isatty())`.

## Verification
- `pytest tests/test_calibrate_progress.py -v` → 4/4 passed.
- `pytest tests/test_cli.py -v` → 8/8 passed (no regression).
- All new functions ≤2 params, ≤20 lines, type-hinted; max 3 public methods per class (NullProgress = 3, RichProgress = 3).

## Working Memory (scratch values the agent carries forward)
- Existing caller of `run_calibration`: `src/cull/cli_subcommands.py:_run_calibrate` — only call site, signature change is safe.
- `_score_p1` is opaque (delegates to `tests._golden_helpers.score_corpus_via_batched_path`); represent as indeterminate phase (`total=None` spinner).
- `_score_p4lite` iterates per-photo; instrument with `progress.advance()` per photo (`total=N`).
- Existing rich pattern: `cli_subcommands.py` already imports `rich.console.Console`, `rich.panel.Panel`, `rich.table.Table`. New module adds `rich.progress.Progress`.

## Acceptance Criteria
- [ ] `specs/calibration_progress_bar.md` exists and follows the format of `specs/fast_mode_sibling_package.md` (Intent / Constraints / Contract / Integration / Out-of-scope).
- [ ] `tests/test_calibrate_progress.py` runs green and exercises: NullProgress, RichProgress factory branch, end-to-end event sequence (`["start", "end", "start", "advance"×N, "end"]`).
- [ ] `pytest tests/test_calibrate_progress.py -v` passes.
- [ ] `cull --calibrate <corpus>` shows a live Rich progress display during scoring (manual sanity — not part of automated test).
- [ ] All existing tests still pass: `pytest tests/test_cli.py -v` does not regress.
- [ ] All new functions ≤2 params, ≤20 lines, with type hints (CLAUDE.md discipline).

## Transition Rules
- IF current step DONE → increment step, update Registers, continue
- IF all steps DONE → set state: VERIFY, self-check acceptance criteria
- IF verify passes → set state: DONE, update MAP.md flag
- IF verify fails → set state to failed step number, note what failed
