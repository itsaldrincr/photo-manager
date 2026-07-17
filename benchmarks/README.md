# benchmarks/ — A/B harness with commit/rollback gate

Self-improvement loop: every model or pipeline change is evaluated against a
human-labeled ground truth before it may be promoted.

## Layout

- `ground_truth.json` — 40 photos (20 keep / 20 reject) hand-labeled by the
  owner's real curation of the 2024-11 Japan trip (NAS `_curated/_selects` vs
  `_review/_rejected`).
- `eval_set/` — the 40 JPEGs (gitignored, ~500 MB; rebuild from NAS).
- `vlm_eval_runner.py` — scores the eval set with ONE model in a fresh
  subprocess (model memory returned to OS on exit — OOM safety on 18 GB hosts).
- `harness.py` — orchestrates runs, aggregates metrics, applies the gate.
- `runs/log.jsonl` + `LOG.md` — durable, append-only run logs (committed).
- `champion.json` — metrics of the currently promoted configuration.

## Usage

```bash
python3 benchmarks/harness.py eval <model_alias>     # run + log one candidate
python3 benchmarks/harness.py gate <model_alias>     # exit 0 = promote, 1 = reject
python3 benchmarks/harness.py report                 # print LOG.md
# (historical: a .venv-mlx063 venv was used while the main env was pinned to
# mlx-vlm 0.3.12; the main env is now 0.6.3, so no separate venv is needed)
```

## Gate rules

Hard gates (any failure → reject): peak Metal memory ≤ 12 GB,
parse-error rate ≤ 5%, accuracy no worse than champion − 2.5%.
Promotion requires accuracy +2.5% over champion, OR ≥1.10× speedup with
accuracy no worse than champion. Update `champion.json` and the config
default only after a PROMOTE verdict, then commit; revert on reject.
