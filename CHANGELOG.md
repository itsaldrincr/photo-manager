# Changelog

## v1.0.2

Patch release focused on hardening the cmux to Ghostty review handoff.

### Review

- Ghostty review handoff now launches in a fresh blocking window and returns cleanly to the waiting cmux command when the review session exits.
- The handoff path no longer depends on a login shell to reopen the serialized review session.
- Ghostty launch failures now surface as a normal CLI error instead of a traceback.

### Tests

- Tightened review handoff coverage around the final Ghostty launch flags and clean error handling.

## v1.0.1

Patch release focused on review reliability and CLI startup behavior.

### Review

- `cull --review-after` and `cull --review` can hand off cleanly from cmux into a blocking Ghostty session without losing the live pipeline result.
- Review save now shows `Saving review changes...` immediately, then `Save complete. Exiting...` before the TUI closes.
- Save failures stay visible inside the review UI instead of exiting silently.

### CLI

- `cull.cli` now lazy-loads the heavy pipeline, review, and subcommand stacks so help and lightweight entry paths do not import the full ML pipeline up front.
- Added an internal `--review-session` path used by the Ghostty handoff flow to reopen an in-memory review session safely.

### Tests

- Added regression coverage for lazy CLI imports, Ghostty review handoff, review save feedback, and preserving PyTorch's default Stage 2 CPU thread settings.

## v1.0.0

Initial public release.

### Pipeline

- **Stage 1** — Classical filters: blur detection, exposure analysis, horizon/keystone geometry, burst grouping, duplicate detection. Runs in a multiprocessing pool.
- **Stage 2** — Neural IQA scoring: TOPIQ (technical quality anchor), LAION aesthetics, CLIPIQA+, composition scoring (rule-of-thirds, subject clearance, negative space), subject-region sharpness, personalized taste model. Preset-aware routing with genre-specific weights for wedding, portrait, landscape, documentary, wildlife, street, and holiday.
- **Stage 2 Reducer** — Shoot-level coherence: palette outlier detection, exposure drift flagging, EXIF anomaly scoring, scene boundary detection.
- **Stage 3** — VLM tiebreaker for ambiguous photos. In-process inference via mlx-vlm (Qwen3-VL, Gemma-4). Prompt context enriched with Stage 1 and Stage 2 signals.
- **Stage 4** — Curator suite (opt-in via `--curate`): portrait peak-moment detection (blink/smile/gaze), action peak detection (optical flow inflection), pairwise VLM tournament, MMR diversity selection, narrative flow regularizer.

### TUI

- Interactive review mode with active-learning queue ordering (most uncertain first).
- VLM explain modal for on-demand photo analysis.
- Batch similarity actions, score panel, burst view.
- Kitty terminal image protocol support.

### Infrastructure

- Fully offline after one-time `cull setup --allow-network` bootstrap.
- CLIP singleton with on-disk embedding cache shared across taste, search, diversity, and tournament.
- XMP sidecar writer (rating, rotation, crop, perspective corrections).
- Semantic text search and reverse image search via CLIP embeddings.
- Session reports with `--report-card` diagnostics.
- Fast mode (`--fast`) with MUSIQ single-pass scoring.
