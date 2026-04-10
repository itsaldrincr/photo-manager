# Changelog

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
