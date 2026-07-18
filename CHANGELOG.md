# Changelog

## v1.4.0

Calibration + TUI hardening. Everything trainable from existing data is now
trained or calibrated; remaining gaps require new labels (documented below).

### Changed

- Routing thresholds calibrated against 293 owner-triaged photos:
  KEEPER_MIN 0.72 -> 0.94, AMBIGUOUS_MIN 0.48 -> 0.85. The old values sat
  below the 5th percentile of real composite scores, auto-keeping 93% of
  photos at a 72% error rate; calibrated routing errs 13.6% at ~35% VLM
  share. Expect materially stricter keeps and more stage-3 activity.
- Genre preset weights were fitted on the same corpus but FAILED the
  promotion gate (+1.06 CV points, needs +3.0) — left unchanged; the full
  fit is archived in benchmarks/runs/calibration_report.md.
- VLM self-reported confidence measured uninformative (0.85-0.95 on every
  call at flat accuracy); the confidence threshold is documented as a dead
  branch rather than retuned.

### Fixed

- TUI no longer breaks on window resize: resize events are debounced (one
  kitty clear+put at settle instead of a mid-storm write storm), identical
  photos reuse their uploaded image id instead of re-uploading, and
  terminals below 40x12 get a clean placeholder instead of corrupt layout.

### Added

- Score panel shows eye state (open/squinting/closed) in the portrait block.

### Known label gaps (cannot be trained without new data)

- Reverence/expression accuracy: needs owner-labeled liturgy faces.
- Occlusion threshold: calibrated on synthetic occluders only.
- Non-holiday genre weights: need triaged corpora per genre (the Weddings
  archive's Edited/ folders are a candidate implicit-label source).

## v1.3.0

Ship-blocker sweep: a full audit traced every user-visible metric to its
producer and revived or fixed six signals that could never fire.

### Fixed

- Occlusion detection works for the first time: a texture-variance signal
  (Laplacian patches at six landmark regions, scaled to inter-ocular
  distance) replaces the dead visibility-based heuristic. F1 0.78-0.80 on a
  synthetic-occlusion eval; the face_occlusion_penalty preset weights now
  actually apply.
- Palette-outlier, EXIF-anomaly, and scene-start signals were structurally
  dead: shoot_stats read model fields that never existed, silently zeroed by
  getattr fallbacks. EXIF + capture time now flow from stage 1 (read once
  beside the existing decode); the palette LAB centroid computes from the
  already-decoded stage-2 image. All three fire with real distributions.
- The taste model never trained: retrain fired on a single override (crashed,
  swallowed) and 'select' decisions were mislabeled as rejects. Retrain now
  uses the full override history; train and inference share one canonical
  15-dim feature row (the old 772-dim inference path would have crashed on
  any real profile); dimension mismatches degrade to warmstart. A profile
  trained on the owner's 602 decisions ships the feature live (5-fold CV
  0.676).
- Scene boundaries needed a 120s minimum gap — the old 2s threshold flagged
  nearly every travel photo as a scene start.
- is_squinting is now computed (existed since v1.0, never assigned).

### Changed

- TUI shows 'Taste: untrained' instead of a frozen 0.500 when no profile is
  loaded; the score panel and report card surface expression valence/arousal.

## v1.2.0

### Changed

- Facial expression analysis now runs on EmotiEffLib (EfficientNet-B0, ONNX)
  instead of DeepFace: ~50x faster per face (21 ms vs 1050 ms), ~5.8 GB less
  resident memory, and TensorFlow leaves the dependency tree entirely. The
  swap was gated by a three-layer fair test (labeled RAF-DB accuracy: macro-F1
  0.578 vs 0.256; perturbation stability; wedding-vs-liturgy context
  separation on the owner's own corpora) — see
  benchmarks/runs/expression_fair_test_report.md.
- Portrait results now carry optional valence/arousal fields alongside the
  existing 7-class emotion label. Scoring math is unchanged; the dimensional
  signal is available for future preset-aware use.

### Fixed

- Stage 2 crashed on any real photo with a detected face when portrait mode
  was enabled: MediaPipe's FaceLandmarker never populates landmark.visibility,
  and detect_occlusion compared None against a float. All prior tests mocked
  visibility as a float, masking the crash. The occlusion heuristic is now
  explicitly a no-op on real photos until a geometric signal replaces it.

## v1.1.0

Correctness, performance, and a model swap validated by a new A/B harness.

### Fixed

- Re-running `cull` on an already-culled folder no longer re-ingests `_review/`
  and `_curated/` contents or nests output directories inside each other.
- Stage 2's reducer repatch fed the fused composite back in as the exposure
  input, corrupting the score of every re-routed photo. It now uses the
  documented neutral default.
- TUI state and `session_report.json` are written atomically with backup
  rotation and recovery; a crash mid-write no longer discards review decisions.
- Stage 3 retries genuine VLM inference failures instead of aborting the run;
  burst detection survives photos that vanish mid-run.
- XMP sidecars follow a photo's current location instead of its original path,
  and sidecar move failures are reported rather than dropped.
- `discover_vlms` read `VLM_MODELS_ROOT` at import time, ignoring environment
  overrides for default-argument callers.
- The dashboard permanently silenced all process logging after first use.
- `detect_expression` silently returned an empty string because DeepFace's
  backend needs `tf-keras` on TensorFlow >= 2.16; it is now a declared
  dependency.

### Performance

- Full-pipeline wall clock on a 30-photo corpus: 7m30s to 5m03s (-33%), with
  Stage 2 down 35% and peak RSS down to ~7.25 GB.
- Photos are decoded once per stage instead of up to four (Stage 1) and three
  (portrait analysis) times; taste scoring reuses batch CLIP embeddings; the
  TOPIQ composition metric is batched like its siblings.
- The Stage 3 VLM now loads only after Stage 2 unloads its models, so the two
  never sit in memory together, and images are passed to `mlx-vlm` in memory
  instead of via temporary files.

### Changed

- Default Stage 3 VLM is now `gemma-4-12b`, promoted over `qwen3-vl-4b` after
  scoring 5 points higher on two disjoint human-labeled holdout sets.
  `qwen3-vl-4b` remains available via `--model`.
- `mlx-vlm` pinned to 0.6.3 (Gemma 4 vision support; 0.6.4 has known upstream
  regressions).
- VLM models are read from `<repo>/models` by default.
- The golden fixture corpus moved into the repo at `fixtures/` (photos are
  gitignored, manifests are tracked). `PERF_CORPUS_PATH` defaults there, so
  golden-baseline tests no longer skip silently.
- Golden baselines re-captured with a `corpus_fingerprint` integrity guard;
  all 684 score entries are unchanged from the pre-refactor baselines.

### Added

- Two-tier near-duplicate detection: DINOv2-small embeddings now run alongside
  the existing MobileNetV3 pass and merge into its groups. Cropped near-dupes
  go from 10/20 to 20/20 recall and rotated ones from 19/20 to 20/20, for
  +29 ms/photo. The CNN pass is retained because Stage 4's clustering
  thresholds are calibrated to its embedding distribution.
- `benchmarks/`: an A/B harness that scores VLM candidates against
  human-curated keep/reject labels in an isolated subprocess, applies memory
  and accuracy gates, and appends every run to a durable log. Includes
  evaluations of dedup backbones and facial-expression models.

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
