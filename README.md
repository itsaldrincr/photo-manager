# cull

Local AI photo culling for macOS. Automatically selects the best photos from a
shoot using classical filters, neural image quality assessment, and an optional
vision language model tiebreaker. Runs entirely offline after a one-time model
download.

## How it works

```
Stage 1 ── Classical filters (blur, exposure, geometry, burst, duplicates)
Stage 2 ── Neural IQA scoring (TOPIQ, LAION aesthetics, CLIP taste, composition)
         ├─ Shoot-level reducer (palette coherence, exposure drift, EXIF anomalies)
Stage 3 ── VLM tiebreaker for ambiguous photos (optional, runs in-process via mlx-vlm)
Stage 4 ── Curator: peak-moment selection, diversity, narrative flow (opt-in)
  TUI  ── Interactive review with active learning and manual override
```

Every photo gets a composite score. Photos above the keeper threshold are
selected automatically. Ambiguous photos between the keeper and reject
thresholds go to the VLM for a second opinion. The rest are rejected.

## Requirements

- macOS (Apple Silicon recommended for VLM inference)
- Python 3.11+
- ~3 GB disk for model cache (one-time download)

## Install

```bash
pip install -e ".[dev]"
```

This installs the `cull` command. `mlx-vlm` is included for in-process VLM
inference — no server or daemon required.

## Setup

Bootstrap the offline model cache (one-time, requires network):

```bash
cull setup --allow-network
```

This downloads and verifies:

- **CLIP ViT-L/14** — shared embedding backbone for taste scoring, search, and diversity
- **LAION Aesthetics V2** — linear head for aesthetic scoring
- **MediaPipe FaceLandmarker** — portrait and expression analysis
- **DeepFace emotion** — facial expression classification
- **pyiqa weights** — TOPIQ, CLIPIQA+ quality metrics

After setup, every `cull` invocation is fully offline.

## Usage

```bash
# Basic cull — stages 1-3
cull /path/to/photos

# With preset tuning for genre
cull --preset wedding /path/to/photos

# Full pipeline with Stage 4 curator (top 30 picks)
cull --curate /path/to/photos

# Curate to specific count
cull --curate 50 /path/to/photos

# Skip VLM (stages 1-2 only)
cull --no-vlm /path/to/photos

# Review previous session interactively
cull --review /path/to/photos

# Pipeline then immediate review
cull --review-after /path/to/photos

# Dry run (no file moves)
cull --dry-run /path/to/photos

# Semantic search across photos
cull --search "bride laughing" /path/to/photos

# Find similar photos to a reference
cull --similar /path/to/reference.jpg /path/to/photos

# VLM explanation of a single photo
cull --explain /path/to/photo.jpg

# Diagnostic report card from a session
cull --report-card /path/to/photos
```

### Presets

Presets tune scoring weights for different genres:

`general` (default), `wedding`, `documentary`, `wildlife`, `landscape`, `street`, `holiday`

### VLM model selection

Stage 3 and Stage 4 run an in-process VLM via `mlx-vlm`. Any MLX-converted
vision-language model works — point `PHOTO_MANAGER_VLM_ROOT` at a directory
containing your model folders. Any subdirectory with a `config.json` containing
a `vision_config` key is auto-discovered.

```bash
cull --vlms                        # list discovered models
cull --model <alias> /path/to/photos  # select a specific model
```

Aliases are defined in `src/cull/config.py::VLM_ALIASES` — edit or extend them
to match your local models.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PHOTO_MANAGER_VLM_ROOT` | — | Directory containing MLX VLM model folders |
| `PHOTO_MANAGER_CACHE` | `~/.cache/photo-manager/models` | Model cache root |

## Output

By default, `cull` writes a `session_report.json` alongside the source
directory and moves photos into `_review/` and `_curated/_selects/`
subdirectories based on their scores. XMP sidecar files (`.xmp`) are written
next to source images with ratings and geometry corrections. Disable sidecars
with `--no-sidecars`.

## Tests

```bash
pytest tests/ -v
```

Tests run fully offline using mocked models and fixtures.

## License

[MIT](LICENSE)
