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

Models are discovered by scanning `/Applications/oMLX/models` (override with
`PHOTO_MANAGER_VLM_ROOT`). Built-in aliases:

| Alias | Model directory |
|---|---|
| `qwen3-vl-4b` (default) | `Qwen3-VL-4B-Instruct-MLX-8bit` |
| `gemma-4-e2b` | `gemma-4-e2b-it-4bit` |
| `gemma-4-e4b` | `gemma-4-e4b-it-8bit` |
| `qwen3.5-9b` | `Qwen3.5-9B-MLX-8bit` |

```bash
cull --model gemma-4-e4b /path/to/photos
cull --vlms   # list all discovered models
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PHOTO_MANAGER_VLM_ROOT` | `/Applications/oMLX/models` | VLM model directory |
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
