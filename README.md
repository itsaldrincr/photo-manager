# cull

Local AI photo culling for macOS. Automatically selects the best photos from a
shoot using classical filters, neural image quality assessment, and an optional
vision language model tiebreaker. Runs entirely offline after a one-time model
download.

## How it works

```
Stage 1 ── Classical filters (blur, exposure, geometry, burst, duplicates)
         └─ Two-tier dedup: MobileNetV3 CNN + DINOv2-small (catches crops/rotations)
Stage 2 ── Neural IQA scoring (TOPIQ, LAION aesthetics, CLIP taste, composition,
           face/eye quality + expression via MediaPipe + EmotiEffLib)
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
- ~4 GB disk for the scoring model cache (one-time download), plus ~6 GB for
  the Stage 3 VLM if you use it
- ~16 GB RAM recommended (peak ~8 GB with the default VLM)

## Install

```bash
uv venv --python 3.13 .venv && . .venv/bin/activate
uv pip install -e ".[dev]"
```

Use `uv`, not plain `pip`. `pyproject.toml` overrides the `transformers<5`
pin of `simple-aesthetics-predictor`, and only `uv` reads that override.
Without it, the dependencies do not resolve.

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

`general` (default), `wedding`, `documentary`, `wildlife`, `landscape`, `street`, `holiday`, `event`

### Event preset

Use `event` for people shoots such as mixers, parties and receptions:

```bash
cull --preset event --curate 100 /path/to/photos
```

The VLM rates every photo from 1 to 5 against an event-photography prompt,
and the rating decides the outcome. A photo rated 5 is a keeper, one rated 4
goes to review, and one rated 3 or lower is rejected. In each moment stack
(frames of one moment taken within 10 s of each other), only the
highest-rated frame is kept. The curator picks N photos from those stack
winners, with a diversity penalty.

The technical-quality composite does not predict what a person would keep on
event photos. On a 488-photo singles mixer the old composite reached a keeper
AUC of 0.56, and the `event` preset raised curation quality as follows:

| Curate 100 | Keepers | Bad picks | Heroes found |
|---|---|---|---|
| Old composite | 36% | 28 | 8 of 12 |
| `event` with `qwen3-8-27b-mlx-4bit` | 63% | 6 | 10 of 12 |
| `event` with `--model gemma-4-12b` | 58% | 9 | 10 of 12 |

The default judge, Qwen3.8-27B, needs about 21 GB of GPU memory and takes
about 16 s per photo on an M6 Mac mini. That is about 2.2 h of rating for
500 photos, so run it on odysseus. `--model gemma-4-12b` needs 8 GB and takes
about 5 s per photo. Both figures were measured on odysseus.

### VLM model selection

Stage 3 and Stage 4 run an in-process VLM via `mlx-vlm`. Models are read from
`models/` in the repo root by default; override with `PHOTO_MANAGER_VLM_ROOT`.
Any subdirectory with a `config.json` containing a `vision_config` key is
auto-discovered, so any MLX-converted vision-language model works.

The default is `gemma-4-12b` (`mlx-community/gemma-4-12B-it-4bit`, ~6.3 GB),
chosen by A/B evaluation against human-curated keep/reject labels — see
`benchmarks/`. `qwen3-vl-4b` (`mlx-community/Qwen3-VL-4B-Instruct-MLX-8bit`)
is a faster, slightly less accurate alternative. Download either with:

```bash
huggingface-cli download mlx-community/gemma-4-12B-it-4bit \
  --local-dir models/gemma-4-12B-it-4bit
```

The `event` preset defaults to Qwen3.8-27B (~16 GB on disk). Download it with:

```bash
huggingface-cli download lmstudio-community/Qwen3.8-27B-MLX-4bit \
  --local-dir models/Qwen3.8-27B-MLX-4bit
```

```bash
cull --vlms                        # list discovered models
cull --model <alias> /path/to/photos  # select a specific model
```

Aliases are defined in `src/cull/config.py::VLM_ALIASES` — edit or extend them
to match your local models.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PHOTO_MANAGER_VLM_ROOT` | `<repo>/models` | Directory containing MLX VLM model folders |
| `PHOTO_MANAGER_CACHE` | `~/.cache/photo-manager/models` | Model cache root |
| `PERF_CORPUS_PATH` | `<repo>/fixtures/easter_vigil` | Golden-baseline test corpus |

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
