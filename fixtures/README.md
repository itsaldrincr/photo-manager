# photo-manager fixture corpora

This directory holds **fixture corpora** for the [photo-manager](../../Library/CloudStorage/GoogleDrive-al.relador@gmail.com/My%20Drive/1%20Projects/OpenCode%20Projects/photo-manager) project's performance-track golden-baseline tests.

A **corpus** is a frozen set of photographs with a known-good baseline score map. Running Stage 1 / Stage 2 / aesthetic scoring on the corpus must produce byte-identical results to the baseline (within 1e-9) — any drift fails the test loudly with a per-category breakdown. Corpora exist to guarantee non-regressive refactors: if a refactor changes the score on any photo, the gate catches it before the PR lands.

## Why this directory is outside the project tree

1. Corpora are large (the default `easter_vigil/` is 1.5 GB across 123 JPEGs). Google Drive sync of a project-local fixture is impractical.
2. Git has no business tracking binary photo blobs. Baselines (~30 KB JSON each) live inside the project at `tests/fixtures/p*_baseline_<corpus_name>.json` and ARE committed to git. The corpora themselves live here and stay out of git.
3. Decoupling storage lets you swap, delete, or regenerate corpora without touching the git history of photo-manager.

---

## Layout convention

```
photo-manager-fixtures/              ← you are here
  README.md                          ← this file
  easter_vigil/                      ← default corpus; one subdirectory = one corpus
    DSCF*.JPG                        ← real photos
    synth_*.JPG                      ← synthetic variants (see "Generating synthetic variants" below)
    burst_*.JPG                      ← real consecutive-frame burst sequence
    manifest.json                    ← sha256 + category + byte count per file
  wedding_shoot/                     ← future corpus
    ...
    manifest.json
```

Every corpus directory contains a `manifest.json`. It is the source of truth for what files the corpus holds, their sha256 hashes, their categories, and their byte sizes. Photo-manager's golden baseline tests read the manifest first and only touch files listed in it.

The **corpus directory's basename** (e.g. `easter_vigil`) is the corpus's identity for naming purposes. Baselines for a corpus `<name>` are stored at:

```
<photo-manager>/tests/fixtures/p1_baseline_<name>.json      # Stage 2 scoring baseline
<photo-manager>/tests/fixtures/p3_baseline_<name>.json      # Stage 1 scoring baseline
<photo-manager>/tests/fixtures/p4lite_baseline_<name>.json  # Shared-CLIP aesthetic baseline
```

Rename a corpus directory → you must rename the baseline files to match, OR just regenerate the baselines under the new name.

---

## The default corpus: `easter_vigil/`

The default corpus is a 123-file mix built from an Easter Vigil shoot:

| Category       | Count | Source |
|----------------|-------|--------|
| `real`         | 100   | Randomly sampled from a 977-JPEG Fujifilm shoot |
| `synth_blur`   | 3     | Gaussian blur, radius 8 |
| `synth_motion` | 3     | Horizontal motion blur, kernel size 15 |
| `synth_overexp`| 3     | Brightness × 2.5 |
| `synth_underexp`| 3    | Brightness × 0.2 |
| `synth_noisy`  | 3     | Gaussian noise, σ = 40 |
| `synth_dupe`   | 3     | 3% center crop + brightness × 1.05 (triggers imagededup) |
| `burst`        | 5     | Real consecutive frames `DSCF0011..0015.JPG` (guaranteed real EXIF burst) |

The synthetic variants exist to guarantee that each classical-filter scoring branch (blur detection, exposure clipping, noise, near-duplicate detection) fires on at least three photos in the corpus. Without them, a real shoot might not exercise every branch and the golden test would miss regressions on those code paths.

---

## Config integration

`src/cull/config.py` holds:

```python
PERF_CORPUS_PATH: Path = Path(
    os.environ.get(
        "PERF_CORPUS_PATH",
        "/Users/alrelador/Documents/Claude/photo-manager-fixtures/easter_vigil"
    )
)
```

`PERF_CORPUS_PATH` defaults to the Easter Vigil corpus. Override at runtime by either:

1. **Setting the env var**: `PERF_CORPUS_PATH=/path/to/other/corpus pytest tests/...`
2. **Passing the pytest flag**: `pytest tests/test_perf_p1_batch_golden.py --corpus /path/to/other/corpus`

The pytest flag is the ergonomic path — the `--corpus` flag is added by `tests/conftest.py` and exposed via the `corpus_path` fixture.

---

## How to add a new corpus

Goal: you run a shoot or curate a photo set, drop it here, generate a manifest + baselines, and the golden tests Just Work against it via `--corpus`.

### 1. Create the directory and drop photos

```bash
mkdir -p ~/Documents/Claude/photo-manager-fixtures/wedding_shoot
cp ~/Desktop/wedding-shoot-raw/*.JPG ~/Documents/Claude/photo-manager-fixtures/wedding_shoot/
```

Any JPEG count ≥50 works. For diagnostic power, aim for a mix that will exercise every scoring branch. If your shoot is narrow (e.g. all tack-sharp, well-exposed portraits), add synthetic variants — see next section.

### 2. Generate synthetic variants (optional but recommended)

Use the Python script in the "Generating synthetic variants" section below. Drop the generated `synth_*.JPG` files alongside the real ones.

### 3. Generate the manifest

Use `tests/fixtures/task_301_build_perf_fixture_corpus.md` as a reference, or run this inline:

```python
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CORPUS = Path("/Users/alrelador/Documents/Claude/photo-manager-fixtures/wedding_shoot")

def categorise(name: str) -> str:
    if name.startswith("synth_"):
        # e.g. synth_blur_01_*.JPG -> synth_blur
        parts = name.split("_")
        return f"synth_{parts[1]}"
    if name.startswith("burst_"):
        return "burst"
    return "real"

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

files = sorted(f for f in CORPUS.iterdir() if f.suffix.upper() in (".JPG", ".JPEG"))
entries = [
    {
        "name": f.name,
        "sha256": sha256_of(f),
        "bytes": f.stat().st_size,
        "category": categorise(f.name),
    }
    for f in files
]
manifest = {
    "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    "count": len(entries),
    "files": entries,
}
(CORPUS / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"Wrote manifest with {len(entries)} entries")
```

### 4. Generate baselines for the new corpus

From the project root:

```bash
python tests/_capture_p1_baseline.py --corpus /Users/alrelador/Documents/Claude/photo-manager-fixtures/wedding_shoot
python tests/_capture_p3_baseline.py --corpus /Users/alrelador/Documents/Claude/photo-manager-fixtures/wedding_shoot
python tests/_capture_p4lite_baseline.py --corpus /Users/alrelador/Documents/Claude/photo-manager-fixtures/wedding_shoot
```

This produces:
- `tests/fixtures/p1_baseline_wedding_shoot.json`
- `tests/fixtures/p3_baseline_wedding_shoot.json`
- `tests/fixtures/p4lite_baseline_wedding_shoot.json`

**Each capture takes several minutes** (5–15 minutes for 123 photos on Apple Silicon MPS; proportionally more for larger corpora). The scripts log per-photo progress to stderr.

Commit the baseline JSONs to git. The corpus itself stays out of git.

### 5. Run the golden tests against the new corpus

```bash
pytest tests/test_perf_p1_batch_golden.py --corpus /Users/alrelador/Documents/Claude/photo-manager-fixtures/wedding_shoot -v
pytest tests/test_perf_p3_pool_golden.py --corpus /Users/alrelador/Documents/Claude/photo-manager-fixtures/wedding_shoot -v
pytest tests/test_perf_p4lite_clip_golden.py --corpus /Users/alrelador/Documents/Claude/photo-manager-fixtures/wedding_shoot -v
```

First run should be all-green (it's identical to the baseline you just captured). Any refactor that breaks scoring will then fail with a per-category divergence breakdown.

---

## Generating synthetic variants

Synthetic variants are photos derived from real photos by applying deterministic transforms. They guarantee that each classical-filter scoring branch fires at least three times per corpus, so a refactor that breaks one branch is caught even if the real photos happen not to trigger it.

The script below was used to build `easter_vigil/`. Run it with a different `CORPUS` and source pool to build synthetics for any corpus. Requires `Pillow`, `numpy`, `scipy` (for motion blur), all in the photo-manager dev environment.

```python
"""
Generate synthetic variants for a corpus: blur, motion, overexp, underexp, noisy, dupe.
Drops the outputs alongside the source photos. Seed is fixed for reproducibility.
"""
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from scipy.ndimage import convolve

CORPUS = Path("/Users/alrelador/Documents/Claude/photo-manager-fixtures/wedding_shoot")  # <— change

random.seed(42)  # reproducible selection
np.random.seed(42)  # reproducible noise

real_files = sorted(f.name for f in CORPUS.iterdir() if f.suffix.upper() in (".JPG", ".JPEG"))
if len(real_files) < 18:
    sys.exit("Need at least 18 source photos to pick 3 per category × 6 categories.")

# 3 sources per category × 6 categories = 18 sources
pool = random.sample(real_files, 18)
sources = {
    "blur":     pool[0:3],
    "motion":   pool[3:6],
    "overexp":  pool[6:9],
    "underexp": pool[9:12],
    "noisy":    pool[12:15],
    "dupe":     pool[15:18],
}

def save_with_exif(img: Image.Image, src_path: Path, dst_path: Path) -> None:
    """Save img to dst_path, preserving source EXIF bytes if present."""
    src_img = Image.open(src_path)
    exif = src_img.info.get("exif", b"")
    if exif:
        img.save(dst_path, "JPEG", quality=92, exif=exif)
    else:
        img.save(dst_path, "JPEG", quality=92)

def gaussian_blur(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB").filter(ImageFilter.GaussianBlur(radius=8))
    save_with_exif(img, src, dst)

def motion_blur(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    arr = np.array(img).astype(np.float32)
    for c in range(3):
        arr[:, :, c] = convolve(arr[:, :, c], kernel, mode="nearest")
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    save_with_exif(out, src, dst)

def overexpose(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(2.5)
    save_with_exif(img, src, dst)

def underexpose(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(0.2)
    save_with_exif(img, src, dst)

def add_noise(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 40, arr.shape)
    arr += noise
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    save_with_exif(out, src, dst)

def near_dupe(src: Path, dst: Path) -> None:
    """3% center crop + slight brightness bump — similar enough to trigger imagededup."""
    img = Image.open(src).convert("RGB")
    w, h = img.size
    crop_pct = 0.03
    left = int(w * crop_pct)
    top = int(h * crop_pct)
    img = img.crop((left, top, w - left, h - top)).resize((w, h), Image.LANCZOS)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    save_with_exif(img, src, dst)

TRANSFORMS = {
    "blur":     gaussian_blur,
    "motion":   motion_blur,
    "overexp":  overexpose,
    "underexp": underexpose,
    "noisy":    add_noise,
    "dupe":     near_dupe,
}

count = 0
for category, fn in TRANSFORMS.items():
    for i, src_name in enumerate(sources[category], 1):
        src = CORPUS / src_name
        stem = Path(src_name).stem
        dst = CORPUS / f"synth_{category}_{i:02d}_{stem}.JPG"
        fn(src, dst)
        count += 1
        print(f"  {category:9s} {i}/3  -> {dst.name}", file=sys.stderr)

print(f"\nGenerated {count} synthetic variants", file=sys.stderr)
```

### Transform parameters (tuning reference)

| Category        | Transform | Parameter | Why this value |
|-----------------|-----------|-----------|----------------|
| `synth_blur`    | PIL `GaussianBlur` | radius=8 | Strong enough that blur_detector always reports "blurry" even on textured images |
| `synth_motion`  | 15-px horizontal convolution | kernel_size=15 | Long enough to look like a real camera shake; short enough to stay fast |
| `synth_overexp` | PIL `Brightness` enhance | factor=2.5 | Blows highlights to 255 on most images; triggers the clipping branch |
| `synth_underexp`| PIL `Brightness` enhance | factor=0.2 | Crushes shadows; triggers the low-clip branch |
| `synth_noisy`   | Gaussian noise | σ=40 | Noise floor high enough that the Laplacian-of-Gaussian estimator fires; low enough to keep the image semantically recognisable |
| `synth_dupe`    | Center crop 3% + brightness ×1.05 | — | Stays perceptually identical → triggers imagededup; but is not a byte-for-byte copy so duplicate-by-hash detection is NOT triggered |

If the tool gets new scoring branches (e.g. a sharpness-gradient check), add a new category here and commit the corresponding transform.

### Bursts and near-dupes — prefer real

For the `burst` category, **prefer real consecutive frames** from a source shoot over synthetic. Real bursts have correct EXIF `DateTimeOriginal` timestamps, which is what burst-detection logic reads. Synthetic bursts require EXIF surgery to fake timestamps, which is brittle. Copying 5 consecutive `DSCF00{11..15}.JPG` from a live shoot is the simplest path:

```python
# Find 5 consecutive frame numbers in the source shoot
all_source = sorted(f for f in os.listdir(SOURCE_SHOOT_DIR) if f.lower().endswith(".jpg"))

def frame_num(name: str) -> int:
    return int("".join(c for c in Path(name).stem if c.isdigit()) or "0")

for i in range(len(all_source) - 4):
    nums = [frame_num(all_source[i + j]) for j in range(5)]
    if all(nums[j + 1] - nums[j] == 1 for j in range(4)):
        for f in all_source[i:i + 5]:
            src = SOURCE_SHOOT_DIR / f
            dst = CORPUS / f"burst_{f}"
            shutil.copy2(src, dst)
        break
```

For `synth_dupe`, the `near_dupe` transform above is sufficient — imagededup's CNN-based similarity detection doesn't care about EXIF, only pixel content.

---

## How the integrity guard works

Every baseline JSON stores a `corpus_fingerprint` field: sha256 of the sorted `name\tsha256` lines from the corpus's `manifest.json` (excluding the manifest entry itself). This is a hash-of-hashes.

At test time:
1. The test loads the baseline JSON.
2. It loads the current `manifest.json` and recomputes the fingerprint.
3. If the fingerprints don't match → `pytest.fail` with both values and an explicit "do not modify the baseline" instruction.

What this catches:
- Someone edits a photo in place without regenerating the manifest → fingerprint mismatch on next test run.
- Someone regenerates the manifest but forgets to regenerate the baseline → fingerprint mismatch.
- Someone swaps corpora without updating the baseline → handled by the filename scheme (different corpus name → different baseline filename), but the fingerprint is a second line of defence if the name happens to match.

What this does NOT catch:
- Someone regenerates both manifest AND baseline against a deliberately corrupted corpus → the test will pass against the corrupted state. No automated system can catch intentional data poisoning. Code review is the only defence there.

## How to regenerate a baseline (intentionally)

If you changed the corpus (added/removed files, edited a photo, rebuilt the manifest) and you know the new state is correct, regenerate the baseline for that corpus:

```bash
python tests/_capture_p1_baseline.py --corpus /Users/alrelador/Documents/Claude/photo-manager-fixtures/easter_vigil
python tests/_capture_p3_baseline.py --corpus /Users/alrelador/Documents/Claude/photo-manager-fixtures/easter_vigil
python tests/_capture_p4lite_baseline.py --corpus /Users/alrelador/Documents/Claude/photo-manager-fixtures/easter_vigil
```

Then commit the new baseline JSONs. The test will pass on the next run because both the fingerprint and the scores are re-captured against the current code.

**Do NOT** regenerate baselines just to make a failing test pass. The test is failing because the scoring code changed behaviour — that's a regression to investigate, not a baseline to update.

---

## Reproducibility notes

- All synthetic transforms use seeded RNGs (`random.seed(42)`, `np.random.seed(42)`) — re-running the generator on the same source photos produces byte-identical outputs.
- The gaussian noise transform uses numpy's RNG, so the noise pattern is deterministic given the seed.
- EXIF is preserved on all synthetic outputs (`save_with_exif`), so burst/timestamp-based logic sees realistic dates.
- Pillow version, numpy version, and scipy version all affect the exact pixel output of the transforms. If you upgrade any of them, re-running the generator may produce different bytes → the manifest sha256 changes → you'll need to regenerate baselines.

Pinning those versions in `pyproject.toml`'s `[dev]` extras would make the corpus fully reproducible across machines. Not currently pinned — worth doing if corpus reproducibility becomes important for CI.
