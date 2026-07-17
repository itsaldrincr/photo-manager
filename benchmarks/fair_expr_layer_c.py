"""Layer C: context-distribution separation on the owner's own photos.

Three face-rich corpora: joy-expected (weddings), solemn-expected (the 30
staged Vigil faces, already local), neutral/mixed control (Friends). Both
models score IDENTICAL mediapipe-derived face crops per corpus (same
fairness contract as Layer B), and we report per-corpus label distributions
plus a wedding-vs-vigil separation effect size for each model.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from pydantic import BaseModel

from fair_expr_layer_b import VIGIL_FACES_DIR
from fair_expr_metrics import CohensDInput, cohens_d, happy_rate, label_histogram, mean_std, negative_rate
from fair_expr_models import ModelReading, RunnerManifest, canonicalize
from fair_expr_subprocess import StageInvocation, run_stage

NAS_HOST: str = "atlas-server"
WEDDINGS_REMOTE_DIR: str = "media/Photos/from-external/1 Photography/4 Weddings"
FRIENDS_REMOTE_DIR: str = "media/Photos/from-external/1 Photography/7 Friends"
WEDDINGS_SAMPLE_COUNT: int = 80
FRIENDS_SAMPLE_COUNT: int = 40
NAS_OVERSAMPLE_FACTOR: int = 2
NAS_PULL_TIMEOUT_SECONDS: int = 300


class NasCorpusSpec(BaseModel):
    """A named face-rich corpus: either pulled from the NAS or already local."""

    name: str
    remote_dir: str | None
    sample_count: int


LAYER_C_CORPORA: list[NasCorpusSpec] = [
    NasCorpusSpec(name="weddings", remote_dir=WEDDINGS_REMOTE_DIR, sample_count=WEDDINGS_SAMPLE_COUNT),
    NasCorpusSpec(name="vigil", remote_dir=None, sample_count=30),
    NasCorpusSpec(name="friends", remote_dir=FRIENDS_REMOTE_DIR, sample_count=FRIENDS_SAMPLE_COUNT),
]


def _remote_pull_command(spec: NasCorpusSpec, dest_dir: Path) -> str:
    """Build the ssh-find-shuf-tar | local-tar pipeline for one NAS corpus."""
    remote_find = (
        f"cd {shlex.quote(spec.remote_dir)} && "
        r'find . -type f -readable \( -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | '
        f"shuf -n {spec.sample_count * NAS_OVERSAMPLE_FACTOR} | "
        r'tar -cf - --transform="s|.*/||" -T - 2>/dev/null'
    )
    return f"ssh {NAS_HOST} {shlex.quote(remote_find)} | tar -xf - -C {shlex.quote(str(dest_dir))}"


def _pull_nas_corpus(spec: NasCorpusSpec, dest_dir: Path) -> list[Path]:
    """Pull an oversampled batch of JPEGs from the NAS, then trim to sample_count."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(_remote_pull_command(spec, dest_dir), shell=True, timeout=NAS_PULL_TIMEOUT_SECONDS, check=False)
    pulled = sorted(dest_dir.glob("*"))
    return pulled[:spec.sample_count]


def _local_vigil_corpus() -> list[Path]:
    """Return the 30 already-staged Vigil face photos."""
    return sorted(p for p in VIGIL_FACES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})


def gather_corpus(spec: NasCorpusSpec, dest_dir: Path) -> list[Path]:
    """Materialize one corpus's photo paths, pulling from the NAS if needed."""
    return _local_vigil_corpus() if spec.remote_dir is None else _pull_nas_corpus(spec, dest_dir)


class GatedCorpus(BaseModel):
    """One corpus's face crops after mediapipe gating, plus the pre-gate photo count."""

    name: str
    photo_count: int
    crop_paths: list[Path]


class GateRequest(BaseModel):
    """A corpus's photos, paired with the crop-output directory for gating."""

    name: str
    photos: list[Path]
    crop_dir: Path


def face_gate_corpus(request: GateRequest) -> GatedCorpus:
    """Run the mediapipe_bbox stage over one corpus and keep only face-bearing crops."""
    output_path = request.crop_dir.parent / f"{request.name}_gate.json"
    output = run_stage(StageInvocation(
        stage="mediapipe_bbox",
        manifest=RunnerManifest(image_paths=request.photos, crop_dir=request.crop_dir),
        output_path=output_path,
    ))
    crop_paths = [Path(b.crop_path) for b in output.bboxes if b.has_face and b.crop_path]
    return GatedCorpus(name=request.name, photo_count=len(request.photos), crop_paths=crop_paths)


# ---------------------------------------------------------------------------
# Distribution / separation metrics
# ---------------------------------------------------------------------------


class CorpusDistribution(BaseModel):
    """One model's label distribution and (if applicable) VA summary for one corpus."""

    corpus: str
    model: str
    n: int
    label_histogram: dict[str, int]
    happy_rate: float
    negative_rate: float
    mean_valence: float | None = None
    std_valence: float | None = None
    mean_arousal: float | None = None
    std_arousal: float | None = None


class DistributionRequest(BaseModel):
    """A corpus name and model name, paired with that model's readings for it."""

    corpus: str
    model: str
    readings: list[ModelReading]
    label_map: dict[str, str]


def _corpus_distribution(request: DistributionRequest) -> CorpusDistribution:
    """Reduce one corpus's per-model readings into a CorpusDistribution."""
    labels = [canonicalize(r.raw_label or "", request.label_map) for r in request.readings if r.has_face]
    dist = CorpusDistribution(
        corpus=request.corpus, model=request.model, n=len(labels), label_histogram=label_histogram(labels),
        happy_rate=happy_rate(labels), negative_rate=negative_rate(labels),
    )
    if request.model != "emotiefflib":
        return dist
    valences = [r.valence for r in request.readings if r.has_face and r.valence is not None]
    arousals = [r.arousal for r in request.readings if r.has_face and r.arousal is not None]
    dist.mean_valence, dist.std_valence = mean_std(valences)
    dist.mean_arousal, dist.std_arousal = mean_std(arousals)
    return dist


class SeparationEffect(BaseModel):
    """Weddings-vs-vigil separation for one model: happy-rate diff, and Cohen's d if VA exists."""

    model: str
    happy_rate_diff: float
    cohens_d_valence: float | None = None


class SeparationRequest(BaseModel):
    """One model's per-corpus distributions, plus raw valence samples for Cohen's d."""

    distributions: list[CorpusDistribution]
    valence_by_corpus: dict[str, list[float]]


def separation_effect(request: SeparationRequest) -> SeparationEffect:
    """Compute happy-rate diff and (for EmotiEffLib) Cohen's d on valence, weddings vs vigil."""
    by_corpus = {d.corpus: d for d in request.distributions}
    weddings, vigil = by_corpus["weddings"], by_corpus["vigil"]
    happy_diff = round(weddings.happy_rate - vigil.happy_rate, 4)
    if weddings.model != "emotiefflib":
        return SeparationEffect(model=weddings.model, happy_rate_diff=happy_diff)
    d_valence = cohens_d(CohensDInput(
        sample_a=request.valence_by_corpus["weddings"], sample_b=request.valence_by_corpus["vigil"],
    ))
    return SeparationEffect(model=weddings.model, happy_rate_diff=happy_diff, cohens_d_valence=d_valence)
