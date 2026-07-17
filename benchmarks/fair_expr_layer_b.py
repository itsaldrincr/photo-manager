"""Layer B: label-free robustness under deterministic perturbations.

Takes the 30 staged Vigil faces + 20 Layer A crops, perturbs each
deterministically (downscale, brightness, blur), and measures how often each
model's prediction FLIPS vs its own unperturbed baseline. This measures
stability, not correctness — Layer A carries the accuracy claim.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from pydantic import BaseModel

from fair_expr_metrics import DriftInput, FlipRateInput, flip_rate, mean_abs_drift
from fair_expr_models import (
    BLUR_RADIUS_PX, BRIGHTNESS_BRIGHT_FACTOR, BRIGHTNESS_DIM_FACTOR,
    FACE_HEIGHT_LARGE_PX, FACE_HEIGHT_SMALL_PX, ModelReading, PERTURBATION_NAMES, RunnerManifest,
)
from fair_expr_subprocess import StageInvocation, run_stage

VIGIL_FACES_DIR: Path = Path(
    "/private/tmp/claude-501/-Users-alrelador/85225a69-a5ad-4e09-b879-f167553ba959/scratchpad/ev_faces"
)
LAYER_B_VIGIL_COUNT: int = 30
LAYER_B_LAYER_A_COUNT: int = 20
BLUR_KERNEL_PX: int = 2 * BLUR_RADIUS_PX + 1


class PerturbationSource(BaseModel):
    """One source crop, ready for perturbation."""

    name: str
    image_path: Path


def _downscale_to_height(image: "cv2.typing.MatLike", target_height: int) -> "cv2.typing.MatLike":
    """Resize an image so its height equals target_height, preserving aspect ratio."""
    height, width = image.shape[:2]
    scale = target_height / height
    return cv2.resize(image, (max(1, round(width * scale)), target_height), interpolation=cv2.INTER_AREA)


def _apply_brightness(image: "cv2.typing.MatLike", factor: float) -> "cv2.typing.MatLike":
    """Scale pixel intensities by factor, clipped to the valid range."""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def _apply_blur(image: "cv2.typing.MatLike") -> "cv2.typing.MatLike":
    """Apply a fixed-radius Gaussian blur."""
    return cv2.GaussianBlur(image, (BLUR_KERNEL_PX, BLUR_KERNEL_PX), sigmaX=BLUR_RADIUS_PX)


_PERTURBATION_FNS: dict[str, object] = {
    "downscale_64": lambda img: _downscale_to_height(img, FACE_HEIGHT_SMALL_PX),
    "downscale_96": lambda img: _downscale_to_height(img, FACE_HEIGHT_LARGE_PX),
    "brightness_0.6": lambda img: _apply_brightness(img, BRIGHTNESS_DIM_FACTOR),
    "brightness_1.4": lambda img: _apply_brightness(img, BRIGHTNESS_BRIGHT_FACTOR),
    "blur_r2": _apply_blur,
}


class PerturbationTarget(BaseModel):
    """A perturbation name paired with the directory to write its outputs into."""

    perturbation: str
    out_dir: Path


def _write_perturbed(source: PerturbationSource, target: PerturbationTarget) -> Path:
    """Apply one perturbation to a source crop and write it to target.out_dir."""
    image = cv2.imread(str(source.image_path))
    perturbed = _PERTURBATION_FNS[target.perturbation](image)
    out_path = target.out_dir / target.perturbation / source.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), perturbed)
    return out_path


class LayerBManifests(BaseModel):
    """Per-perturbation image manifests, plus the baseline manifest, for one layer."""

    baseline: list[Path]
    by_perturbation: dict[str, list[Path]]


def build_layer_b_manifests(sources: list[PerturbationSource], out_dir: Path) -> LayerBManifests:
    """Generate baseline + all perturbed crops for every source face."""
    by_perturbation = {
        name: [_write_perturbed(source, PerturbationTarget(perturbation=name, out_dir=out_dir)) for source in sources]
        for name in PERTURBATION_NAMES
    }
    return LayerBManifests(baseline=[s.image_path for s in sources], by_perturbation=by_perturbation)


class RobustnessResult(BaseModel):
    """One model's flip-rate and VA-drift per perturbation type."""

    model: str
    n_faces: int
    flip_rate_by_perturbation: dict[str, float]
    valence_drift_by_perturbation: dict[str, float] | None = None
    arousal_drift_by_perturbation: dict[str, float] | None = None


class ManifestStageRequest(BaseModel):
    """A model stage, its input image paths, and where/what to name the output file."""

    stage: str
    paths: list[Path]
    output_path: Path


def _run_manifest_stage(request: ManifestStageRequest) -> list[ModelReading]:
    """Run one model stage over one manifest of image paths."""
    output = run_stage(StageInvocation(
        stage=request.stage, manifest=RunnerManifest(image_paths=request.paths), output_path=request.output_path,
    ))
    return output.readings


def _readings_by_name(readings: list[ModelReading]) -> dict[str, ModelReading]:
    """Index readings by source photo name for cross-perturbation joins."""
    return {r.name: r for r in readings}


class RobustnessRequest(BaseModel):
    """A model stage's manifests, paired with the directory to write stage outputs into."""

    stage: str
    manifests: LayerBManifests
    work_dir: Path


def _robustness_for_model(request: RobustnessRequest) -> RobustnessResult:
    """Compute one model's flip-rate (and VA drift, if applicable) across all perturbations."""
    stage, manifests = request.stage, request.manifests
    baseline_out = request.work_dir / f"fair_expr_layer_b_{stage}_baseline.json"
    baseline = _readings_by_name(_run_manifest_stage(
        ManifestStageRequest(stage=stage, paths=manifests.baseline, output_path=baseline_out)
    ))
    flip_rates: dict[str, float] = {}
    valence_drift: dict[str, float] = {}
    arousal_drift: dict[str, float] = {}
    has_va = stage == "emotiefflib"
    for perturbation, paths in manifests.by_perturbation.items():
        tag = perturbation.replace(".", "_")
        perturbed_out = request.work_dir / f"fair_expr_layer_b_{stage}_{tag}.json"
        perturbed = _readings_by_name(_run_manifest_stage(
            ManifestStageRequest(stage=stage, paths=paths, output_path=perturbed_out)
        ))
        common = [name for name in baseline if name in perturbed]
        flip_rates[perturbation] = flip_rate(FlipRateInput(
            baseline_labels=[baseline[n].raw_label or "" for n in common],
            perturbed_labels=[perturbed[n].raw_label or "" for n in common],
        ))
        if has_va:
            valence_drift[perturbation] = mean_abs_drift(DriftInput(
                baseline_values=[baseline[n].valence or 0.0 for n in common],
                perturbed_values=[perturbed[n].valence or 0.0 for n in common],
            ))
            arousal_drift[perturbation] = mean_abs_drift(DriftInput(
                baseline_values=[baseline[n].arousal or 0.0 for n in common],
                perturbed_values=[perturbed[n].arousal or 0.0 for n in common],
            ))
    return RobustnessResult(
        model=stage, n_faces=len(manifests.baseline), flip_rate_by_perturbation=flip_rates,
        valence_drift_by_perturbation=valence_drift if has_va else None,
        arousal_drift_by_perturbation=arousal_drift if has_va else None,
    )


class LayerBRequest(BaseModel):
    """Source faces plus the working directory for all Layer B artifacts."""

    sources: list[PerturbationSource]
    work_dir: Path


def run_layer_b(request: LayerBRequest) -> list[RobustnessResult]:
    """Build all perturbations, then run both models and return their robustness results."""
    manifests = build_layer_b_manifests(request.sources, request.work_dir / "images")
    return [
        _robustness_for_model(RobustnessRequest(stage=stage, manifests=manifests, work_dir=request.work_dir))
        for stage in ("deepface", "emotiefflib")
    ]
