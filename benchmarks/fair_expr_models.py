"""Shared models and pure taxonomy/config for the three-layer fair test.

No heavy imports (torch/tensorflow/onnxruntime/mediapipe) live here so every
stage — prep, subprocess runner, metrics, report — can import it cheaply.
Mirrors the structure of expression_eval_models.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

CanonicalLabel = Literal[
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "contempt", "unknown",
]

# The 7-class taxonomy shared by RAF-DB and DeepFace's emotion head.
CANONICAL_CLASSES: list[CanonicalLabel] = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
]

# Classes used for the sad/fear/neutral/angry sub-analysis called out in the task.
SUB_ANALYSIS_CLASSES: list[CanonicalLabel] = ["sad", "fear", "neutral", "angry"]

# RAF-DB's raw class_label names -> canonical taxonomy.
RAFDB_LABEL_MAP: dict[str, CanonicalLabel] = {
    "anger": "angry", "disgust": "disgust", "fear": "fear", "happiness": "happy",
    "neutral": "neutral", "sadness": "sad", "surprise": "surprise",
}

# DeepFace's raw dominant_emotion strings already equal the canonical names.
DEEPFACE_LABEL_MAP: dict[str, CanonicalLabel] = {c: c for c in CANONICAL_CLASSES}

# EmotiEffLib's 8 raw class strings (enet_b0_8_va_mtl) -> canonical taxonomy.
# "Contempt" has no RAF-DB/DeepFace equivalent — kept unmapped and reported
# separately rather than silently folded into a lookalike class.
EMOTIEFFLIB_LABEL_MAP: dict[str, CanonicalLabel] = {
    "Anger": "angry", "Disgust": "disgust", "Fear": "fear", "Happiness": "happy",
    "Neutral": "neutral", "Sadness": "sad", "Surprise": "surprise", "Contempt": "contempt",
}

# ---------------------------------------------------------------------------
# Layer A dataset config
# ---------------------------------------------------------------------------

LAYER_A_DATASET_ID: str = "deanngkl/raf-db-7emotions"
LAYER_A_SAMPLES_PER_CLASS: int = 60
LAYER_A_SEED: int = 42

# ---------------------------------------------------------------------------
# Perturbation config (Layer B)
# ---------------------------------------------------------------------------

FACE_HEIGHT_SMALL_PX: int = 64
FACE_HEIGHT_LARGE_PX: int = 96
BRIGHTNESS_DIM_FACTOR: float = 0.6
BRIGHTNESS_BRIGHT_FACTOR: float = 1.4
BLUR_RADIUS_PX: int = 2

PerturbationName = Literal[
    "baseline", "downscale_64", "downscale_96", "brightness_0.6", "brightness_1.4", "blur_r2",
]
PERTURBATION_NAMES: list[PerturbationName] = [
    "downscale_64", "downscale_96", "brightness_0.6", "brightness_1.4", "blur_r2",
]

# ---------------------------------------------------------------------------
# Face-crop config (mediapipe_bbox stage, Layers B/C)
# ---------------------------------------------------------------------------

FACE_CROP_MARGIN_FRACTION: float = 0.25

# ---------------------------------------------------------------------------
# Verdict rubric thresholds (per the task's REPLACE/KEEP/MIXED rubric)
# ---------------------------------------------------------------------------

MACRO_F1_TOLERANCE: float = 0.03
SEPARATION_TOLERANCE_FRACTION: float = 0.20


def canonicalize(raw_label: str, label_map: dict[str, CanonicalLabel]) -> CanonicalLabel:
    """Map a model's raw label string to the shared canonical taxonomy."""
    return label_map.get(raw_label, "unknown")


# ---------------------------------------------------------------------------
# Stage I/O models shared between prep and the subprocess runner
# ---------------------------------------------------------------------------


class RunnerManifest(BaseModel):
    """Input to fair_expr_runner.py: image paths, plus an optional crop output dir."""

    image_paths: list[Path]
    crop_dir: Path | None = None


class ModelReading(BaseModel):
    """One image's raw prediction from one model, before canonicalization."""

    name: str
    has_face: bool
    latency_seconds: float = 0.0
    raw_label: str | None = None
    valence: float | None = None
    arousal: float | None = None


class BboxReading(BaseModel):
    """One photo's mediapipe face bbox/crop result."""

    name: str
    path: str
    has_face: bool
    crop_path: str | None = None
    face_height_px: int | None = None


class RunnerStageOutput(BaseModel):
    """Full output of one fair_expr_runner.py subprocess invocation."""

    stage: str
    readings: list[ModelReading] = []
    bboxes: list[BboxReading] = []
