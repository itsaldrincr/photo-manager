"""Runtime configuration and named constants for the cull pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

STAGE_IQA: int = 2
STAGE_VLM: int = 3

# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------

VLM_DEFAULT_ALIAS: str = "qwen3-vl-4b"
VLM_MODELS_ROOT: Path = Path(os.environ.get("PHOTO_MANAGER_VLM_ROOT", "/Applications/oMLX/models"))
VLM_ALIASES: dict[str, str] = {
    "qwen3-vl-4b": "Qwen3-VL-4B-Instruct-MLX-8bit",
    "gemma-4-e2b": "gemma-4-e2b-it-4bit",
    "gemma-4-e4b": "gemma-4-e4b-it-8bit",
    "qwen3.5-9b": "Qwen3.5-9B-MLX-8bit",
}

VLM_TEMPERATURE: float = 0.0
VLM_MAX_TOKENS: int = 512
VLM_MAX_RETRIES: int = 3
VLM_CONFIDENCE_THRESHOLD: float = 0.70
VLM_IMAGE_MAX_PX: int = 1024
VLM_JPEG_QUALITY: int = 85

# ---------------------------------------------------------------------------
# Stage 2 routing thresholds
# ---------------------------------------------------------------------------

ROUTING_KEEPER_MIN: float = 0.72
ROUTING_AMBIGUOUS_MIN: float = 0.48

# ---------------------------------------------------------------------------
# Stage 2 default genre weights (TOPIQ, LAION, CLIPIQA, Exposure)
# ---------------------------------------------------------------------------

COMPOSITION_WEIGHT_DEFAULT: float = 0.15
TASTE_WEIGHT_DEFAULT: float = 0.15
TILT_PENALTY_WEIGHT_DEFAULT: float = 0.10
TILT_PENALTY_WEIGHT_LANDSCAPE: float = 0.20
TILT_PENALTY_WEIGHT_PEOPLE: float = 0.05

PALETTE_OUTLIER_WEIGHT_DEFAULT: float = 0.05
EXPOSURE_DRIFT_WEIGHT_DEFAULT: float = 0.05
EXIF_ANOMALY_WEIGHT_DEFAULT: float = 0.03
SCENE_START_BONUS_WEIGHT_DEFAULT: float = 0.04

GENRE_WEIGHTS: dict[str, dict[str, float]] = {
    "general": {
        "topiq": 0.35,
        "laion_aesthetic": 0.35,
        "clipiqa": 0.20,
        "exposure": 0.10,
        "composition": COMPOSITION_WEIGHT_DEFAULT,
        "taste": TASTE_WEIGHT_DEFAULT,
        "tilt_penalty": TILT_PENALTY_WEIGHT_DEFAULT,
        "palette_outlier": PALETTE_OUTLIER_WEIGHT_DEFAULT,
        "exposure_drift": EXPOSURE_DRIFT_WEIGHT_DEFAULT,
        "exif_anomaly": EXIF_ANOMALY_WEIGHT_DEFAULT,
        "scene_start_bonus": SCENE_START_BONUS_WEIGHT_DEFAULT,
    },
    "wedding": {
        "topiq": 0.25,
        "laion_aesthetic": 0.40,
        "clipiqa": 0.25,
        "exposure": 0.10,
        "composition": COMPOSITION_WEIGHT_DEFAULT,
        "taste": TASTE_WEIGHT_DEFAULT,
        "tilt_penalty": TILT_PENALTY_WEIGHT_PEOPLE,
        "palette_outlier": PALETTE_OUTLIER_WEIGHT_DEFAULT,
        "exposure_drift": EXPOSURE_DRIFT_WEIGHT_DEFAULT,
        "exif_anomaly": EXIF_ANOMALY_WEIGHT_DEFAULT,
        "scene_start_bonus": SCENE_START_BONUS_WEIGHT_DEFAULT,
    },
    "documentary": {
        "topiq": 0.30,
        "laion_aesthetic": 0.30,
        "clipiqa": 0.30,
        "exposure": 0.10,
        "composition": COMPOSITION_WEIGHT_DEFAULT,
        "taste": TASTE_WEIGHT_DEFAULT,
        "tilt_penalty": TILT_PENALTY_WEIGHT_PEOPLE,
        "palette_outlier": PALETTE_OUTLIER_WEIGHT_DEFAULT,
        "exposure_drift": EXPOSURE_DRIFT_WEIGHT_DEFAULT,
        "exif_anomaly": EXIF_ANOMALY_WEIGHT_DEFAULT,
        "scene_start_bonus": SCENE_START_BONUS_WEIGHT_DEFAULT,
    },
    "wildlife": {
        "topiq": 0.50,
        "laion_aesthetic": 0.20,
        "clipiqa": 0.20,
        "exposure": 0.10,
        "composition": COMPOSITION_WEIGHT_DEFAULT,
        "taste": TASTE_WEIGHT_DEFAULT,
        "tilt_penalty": TILT_PENALTY_WEIGHT_PEOPLE,
        "palette_outlier": PALETTE_OUTLIER_WEIGHT_DEFAULT,
        "exposure_drift": EXPOSURE_DRIFT_WEIGHT_DEFAULT,
        "exif_anomaly": EXIF_ANOMALY_WEIGHT_DEFAULT,
        "scene_start_bonus": SCENE_START_BONUS_WEIGHT_DEFAULT,
    },
    "landscape": {
        "topiq": 0.30,
        "laion_aesthetic": 0.35,
        "clipiqa": 0.15,
        "exposure": 0.20,
        "composition": COMPOSITION_WEIGHT_DEFAULT,
        "taste": TASTE_WEIGHT_DEFAULT,
        "tilt_penalty": TILT_PENALTY_WEIGHT_LANDSCAPE,
        "palette_outlier": PALETTE_OUTLIER_WEIGHT_DEFAULT,
        "exposure_drift": EXPOSURE_DRIFT_WEIGHT_DEFAULT,
        "exif_anomaly": EXIF_ANOMALY_WEIGHT_DEFAULT,
        "scene_start_bonus": SCENE_START_BONUS_WEIGHT_DEFAULT,
    },
    "street": {
        "topiq": 0.30,
        "laion_aesthetic": 0.30,
        "clipiqa": 0.30,
        "exposure": 0.10,
        "composition": COMPOSITION_WEIGHT_DEFAULT,
        "taste": TASTE_WEIGHT_DEFAULT,
        "tilt_penalty": TILT_PENALTY_WEIGHT_DEFAULT,
        "palette_outlier": PALETTE_OUTLIER_WEIGHT_DEFAULT,
        "exposure_drift": EXPOSURE_DRIFT_WEIGHT_DEFAULT,
        "exif_anomaly": EXIF_ANOMALY_WEIGHT_DEFAULT,
        "scene_start_bonus": SCENE_START_BONUS_WEIGHT_DEFAULT,
    },
    "holiday": {
        "topiq": 0.30,
        "laion_aesthetic": 0.325,
        "clipiqa": 0.225,
        "exposure": 0.15,
        "composition": COMPOSITION_WEIGHT_DEFAULT,
        "taste": TASTE_WEIGHT_DEFAULT,
        "tilt_penalty": TILT_PENALTY_WEIGHT_DEFAULT,
        "palette_outlier": PALETTE_OUTLIER_WEIGHT_DEFAULT,
        "exposure_drift": EXPOSURE_DRIFT_WEIGHT_DEFAULT,
        "exif_anomaly": EXIF_ANOMALY_WEIGHT_DEFAULT,
        "scene_start_bonus": SCENE_START_BONUS_WEIGHT_DEFAULT,
    },
}

# ---------------------------------------------------------------------------
# Stage 2 preset-aware quality policy
# ---------------------------------------------------------------------------

# These knobs tune how strongly local subject sharpness can rescue a frame,
# how much intentional bokeh is tolerated, and how face-quality signals alter
# routing inside a given preset.
PRESET_QUALITY_POLICY: dict[str, dict[str, float]] = {
    "general": {
        "subject_blur_blend": 0.25,
        "bokeh_bonus": 0.03,
        "portrait_sharpness_bonus": 0.05,
        "eyes_closed_penalty": 0.12,
        "face_occlusion_penalty": 0.08,
    },
    "wedding": {
        "subject_blur_blend": 0.30,
        "bokeh_bonus": 0.05,
        "portrait_sharpness_bonus": 0.08,
        "eyes_closed_penalty": 0.18,
        "face_occlusion_penalty": 0.12,
    },
    "documentary": {
        "subject_blur_blend": 0.22,
        "bokeh_bonus": 0.02,
        "portrait_sharpness_bonus": 0.05,
        "eyes_closed_penalty": 0.10,
        "face_occlusion_penalty": 0.08,
    },
    "wildlife": {
        "subject_blur_blend": 0.35,
        "bokeh_bonus": 0.01,
        "portrait_sharpness_bonus": 0.00,
        "eyes_closed_penalty": 0.00,
        "face_occlusion_penalty": 0.00,
    },
    "landscape": {
        "subject_blur_blend": 0.10,
        "bokeh_bonus": 0.00,
        "portrait_sharpness_bonus": 0.00,
        "eyes_closed_penalty": 0.00,
        "face_occlusion_penalty": 0.00,
    },
    "street": {
        "subject_blur_blend": 0.20,
        "bokeh_bonus": 0.03,
        "portrait_sharpness_bonus": 0.05,
        "eyes_closed_penalty": 0.08,
        "face_occlusion_penalty": 0.06,
    },
    "holiday": {
        "subject_blur_blend": 0.28,
        "bokeh_bonus": 0.04,
        "portrait_sharpness_bonus": 0.06,
        "eyes_closed_penalty": 0.12,
        "face_occlusion_penalty": 0.08,
    },
}

# ---------------------------------------------------------------------------
# Stage 1 blur thresholds
# ---------------------------------------------------------------------------

BLUR_DHASH_HAMMING_MAX: int = 8
BLUR_CNN_SIMILARITY_BURST: float = 0.90
BLUR_CNN_SIMILARITY_EXACT: float = 0.98
BLUR_MOTION_ANISOTROPY_RATIO: float = 3.0
BLUR_SPATIAL_CENTER_FRACTION: float = 0.40

# ---------------------------------------------------------------------------
# Stage 1 exposure thresholds
# ---------------------------------------------------------------------------

EXPOSURE_HIGHLIGHT_CLIP_VALUE: int = 253
EXPOSURE_SHADOW_CLIP_VALUE: int = 2
EXPOSURE_HIGHLIGHT_CLIP_PCT_MAX: float = 0.03
EXPOSURE_SHADOW_CLIP_PCT_MAX: float = 0.20
EXPOSURE_DR_MIN: float = 0.45
EXPOSURE_MIDTONE_L_LOW: int = 64
EXPOSURE_MIDTONE_L_HIGH: int = 192
EXPOSURE_MIDTONE_PCT_MIN: float = 0.35
EXPOSURE_CAST_A_MAX: float = 8.0
EXPOSURE_CAST_B_MAX: float = 12.0
EXPOSURE_CAST_COMBINED_MAX: float = 18.0

# ---------------------------------------------------------------------------
# Stage 1 noise thresholds
# ---------------------------------------------------------------------------

NOISE_SCORE_REJECT_MIN: float = 0.70

# ---------------------------------------------------------------------------
# Stage 1 multiprocessing
# ---------------------------------------------------------------------------

STAGE1_WORKER_MAX: int = 6
STAGE1_WORKER_COUNT: int = min(os.cpu_count() or 1, STAGE1_WORKER_MAX)  # Capped at 6 to keep peak RSS under ~1.5 GB on 24MP corpora.

# ---------------------------------------------------------------------------
# Stage 4 curation thresholds
# ---------------------------------------------------------------------------

# Per-preset cosine-distance thresholds for MobileNetV3 embedding clusters.
# MobileNetV3 embeddings are L2-normalised, so cosine distance = 1 − similarity.
# Lower value → stricter grouping (fewer, more-similar images per cluster).
# Wedding and faces: near-identical poses must cluster tightly so only the
#   sharpest representative survives.  0.10 keeps same-moment variants together.
# General / documentary / wildlife: moderate threshold (0.15) balances diversity
#   with eliminating redundant near-duplicates.
# Landscape and street: looser threshold (0.20) preserves compositional variety
#   and decisive-moment differences typical of those genres.
CLUSTER_THRESHOLD: dict[str, float] = {
    "general": 0.15,       # moderate — balanced duplicate removal
    "wedding": 0.10,       # strict  — identical-pose clusters must be tight
    "documentary": 0.15,   # moderate — story continuity needs some variety
    "wildlife": 0.15,      # moderate — action sequences vary, keep range
    "landscape": 0.20,     # loose   — compositional diversity valued
    "street": 0.20,        # loose   — decisive moments differ visually
    "holiday": 0.18,       # mid     — blend of documentary and landscape
}

CURATE_DEFAULT_TARGET: int = 30
CURATE_VLM_TIEBREAK_THRESHOLD: float = 0.02

# ---------------------------------------------------------------------------
# Portrait mode thresholds
# ---------------------------------------------------------------------------

PORTRAIT_NUM_FACES_MAX: int = 10
PORTRAIT_FACE_DETECTION_CONFIDENCE_MIN: float = 0.5
PORTRAIT_LANDMARK_VISIBILITY_THRESHOLD: float = 0.5
PORTRAIT_EAR_CLOSED_MAX: float = 0.20
PORTRAIT_EAR_SQUINT_MAX: float = 0.25
PORTRAIT_FACE_OCCLUSION_MIN: float = 0.70
PORTRAIT_PRESENCE_SCORE_MIN: float = 0.70

# ---------------------------------------------------------------------------
# Burst grouping defaults
# ---------------------------------------------------------------------------

BURST_GAP_DEFAULT_SECONDS: float = 0.5
BURST_GAP_NO_SUBSEC_SECONDS: float = 2.0

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

IMAGE_LONG_EDGE_PX: int = 1280
JPEG_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg"})
STAGE2_BATCH_SIZE: int = 8

# ---------------------------------------------------------------------------
# Stage 2 IQA defaults
# ---------------------------------------------------------------------------

IQA_EXPOSURE_DEFAULT: float = 1.0

# ---------------------------------------------------------------------------
# Dashboard refresh timing
# ---------------------------------------------------------------------------

DASHBOARD_REFRESH_SLEEP_SECONDS: float = 0.1

# ---------------------------------------------------------------------------
# TUI auto-save interval
# ---------------------------------------------------------------------------

TUI_AUTOSAVE_INTERVAL_SECONDS: int = 30
TUI_AUTOSAVE_BATCH_CONFIDENCE: float = 0.60

# ---------------------------------------------------------------------------
# Override log — append-only JSONL for training data
# ---------------------------------------------------------------------------

OVERRIDE_LOG_DIR: Path = Path.home() / ".cull"
OVERRIDE_LOG_PATH: Path = OVERRIDE_LOG_DIR / "overrides.jsonl"

# ---------------------------------------------------------------------------
# Performance fixture corpus (external, not in git)
# ---------------------------------------------------------------------------

# Default corpus for golden-baseline tests. Override via --corpus CLI flag
# (pytest / capture scripts) or PERF_CORPUS_PATH env var.
PERF_CORPUS_PATH: Path = Path(
    os.environ.get(
        "PERF_CORPUS_PATH",
        ""
    )
)

# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

SEARCH_TOP_K_DEFAULT: int = 20
EMBEDDING_CACHE_FILENAME: str = ".cull_embeddings.npy"
EMBEDDING_INDEX_FILENAME: str = ".cull_embeddings_index.json"
CLIP_MODEL_ID: str = "openai/clip-vit-large-patch14"

# ---------------------------------------------------------------------------
# Offline model cache
# ---------------------------------------------------------------------------

PHOTO_MANAGER_CACHE_ENV: str = "PHOTO_MANAGER_CACHE"
DEFAULT_CACHE_ROOT: Path = Path.home() / ".cache" / "photo-manager" / "models"

CLIP_REPO_ID: str = CLIP_MODEL_ID
AESTHETIC_REPO_ID: str = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"

FACE_LANDMARKER_FILENAME: str = "face_landmarker.task"
FACE_LANDMARKER_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

DEEPFACE_EMOTION_FILENAME: str = "facial_expression_model_weights.h5"
DEEPFACE_EMOTION_URL: str = (
    "https://github.com/serengil/deepface_models/releases/download/"
    "v1.0/facial_expression_model_weights.h5"
)

# SHA-256 pinning is deferred to v2 (see comments in cull.model_cache).
# Until pinned, file-existence is the only integrity check on cached assets.
# To pin a hash, replace the empty string with the 64-char hex digest from
# `shasum -a 256 <file>` after a known-good `cull setup --allow-network` run;
# subsequent runs will then verify the hash via cull.model_cache.check_manifest_entry.
CLIP_MODEL_SHA256: str = ""
AESTHETIC_HEAD_SHA256: str = ""
FACE_LANDMARKER_SHA256: str = ""
DEEPFACE_EMOTION_SHA256: str = ""

# Cache subdir tokens — kept here so config.py can build MODEL_MANIFEST
# without importing from cull.model_cache (which would create a cycle).
SUBDIR_HF: str = "hf"
SUBDIR_MEDIAPIPE: str = "mediapipe"
SUBDIR_DEEPFACE: str = "deepface"

# ---------------------------------------------------------------------------
# Rejection explanations
# ---------------------------------------------------------------------------

EXPLAIN_MAX_TOKENS: int = 400
EXPLAIN_TEMPERATURE: float = 0.1

# ---------------------------------------------------------------------------
# Report card
# ---------------------------------------------------------------------------

REPORT_CARD_EXIF_SAMPLE_SIZE: int = 30

# ---------------------------------------------------------------------------
# Geometry / composition analysis
# ---------------------------------------------------------------------------

TILT_PENALTY_DEGREES: float = 3.0
KEYSTONE_PENALTY_DEGREES: float = 5.0

# ---------------------------------------------------------------------------
# Colour palette and exposure drift
# ---------------------------------------------------------------------------

PALETTE_OUTLIER_SIGMA: float = 2.5
EXPOSURE_DRIFT_SIGMA: float = 2.0

# ---------------------------------------------------------------------------
# EXIF anomaly detection
# ---------------------------------------------------------------------------

EXIF_ANOMALY_SIGMA: float = 3.0

# ---------------------------------------------------------------------------
# Taste model
# ---------------------------------------------------------------------------

TASTE_MIN_LABELS: int = 5
TASTE_RAMP_LABELS: int = 20
TASTE_RETRAIN_BATCH: int = 50
DPP_QUALITY_TRADEOFF: float = 0.5
TASTE_PROFILE_PATH: Path = Path.home() / ".cull" / "taste_profile.joblib"

# ---------------------------------------------------------------------------
# Portrait / face quality
# ---------------------------------------------------------------------------

EYE_OPEN_THRESHOLD: float = 0.25
SMILE_THRESHOLD: float = 0.40

# ---------------------------------------------------------------------------
# Narrative flow — shot-type thresholds
# ---------------------------------------------------------------------------

# Face-bbox-area-to-frame-area ratio thresholds for shot classification.
# Close-up: face occupies ≥ 15 % of the frame.
# Medium shot: face occupies 3–15 % of the frame.
# Wide shot: face occupies < 3 % (or no face — saliency fallback used).
SHOT_CLOSE_UP_RATIO: float = 0.15
SHOT_MEDIUM_RATIO: float = 0.03

# Saliency-extent fallback: fraction of the heatmap grid covered by the
# thresholded bbox (x1/w .. x0/w, y1/h .. y0/h).
# Values ≥ this threshold → wide (subject fills much of frame).
SHOT_SALIENCY_WIDE_RATIO: float = 0.50

# Minimum variety score below which a swap is proposed.
NARRATIVE_VARIETY_MIN: float = 0.60

# ---------------------------------------------------------------------------
# Motion / saliency
# ---------------------------------------------------------------------------

MOTION_PEAK_WINDOW: int = 5
SALIENCY_TARGET_PX: int = 512
SHARED_DECODE_CLIP_PX: int = 224  # Shared CLIP forward input edge size.
SHARED_DECODE_PIXEL_PX: int = 1280  # Long-edge LANCZOS target for pixel-quality consumers.
CLIP_PATCH_GRID: int = 16  # CLIP-L/14 patch grid for 224 input.

# ---------------------------------------------------------------------------
# Sidecar XMP namespace
# ---------------------------------------------------------------------------

SIDECAR_NAMESPACE_CRS: str = "http://ns.adobe.com/camera-raw-settings/1.0/"

# ---------------------------------------------------------------------------
# TUI overlay rendering
# ---------------------------------------------------------------------------

OVERLAY_HORIZON_COLOR: tuple[int, int, int] = (255, 80, 0)
OVERLAY_CROP_COLOR: tuple[int, int, int] = (0, 200, 255)
OVERLAY_LINE_THICKNESS: int = 2
OVERLAY_LABEL_OFFSET_PX: int = 4

# ---------------------------------------------------------------------------
# Cull default threshold
# ---------------------------------------------------------------------------

CULL_DEFAULT_THRESHOLD: float = 0.65

# ---------------------------------------------------------------------------
# Preset type alias
# ---------------------------------------------------------------------------

PresetName = Literal[
    "general", "wedding", "documentary", "wildlife", "landscape", "street", "holiday"
]

# ---------------------------------------------------------------------------
# Model cache config
# ---------------------------------------------------------------------------


class ModelCacheConfig(BaseModel):
    """Resolved on-disk paths for the offline model cache."""

    root: Path
    hf_home: Path
    torch_home: Path
    deepface_home: Path
    mediapipe_dir: Path

    @classmethod
    def from_env(cls) -> "ModelCacheConfig":
        """Resolve cache paths from PHOTO_MANAGER_CACHE or default."""
        override = os.environ.get(PHOTO_MANAGER_CACHE_ENV)
        root = Path(override) if override else DEFAULT_CACHE_ROOT
        return cls(
            root=root,
            hf_home=root / "hf",
            torch_home=root / "torch",
            deepface_home=root / "deepface",
            mediapipe_dir=root / "mediapipe",
        )


class ModelManifestEntry(BaseModel):
    """One cached model asset — HF repo or URL file."""

    name: str
    kind: Literal["hf_repo", "url_file"]
    subdir: str
    repo_id: str | None = None
    url: str | None = None
    filename: str | None = None
    sha256: str | None = None


MODEL_MANIFEST: dict[str, ModelManifestEntry] = {
    "clip": ModelManifestEntry(
        name="clip",
        kind="hf_repo",
        repo_id=CLIP_REPO_ID,
        sha256=CLIP_MODEL_SHA256 or None,
        subdir=SUBDIR_HF,
    ),
    "aesthetic": ModelManifestEntry(
        name="aesthetic",
        kind="hf_repo",
        repo_id=AESTHETIC_REPO_ID,
        sha256=AESTHETIC_HEAD_SHA256 or None,
        subdir=SUBDIR_HF,
    ),
    "face_landmarker": ModelManifestEntry(
        name="face_landmarker",
        kind="url_file",
        url=FACE_LANDMARKER_URL,
        filename=FACE_LANDMARKER_FILENAME,
        sha256=FACE_LANDMARKER_SHA256 or None,
        subdir=SUBDIR_MEDIAPIPE,
    ),
    "deepface_emotion": ModelManifestEntry(
        name="deepface_emotion",
        kind="url_file",
        url=DEEPFACE_EMOTION_URL,
        filename=DEEPFACE_EMOTION_FILENAME,
        sha256=DEEPFACE_EMOTION_SHA256 or None,
        subdir=SUBDIR_DEEPFACE,
    ),
}

# ---------------------------------------------------------------------------
# Runtime config model
# ---------------------------------------------------------------------------


class CullConfig(BaseModel):
    """Runtime configuration for a single cull session."""

    threshold: float = Field(
        default=CULL_DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Keeper score threshold override.",
    )
    burst_gap: float = Field(
        default=BURST_GAP_DEFAULT_SECONDS,
        gt=0.0,
        description="Burst grouping time window in seconds.",
    )
    preset: PresetName = Field(
        default="general",
        description="Genre preset for Stage 2 score fusion weights.",
    )
    model: str = Field(
        default=VLM_DEFAULT_ALIAS,
        description="VLM alias (see VLM_ALIASES)",
    )
    is_portrait: bool = Field(
        default=True,
        description="Enable portrait mode (face/eye quality analysis).",
    )
    is_dry_run: bool = Field(
        default=False,
        description="Show decisions without moving any files.",
    )
    stages: list[int] = Field(
        default=[1, 2, 3],
        description="Pipeline stages to run.",
    )
    curate_target: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of photos to select in Stage 4 curation. "
            "None = curation stage disabled."
        ),
    )
    curate_vlm_threshold: float = Field(
        default=CURATE_VLM_TIEBREAK_THRESHOLD,
        ge=0.0,
        le=1.0,
        description=(
            "Composite-score gap below which Stage 4 calls the VLM "
            "to break a cluster-winner tie."
        ),
    )
    is_sidecars: bool = Field(
        default=True,
        description="Write XMP sidecar files alongside source images.",
    )
