"""Stage 2 taste scorer — logistic over CLIP embedding + scalar features."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from cull.config import TASTE_MIN_LABELS, TASTE_PROFILE_PATH
from cull.models import TasteScore

logger = logging.getLogger(__name__)

WARMSTART_PROBABILITY: float = 0.5
WARMSTART_WEIGHT: float = 0.0
WARMSTART_VERSION: str = "warmstart"


@dataclass
class TasteFeatureVector:
    """CLIP embedding + scalar feature row used as taste model input."""

    clip_embedding: np.ndarray
    scalar_scores: np.ndarray


class TasteScoreInput(BaseModel):
    """Public input bundle for taste scoring."""

    model_config = {"arbitrary_types_allowed": True}

    image_path: Path
    scalar_features: np.ndarray


class _TasteProfile(BaseModel):
    """Loaded taste model artifact returned by joblib."""

    model_config = {"arbitrary_types_allowed": True}

    estimator: object
    label_count: int
    version: str


_profile_cache: _TasteProfile | None = None
_load_attempted: bool = False


def _load_profile() -> _TasteProfile | None:
    """Lazily load and cache the joblib taste profile from disk."""
    global _profile_cache, _load_attempted
    if _load_attempted:
        return _profile_cache
    _load_attempted = True
    if not TASTE_PROFILE_PATH.exists():
        return None
    try:
        import joblib  # noqa: PLC0415

        data = joblib.load(TASTE_PROFILE_PATH)
        _profile_cache = _TasteProfile(**data)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Failed to load taste profile: %s", exc)
        _profile_cache = None
    return _profile_cache


def _reset_profile_cache() -> None:
    """Test hook to clear the lazy profile cache."""
    global _profile_cache, _load_attempted
    _profile_cache = None
    _load_attempted = False


def _warmstart_score() -> TasteScore:
    """Return the neutral taste score used while the model is cold."""
    return TasteScore(
        probability=WARMSTART_PROBABILITY,
        label_count_at_score=0,
        weight_applied=WARMSTART_WEIGHT,
        model_version=WARMSTART_VERSION,
    )


def _embed_clip(image_path: Path) -> np.ndarray:
    """Run the CLIP singleton on one image and return its embedding row."""
    from cull.clip_loader import get_clip_model, get_clip_processor  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    model = get_clip_model()
    processor = get_clip_processor()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    features = model.get_image_features(**inputs).pooler_output
    return features.detach().cpu().numpy().reshape(-1)


def _build_feature_row(score_in: TasteScoreInput) -> np.ndarray:
    """Concatenate CLIP embedding + scalar features into a single 1-D row."""
    embedding = _embed_clip(score_in.image_path)
    scalar = np.asarray(score_in.scalar_features, dtype=np.float32).reshape(-1)
    return np.concatenate([embedding.astype(np.float32), scalar], axis=0)


def _scored(profile: _TasteProfile, row: np.ndarray) -> TasteScore:
    """Run the loaded estimator on one feature row and wrap the output."""
    probability = float(profile.estimator.predict_proba(row.reshape(1, -1))[0, 1])
    return TasteScore(
        probability=probability,
        label_count_at_score=profile.label_count,
        weight_applied=1.0,
        model_version=profile.version,
    )


def score_one(score_in: TasteScoreInput) -> TasteScore:
    """Score one photo's taste probability; warm-starts when no profile exists."""
    profile = _load_profile()
    if profile is None or profile.label_count < TASTE_MIN_LABELS:
        return _warmstart_score()
    row = _build_feature_row(score_in)
    return _scored(profile, row)


def score_batch(score_inputs: list[TasteScoreInput]) -> list[TasteScore]:
    """Score a batch of photos by delegating to score_one for each input."""
    return [score_one(item) for item in score_inputs]
