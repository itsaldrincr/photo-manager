"""Stage 2 taste scorer — logistic model over the canonical scalar feature row.

See cull.taste_features for the shared row layout used here and in
taste_trainer.py.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel

from cull.config import TASTE_MIN_LABELS, TASTE_PROFILE_PATH
from cull.models import TasteScore

logger = logging.getLogger(__name__)

WARMSTART_PROBABILITY: float = 0.5
WARMSTART_WEIGHT: float = 0.0
WARMSTART_VERSION: str = "warmstart"


class TasteScoreInput(BaseModel):
    """Public input bundle for taste scoring: one canonical feature row."""

    model_config = {"arbitrary_types_allowed": True}

    feature_row: np.ndarray


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

        # Safe: TASTE_PROFILE_PATH is this app's own locally-trained artifact
        # (written by taste_trainer.retrain), never an externally-sourced file.
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


def _expected_feature_count(profile: _TasteProfile) -> int | None:
    """Return the profile's fitted feature count, or None if unavailable."""
    return getattr(profile.estimator, "n_features_in_", None)


def _shape_matches(profile: _TasteProfile, row: np.ndarray) -> bool:
    """Return True if the profile's fitted feature count matches the row length."""
    n_expected = _expected_feature_count(profile)
    return n_expected is None or n_expected == row.shape[0]


def _scored(profile: _TasteProfile, row: np.ndarray) -> TasteScore:
    """Run the loaded estimator on one feature row and wrap the output."""
    probability = float(profile.estimator.predict_proba(row.reshape(1, -1))[0, 1])
    return TasteScore(
        probability=probability,
        label_count_at_score=profile.label_count,
        weight_applied=1.0,
        model_version=profile.version,
    )


def _guarded_score(profile: _TasteProfile, row: np.ndarray) -> TasteScore:
    """Score row against profile, falling back to warmstart on a shape mismatch."""
    if _shape_matches(profile, row):
        return _scored(profile, row)
    logger.warning(
        "Taste profile shape mismatch: model expects %s features, got %d — using warmstart",
        _expected_feature_count(profile), row.shape[0],
    )
    return _warmstart_score()


def score_one(score_in: TasteScoreInput) -> TasteScore:
    """Score one photo's taste probability; warm-starts when no profile exists."""
    profile = _load_profile()
    if profile is None or profile.label_count < TASTE_MIN_LABELS:
        return _warmstart_score()
    row = np.asarray(score_in.feature_row, dtype=np.float32).reshape(-1)
    return _guarded_score(profile, row)


def score_batch(score_inputs: list[TasteScoreInput]) -> list[TasteScore]:
    """Score a batch of photos by delegating to score_one for each input."""
    return [score_one(item) for item in score_inputs]
