"""Incremental + batch retrainer for the taste model from override entries."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from cull.config import TASTE_RETRAIN_BATCH
from cull.models import OverrideEntry

logger = logging.getLogger(__name__)

KEEPER_LABEL: str = "keeper"
COUNTER_SUFFIX: str = ".counter"
PROFILE_VERSION_PREFIX: str = "logreg-v"


class TasteTrainerInput(BaseModel):
    """Input bundle for the taste trainer entry points."""

    model_config = {"arbitrary_types_allowed": True}

    overrides: list[OverrideEntry]
    profile_path: Path


def _label_for(entry: OverrideEntry) -> int:
    """Map a final user_decision to a 1/0 keeper label."""
    return 1 if entry.user_decision == KEEPER_LABEL else 0


def _features_for(entry: OverrideEntry) -> np.ndarray:
    """Flatten an OverrideEntry's stage1_scores dict into a stable vector."""
    keys = sorted(entry.stage1_scores.keys())
    return np.asarray([entry.stage1_scores[k] for k in keys], dtype=np.float32)


def _build_matrix(overrides: list[OverrideEntry]) -> tuple[np.ndarray, np.ndarray]:
    """Stack feature rows + labels from a list of overrides."""
    rows = [_features_for(o) for o in overrides]
    labels = [_label_for(o) for o in overrides]
    return np.stack(rows, axis=0), np.asarray(labels, dtype=np.int64)


def _persist(estimator: object, ctx: TasteTrainerInput) -> Path:
    """Persist the trained estimator + label count to ctx.profile_path via joblib."""
    import joblib  # noqa: PLC0415

    payload = {
        "estimator": estimator,
        "label_count": len(ctx.overrides),
        "version": f"{PROFILE_VERSION_PREFIX}{len(ctx.overrides)}",
    }
    ctx.profile_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, ctx.profile_path)
    return ctx.profile_path


def retrain(ctx: TasteTrainerInput) -> Path:
    """Batch-retrain the taste model from all overrides and persist it."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    matrix, labels = _build_matrix(ctx.overrides)
    estimator = LogisticRegression(class_weight="balanced", max_iter=1000)
    estimator.fit(matrix, labels)
    return _persist(estimator, ctx)


def _counter_path_for(profile_path: Path) -> Path:
    """Return the on-disk counter file companion to a profile path."""
    return profile_path.with_suffix(profile_path.suffix + COUNTER_SUFFIX)


def _read_counter(counter_path: Path) -> int:
    """Read the persisted retrain counter, defaulting to zero on miss."""
    if not counter_path.exists():
        return 0
    try:
        return int(counter_path.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return 0


def _write_counter(counter_path: Path, value: int) -> None:
    """Persist the retrain counter integer to disk."""
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    counter_path.write_text(str(value), encoding="utf-8")


def maybe_retrain(ctx: TasteTrainerInput) -> Path | None:
    """Retrain only when the counter has accumulated TASTE_RETRAIN_BATCH labels."""
    counter_path = _counter_path_for(ctx.profile_path)
    new_count = _read_counter(counter_path) + len(ctx.overrides)
    if new_count < TASTE_RETRAIN_BATCH:
        _write_counter(counter_path, new_count)
        return None
    _write_counter(counter_path, 0)
    return retrain(ctx)


def _stream_partial_fit(estimator: object, ctx: TasteTrainerInput) -> object:
    """Apply river-based streaming partial fit to an existing estimator."""
    from river import linear_model  # noqa: PLC0415

    online = estimator if isinstance(estimator, linear_model.LogisticRegression) else linear_model.LogisticRegression()
    for entry in ctx.overrides:
        features = {str(i): float(v) for i, v in enumerate(_features_for(entry))}
        online.learn_one(features, _label_for(entry))
    return online
