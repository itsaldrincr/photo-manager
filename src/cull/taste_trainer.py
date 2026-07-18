"""Incremental + batch retrainer for the taste model from override entries."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from cull.config import TASTE_RETRAIN_BATCH
from cull.models import OverrideEntry
from cull.override_log import load_overrides
from cull.taste_features import TasteFeatureInputs, build_taste_feature_row

logger = logging.getLogger(__name__)

# "select" is the curated-keeper queue (see report_card._KEEPER_LABELS /
# router.SIDECAR_DECISIONS) — it must count as a positive taste label too.
KEEPER_LABELS: frozenset[str] = frozenset({"keeper", "select"})
COUNTER_SUFFIX: str = ".counter"
PROFILE_VERSION_PREFIX: str = "logreg-v"
MIN_CLASSES_TO_FIT: int = 2


class TasteTrainerInput(BaseModel):
    """Input bundle for the taste trainer entry points."""

    model_config = {"arbitrary_types_allowed": True}

    overrides: list[OverrideEntry]
    profile_path: Path


def _label_for(entry: OverrideEntry) -> int:
    """Map a final user_decision to a 1/0 keeper label."""
    return 1 if entry.user_decision in KEEPER_LABELS else 0


def _taste_inputs_from_entry(entry: OverrideEntry) -> TasteFeatureInputs:
    """Extract canonical taste feature inputs from a logged override entry."""
    composition = entry.stage2_composition
    subject_blur = entry.stage2_subject_blur
    return TasteFeatureInputs(
        stage1_scores=entry.stage1_scores,
        stage2_composite=entry.stage2_composite,
        composition_composite=composition.composite if composition else None,
        thirds_alignment=composition.thirds_alignment if composition else None,
        negative_space_balance=composition.negative_space_balance if composition else None,
        subject_blur_tenengrad=subject_blur.tenengrad if subject_blur else None,
    )


def _features_for(entry: OverrideEntry) -> np.ndarray:
    """Return an entry's canonical taste feature row.

    Prefers the row logged at write time (entry.feature_row); falls back to
    recomputing it from the entry's stage1/stage2 parts for older records
    that predate the feature_row field.
    """
    if entry.feature_row:
        return np.asarray(entry.feature_row, dtype=np.float32)
    return build_taste_feature_row(_taste_inputs_from_entry(entry))


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


def _has_enough_classes(labels: np.ndarray) -> bool:
    """Check whether labels contain both keeper and non-keeper examples."""
    return len(np.unique(labels)) >= MIN_CLASSES_TO_FIT


def retrain(ctx: TasteTrainerInput) -> Path | None:
    """Batch-retrain the taste model from all overrides and persist it.

    Returns None (without raising) when the override history is single-class
    (all-keeper or all-reject) — LogisticRegression cannot fit one class.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    matrix, labels = _build_matrix(ctx.overrides)
    if not _has_enough_classes(labels):
        logger.warning(
            "Skipping taste retrain: %d overrides are all one class, need both keeper and reject examples",
            len(labels),
        )
        return None
    # StandardScaler + LogisticRegression as one Pipeline: standardizing the
    # unscaled scalar row (tilt degrees, pixel-scale exposure counts, etc.)
    # fixes the solver's ConvergenceWarning, and the fitted scaler travels
    # with the estimator in a single joblib artifact — no separate file to
    # keep in sync at inference time.
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
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


def _full_history_ctx(profile_path: Path) -> TasteTrainerInput:
    """Build a trainer input from the complete on-disk override history.

    The incremental counter only tracks WHEN to retrain; the actual training
    set is always the full override log, not the handful of new entries that
    tripped the threshold.
    """
    return TasteTrainerInput(overrides=load_overrides(), profile_path=profile_path)


def maybe_retrain(ctx: TasteTrainerInput) -> Path | None:
    """Retrain on the full override history once the counter reaches TASTE_RETRAIN_BATCH."""
    counter_path = _counter_path_for(ctx.profile_path)
    new_count = _read_counter(counter_path) + len(ctx.overrides)
    if new_count < TASTE_RETRAIN_BATCH:
        _write_counter(counter_path, new_count)
        return None
    _write_counter(counter_path, 0)
    return retrain(_full_history_ctx(ctx.profile_path))


def _stream_partial_fit(estimator: object, ctx: TasteTrainerInput) -> object:
    """Apply river-based streaming partial fit to an existing estimator."""
    from river import linear_model  # noqa: PLC0415

    online = estimator if isinstance(estimator, linear_model.LogisticRegression) else linear_model.LogisticRegression()
    for entry in ctx.overrides:
        features = {str(i): float(v) for i, v in enumerate(_features_for(entry))}
        online.learn_one(features, _label_for(entry))
    return online
