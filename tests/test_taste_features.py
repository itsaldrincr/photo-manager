"""Tests for cull.taste_features — the canonical train/serve taste feature row."""

from __future__ import annotations

import numpy as np

from cull.models import CompositionScore, OverrideEntry, SubjectBlurScore
from cull.taste_features import (
    NEUTRAL_COMPOSITE,
    NEUTRAL_COMPOSITION_METRIC,
    NEUTRAL_SUBJECT_BLUR,
    TasteFeatureInputs,
    build_taste_feature_row,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE1_SCORES: dict[str, float] = {
    "tenengrad": 0.72,
    "fft_ratio": 0.55,
    "dr_score": 0.80,
    "clipping_highlight": 0.01,
    "clipping_shadow": 0.02,
    "midtone_pct": 0.60,
    "color_cast_score": 0.03,
    "noise_score": 0.10,
    "tilt_degrees": 0.5,
    "keystone_degrees": 0.2,
}
STAGE2_COMPOSITE: float = 0.68
COMPOSITION_COMPOSITE: float = 0.75
THIRDS_ALIGNMENT: float = 0.80
NEGATIVE_SPACE: float = 0.65
SUBJECT_BLUR_TENENGRAD: float = 850.0
EXPECTED_ROW_LENGTH: int = 15  # 10 stage1 keys + 5 stage2 scalars


def _full_inputs() -> TasteFeatureInputs:
    """Build a fully-populated TasteFeatureInputs for one logical photo."""
    return TasteFeatureInputs(
        stage1_scores=STAGE1_SCORES,
        stage2_composite=STAGE2_COMPOSITE,
        composition_composite=COMPOSITION_COMPOSITE,
        thirds_alignment=THIRDS_ALIGNMENT,
        negative_space_balance=NEGATIVE_SPACE,
        subject_blur_tenengrad=SUBJECT_BLUR_TENENGRAD,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_row_length_and_dtype() -> None:
    """The canonical row has one column per stage1 key plus five stage2 scalars."""
    row = build_taste_feature_row(_full_inputs())
    assert row.shape == (EXPECTED_ROW_LENGTH,)
    assert row.dtype == np.float32


def test_missing_stage2_scalars_use_neutral_fallback() -> None:
    """Absent stage2 scalars fall back to their documented neutral constants."""
    inputs = TasteFeatureInputs(stage1_scores=STAGE1_SCORES)
    row = build_taste_feature_row(inputs)
    tail = row[-5:]
    expected_tail = np.asarray(
        [
            NEUTRAL_COMPOSITE,
            NEUTRAL_COMPOSITION_METRIC,
            NEUTRAL_COMPOSITION_METRIC,
            NEUTRAL_COMPOSITION_METRIC,
            NEUTRAL_SUBJECT_BLUR,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(tail, expected_tail)


def test_train_serve_row_identity_for_same_logical_photo() -> None:
    """taste_trainer's parts-based path and stage2_scoring's live-scoring path
    build byte-identical rows for the same logical photo's data."""
    from cull.taste_trainer import _taste_inputs_from_entry  # noqa: PLC0415

    entry = OverrideEntry(
        photo_path="/tmp/x.jpg",
        filename="x.jpg",
        original_decision="uncertain",
        user_decision="keeper",
        stage1_scores=STAGE1_SCORES,
        stage2_composite=STAGE2_COMPOSITE,
        session_source="test",
        override_origin="unit",
        stage2_composition=CompositionScore(
            thirds_alignment=THIRDS_ALIGNMENT,
            edge_clearance=0.5,
            negative_space_balance=NEGATIVE_SPACE,
            topiq_iaa=0.5,
            composite=COMPOSITION_COMPOSITE,
        ),
        stage2_subject_blur=SubjectBlurScore(
            tenengrad=SUBJECT_BLUR_TENENGRAD,
            subject_region_source="face",
            has_subject=True,
        ),
    )

    train_row = build_taste_feature_row(_taste_inputs_from_entry(entry))
    serve_row = build_taste_feature_row(_full_inputs())
    np.testing.assert_array_equal(train_row, serve_row)


def test_row_is_stable_regardless_of_stage1_dict_insertion_order() -> None:
    """Row values depend on sorted stage1 keys, not dict insertion order."""
    reordered = dict(reversed(list(STAGE1_SCORES.items())))
    row_a = build_taste_feature_row(TasteFeatureInputs(stage1_scores=STAGE1_SCORES))
    row_b = build_taste_feature_row(TasteFeatureInputs(stage1_scores=reordered))
    np.testing.assert_array_equal(row_a, row_b)
