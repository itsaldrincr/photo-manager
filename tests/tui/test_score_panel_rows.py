"""Unit tests for the five new score rows in score_panel.py."""

from __future__ import annotations

from pathlib import Path

from cull.models import (
    BlurScores,
    BurstInfo,
    CompositionScore,
    ExposureScores,
    GeometryScore,
    PhotoDecision,
    PhotoMeta,
    PortraitScores,
    ShootStatsScore,
    Stage1Result,
    Stage2Result,
    SubjectBlurScore,
    TasteScore,
)
from cull.tui.score_panel import EM_DASH, render_score_text

_PHOTO_PATH = Path("/tmp/test.jpg")


def _make_photo_meta() -> PhotoMeta:
    return PhotoMeta(path=_PHOTO_PATH, filename="test.jpg")


def _make_blur_scores() -> BlurScores:
    return BlurScores(tenengrad=120.0, fft_ratio=0.85, blur_tier=1)


def _make_exposure_scores() -> ExposureScores:
    return ExposureScores(
        dr_score=0.9,
        clipping_highlight=0.01,
        clipping_shadow=0.02,
        midtone_pct=0.65,
        color_cast_score=0.1,
    )


def _make_stage1_with_geometry() -> Stage1Result:
    return Stage1Result(
        photo_path=_PHOTO_PATH,
        blur=_make_blur_scores(),
        exposure=_make_exposure_scores(),
        noise_score=0.05,
        geometry=GeometryScore(
            tilt_degrees=2.5,
            keystone_degrees=0.3,
            confidence=0.95,
            has_horizon=True,
            has_verticals=False,
        ),
    )


def _make_stage1_no_geometry() -> Stage1Result:
    return Stage1Result(
        photo_path=_PHOTO_PATH,
        blur=_make_blur_scores(),
        exposure=_make_exposure_scores(),
        noise_score=0.05,
        geometry=None,
    )


def _make_stage2_populated() -> Stage2Result:
    return Stage2Result(
        photo_path=_PHOTO_PATH,
        topiq=0.72,
        laion_aesthetic=0.68,
        clipiqa=0.81,
        composite=0.74,
        composition=CompositionScore(
            thirds_alignment=0.6,
            edge_clearance=0.8,
            negative_space_balance=0.5,
            topiq_iaa=0.7,
            composite=0.650,
        ),
        subject_blur=SubjectBlurScore(
            tenengrad=145.3,
            subject_region_source="face",
            has_subject=True,
        ),
        taste=TasteScore(
            probability=0.823,
            label_count_at_score=3,
            weight_applied=1.0,
            model_version="v1",
        ),
        shoot_stats=ShootStatsScore(
            palette_outlier_score=0.7,
            exposure_drift_score=0.3,
            exif_anomaly_score=0.6,
            scene_start_bonus=0.0,
            scene_id=1,
        ),
    )


def _make_stage2_nulls() -> Stage2Result:
    return Stage2Result(
        photo_path=_PHOTO_PATH,
        topiq=0.5,
        laion_aesthetic=0.5,
        clipiqa=0.5,
        composite=0.5,
        composition=None,
        subject_blur=None,
        taste=None,
        shoot_stats=None,
    )


def _make_decision(stage1: Stage1Result, stage2: Stage2Result) -> PhotoDecision:
    return PhotoDecision(
        photo=_make_photo_meta(),
        decision="keeper",
        stage1=stage1,
        stage2=stage2,
    )


def test_populated_rows_appear() -> None:
    """Rendered text contains all five new score rows when fields are populated."""
    decision = _make_decision(_make_stage1_with_geometry(), _make_stage2_populated())
    text = render_score_text(decision)
    assert "Tilt (deg)" in text
    assert "2.5" in text
    assert "Composition" in text
    assert "0.650" in text
    assert "Subject sharpness" in text
    assert "145.30" in text
    assert "Taste p" in text
    assert "0.823" in text
    assert "Shoot anomaly" in text
    assert "palette" in text
    assert "exif" in text


def test_none_fields_render_em_dash() -> None:
    """Rows fall back to em dash when underlying stage fields are None."""
    decision = _make_decision(_make_stage1_no_geometry(), _make_stage2_nulls())
    text = render_score_text(decision)
    assert EM_DASH in text
    assert "Tilt (deg)" in text
    assert "Composition" in text
    assert "Subject sharpness" in text
    assert "Taste p" in text
    assert "Shoot anomaly" in text
    em_count = text.count(EM_DASH)
    assert em_count >= 5


def test_taste_untrained_warmstart() -> None:
    """Taste renders 'untrained' when weight_applied is 0.0."""
    stage2 = Stage2Result(
        photo_path=_PHOTO_PATH,
        topiq=0.72,
        laion_aesthetic=0.68,
        clipiqa=0.81,
        composite=0.74,
        taste=TasteScore(
            probability=0.5,
            label_count_at_score=0,
            weight_applied=0.0,
            model_version="warmstart",
        ),
    )
    decision = _make_decision(_make_stage1_no_geometry(), stage2)
    text = render_score_text(decision)
    assert "untrained" in text
    assert "0.500" not in text


def test_taste_trained_shows_probability() -> None:
    """Taste shows probability when model is trained (weight > 0)."""
    stage2 = Stage2Result(
        photo_path=_PHOTO_PATH,
        topiq=0.72,
        laion_aesthetic=0.68,
        clipiqa=0.81,
        composite=0.74,
        taste=TasteScore(
            probability=0.823,
            label_count_at_score=5,
            weight_applied=1.0,
            model_version="v1",
        ),
    )
    decision = _make_decision(_make_stage1_no_geometry(), stage2)
    text = render_score_text(decision)
    assert "0.823" in text
    assert "untrained" not in text


def test_portrait_emotion_with_valence_arousal() -> None:
    """Expression line shows emotion with valence/arousal when present."""
    stage2 = Stage2Result(
        photo_path=_PHOTO_PATH,
        topiq=0.72,
        laion_aesthetic=0.68,
        clipiqa=0.81,
        composite=0.74,
        portrait=PortraitScores(
            eye_sharpness_left=0.8,
            eye_sharpness_right=0.75,
            dominant_emotion="happy",
            valence=0.75,
            arousal=0.60,
        ),
    )
    decision = _make_decision(_make_stage1_no_geometry(), stage2)
    text = render_score_text(decision)
    assert "Expression:" in text
    assert "happy" in text
    assert "(v +0.75 a +0.60)" in text


def test_portrait_emotion_without_valence_arousal() -> None:
    """Expression line shows just emotion when valence/arousal absent."""
    stage2 = Stage2Result(
        photo_path=_PHOTO_PATH,
        topiq=0.72,
        laion_aesthetic=0.68,
        clipiqa=0.81,
        composite=0.74,
        portrait=PortraitScores(
            eye_sharpness_left=0.8,
            dominant_emotion="neutral",
        ),
    )
    decision = _make_decision(_make_stage1_no_geometry(), stage2)
    text = render_score_text(decision)
    assert "Expression:" in text
    assert "neutral" in text
    assert "(v" not in text


def test_no_portrait_emotion_omits_expression_line() -> None:
    """Expression line is omitted when portrait data absent."""
    stage2 = _make_stage2_nulls()
    decision = _make_decision(_make_stage1_no_geometry(), stage2)
    text = render_score_text(decision)
    lines = text.split("\n")
    expression_lines = [l for l in lines if "Expression:" in l]
    assert len(expression_lines) == 0
