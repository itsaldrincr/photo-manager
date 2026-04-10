"""Tests for Stage 2 score fusion and routing logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from cull.config import CullConfig, GENRE_WEIGHTS
from cull.models import PortraitScores
from cull.stage2.fusion import IqaScores, compute_composite


PHOTO_PATH = Path("/tmp/test_photo.jpg")


def _make_scores(topiq: float, laion: float, clipiqa: float, exposure: float) -> IqaScores:
    return IqaScores(
        photo_path=PHOTO_PATH,
        topiq=topiq,
        laion_aesthetic=laion,
        clipiqa=clipiqa,
        exposure=exposure,
    )


def _uniform_scores(value: float) -> IqaScores:
    """All metrics equal to `value` so composite == value regardless of weights."""
    return _make_scores(value, value, value, value)


def test_routes_keeper_at_high_composite() -> None:
    """Composite of 0.80 should route to KEEPER."""
    scores = _uniform_scores(0.80)
    result = compute_composite(scores, CullConfig())
    assert result.routing == "KEEPER"
    assert result.stage2.composite == pytest.approx(0.80, abs=1e-6)


def test_routes_ambiguous_at_mid_composite() -> None:
    """Composite of 0.60 should route to AMBIGUOUS."""
    scores = _uniform_scores(0.60)
    result = compute_composite(scores, CullConfig())
    assert result.routing == "AMBIGUOUS"
    assert result.stage2.composite == pytest.approx(0.60, abs=1e-6)


def test_routes_reject_at_low_composite() -> None:
    """Composite of 0.30 should route to REJECT."""
    scores = _uniform_scores(0.30)
    result = compute_composite(scores, CullConfig())
    assert result.routing == "REJECT"
    assert result.stage2.composite == pytest.approx(0.30, abs=1e-6)


def test_keeper_boundary_exact() -> None:
    """Score exactly at ROUTING_KEEPER_MIN (0.72) should route to KEEPER."""
    scores = _uniform_scores(0.72)
    result = compute_composite(scores, CullConfig())
    assert result.routing == "KEEPER"


def test_ambiguous_boundary_exact() -> None:
    """Score exactly at ROUTING_AMBIGUOUS_MIN (0.48) should route to AMBIGUOUS."""
    scores = _uniform_scores(0.48)
    result = compute_composite(scores, CullConfig())
    assert result.routing == "AMBIGUOUS"


def test_just_below_keeper_boundary() -> None:
    """Score just below ROUTING_KEEPER_MIN (0.719) should route to AMBIGUOUS."""
    scores = _uniform_scores(0.719)
    result = compute_composite(scores, CullConfig())
    assert result.routing == "AMBIGUOUS"


def test_just_below_ambiguous_boundary() -> None:
    """Score just below ROUTING_AMBIGUOUS_MIN (0.479) should route to REJECT."""
    scores = _uniform_scores(0.479)
    result = compute_composite(scores, CullConfig())
    assert result.routing == "REJECT"


def test_genre_weights_change_composite() -> None:
    """Different genre presets must produce different composite values for skewed inputs."""
    # Give topiq a high value and laion_aesthetic a low value.
    # wildlife preset weights topiq=0.50, laion=0.20 → composite skews high.
    # wedding preset weights topiq=0.25, laion=0.40 → composite skews low.
    scores = _make_scores(topiq=0.9, laion=0.1, clipiqa=0.5, exposure=0.5)
    wildlife_config = CullConfig(preset="wildlife")
    wedding_config = CullConfig(preset="wedding")
    wildlife_result = compute_composite(scores, wildlife_config)
    wedding_result = compute_composite(scores, wedding_config)
    assert wildlife_result.stage2.composite != pytest.approx(
        wedding_result.stage2.composite, abs=1e-6
    )
    assert wildlife_result.stage2.composite > wedding_result.stage2.composite


def test_preset_recorded_in_stage2() -> None:
    """The preset used should be recorded in the Stage2Result."""
    scores = _uniform_scores(0.80)
    result = compute_composite(scores, CullConfig(preset="wedding"))
    assert result.stage2.preset_used == "wedding"


# ---------------------------------------------------------------------------
# Tilt penalty tests (no real ML)
# ---------------------------------------------------------------------------


def test_tilt_penalty_none_does_not_change_composite() -> None:
    """When tilt_penalty is None the composite must equal the no-penalty value."""
    baseline = compute_composite(_uniform_scores(0.80), CullConfig())
    with_none = _uniform_scores(0.80)
    with_none.tilt_penalty = None
    result = compute_composite(with_none, CullConfig())
    assert result.stage2.composite == pytest.approx(baseline.stage2.composite, abs=1e-9)


def test_tilt_penalty_subtracts_from_composite() -> None:
    """A non-null tilt_penalty must subtract weight*penalty from the composite."""
    scores = _uniform_scores(0.80)
    scores.tilt_penalty = 1.0
    result = compute_composite(scores, CullConfig(preset="general"))
    expected_drop = GENRE_WEIGHTS["general"]["tilt_penalty"]
    assert result.stage2.composite < 0.80
    assert (0.80 - result.stage2.composite) == pytest.approx(expected_drop, abs=1e-6)


def test_tilt_penalty_landscape_weighted_higher_than_wedding() -> None:
    """Landscape preset penalises tilt more aggressively than wedding."""
    scores_landscape = _uniform_scores(0.80)
    scores_wedding = _uniform_scores(0.80)
    scores_landscape.tilt_penalty = 1.0
    scores_wedding.tilt_penalty = 1.0
    landscape_result = compute_composite(scores_landscape, CullConfig(preset="landscape"))
    wedding_result = compute_composite(scores_wedding, CullConfig(preset="wedding"))
    assert landscape_result.stage2.composite < wedding_result.stage2.composite


def test_every_genre_has_tilt_penalty_key() -> None:
    """Every preset in GENRE_WEIGHTS must define a tilt_penalty weight key."""
    for preset, weights in GENRE_WEIGHTS.items():
        assert "tilt_penalty" in weights, f"missing tilt_penalty for preset {preset}"


def test_subject_blur_blends_with_topiq_instead_of_overriding() -> None:
    """Subject sharpness should raise the score, but not replace weak global IQA."""
    scores = _make_scores(topiq=0.2, laion=0.5, clipiqa=0.5, exposure=0.5)
    scores.subject_blur = 1000.0
    result = compute_composite(scores, CullConfig(preset="holiday"))
    assert result.stage2.composite > 0.2
    assert result.stage2.composite < 0.6


def test_bokeh_bonus_is_preset_aware() -> None:
    """Holiday should tolerate intentional bokeh more than landscape."""
    holiday_scores = _make_scores(topiq=0.4, laion=0.5, clipiqa=0.5, exposure=0.5)
    landscape_scores = holiday_scores.model_copy(deep=True)
    holiday_scores.subject_blur = 1000.0
    landscape_scores.subject_blur = 1000.0
    holiday_scores.is_bokeh = True
    landscape_scores.is_bokeh = True
    holiday = compute_composite(holiday_scores, CullConfig(preset="holiday"))
    landscape = compute_composite(landscape_scores, CullConfig(preset="landscape"))
    assert holiday.stage2.composite > landscape.stage2.composite


def test_portrait_penalties_and_bonus_affect_routing() -> None:
    """Portrait quality should reward sharp eyes and penalize closed or occluded faces."""
    base = _make_scores(topiq=0.68, laion=0.68, clipiqa=0.68, exposure=0.68)
    improved = base.model_copy(deep=True)
    improved.portrait = PortraitScores(
        eye_sharpness_left=900.0,
        eye_sharpness_right=900.0,
        is_eyes_closed=False,
        is_face_occluded=False,
    )
    penalized = base.model_copy(deep=True)
    penalized.portrait = PortraitScores(
        eye_sharpness_left=900.0,
        eye_sharpness_right=900.0,
        is_eyes_closed=True,
        is_face_occluded=True,
    )
    improved_result = compute_composite(improved, CullConfig(preset="wedding"))
    penalized_result = compute_composite(penalized, CullConfig(preset="wedding"))
    assert improved_result.stage2.composite > penalized_result.stage2.composite
    assert improved_result.routing != "REJECT"
