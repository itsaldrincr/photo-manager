"""Parametrized tests for _rescale_preset_weights and build_iqa_from_musiq."""

from __future__ import annotations

from pathlib import Path

import pytest

from cull_fast.fusion_fast import (
    FAST_CLIPIQA_SENTINEL,
    _FusionFastInput,
    _rescale_preset_weights,
    build_iqa_from_musiq,
)
from cull_fast.musiq import MusiQScorePair
from cull.config import GENRE_WEIGHTS

ALL_PRESETS: tuple[str, ...] = (
    "general",
    "wedding",
    "documentary",
    "wildlife",
    "landscape",
    "street",
    "holiday",
)

_TECHNICAL_SCORE: float = 0.73
_AESTHETIC_SCORE: float = 0.41
_EXPOSURE_SCORE: float = 0.88
FAKE_PHOTO_PATH: Path = Path("/tmp/x.jpg")


def _make_fast_input() -> _FusionFastInput:
    """Build a canonical _FusionFastInput for build_iqa tests."""
    pair = MusiQScorePair(
        photo_path=FAKE_PHOTO_PATH,
        technical=_TECHNICAL_SCORE,
        aesthetic=_AESTHETIC_SCORE,
    )
    return _FusionFastInput(musiq_scores=pair, exposure_score=_EXPOSURE_SCORE)


def _rescaled(preset: str) -> dict[str, float]:
    """Return rescaled weights for a preset."""
    return _rescale_preset_weights(preset)


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_rescale_sums_to_one(preset: str) -> None:
    """topiq + laion_aesthetic + exposure must sum to 1.0 after rescaling."""
    weights = _rescaled(preset)
    total = weights["topiq"] + weights["laion_aesthetic"] + weights["exposure"]
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_rescale_drops_clipiqa(preset: str) -> None:
    """clipiqa weight must be zeroed in fast mode."""
    weights = _rescaled(preset)
    assert weights["clipiqa"] == 0.0


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_rescale_zeroes_other_terms(preset: str) -> None:
    """composition, taste, and tilt_penalty must all be zeroed."""
    weights = _rescaled(preset)
    assert weights["composition"] == 0.0
    assert weights["taste"] == 0.0
    assert weights["tilt_penalty"] == 0.0


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_rescale_preserves_proportions(preset: str) -> None:
    """Proportional ratio of topiq/laion_aesthetic is unchanged after rescaling (§5C)."""
    original = GENRE_WEIGHTS[preset]
    weights = _rescaled(preset)
    original_ratio = original["topiq"] / original["laion_aesthetic"]
    rescaled_ratio = weights["topiq"] / weights["laion_aesthetic"]
    assert rescaled_ratio == pytest.approx(original_ratio)


def test_rescale_unknown_preset_raises() -> None:
    """_rescale_preset_weights raises ValueError for unknown preset names."""
    with pytest.raises(ValueError):
        _rescale_preset_weights("nonexistent")


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_build_iqa_fields_in_range(preset: str) -> None:
    """All required IqaScores fields are populated, sentinel set, values in [0,1]."""
    fast_in = _make_fast_input()
    iqa = build_iqa_from_musiq(fast_in)
    required_values: list[float] = [
        iqa.topiq,
        iqa.laion_aesthetic,
        iqa.clipiqa,
        iqa.exposure,
    ]
    assert iqa.photo_path == fast_in.musiq_scores.photo_path
    assert iqa.clipiqa == FAST_CLIPIQA_SENTINEL
    assert all(0.0 <= v <= 1.0 for v in required_values)
