"""Tests for stage4.peak_portrait — mocked blendshape burst selection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cull.config import CullConfig

BURST_SIZE: int = 5
WINNER_INDEX: int = 2

_BURST_BLENDSHAPES: list[dict[str, float]] = [
    {
        "eyeBlinkLeft": 0.80,
        "eyeBlinkRight": 0.75,
        "mouthSmileLeft": 0.10,
        "mouthSmileRight": 0.10,
        "eyeLookOutLeft": 0.20,
        "eyeLookOutRight": 0.20,
    },
    {
        "eyeBlinkLeft": 0.60,
        "eyeBlinkRight": 0.55,
        "mouthSmileLeft": 0.20,
        "mouthSmileRight": 0.25,
        "eyeLookOutLeft": 0.15,
        "eyeLookOutRight": 0.15,
    },
    {
        "eyeBlinkLeft": 0.05,
        "eyeBlinkRight": 0.05,
        "mouthSmileLeft": 0.80,
        "mouthSmileRight": 0.85,
        "eyeLookOutLeft": 0.05,
        "eyeLookOutRight": 0.05,
    },
    {
        "eyeBlinkLeft": 0.50,
        "eyeBlinkRight": 0.45,
        "mouthSmileLeft": 0.30,
        "mouthSmileRight": 0.30,
        "eyeLookOutLeft": 0.40,
        "eyeLookOutRight": 0.40,
    },
    {
        "eyeBlinkLeft": 0.70,
        "eyeBlinkRight": 0.65,
        "mouthSmileLeft": 0.15,
        "mouthSmileRight": 0.10,
        "eyeLookOutLeft": 0.30,
        "eyeLookOutRight": 0.30,
    },
]


def _make_burst_paths() -> list[Path]:
    """Return a synthetic burst of BURST_SIZE fake paths."""
    return [Path(f"/fake/burst/frame_{i:03d}.jpg") for i in range(BURST_SIZE)]


def _mock_detect_blendshapes(image_path: Path) -> dict[str, float]:
    """Return deterministic blendshapes keyed by path index suffix."""
    idx = int(image_path.stem.split("_")[-1])
    return _BURST_BLENDSHAPES[idx]


@pytest.fixture
def burst_paths() -> list[Path]:
    """Return synthetic 5-frame burst path list."""
    return _make_burst_paths()


def test_winner_has_highest_combined_score(burst_paths: list[Path]) -> None:
    """Frame with max eye-open + smile + gaze score is selected as winner."""
    from cull.stage4.peak_portrait import PeakPortraitInput, pick_winner

    config = CullConfig()
    peak_input = PeakPortraitInput(burst_members=burst_paths, config=config)
    with patch(
        "cull.stage4.peak_portrait._detect_blendshapes",
        side_effect=_mock_detect_blendshapes,
    ):
        winner, score = pick_winner(peak_input)
    assert winner == burst_paths[WINNER_INDEX]
    assert score.eyes_open_score > 0.9
    assert score.smile_score > 0.8
    assert score.peak_type == "portrait"


def test_tie_breaking_is_deterministic(burst_paths: list[Path]) -> None:
    """Identical blendshapes → first-listed path wins consistently."""
    from cull.stage4.peak_portrait import PeakPortraitInput, pick_winner

    tied_blendshapes: dict[str, float] = {
        "eyeBlinkLeft": 0.10,
        "eyeBlinkRight": 0.10,
        "mouthSmileLeft": 0.50,
        "mouthSmileRight": 0.50,
        "eyeLookOutLeft": 0.10,
        "eyeLookOutRight": 0.10,
    }
    config = CullConfig()
    peak_input = PeakPortraitInput(burst_members=burst_paths, config=config)
    with patch(
        "cull.stage4.peak_portrait._detect_blendshapes",
        return_value=tied_blendshapes,
    ):
        winner1, _ = pick_winner(peak_input)
        winner2, _ = pick_winner(peak_input)
    assert winner1 == winner2
    assert winner1 == burst_paths[0]


def test_empty_blendshapes_returns_first(burst_paths: list[Path]) -> None:
    """When no face detected (empty dict), first frame is returned as fallback."""
    from cull.stage4.peak_portrait import PeakPortraitInput, pick_winner

    config = CullConfig()
    peak_input = PeakPortraitInput(burst_members=burst_paths, config=config)
    with patch(
        "cull.stage4.peak_portrait._detect_blendshapes",
        return_value={},
    ):
        winner, score = pick_winner(peak_input)
    assert winner == burst_paths[0]
    assert score.peak_type == "portrait"


def test_pick_winner_raises_on_empty_burst() -> None:
    """pick_winner raises ValueError when burst_members is empty."""
    from cull.stage4.peak_portrait import PeakPortraitInput, pick_winner

    config = CullConfig()
    peak_input = PeakPortraitInput(burst_members=[], config=config)
    with pytest.raises(ValueError, match="burst_members"):
        pick_winner(peak_input)
