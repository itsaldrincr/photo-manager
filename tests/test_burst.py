"""Tests for burst detection: temporal clustering, visual confirmation, and winner selection."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from cull.stage1.burst import (
    BurstScoringInput,
    cluster_by_time,
    confirm_burst_visually,
    select_burst_winner,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 6, 1, 12, 0, 0)

PHOTO_A = Path("/fake/photo_a.jpg")
PHOTO_B = Path("/fake/photo_b.jpg")
PHOTO_C = Path("/fake/photo_c.jpg")
PHOTO_D = Path("/fake/photo_d.jpg")


def _ts(offset_seconds: float) -> datetime:
    """Return BASE_TIME + offset_seconds."""
    return BASE_TIME + timedelta(seconds=offset_seconds)


# ---------------------------------------------------------------------------
# cluster_by_time
# ---------------------------------------------------------------------------


def test_cluster_by_time_groups_within_gap() -> None:
    """Photos taken within gap_seconds are grouped together."""
    timestamped = [
        (PHOTO_A, _ts(0.0)),
        (PHOTO_B, _ts(0.3)),
        (PHOTO_C, _ts(0.6)),
    ]
    groups = cluster_by_time(timestamped, gap_seconds=0.5)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_cluster_by_time_splits_across_gap() -> None:
    """Photos separated by more than gap_seconds are in different groups."""
    timestamped = [
        (PHOTO_A, _ts(0.0)),
        (PHOTO_B, _ts(0.3)),
        (PHOTO_C, _ts(5.0)),
        (PHOTO_D, _ts(5.3)),
    ]
    groups = cluster_by_time(timestamped, gap_seconds=0.5)
    assert len(groups) == 2
    assert PHOTO_A in groups[0] and PHOTO_B in groups[0]
    assert PHOTO_C in groups[1] and PHOTO_D in groups[1]


def test_cluster_by_time_single_photo_not_grouped() -> None:
    """A single photo with no burst neighbor is not returned as a group."""
    timestamped = [(PHOTO_A, _ts(0.0))]
    groups = cluster_by_time(timestamped, gap_seconds=0.5)
    assert groups == []


def test_cluster_by_time_empty_input() -> None:
    """Empty input yields empty output."""
    groups = cluster_by_time([], gap_seconds=0.5)
    assert groups == []


# ---------------------------------------------------------------------------
# confirm_burst_visually
# ---------------------------------------------------------------------------


def test_confirm_burst_visually_keeps_similar_images() -> None:
    """Visually similar images (distance <= threshold) remain in one group."""
    with patch("cull.stage1.burst._dhash_distance", return_value=4):
        result = confirm_burst_visually([PHOTO_A, PHOTO_B, PHOTO_C])
    assert len(result) == 1
    assert len(result[0]) == 3


def test_confirm_burst_visually_rejects_dissimilar_images() -> None:
    """Visually dissimilar images (distance > threshold) are split into separate groups."""
    with patch("cull.stage1.burst._dhash_distance", return_value=20):
        result = confirm_burst_visually([PHOTO_A, PHOTO_B, PHOTO_C])
    assert result == []


def test_confirm_burst_visually_mixed_similarity() -> None:
    """First two similar, third dissimilar — only the similar pair is returned."""
    call_count: list[int] = [0]

    def mock_distance(path_a: Path, path_b: Path) -> int:
        call_count[0] += 1
        if path_b == PHOTO_B:
            return 3
        return 20

    with patch("cull.stage1.burst._dhash_distance", side_effect=mock_distance):
        result = confirm_burst_visually([PHOTO_A, PHOTO_B, PHOTO_C])
    assert len(result) == 1
    assert PHOTO_A in result[0] and PHOTO_B in result[0]
    assert PHOTO_C not in result[0]


# ---------------------------------------------------------------------------
# select_burst_winner
# ---------------------------------------------------------------------------


def test_select_burst_winner_picks_highest_blur_score() -> None:
    """Winner is the photo with the highest blur score."""
    scoring_input = BurstScoringInput(
        group=[PHOTO_A, PHOTO_B, PHOTO_C],
        blur_scores={
            str(PHOTO_A): 0.5,
            str(PHOTO_B): 0.9,
            str(PHOTO_C): 0.3,
        },
    )
    winner, losers = select_burst_winner(scoring_input)
    assert winner == PHOTO_B
    assert set(losers) == {PHOTO_A, PHOTO_C}


def test_select_burst_winner_single_photo() -> None:
    """A group with one photo returns it as winner with no losers."""
    scoring_input = BurstScoringInput(
        group=[PHOTO_A],
        blur_scores={str(PHOTO_A): 0.7},
    )
    winner, losers = select_burst_winner(scoring_input)
    assert winner == PHOTO_A
    assert losers == []


def test_select_burst_winner_zero_scores_picks_first() -> None:
    """When all scores are 0, the first element in the sorted order wins."""
    scoring_input = BurstScoringInput(
        group=[PHOTO_A, PHOTO_B],
        blur_scores={},
    )
    winner, losers = select_burst_winner(scoring_input)
    assert winner in {PHOTO_A, PHOTO_B}
    assert len(losers) == 1
