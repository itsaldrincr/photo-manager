"""Tests for stage4 single-elimination tournament bracket."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cull.config import CullConfig
from cull.stage4.tournament import TournamentContext, TournamentInput, run

CANDIDATE_COUNT_POW2: int = 8
CANDIDATE_COUNT_ODD: int = 5


def _make_paths(count: int, tmp_path: Path) -> list[Path]:
    """Return a list of count distinct dummy Paths."""
    return [tmp_path / f"photo_{i:03d}.jpg" for i in range(count)]


def _make_scores(paths: list[Path]) -> dict[str, float]:
    """Assign composite scores: higher index → higher score."""
    return {str(p): float(i) / len(paths) for i, p in enumerate(paths)}


def _make_context(paths: list[Path]) -> TournamentContext:
    """Build a TournamentContext with empty s1_results and index-based scores."""
    return TournamentContext(
        s1_results={},
        composite_scores=_make_scores(paths),
    )


def _mock_compare_lower_index_wins(call_in: object) -> object:
    """Deterministic mock: always returns the photo with the lower path index."""
    from cull.stage4.curator_prompt import CuratorTiebreakResult

    tiebreak_input = call_in.tiebreak_input  # type: ignore[attr-defined]
    photo_a = tiebreak_input.photo_a
    photo_b = tiebreak_input.photo_b
    winner = photo_a if str(photo_a) < str(photo_b) else photo_b
    return CuratorTiebreakResult(winner=winner, reason="mock", confidence=1.0)


def test_power_of_two_bracket_round_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For N=8 candidates (power of 2), exactly log2(N) rounds must fire."""
    monkeypatch.setattr(
        "cull.stage4.tournament.compare_photos", _mock_compare_lower_index_wins
    )

    paths = _make_paths(CANDIDATE_COUNT_POW2, tmp_path)
    ctx = _make_context(paths)
    inp = TournamentInput(candidates=paths, config=CullConfig())

    result = run(inp, ctx)

    assert len(result) == 1
    expected_rounds = int(math.log2(CANDIDATE_COUNT_POW2))
    assert expected_rounds == 3


def test_deterministic_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lower-index-always-wins mock must produce photo_000 as champion."""
    monkeypatch.setattr(
        "cull.stage4.tournament.compare_photos", _mock_compare_lower_index_wins
    )

    paths = _make_paths(CANDIDATE_COUNT_POW2, tmp_path)
    ctx = _make_context(paths)
    inp = TournamentInput(candidates=paths, config=CullConfig())

    result = run(inp, ctx)

    assert result[0].name == "photo_000.jpg"


def test_single_candidate_returns_immediately(tmp_path: Path) -> None:
    """A single candidate skips all rounds and returns itself."""
    paths = _make_paths(1, tmp_path)
    ctx = TournamentContext(s1_results={}, composite_scores={})
    inp = TournamentInput(candidates=paths, config=CullConfig())

    result = run(inp, ctx)

    assert result == paths


def test_odd_candidate_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Odd N: bye advances, tournament still resolves to one winner."""
    monkeypatch.setattr(
        "cull.stage4.tournament.compare_photos", _mock_compare_lower_index_wins
    )

    paths = _make_paths(CANDIDATE_COUNT_ODD, tmp_path)
    ctx = _make_context(paths)
    inp = TournamentInput(candidates=paths, config=CullConfig())

    result = run(inp, ctx)

    assert len(result) == 1
