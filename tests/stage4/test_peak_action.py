"""Tests for cull.stage4.peak_action — cv2 DIS flow is never invoked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cull.config import CullConfig
from cull.models import PeakMomentScore
from cull.stage4.peak_action import PeakActionInput, pick_winner

BURST_SIZE: int = 5
FRAME_H: int = 64
FRAME_W: int = 64
PEAK_FRAME_IDX: int = 2
LOW_FLOW: float = 1.0
HIGH_FLOW: float = 8.0


def _make_burst_paths(tmp_path: Path) -> list[Path]:
    """Return 5 synthetic path objects (files need not exist — load is mocked)."""
    return [tmp_path / f"frame_{i:02d}.jpg" for i in range(BURST_SIZE)]


def _make_gray_frame(value: int) -> np.ndarray:
    """Return a FRAME_H x FRAME_W uint8 array filled with value."""
    return np.full((FRAME_H, FRAME_W), value, dtype=np.uint8)


def _precomputed_magnitudes() -> list[float]:
    """Return flow magnitudes with engineered peak at PEAK_FRAME_IDX."""
    mags = [LOW_FLOW] * (BURST_SIZE - 1)
    mags[PEAK_FRAME_IDX] = HIGH_FLOW
    return mags


@pytest.fixture()
def burst_ctx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_scorers: None,
) -> dict:
    """Patch load + flow helpers; return paths and expected winner."""
    paths = _make_burst_paths(tmp_path)
    grays = [_make_gray_frame(i * 10) for i in range(BURST_SIZE)]
    mags = _precomputed_magnitudes()
    call_count = [0]

    def _fake_load(path: Path) -> np.ndarray:
        idx = paths.index(path)
        return grays[idx]

    def _fake_flow(prev: np.ndarray, nxt: np.ndarray) -> float:
        i = call_count[0]
        call_count[0] += 1
        return mags[i]

    monkeypatch.setattr("cull.stage4.peak_action._load_gray", _fake_load)
    monkeypatch.setattr("cull.stage4.peak_action._compute_flow_pair", _fake_flow)
    return {"paths": paths, "expected_winner": paths[PEAK_FRAME_IDX]}


def test_pick_winner_returns_peak_frame(burst_ctx: dict) -> None:
    """pick_winner must return the frame at the engineered motion peak."""
    inp = PeakActionInput(burst_members=burst_ctx["paths"], config=CullConfig())
    winner_path, score = pick_winner(inp)
    assert winner_path == burst_ctx["expected_winner"]
    assert isinstance(score, PeakMomentScore)
    assert score.peak_type == "action"
    assert score.motion_peak_score == HIGH_FLOW


def test_pick_winner_single_member(tmp_path: Path) -> None:
    """pick_winner with one burst member must return that member without cv2."""
    paths = [tmp_path / "solo.jpg"]
    inp = PeakActionInput(burst_members=paths, config=CullConfig())
    winner_path, score = pick_winner(inp)
    assert winner_path == paths[0]
    assert score.motion_peak_score == 0.0
