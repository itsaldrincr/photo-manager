"""Mocked tests for stage4/vlm_tiebreak.py — offline, stub-based, no real weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from cull.stage4.vlm_tiebreak import (
    CuratorTiebreakCallInput,
    CuratorTiebreakInput,
    compare_photos,
)
from cull.stage3.prompt import PromptContext
from cull.vlm_session import VlmGenerateInput

PHOTO_A_NAME: str = "photo_a.jpg"
PHOTO_B_NAME: str = "photo_b.jpg"
EMPTY_BYTES: bytes = b""


class FakeVlmSession(BaseModel):
    """Stub VlmSession that returns canned JSON responses."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    responses: list[str]
    recorded_calls: list[VlmGenerateInput] = []

    def generate(self, call_in: VlmGenerateInput) -> str:
        """Return next canned response and record the call."""
        self.recorded_calls.append(call_in)
        if not self.responses:
            raise ValueError("FakeVlmSession exhausted all canned responses")
        return self.responses.pop(0)


class _PhotoPair(BaseModel):
    """Bundle of two real photo paths created on disk for tests."""

    photo_a: Path
    photo_b: Path


@pytest.fixture
def photo_pair(tmp_path: Path) -> _PhotoPair:
    """Create two empty real files in tmp_path so existence checks pass."""
    photo_a = tmp_path / PHOTO_A_NAME
    photo_b = tmp_path / PHOTO_B_NAME
    photo_a.write_bytes(EMPTY_BYTES)
    photo_b.write_bytes(EMPTY_BYTES)
    return _PhotoPair(photo_a=photo_a, photo_b=photo_b)


def _make_context() -> PromptContext:
    """Return a minimal PromptContext with all flags false."""
    return PromptContext(
        motion_blur_detected=False,
        dominant_emotion=None,
        eyes_closed=False,
        has_highlight_clip=False,
        has_shadow_clip=False,
        has_color_cast=False,
    )


def _make_tiebreak_input(pair: _PhotoPair) -> CuratorTiebreakInput:
    """Return a CuratorTiebreakInput with both photos set."""
    return CuratorTiebreakInput(
        photo_a=pair.photo_a,
        photo_b=pair.photo_b,
        context=_make_context(),
        model="qwen3-vl-4b",
    )


def _make_call_input(
    session: FakeVlmSession, pair: _PhotoPair
) -> CuratorTiebreakCallInput:
    """Bundle a FakeVlmSession with the default tiebreak input."""
    return CuratorTiebreakCallInput(
        tiebreak_input=_make_tiebreak_input(pair),
        session=session,
    )


def _canned_winner(winner: str, confidence: float = 0.85) -> str:
    """Return a canned JSON response string for the given winner."""
    return json.dumps({"winner": winner, "reason": f"Photo {winner} is better.", "confidence": confidence})


def test_compare_photos_winner_a(photo_pair: _PhotoPair) -> None:
    """compare_photos returns photo_a as winner when VLM says 'A'."""
    session = FakeVlmSession(responses=[_canned_winner("A")])
    result = compare_photos(_make_call_input(session, photo_pair))
    assert result.winner == photo_pair.photo_a
    assert result.confidence == pytest.approx(0.85)


def test_compare_photos_winner_b(photo_pair: _PhotoPair) -> None:
    """compare_photos returns photo_b as winner when VLM says 'B'."""
    session = FakeVlmSession(responses=[_canned_winner("B", 0.72)])
    result = compare_photos(_make_call_input(session, photo_pair))
    assert result.winner == photo_pair.photo_b
    assert result.confidence == pytest.approx(0.72)


def test_compare_photos_parse_error_retries(photo_pair: _PhotoPair) -> None:
    """compare_photos retries when VLM returns unparseable output, succeeds on retry."""
    session = FakeVlmSession(responses=["not json at all", _canned_winner("A")])
    result = compare_photos(_make_call_input(session, photo_pair))
    assert result.winner == photo_pair.photo_a
    assert len(session.recorded_calls) == 2


def test_compare_photos_exhausts_retries_falls_back_to_a(
    photo_pair: _PhotoPair,
) -> None:
    """compare_photos returns Photo A with confidence 0.0 after MAX_PARSE_RETRIES failed parses."""
    bad = "no json here"
    session = FakeVlmSession(responses=[bad, bad, bad])
    result = compare_photos(_make_call_input(session, photo_pair))
    assert result.winner == photo_pair.photo_a
    assert result.confidence == 0.0
    assert "tiebreak parse failed" in result.reason


def test_compare_photos_passes_two_images_to_session(
    photo_pair: _PhotoPair,
) -> None:
    """compare_photos passes images=[photo_a, photo_b] to the VLM session."""
    session = FakeVlmSession(responses=[_canned_winner("B")])
    compare_photos(_make_call_input(session, photo_pair))
    assert len(session.recorded_calls) == 1
    recorded = session.recorded_calls[0]
    assert recorded.images == [photo_pair.photo_a, photo_pair.photo_b]


def test_compare_photos_missing_photo_raises_file_not_found(tmp_path: Path) -> None:
    """compare_photos raises FileNotFoundError when either photo is missing."""
    missing_a = tmp_path / "nope_a.jpg"
    missing_b = tmp_path / "nope_b.jpg"
    pair = _PhotoPair(photo_a=missing_a, photo_b=missing_b)
    session = FakeVlmSession(responses=[_canned_winner("A")])
    with pytest.raises(FileNotFoundError, match="tiebreak photo missing"):
        compare_photos(_make_call_input(session, pair))
