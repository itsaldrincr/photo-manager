"""Tests for cull.stage3.explain — VLM photo explanation module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cull.models import BlurScores, BurstInfo, ExplainRequest, ExposureScores, Stage1Result
from cull.stage3.explain import (
    EXPLAIN_MAX_TOKENS,
    _build_hint_block,
    _parse_explain_response,
    explain_photo,
)
from cull.stage3.vlm_explain import ExplainCallInput
from cull.vlm_registry import VLMEntry
from cull.vlm_session import VlmGenerateInput, VlmSession

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_DUMMY_PATH: Path = Path(tempfile.gettempdir()) / "test_photo.jpg"
_DUMMY_MODEL: str = "test-model"
_DUMMY_DIR: Path = Path(tempfile.gettempdir())

VALID_EXPLAIN_PAYLOAD: dict = {
    "explanation": "Sharp focus and pleasing composition.",
    "weaknesses": [],
    "strengths": ["sharp focus", "good exposure"],
    "confidence": 0.88,
}


class FakeVlmSession(VlmSession):
    """VlmSession stub that records generate calls and returns scripted responses."""

    _responses: list[str] = []
    _call_count: int = 0
    _captured_prompts: list[str] = []

    def __init__(self, responses: list[str]) -> None:
        entry = VLMEntry(alias=_DUMMY_MODEL, directory=_DUMMY_DIR, display_name="Test")
        super().__init__(entry=entry)
        object.__setattr__(self, "_responses", list(responses))
        object.__setattr__(self, "_call_count", 0)
        object.__setattr__(self, "_captured_prompts", [])

    def generate(self, call_in: VlmGenerateInput) -> str:
        """Return next scripted response; record prompt and increment count."""
        count = object.__getattribute__(self, "_call_count")
        prompts = object.__getattribute__(self, "_captured_prompts")
        responses = object.__getattribute__(self, "_responses")
        prompts.append(call_in.prompt)
        object.__setattr__(self, "_call_count", count + 1)
        idx = min(count, len(responses) - 1)
        return responses[idx]

    @property
    def call_count(self) -> int:
        """Return number of times generate was called."""
        return object.__getattribute__(self, "_call_count")

    @property
    def captured_prompts(self) -> list[str]:
        """Return list of prompts passed to generate."""
        return object.__getattribute__(self, "_captured_prompts")


def _make_stage1_result(is_motion_blur: bool = False) -> Stage1Result:
    """Return a Stage1Result with controllable blur flag."""
    blur = BlurScores(
        tenengrad=100.0,
        fft_ratio=0.5,
        blur_tier=1,
        is_motion_blur=is_motion_blur,
    )
    exposure = ExposureScores(
        dr_score=0.8,
        clipping_highlight=0.01,
        clipping_shadow=0.05,
        midtone_pct=0.6,
        color_cast_score=5.0,
    )
    return Stage1Result(
        photo_path=_DUMMY_PATH,
        blur=blur,
        exposure=exposure,
        noise_score=0.0,
    )


def _make_request(stage1: Stage1Result | None = None) -> ExplainRequest:
    """Return a minimal ExplainRequest for testing."""
    return ExplainRequest(
        image_path=_DUMMY_PATH,
        stage1_result=stage1,
        model="test-model",
    )


# ---------------------------------------------------------------------------
# _build_hint_block tests
# ---------------------------------------------------------------------------


def test_build_hint_block_with_motion_blur() -> None:
    """_build_hint_block includes motion blur hint when is_motion_blur=True."""
    stage1 = _make_stage1_result(is_motion_blur=True)
    request = _make_request(stage1=stage1)
    hint = _build_hint_block(request)
    assert "motion blur" in hint.lower()


def test_build_hint_block_empty() -> None:
    """_build_hint_block returns 'No prior flags.' when stage1_result is None."""
    request = _make_request(stage1=None)
    hint = _build_hint_block(request)
    assert hint == "No prior flags."


# ---------------------------------------------------------------------------
# _parse_explain_response tests
# ---------------------------------------------------------------------------


def test_parse_explain_response_valid_json() -> None:
    """_parse_explain_response populates all fields from valid JSON text."""
    text = json.dumps(VALID_EXPLAIN_PAYLOAD)
    result = _parse_explain_response(text, _DUMMY_PATH)
    assert result.is_parse_error is False
    assert result.explanation == "Sharp focus and pleasing composition."
    assert "sharp focus" in result.strengths
    assert result.confidence == pytest.approx(0.88)


def test_parse_explain_response_missing_field() -> None:
    """_parse_explain_response sets is_parse_error=True when 'weaknesses' absent."""
    payload = {k: v for k, v in VALID_EXPLAIN_PAYLOAD.items() if k != "weaknesses"}
    result = _parse_explain_response(json.dumps(payload), _DUMMY_PATH)
    assert result.is_parse_error is True


def test_parse_explain_response_out_of_range_confidence() -> None:
    """_parse_explain_response clamps confidence > 1.0 to 1.0."""
    payload = {**VALID_EXPLAIN_PAYLOAD, "confidence": 1.5}
    result = _parse_explain_response(json.dumps(payload), _DUMMY_PATH)
    assert result.is_parse_error is False
    assert result.confidence == pytest.approx(1.0)


def test_parse_explain_response_non_json() -> None:
    """_parse_explain_response sets is_parse_error=True on non-JSON text."""
    result = _parse_explain_response("This photo looks great!", _DUMMY_PATH)
    assert result.is_parse_error is True


# ---------------------------------------------------------------------------
# explain_photo integration tests (FakeVlmSession stub)
# ---------------------------------------------------------------------------


def test_explain_photo_calls_session_generate(tmp_path: Path) -> None:
    """explain_photo calls session.generate exactly once on a valid response."""
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    fake = FakeVlmSession(responses=[json.dumps(VALID_EXPLAIN_PAYLOAD)])
    call_in = ExplainCallInput(
        request=ExplainRequest(image_path=image, model=_DUMMY_MODEL),
        session=fake,
    )
    result = explain_photo(call_in)
    assert result.is_parse_error is False
    assert result.explanation == VALID_EXPLAIN_PAYLOAD["explanation"]
    assert result.strengths == VALID_EXPLAIN_PAYLOAD["strengths"]
    assert fake.call_count == 1


def test_explain_photo_retries_on_parse_error(tmp_path: Path) -> None:
    """explain_photo retries when first response is garbage; succeeds on second."""
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    fake = FakeVlmSession(responses=["NOT JSON", json.dumps(VALID_EXPLAIN_PAYLOAD)])
    call_in = ExplainCallInput(
        request=ExplainRequest(image_path=image, model=_DUMMY_MODEL),
        session=fake,
    )
    result = explain_photo(call_in)
    assert result.is_parse_error is False
    assert fake.call_count == 2


def test_explain_photo_missing_file(tmp_path: Path) -> None:
    """explain_photo returns is_parse_error=True without calling generate when file absent."""
    image = tmp_path / "nope.jpg"
    fake = FakeVlmSession(responses=[json.dumps(VALID_EXPLAIN_PAYLOAD)])
    call_in = ExplainCallInput(
        request=ExplainRequest(image_path=image, model=_DUMMY_MODEL),
        session=fake,
    )
    result = explain_photo(call_in)
    assert result.is_parse_error is True
    assert fake.call_count == 0


def test_explain_photo_prompt_includes_stage1_scores(tmp_path: Path) -> None:
    """explain_photo prompt contains stage1 motion-blur hint and stage2 composite score."""
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    stage1 = _make_stage1_result(is_motion_blur=True)
    fake = FakeVlmSession(responses=[json.dumps(VALID_EXPLAIN_PAYLOAD)])
    call_in = ExplainCallInput(
        request=ExplainRequest(
            image_path=image,
            model=_DUMMY_MODEL,
            stage1_result=stage1,
            stage2_composite=0.75,
        ),
        session=fake,
    )
    explain_photo(call_in)
    assert fake.call_count == 1
    combined = fake.captured_prompts[0].lower()
    assert "motion blur" in combined
    assert "0.75" in fake.captured_prompts[0]
