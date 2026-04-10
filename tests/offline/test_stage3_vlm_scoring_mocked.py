"""Mocked tests for stage3/vlm_scoring.py — stub session only, no real VLM or network."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict

from cull.stage3.prompt import PromptContext
from cull.stage3.vlm_scoring import VlmRequest, VlmScoreCallInput, score_photo

_VALID_JSON = (
    '{"sharpness": 0.9, "exposure": 0.85, "composition": 0.8, '
    '"expression": 0.75, "keeper": true, "confidence": 0.88, "flags": []}'
)
_GARBAGE_TEXT = "not json at all"


class FakeVlmSession(BaseModel):
    """Stub VlmSession that returns canned text from generate()."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _responses: list[str] = []
    _call_count: int = 0

    def configure(self, responses: list[str]) -> None:
        """Set the canned response queue."""
        object.__setattr__(self, "_responses", list(responses))
        object.__setattr__(self, "_call_count", 0)

    def generate(self, call_in: Any) -> str:  # noqa: ANN401
        """Return the next canned response."""
        idx = self._call_count
        object.__setattr__(self, "_call_count", idx + 1)
        return self._responses[min(idx, len(self._responses) - 1)]

    @property
    def call_count(self) -> int:
        """Return the number of generate() calls made."""
        return self._call_count


def _make_call_in(
    session: FakeVlmSession,
    image_path: Path,
) -> VlmScoreCallInput:
    """Build a VlmScoreCallInput with a stub session."""
    request = VlmRequest(image_path=image_path, context=PromptContext(), model="test-model")
    return VlmScoreCallInput(request=request, session=session)  # type: ignore[arg-type]


def test_score_photo_parses_valid_response(tmp_path: Path) -> None:
    """score_photo returns a valid Stage3Result when VLM returns parseable JSON."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-image")

    session = FakeVlmSession()
    session.configure([_VALID_JSON])

    call_in = _make_call_in(session, img)

    from unittest.mock import patch

    with patch("cull.stage3.vlm_scoring.load_image_b64", return_value="b64data"):
        result = score_photo(call_in)

    assert result.is_parse_error is False
    assert result.photo_path == img
    assert result.model_used == "test-model"
    assert result.is_keeper is True


def test_score_photo_retries_on_parse_error(tmp_path: Path) -> None:
    """score_photo retries once after a parse error and succeeds on second attempt."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-image")

    session = FakeVlmSession()
    session.configure([_GARBAGE_TEXT, _VALID_JSON])

    call_in = _make_call_in(session, img)

    from unittest.mock import patch

    with (
        patch("cull.stage3.vlm_scoring.load_image_b64", return_value="b64data"),
        patch("cull.stage3.vlm_scoring.time.sleep"),
    ):
        result = score_photo(call_in)

    assert result.is_parse_error is False
    assert session.call_count == 2


def test_score_photo_exhausts_retries(tmp_path: Path) -> None:
    """score_photo returns is_parse_error=True after all retries produce garbage."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-image")

    session = FakeVlmSession()
    session.configure([_GARBAGE_TEXT])

    call_in = _make_call_in(session, img)

    from unittest.mock import patch

    with (
        patch("cull.stage3.vlm_scoring.load_image_b64", return_value="b64data"),
        patch("cull.stage3.vlm_scoring.time.sleep"),
    ):
        result = score_photo(call_in)

    assert result.is_parse_error is True


def test_score_photo_missing_file() -> None:
    """score_photo returns is_parse_error=True immediately when image_path is missing."""
    missing = Path("/tmp/does_not_exist_abc123.jpg")

    session = FakeVlmSession()
    session.configure([_VALID_JSON])

    call_in = _make_call_in(session, missing)
    result = score_photo(call_in)

    assert result.is_parse_error is True
    assert session.call_count == 0
