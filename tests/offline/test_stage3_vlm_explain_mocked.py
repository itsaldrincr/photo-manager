# stdlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

# third-party
import pytest

# local
from cull.models import ExplainRequest, ExplainResult
from cull.stage3.vlm_explain import ExplainCallInput, explain_photo
from cull.vlm_registry import VLMEntry
from cull.vlm_session import VlmGenerateInput, VlmSession

_DUMMY_DIR: Path = Path(tempfile.gettempdir())
_DUMMY_MODEL: str = "test-model"

_VALID_RESPONSE: dict = {
    "explanation": "Sharp focus and good composition.",
    "weaknesses": [],
    "strengths": ["sharp focus", "good exposure"],
    "confidence": 0.88,
}

_INVALID_RESPONSE: str = "not json at all"


def _make_fake_entry() -> VLMEntry:
    """Return a minimal VLMEntry for test sessions."""
    return VLMEntry(alias=_DUMMY_MODEL, directory=_DUMMY_DIR, display_name="Test")


def _make_session(generate_fn: Callable[[VlmGenerateInput], str]) -> VlmSession:
    """Build a VlmSession stub with a patched generate method."""
    entry = _make_fake_entry()
    session = VlmSession(entry=entry)
    session.generate = generate_fn  # type: ignore[method-assign]
    return session


def _make_request(image_path: Path) -> ExplainRequest:
    """Build a minimal ExplainRequest for the given image path."""
    return ExplainRequest(image_path=image_path, model=_DUMMY_MODEL)


def _make_call_in(image_path: Path, generate_fn: Callable[[VlmGenerateInput], str]) -> ExplainCallInput:
    """Build an ExplainCallInput with the given image path and generate stub."""
    return ExplainCallInput(
        request=_make_request(image_path),
        session=_make_session(generate_fn),
    )


def test_explain_photo_parses_valid_response(tmp_path: Path) -> None:
    """explain_photo returns a populated ExplainResult when VLM returns valid JSON."""
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff")

    def generate(_: VlmGenerateInput) -> str:
        return json.dumps(_VALID_RESPONSE)

    call_in = _make_call_in(image, generate)
    result = explain_photo(call_in)

    assert not result.is_parse_error
    assert result.explanation == _VALID_RESPONSE["explanation"]
    assert result.strengths == _VALID_RESPONSE["strengths"]
    assert result.confidence == pytest.approx(_VALID_RESPONSE["confidence"])
    assert result.model_used == _DUMMY_MODEL


def test_explain_photo_retries_on_parse_error(tmp_path: Path) -> None:
    """explain_photo retries when first response is unparseable, succeeds on second."""
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff")

    call_count = 0

    def generate(_: VlmGenerateInput) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _INVALID_RESPONSE
        return json.dumps(_VALID_RESPONSE)

    call_in = _make_call_in(image, generate)
    result = explain_photo(call_in)

    assert not result.is_parse_error
    assert call_count == 2


def test_explain_photo_exhausts_retries(tmp_path: Path) -> None:
    """explain_photo returns is_parse_error=True after all retries produce bad JSON."""
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff")

    def generate(_: VlmGenerateInput) -> str:
        return _INVALID_RESPONSE

    call_in = _make_call_in(image, generate)
    result = explain_photo(call_in)

    assert result.is_parse_error
    assert result.photo_path == image


def test_explain_photo_missing_file(tmp_path: Path) -> None:
    """explain_photo returns is_parse_error=True immediately when file is absent."""
    image = tmp_path / "nonexistent.jpg"

    call_count = 0

    def generate(_: VlmGenerateInput) -> str:
        nonlocal call_count
        call_count += 1
        return json.dumps(_VALID_RESPONSE)

    call_in = _make_call_in(image, generate)
    result = explain_photo(call_in)

    assert result.is_parse_error
    assert result.photo_path == image
    assert call_count == 0
