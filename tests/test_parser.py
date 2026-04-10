"""Tests for cull.stage3.parser — VLM response parsing and validation."""

from __future__ import annotations

import json

import pytest

from cull.stage3.parser import parse_vlm_response

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

VALID_PAYLOAD: dict = {
    "sharpness": 0.85,
    "exposure": 0.90,
    "composition": 0.75,
    "expression": 1.0,
    "keeper": True,
    "confidence": 0.82,
    "flags": [],
}


def _make_json_text(payload: dict) -> str:
    """Return a clean JSON string for the given payload."""
    return json.dumps(payload)


def _make_prose_json(payload: dict) -> str:
    """Embed JSON inside natural-language prose as a VLM might output."""
    return f"Here is my assessment:\n{json.dumps(payload)}\nThank you."


# ---------------------------------------------------------------------------
# Valid JSON — happy path
# ---------------------------------------------------------------------------


def test_valid_json_returns_result() -> None:
    """parse_vlm_response extracts all fields from a clean valid JSON response."""
    result = parse_vlm_response(_make_json_text(VALID_PAYLOAD))

    assert result.is_parse_error is False
    assert result.is_keeper is True
    assert abs(result.confidence - 0.82) < 1e-6
    assert abs(result.sharpness - 0.85) < 1e-6
    assert abs(result.exposure - 0.90) < 1e-6
    assert abs(result.composition - 0.75) < 1e-6


def test_valid_json_keeper_false() -> None:
    """parse_vlm_response handles keeper=false correctly."""
    payload = {**VALID_PAYLOAD, "keeper": False, "confidence": 0.91}
    result = parse_vlm_response(_make_json_text(payload))

    assert result.is_parse_error is False
    assert result.is_keeper is False


def test_valid_json_with_flags() -> None:
    """parse_vlm_response preserves flags list from valid response."""
    payload = {**VALID_PAYLOAD, "flags": ["motion_blur", "defocus"]}
    result = parse_vlm_response(_make_json_text(payload))

    assert result.is_parse_error is False
    assert "motion_blur" in result.flags
    assert "defocus" in result.flags


# ---------------------------------------------------------------------------
# JSON embedded in prose
# ---------------------------------------------------------------------------


def test_json_embedded_in_prose() -> None:
    """parse_vlm_response extracts JSON even when surrounded by prose text."""
    result = parse_vlm_response(_make_prose_json(VALID_PAYLOAD))

    assert result.is_parse_error is False
    assert result.is_keeper is True


# ---------------------------------------------------------------------------
# Malformed JSON fallback
# ---------------------------------------------------------------------------


def test_malformed_json_returns_parse_error() -> None:
    """parse_vlm_response returns parse_error sentinel on malformed JSON."""
    result = parse_vlm_response('{"keeper": true, "confidence": 0.8, BROKEN')

    assert result.is_parse_error is True
    assert result.confidence == 0.0


def test_completely_non_json_returns_parse_error() -> None:
    """parse_vlm_response returns parse_error sentinel when no JSON found."""
    result = parse_vlm_response("This image looks great!")

    assert result.is_parse_error is True
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Missing required keys fallback
# ---------------------------------------------------------------------------


def test_missing_keeper_key_returns_parse_error() -> None:
    """parse_vlm_response returns parse_error when 'keeper' key is absent."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "keeper"}
    result = parse_vlm_response(_make_json_text(payload))

    assert result.is_parse_error is True


def test_missing_confidence_key_returns_parse_error() -> None:
    """parse_vlm_response treats missing confidence as a parse error sentinel."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "confidence"}
    result = parse_vlm_response(_make_json_text(payload))

    assert result.is_parse_error is True
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Out-of-range values fallback
# ---------------------------------------------------------------------------


def test_out_of_range_confidence_returns_parse_error() -> None:
    """parse_vlm_response returns parse_error when confidence > 1.0."""
    payload = {**VALID_PAYLOAD, "confidence": 1.5}
    result = parse_vlm_response(_make_json_text(payload))

    assert result.is_parse_error is True


def test_out_of_range_sharpness_returns_parse_error() -> None:
    """parse_vlm_response returns parse_error when sharpness < 0.0."""
    payload = {**VALID_PAYLOAD, "sharpness": -0.1}
    result = parse_vlm_response(_make_json_text(payload))

    assert result.is_parse_error is True


def test_out_of_range_exposure_returns_parse_error() -> None:
    """parse_vlm_response returns parse_error when exposure > 1.0."""
    payload = {**VALID_PAYLOAD, "exposure": 2.0}
    result = parse_vlm_response(_make_json_text(payload))

    assert result.is_parse_error is True


def test_keeper_not_boolean_returns_parse_error() -> None:
    """parse_vlm_response returns parse_error when keeper is not a bool."""
    payload = {**VALID_PAYLOAD, "keeper": "yes"}
    result = parse_vlm_response(_make_json_text(payload))

    assert result.is_parse_error is True
