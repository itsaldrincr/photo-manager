"""Stage 3 VLM response parser — regex-extract JSON, validate, return Stage3Result."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from cull.models import Stage3Result

logger = logging.getLogger(__name__)

SCORE_MIN: float = 0.0
SCORE_MAX: float = 1.0

# expression is now included so a VLM returning 5.0 is caught before _build_gauge_bar
_REQUIRED_SCORE_KEYS = ("sharpness", "exposure", "composition", "expression", "confidence")

_PARSE_ERROR_SENTINEL_PATH = Path("unknown")


def _extract_json_text(text: str) -> str | None:
    """Return the first balanced {...} substring in text, supporting nested objects.

    Walks the string with a brace-depth counter so `{"a": {"b": 1}}` is matched
    as a single unit. Returns None if no complete object is found.
    """
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _clean_json_text(text: str) -> str:
    """Fix common local model JSON mistakes before parsing."""
    text = text.replace("'", '"')
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return text


def _validate_scores(data: dict[str, object]) -> None:
    """Raise ValueError if any required score key is missing or out of range."""
    for key in _REQUIRED_SCORE_KEYS:
        if key not in data:
            raise ValueError(f"missing required field: {key}")
        value = float(data[key])
        if not (SCORE_MIN <= value <= SCORE_MAX):
            raise ValueError(f"{key}={value} out of range [{SCORE_MIN}, {SCORE_MAX}]")


def _build_result(data: dict[str, object]) -> Stage3Result:
    """Construct a Stage3Result from a validated VLM response dict."""
    return Stage3Result(
        photo_path=_PARSE_ERROR_SENTINEL_PATH,
        sharpness=float(data.get("sharpness", SCORE_MIN)),
        exposure=float(data.get("exposure", SCORE_MIN)),
        composition=float(data.get("composition", SCORE_MIN)),
        expression=float(data.get("expression", SCORE_MIN)),
        is_keeper=bool(data["keeper"]),
        confidence=float(data.get("confidence", SCORE_MIN)),
        flags=list(data.get("flags", [])),
    )


def parse_vlm_response(text: str) -> Stage3Result:
    """Regex-extract JSON from VLM text, validate keys/ranges, return Stage3Result."""
    json_text = _extract_json_text(text)
    if json_text is None:
        logger.warning("parse_vlm_response: no JSON found in response")
        return Stage3Result(photo_path=_PARSE_ERROR_SENTINEL_PATH, is_parse_error=True)

    try:
        cleaned = _clean_json_text(json_text)
        data = json.loads(cleaned)
        if "keeper" not in data or not isinstance(data["keeper"], bool):
            raise ValueError("keeper missing or not a boolean")
        _validate_scores(data)
        return _build_result(data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("parse_vlm_response: validation failed — %s", exc)
        return Stage3Result(photo_path=_PARSE_ERROR_SENTINEL_PATH, is_parse_error=True)
