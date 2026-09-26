"""Event preset prompt and its response parser: one 1-5 rating per photo."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cull.config import EVENT_RATING_MAX, EVENT_RATING_MIN
from cull.models import Stage3Result
from cull.stage3.parser import _clean_json_text, _extract_json_text

logger = logging.getLogger(__name__)

# Rejection criteria and reply format are verbatim from the prompt measured on
# the 488-photo singles-mixer shoot (keeper AUC 0.72 from `rating` alone with
# qwen3.8-27b). Only the opening names a generic event and client.
EVENT_PROMPT: str = """\
You are culling photos from an event for the client.
They want flattering, lively candids: people engaged in conversation, genuine
smiles and laughter, pairs and small groups connecting, a few wide room shots.
Guests will see these, so an unflattering frame of a person is a reject.

Reject for: missed focus or blur on the main face, closed or half-closed eyes
on a main subject (a squint from genuine laughter is fine), mid-chew or
mid-word grimace, the main subject cut off badly at the frame edge, backs of
heads dominating the frame, a blurry foreground object blocking the subject,
no clear subject.

Reply with ONLY one JSON object, no other text:
{"rating": <1-5, 5 = hero shot>, "keep": <true|false>, "flags": [<zero or more of
"eyes_closed","blur","awkward_expression","mid_chew","back_of_head","face_cut_off",
"obstructed_fg","cluttered","no_clear_subject","great_expression","connection">]}"""

EVENT_FLAGS: frozenset[str] = frozenset({
    "eyes_closed", "blur", "awkward_expression", "mid_chew", "back_of_head",
    "face_cut_off", "obstructed_fg", "cluttered", "no_clear_subject",
    "great_expression", "connection",
})

_UNKNOWN_PATH = Path("unknown")


def parse_rating(value: object) -> int | None:
    """Return value as an int rating in 1-5, or None when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not number.is_integer():
        return None
    rating = int(number)
    return rating if EVENT_RATING_MIN <= rating <= EVENT_RATING_MAX else None


def _parse_flags(value: object) -> list[str]:
    """Return the known flags in reply order; ignore anything else."""
    if not isinstance(value, list):
        return []
    return [flag for flag in value if isinstance(flag, str) and flag in EVENT_FLAGS]


def _load_reply(text: str) -> dict[str, object] | None:
    """Return the first JSON object in text, or None."""
    json_text = _extract_json_text(text)
    if json_text is None:
        return None
    try:
        data = json.loads(_clean_json_text(json_text))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_event_response(text: str) -> Stage3Result:
    """Return a Stage3Result carrying rating and flags, or a parse-error result."""
    data = _load_reply(text)
    rating = parse_rating(data.get("rating")) if data is not None else None
    if data is None or rating is None:
        logger.warning("parse_event_response: no 1-5 rating in reply: %.120r", text)
        return Stage3Result(photo_path=_UNKNOWN_PATH, is_parse_error=True)
    return Stage3Result(
        photo_path=_UNKNOWN_PATH, rating=rating, flags=_parse_flags(data.get("flags")),
    )
