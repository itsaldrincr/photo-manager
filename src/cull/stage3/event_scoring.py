"""Event preset Stage 3: one VLM rating call per photo on the shared session."""

from __future__ import annotations

import logging

from cull.models import Stage3Result
from cull.stage3.event_prompt import EVENT_PROMPT, parse_event_response
from cull.stage3.event_score import rating_verdict
from cull.stage3.vlm_scoring import VlmScoreCallInput, run_with_retries
from cull.vlm_session import VlmGenerateInput

logger = logging.getLogger(__name__)


def _run_event_attempt(call_in: VlmScoreCallInput) -> Stage3Result:
    """Run one generate() call with the event prompt and parse the rating."""
    image_path = call_in.request.image_path
    raw_text = call_in.session.generate(VlmGenerateInput(prompt=EVENT_PROMPT, images=[image_path]))
    result = parse_event_response(raw_text)
    result.photo_path = image_path
    result.model_used = call_in.request.model
    return result


def rate_photo(call_in: VlmScoreCallInput) -> Stage3Result:
    """Return the photo's event rating; a parse error after retries has no rating."""
    image_path = call_in.request.image_path
    if not image_path.exists():
        logger.warning("Image not found, skipping: %s", image_path)
        return Stage3Result(photo_path=image_path, is_parse_error=True)
    result = run_with_retries(call_in, _run_event_attempt)
    result.is_keeper = rating_verdict(result.rating)
    return result
