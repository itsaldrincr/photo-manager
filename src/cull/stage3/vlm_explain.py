"""Stage 3 VLM explanation — offline in-process mlx_vlm path."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from cull.config import EXPLAIN_MAX_TOKENS, EXPLAIN_TEMPERATURE, VLM_MAX_RETRIES
from cull.models import ExplainRequest, ExplainResult, Stage1Result
from cull.vlm_session import VlmGenerateInput, VlmSession

logger = logging.getLogger(__name__)

CONFIDENCE_MIN: float = 0.0
CONFIDENCE_MAX: float = 1.0
RETRY_BASE_DELAY: float = 1.0
RETRY_BACKOFF_FACTOR: float = 2.0

EXPLAIN_PROMPT_TEMPLATE: str = """\
You are a photo editor analyzing a photograph.

Task: Explain what makes this photo weak or strong.

Prior analysis signals:
{hint_block}

Identify specific technical issues (focus, exposure, noise), compositional issues \
(framing, subject placement, clutter), and write a 2-3 sentence summary in "explanation".

RULES:
- Respond with ONLY a JSON object. No markdown, no explanation, no text before or after.
- "weaknesses" and "strengths" are arrays of short strings (max 5 each).
- "confidence" is a float between 0.0 and 1.0.

EXAMPLE — a sharp, well-lit portrait with strong composition:
{{"explanation": "Strong eye contact and shallow depth of field create visual impact. \
Exposure is balanced with clean highlights. Subject is well-centred with pleasing bokeh.",
"strengths": ["sharp focus on eyes", "good exposure", "clean bokeh"],
"weaknesses": [],
"confidence": 0.91}}

EXAMPLE — a blurry, poorly composed snapshot:
{{"explanation": "Motion blur makes the subject unrecognisably soft. The horizon is tilted \
and the main subject is clipped at the frame edge. Highlight clipping burns the sky.",
"strengths": [],
"weaknesses": ["motion blur", "tilted horizon", "edge-cropped subject", "highlight clipping"],
"confidence": 0.87}}

Now evaluate the photograph. Respond with exactly one JSON object:"""

_JSON_PATTERN: re.Pattern[str] = re.compile(r"\{.*\}", re.DOTALL)
_REQUIRED_KEYS: frozenset[str] = frozenset({"explanation", "weaknesses", "strengths", "confidence"})


class ExplainCallInput(BaseModel):
    """Bundle of request + session for a single explain call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    request: ExplainRequest
    session: VlmSession


def explain_photo(call_in: ExplainCallInput) -> ExplainResult:
    """Load image, call VLM for explanation, return structured ExplainResult."""
    if not call_in.request.image_path.exists():
        logger.warning("Image not found, skipping: %s", call_in.request.image_path)
        return ExplainResult(photo_path=call_in.request.image_path, is_parse_error=True)
    return _explain_retry_loop(call_in)


def _collect_stage1_hints(stage1: Stage1Result) -> list[str]:
    """Return hint strings derived from Stage1Result blur/exposure signals."""
    hints: list[str] = []
    if stage1.blur.is_motion_blur:
        hints.append("Motion blur detected.")
    if stage1.blur.is_bokeh:
        hints.append("Bokeh (intentional background blur) detected.")
    if stage1.exposure.has_highlight_clip:
        hints.append("Highlight clipping detected.")
    if stage1.exposure.has_shadow_clip:
        hints.append("Shadow clipping detected.")
    if stage1.exposure.has_color_cast:
        hints.append("Color cast detected.")
    if stage1.noise_score > 0.0:
        hints.append(f"Noise score: {stage1.noise_score:.2f}.")
    return hints


def _build_hint_block(request: ExplainRequest) -> str:
    """Extract Stage 1 signals from request into a readable hint block."""
    if request.stage1_result is None:
        return "No prior flags."
    hints = _collect_stage1_hints(request.stage1_result)
    if request.stage2_composite is not None:
        hints.append(f"Stage 2 composite IQA score: {request.stage2_composite:.2f}.")
    return "\n".join(hints) if hints else "No prior flags."


def _build_explain_prompt(request: ExplainRequest) -> str:
    """Build the explain prompt string with hint block substituted."""
    hint_block = _build_hint_block(request)
    return EXPLAIN_PROMPT_TEMPLATE.format(hint_block=hint_block)


def _extract_explain_data(text: str) -> dict | None:
    """Regex-extract the first JSON object from text; return None on failure."""
    match = _JSON_PATTERN.search(text)
    if match is None:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _validate_explain_data(data: dict) -> bool:
    """Return True if all required keys are present in the parsed dict."""
    return _REQUIRED_KEYS.issubset(data.keys())


def _clamp_confidence(value: float) -> float:
    """Clamp a confidence value to the valid [0.0, 1.0] range."""
    return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, value))


def _build_explain_result(data: dict, image_path: Path) -> ExplainResult:
    """Construct an ExplainResult from validated VLM response data."""
    return ExplainResult(
        photo_path=image_path,
        explanation=str(data.get("explanation", "")),
        weaknesses=list(data.get("weaknesses", [])),
        strengths=list(data.get("strengths", [])),
        confidence=_clamp_confidence(float(data.get("confidence", CONFIDENCE_MIN))),
    )


def _parse_explain_response(text: str, image_path: Path) -> ExplainResult:
    """Regex-extract JSON from VLM text, validate keys, return ExplainResult."""
    logger.debug("_parse_explain_response: raw VLM text for %s:\n%s", image_path, text[:500])
    data = _extract_explain_data(text)
    if data is None:
        logger.warning("_parse_explain_response: no JSON found for %s", image_path)
        snippet = text[:200] if text else "(empty)"
        return ExplainResult(
            photo_path=image_path, is_parse_error=True,
            explanation=f"VLM returned unparseable response: {snippet}"
        )
    if not _validate_explain_data(data):
        logger.warning("_parse_explain_response: missing required keys for %s", image_path)
        return ExplainResult(
            photo_path=image_path, is_parse_error=True,
            explanation=f"Missing required fields. VLM output: {list(data.keys())}"
        )
    logger.debug("_parse_explain_response: parsed data for %s: %s", image_path, data)
    return _build_explain_result(data, image_path)


def _run_explain_attempt(call_in: ExplainCallInput) -> ExplainResult:
    """Execute one VLM explanation attempt and return an ExplainResult."""
    request = call_in.request
    prompt = _build_explain_prompt(request)
    gen_in = VlmGenerateInput(
        prompt=prompt,
        images=[request.image_path],
        max_tokens=EXPLAIN_MAX_TOKENS,
        temperature=EXPLAIN_TEMPERATURE,
    )
    raw_text = call_in.session.generate(gen_in)
    result = _parse_explain_response(raw_text, request.image_path)
    result.model_used = request.model
    return result


def _explain_retry_loop(call_in: ExplainCallInput) -> ExplainResult:
    """Retry explanation on parse errors up to VLM_MAX_RETRIES times."""
    delay = RETRY_BASE_DELAY
    request = call_in.request
    for attempt in range(1, VLM_MAX_RETRIES + 1):
        logger.info("explain attempt %d/%d model=%s", attempt, VLM_MAX_RETRIES, request.model)
        result = _run_explain_attempt(call_in)
        if not result.is_parse_error:
            return result
        if attempt < VLM_MAX_RETRIES:
            logger.warning("parse_error attempt %d, retry in %.1fs", attempt, delay)
            time.sleep(delay)
            delay *= RETRY_BACKOFF_FACTOR
    logger.error("All %d explain attempts failed for %s", VLM_MAX_RETRIES, request.image_path)
    return ExplainResult(
        photo_path=request.image_path,
        model_used=request.model,
        is_parse_error=True,
    )
