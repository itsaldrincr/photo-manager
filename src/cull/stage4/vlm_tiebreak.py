"""Stage 4 VLM tiebreaker — in-process mlx-vlm session-based comparison."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from cull.stage3.prompt import PromptContext
from cull.vlm_session import VlmGenerateInput

logger = logging.getLogger(__name__)

CURATOR_MAX_TOKENS: int = 300
CURATOR_TEMPERATURE: float = 0.0
MAX_PARSE_RETRIES: int = 3

WINNER_VALUES: frozenset[str] = frozenset({"A", "B"})
CONFIDENCE_MIN: float = 0.0
CONFIDENCE_MAX: float = 1.0

# Balanced-brace scanner handles nested objects; the previous
# r"\{[^{}]*\}" pattern only matched flat JSON and rejected any
# response with a sub-object such as {"winner": "A", "meta": {"k": 1}}.

CURATOR_PROMPT_TEMPLATE: str = """\
You are a photo editor selecting the strongest image for a portfolio.
You are given two photographs labelled Photo A and Photo B.

Think step by step about:
1. Subject focus — which photo has sharper, cleaner subject separation?
2. Expression — which captures a stronger, more authentic moment?
3. Composition — which has better framing, balance, and visual flow?
4. Decisive moment — which freezes the peak of the action or emotion?

Prior context hints:
{hint_block}

Output ONLY a JSON object with exactly these keys:
  "winner": "A" or "B"
  "reason": one sentence explaining the choice
  "confidence": float between 0.0 and 1.0

EXAMPLE — Photo A has sharper focus and stronger expression:
{{"winner": "A", "reason": "Photo A captures a crisper subject with a more genuine smile.", "confidence": 0.85}}

EXAMPLE — Photo B has better moment and composition:
{{"winner": "B", "reason": "Photo B freezes the decisive peak with better framing.", "confidence": 0.78}}

Now evaluate the two photographs. Respond with exactly one JSON object:"""


class CuratorTiebreakInput(BaseModel):
    """Input bundle for a VLM tiebreaker comparison call."""

    photo_a: Path
    photo_b: Path
    context: PromptContext
    context_b: PromptContext | None = None
    model: str


class CuratorTiebreakResult(BaseModel):
    """Result of a VLM tiebreaker comparison."""

    winner: Path
    reason: str
    confidence: float


class CuratorTiebreakCallInput(BaseModel):
    """Bundle of tiebreak parameters and a live VLM session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tiebreak_input: CuratorTiebreakInput
    session: Any


def compare_photos(call_in: CuratorTiebreakCallInput) -> CuratorTiebreakResult:
    """Call the VLM session to compare two photos and return CuratorTiebreakResult."""
    photo_a = call_in.tiebreak_input.photo_a
    photo_b = call_in.tiebreak_input.photo_b
    if not photo_a.exists() or not photo_b.exists():
        raise FileNotFoundError(f"tiebreak photo missing: a={photo_a} b={photo_b}")
    prompt = _build_prompt(call_in.tiebreak_input)
    gen_in = VlmGenerateInput(
        prompt=prompt,
        images=[photo_a, photo_b],
        max_tokens=CURATOR_MAX_TOKENS,
        temperature=CURATOR_TEMPERATURE,
    )
    return _retry_parse(gen_in, call_in)


def _build_prompt(tiebreak_input: CuratorTiebreakInput) -> str:
    """Construct the tiebreaker prompt string with hint block."""
    hint_block = _build_hint_block(tiebreak_input.context, tiebreak_input.context_b)
    return CURATOR_PROMPT_TEMPLATE.format(hint_block=hint_block)


def _retry_parse(
    gen_in: VlmGenerateInput, call_in: CuratorTiebreakCallInput
) -> CuratorTiebreakResult:
    """Call session.generate up to MAX_PARSE_RETRIES times; fall back to Photo A on exhaustion."""
    for attempt in range(MAX_PARSE_RETRIES):
        raw = call_in.session.generate(gen_in)
        try:
            return _parse_response(raw, call_in.tiebreak_input)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("Parse attempt %d/%d failed: %s", attempt + 1, MAX_PARSE_RETRIES, exc)
    logger.error("VLM tiebreak parse exhausted after %d retries; falling back to Photo A", MAX_PARSE_RETRIES)
    return CuratorTiebreakResult(
        winner=call_in.tiebreak_input.photo_a,
        reason="tiebreak parse failed; defaulted to Photo A",
        confidence=0.0,
    )


def _build_hint_block(
    context_a: PromptContext, context_b: PromptContext | None
) -> str:
    """Return per-photo labelled hint strings from one or two PromptContexts."""
    hints: list[str] = list(_format_context_hints(context_a, "Photo A"))
    if context_b is not None:
        hints.extend(_format_context_hints(context_b, "Photo B"))
    return "\n".join(hints) if hints else "No prior flags."


def _format_context_hints(context: PromptContext, label: str) -> list[str]:
    """Return per-flag labelled hint strings for one PromptContext."""
    hints: list[str] = []
    if context.motion_blur_detected:
        hints.append(f"Note: {label} had motion blur in prior analysis.")
    if context.dominant_emotion:
        hints.append(f"Note: {label} expression classifier detected '{context.dominant_emotion}'.")
    if context.eyes_closed:
        hints.append(f"Note: {label} had eye closure in prior analysis.")
    if context.has_highlight_clip:
        hints.append(f"Note: {label} had highlight clipping in prior analysis.")
    if context.has_shadow_clip:
        hints.append(f"Note: {label} had shadow clipping in prior analysis.")
    if context.has_color_cast:
        hints.append(f"Note: {label} had color cast in prior analysis.")
    return hints


def _extract_json_text(text: str) -> str:
    """Return the first balanced {...} substring in text, supporting nested objects."""
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
    raise ValueError(f"No JSON object found in VLM response: {text!r}")


def _clamp_confidence(value: float) -> float:
    """Clamp confidence to [CONFIDENCE_MIN, CONFIDENCE_MAX]."""
    return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, value))


def _parse_response(
    text: str, tiebreak_input: CuratorTiebreakInput
) -> CuratorTiebreakResult:
    """Extract winner/reason/confidence from VLM text and return CuratorTiebreakResult."""
    json_text = _extract_json_text(text)
    data = json.loads(json_text)
    winner_label = str(data.get("winner", "")).upper()
    if winner_label not in WINNER_VALUES:
        raise ValueError(f"Invalid winner value: {winner_label!r}; expected 'A' or 'B'")
    winner_path = tiebreak_input.photo_a if winner_label == "A" else tiebreak_input.photo_b
    reason = str(data.get("reason", ""))
    confidence = _clamp_confidence(float(data.get("confidence", CONFIDENCE_MIN)))
    return CuratorTiebreakResult(winner=winner_path, reason=reason, confidence=confidence)
