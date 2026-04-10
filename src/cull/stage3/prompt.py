"""Stage 3 VLM prompt construction — injects Stage 1/2 signals as context hints."""

from __future__ import annotations

from pydantic import BaseModel


class PromptContext(BaseModel):
    """Stage 1 and Stage 2 signals to inject into the VLM scoring prompt."""

    preset: str | None = None
    stage2_composite: float | None = None
    composition_score: float | None = None
    motion_blur_detected: bool = False
    dominant_emotion: str | None = None
    has_face: bool = False
    eyes_closed: bool = False
    face_occluded: bool = False
    is_bokeh: bool = False
    has_highlight_clip: bool = False
    has_shadow_clip: bool = False
    has_color_cast: bool = False


_PROMPT_TEMPLATE = """\
You are a photo quality assessor. Output a single JSON object. Nothing else.

Preset intent:
{preset_block}

Prior analysis signals:
{hint_block}

REQUIRED FIELDS (all 7 must be present, exactly these keys):
  "sharpness"   — float 0.0-1.0  (edge sharpness of the subject)
  "exposure"    — float 0.0-1.0  (brightness balance, no blown highlights or crushed shadows)
  "composition" — float 0.0-1.0  (framing, balance, visual flow, rule of thirds)
  "expression"  — float 0.0-1.0  (subject expression; 0.5 if no people)
  "keeper"      — boolean        (true or false)
  "confidence"  — float 0.0-1.0  (how confident you are in your assessment)
  "flags"       — array of strings (zero or more from the allowed list)

ALLOWED FLAGS: "motion_blur", "defocus", "overexposed", "underexposed", "eyes_closed", "bad_expression", "no_clear_subject", "cluttered", "color_cast", "noise", "tilted_horizon", "edge_cropped_subject"

OUTPUT RULES:
- Output ONLY the JSON object. No markdown fences, no explanation, no preamble, no text before or after.
- Start with {{ and end with }}. Include every field above.
- If unsure about a field, still output a value. Never omit a field.

EXAMPLE — a sharp, well-exposed portrait with good composition:
{{"sharpness": 0.92, "exposure": 0.85, "composition": 0.78, "expression": 0.90, "keeper": true, "confidence": 0.88, "flags": []}}

EXAMPLE — a blurry, poorly composed shot:
{{"sharpness": 0.25, "exposure": 0.60, "composition": 0.30, "expression": 0.50, "keeper": false, "confidence": 0.91, "flags": ["defocus", "no_clear_subject"]}}

Now evaluate the photograph. Respond with exactly one JSON object containing all 7 required fields:"""


def _collect_hints(context: PromptContext) -> list[str]:
    """Return a list of hint strings derived from the context signals."""
    hints: list[str] = []
    hints.extend(_collect_score_hints(context))
    hints.extend(_collect_portrait_hints(context))
    hints.extend(_collect_exposure_hints(context))
    return hints


def _collect_score_hints(context: PromptContext) -> list[str]:
    """Return generic score and motion hints for the prompt."""
    hints: list[str] = []
    if context.stage2_composite is not None:
        hints.append(f"Note: Stage 2 composite score was {context.stage2_composite:.2f}.")
    if context.composition_score is not None:
        hints.append(f"Note: prior composition analysis scored {context.composition_score:.2f}.")
    if context.motion_blur_detected:
        hints.append("Note: directional frequency analysis suggests possible motion blur.")
    if context.is_bokeh:
        hints.append(
            "Note: shallow depth of field may be intentional; judge subject sharpness without rewarding a weak frame overall."
        )
    return hints


def _collect_portrait_hints(context: PromptContext) -> list[str]:
    """Return portrait and subject-expression hints for the prompt."""
    hints: list[str] = []
    if context.has_face:
        hints.append("Note: a face was detected in the frame.")
    if context.dominant_emotion:
        hints.append(f"Note: expression classifier detected '{context.dominant_emotion}'.")
    if context.eyes_closed:
        hints.append("Note: eye closure was detected — verify if subject's eyes are open.")
    if context.face_occluded:
        hints.append("Note: face occlusion was detected — verify if the subject is obstructed.")
    return hints


def _collect_exposure_hints(context: PromptContext) -> list[str]:
    """Return highlight, shadow, and color hints for the prompt."""
    hints: list[str] = []
    if context.has_highlight_clip:
        hints.append("Note: highlight clipping detected in prior exposure analysis.")
    if context.has_shadow_clip:
        hints.append("Note: shadow clipping detected in prior exposure analysis.")
    if context.has_color_cast:
        hints.append("Note: color cast detected in prior exposure analysis.")
    return hints


def _preset_guidance(context: PromptContext) -> str:
    """Return a concise quality brief for the active preset."""
    preset = (context.preset or "general").lower()
    guidance = {
        "wedding": "Prioritize expressions, open eyes, flattering light, and emotional moments. Technical flaws still matter.",
        "documentary": "Prioritize moment, clarity, and readable subject separation. Do not excuse severe technical flaws.",
        "wildlife": "Prioritize subject sharpness, timing, and clean separation from the background.",
        "landscape": "Prioritize edge-to-edge technical quality, exposure control, horizon discipline, and strong composition.",
        "street": "Prioritize decisive moment and framing, but reject images with obvious blur, clipping, or visual confusion.",
        "holiday": "Balance people, atmosphere, and scene storytelling while still rejecting weak technical execution.",
        "general": "Balance technical quality, composition, and subject impact. Technical flaws should outweigh vague mood.",
    }
    return guidance.get(preset, guidance["general"])


def build_prompt(context: PromptContext) -> str:
    """Inject Stage 1/2 hints into the scoring prompt template and return it."""
    hints = _collect_hints(context)
    hint_block = "\n".join(hints) if hints else "No prior flags."
    return _PROMPT_TEMPLATE.format(
        preset_block=_preset_guidance(context),
        hint_block=hint_block,
    )
