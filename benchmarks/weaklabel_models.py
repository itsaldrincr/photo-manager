"""Shared models, prompts, and agreement logic for Gemma weak-labeling runs.

Weak labels are MODEL-GENERATED (labeler: gemma-4-12b). They are never human
ground truth. Gemma's self-reported confidence is known-uninformative
(benchmarks/LOG.md 2026-07-18), so trust comes from double-pass agreement:
two paraphrased prompts, keep only photos where both passes agree.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

LABELER_NAME: str = "gemma-4-12b"
WEAKLABEL_MAX_TOKENS: int = 96

EXPRESSION_VOCAB: list[str] = [
    "reverent/prayerful", "joyful", "neutral", "distressed",
    "surprised", "eyes-closed-rest", "unclear",
]

_OCCLUSION_SCHEMA_LINE = (
    'Answer with exactly one JSON object and nothing else: '
    '{"face_occluded": true or false, "occluder": "<short name of the covering '
    'object, or null>", "confidence": <number 0.0-1.0>}'
)

_EXPRESSION_SCHEMA_LINE = (
    'Answer with exactly one JSON object and nothing else: '
    '{"expression": "<one label from the list>", "confidence": <number 0.0-1.0>}'
)

_EXPRESSION_VOCAB_LINE = (
    "'reverent/prayerful' (calm, solemn, devout — head bowed or eyes closed in "
    "prayer), 'joyful' (smiling or laughing), 'neutral', 'distressed' (upset, "
    "crying, pained, anxious), 'surprised', 'eyes-closed-rest' (eyes closed from "
    "a blink or rest, not prayer), 'unclear' (cannot tell)."
)

TASK_PROMPTS: dict[str, str] = {
    "occlusion_v1": (
        "Look at this cropped photo of a person's face. Is any part of the face "
        "(eyes, eyebrows, nose, or mouth) physically covered by an object or by "
        "another person — for example a hand, cup, phone, microphone, candle, "
        "veil, hair, mask, or someone else's head or shoulder? A face that is "
        "merely turned away, in profile, blurry, dark, or partly outside the "
        "frame does NOT count as occluded. " + _OCCLUSION_SCHEMA_LINE
    ),
    "occlusion_v2": (
        "You are auditing event photos for face visibility. For the face shown, "
        "decide whether some foreign thing blocks part of it from the camera: "
        "fingers, a drink, a microphone, fabric, an instrument, another "
        "attendee, etc. Ignore motion blur, soft focus, profile poses, harsh "
        "shadows, and frame cropping — none of those are occlusion. "
        + _OCCLUSION_SCHEMA_LINE
    ),
    "expression_v1": (
        "This face was photographed at a live event (a church liturgy or a "
        "wedding). Choose the single label that best describes the person's "
        "expression: " + _EXPRESSION_VOCAB_LINE + " " + _EXPRESSION_SCHEMA_LINE
    ),
    "expression_v2": (
        "Classify the facial expression in this event photo (taken during a "
        "religious service or wedding celebration). Pick exactly one: "
        + _EXPRESSION_VOCAB_LINE + " Base the choice on the whole face — mouth, "
        "eyes, and brow together. " + _EXPRESSION_SCHEMA_LINE
    ),
}


class WeaklabelManifest(BaseModel):
    """Input to gemma_weaklabel_runner.py: which images to label."""

    image_paths: list[Path]


class WeaklabelReading(BaseModel):
    """One image's parsed weak label from one Gemma pass."""

    name: str
    path: str
    is_parse_error: bool = False
    face_occluded: bool | None = None
    occluder: str | None = None
    expression: str | None = None
    confidence: float | None = None
    raw_text: str = ""
    latency_seconds: float = 0.0


class WeaklabelRunOutput(BaseModel):
    """Full output of one gemma_weaklabel_runner.py invocation."""

    task: str
    labeler: str = LABELER_NAME
    prompt: str
    readings: list[WeaklabelReading] = []


class AgreedLabel(BaseModel):
    """A double-pass-agreed weak label for one image."""

    name: str
    path: str
    label: str


class AgreementResult(BaseModel):
    """Double-pass agreement outcome across a corpus."""

    agreed: list[AgreedLabel]
    disagreed_names: list[str]
    parse_error_names: list[str]

    @property
    def disagreement_rate(self) -> float:
        """Share of comparable (both-parsed) photos where the passes disagree."""
        comparable = len(self.agreed) + len(self.disagreed_names)
        return len(self.disagreed_names) / comparable if comparable else 0.0


def _reading_label(reading: WeaklabelReading) -> str | None:
    """Return the comparable label string for a reading, or None if unparsed."""
    if reading.is_parse_error:
        return None
    if reading.expression is not None:
        return reading.expression
    if reading.face_occluded is not None:
        return str(reading.face_occluded)
    return None


def compute_agreement(runs: tuple[WeaklabelRunOutput, WeaklabelRunOutput]) -> AgreementResult:
    """Keep only photos where both paraphrased passes produced the same label."""
    second = {r.name: r for r in runs[1].readings}
    agreed, disagreed, errored = [], [], []
    for first in runs[0].readings:
        label_a = _reading_label(first)
        label_b = _reading_label(second[first.name]) if first.name in second else None
        if label_a is None or label_b is None:
            errored.append(first.name)
        elif label_a == label_b:
            agreed.append(AgreedLabel(name=first.name, path=first.path, label=label_a))
        else:
            disagreed.append(first.name)
    return AgreementResult(agreed=agreed, disagreed_names=disagreed, parse_error_names=errored)
