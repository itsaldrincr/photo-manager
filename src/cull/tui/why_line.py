"""One plain-language line stating the AI decision on a photo and why."""

from __future__ import annotations

from pydantic import BaseModel
from rich.text import Text

from cull.config import ROUTING_AMBIGUOUS_MIN, ROUTING_KEEPER_MIN
from cull.models import DecisionLabel, PhotoDecision

SEPARATOR: str = " · "
DECISION_WORDS: dict[str, str] = {
    "keeper": "KEEP",
    "rejected": "REJECT",
    "uncertain": "UNCERTAIN",
    "duplicate": "DUPLICATE",
    "select": "SELECT",
}
DECISION_COLOURS: dict[str, str] = {
    "keeper": "bold green",
    "rejected": "bold red",
    "uncertain": "bold yellow",
    "duplicate": "bold magenta",
    "select": "bold cyan",
}
STAGE1_REASON_WORDS: dict[str, str] = {"blur": "blurry", "noise": "too noisy"}


class WhyContext(BaseModel):
    """A decision, the AI's own label for it, and its burst winner's filename."""

    decision: PhotoDecision
    ai_label: DecisionLabel
    burst_winner: str | None = None


def _composite(decision: PhotoDecision) -> float | None:
    """Return the Stage 2 composite score, if scored."""
    return decision.stage2.composite if decision.stage2 is not None else None


def _vlm_note(decision: PhotoDecision) -> str | None:
    """Return 'VLM: flag, flag' or the VLM verdict, if Stage 3 ran."""
    stage3 = decision.stage3
    if stage3 is None:
        return None
    if stage3.flags:
        return f"VLM: {', '.join(stage3.flags)}"
    if stage3.is_keeper is None:
        return None
    verdict = "keeper" if stage3.is_keeper else "reject"
    return f"VLM: {verdict} (conf {stage3.confidence:.2f})"


def _stage1_reason(ctx: WhyContext) -> str | None:
    """Return the Stage 1 reason for a reject or duplicate, if Stage 1 made the call."""
    stage1 = ctx.decision.stage1
    if stage1 is None:
        return None
    if stage1.is_duplicate:
        return "Stage 1: near-duplicate"
    burst = stage1.burst
    if burst is not None and not burst.is_burst_winner and ctx.burst_winner:
        return f"Stage 1: burst loser (sharper frame {ctx.burst_winner})"
    if not stage1.is_pass:
        return f"Stage 1: {STAGE1_REASON_WORDS.get(stage1.reject_reason or '', 'failed checks')}"
    return None


def _score_reason(ctx: WhyContext) -> str | None:
    """Return the composite score, with the threshold that routed it when relevant."""
    composite = _composite(ctx.decision)
    if composite is None:
        return None
    if ctx.ai_label == "uncertain":
        return f"composite {composite:.2f} (keep ≥{ROUTING_KEEPER_MIN:.2f})"
    if ctx.ai_label == "rejected" and composite < ROUTING_AMBIGUOUS_MIN:
        return f"composite {composite:.2f} (reject <{ROUTING_AMBIGUOUS_MIN:.2f})"
    return f"composite {composite:.2f}"


def _reasons(ctx: WhyContext) -> list[str]:
    """Return the reasons, most decisive first."""
    stage1 = _stage1_reason(ctx) if ctx.ai_label in ("rejected", "duplicate") else None
    if stage1 is not None:
        return [stage1]
    reasons = ["Stage 4: curated pick"] if ctx.ai_label == "select" else []
    for reason in (_score_reason(ctx), _vlm_note(ctx.decision)):
        if reason is not None:
            reasons.append(reason)
    return reasons


def build_why_text(ctx: WhyContext) -> str:
    """Return e.g. 'UNCERTAIN · composite 0.88 (keep ≥0.94) · VLM: eyes_closed'."""
    parts = [DECISION_WORDS[ctx.ai_label], *_reasons(ctx)]
    text = SEPARATOR.join(parts)
    if ctx.decision.decision != ctx.ai_label:
        text += f"  → you: {DECISION_WORDS[ctx.decision.decision]}"
    return text


def build_why_line(ctx: WhyContext) -> Text:
    """Return the why text coloured by the AI decision."""
    return Text(build_why_text(ctx), style=DECISION_COLOURS[ctx.ai_label], no_wrap=True, overflow="ellipsis")
