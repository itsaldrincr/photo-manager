"""Decision assembly — label routing and PhotoDecision construction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from cull.models import (
    DecisionLabel,
    PhotoDecision,
    PhotoMeta,
    Stage3Result,
)
from cull._pipeline.stage1_runner import _Stage1Output
from cull._pipeline.stage2_runner import _Stage2Output

if TYPE_CHECKING:
    from cull._pipeline.orchestrator import SessionSummary


class _DecisionCtx(BaseModel):
    """Input bundle for building all PhotoDecision objects."""

    paths: list[Path]
    s1_out: _Stage1Output
    s2_out: _Stage2Output | None = None
    s3_results: dict[str, Stage3Result] = Field(default_factory=dict)


def _build_photo_meta(path: Path) -> PhotoMeta:
    """Create a PhotoMeta from a file path."""
    return PhotoMeta(
        path=path,
        filename=path.name,
        fs_mtime=path.stat().st_mtime,
    )


def _decide_label(path: Path, ctx: _DecisionCtx) -> DecisionLabel:
    """Determine final decision label for one photo."""
    key = str(path)
    if key in ctx.s1_out.duplicate_paths:
        return "duplicate"
    if key in ctx.s1_out.burst_losers:
        return "rejected"
    s1 = ctx.s1_out.results.get(key)
    if s1 and not s1.is_pass:
        return "rejected"
    return _decide_from_s2_s3(key, ctx)


def _decide_from_s2_s3(key: str, ctx: _DecisionCtx) -> DecisionLabel:
    """Determine label from Stage 2/3 results."""
    if ctx.s2_out is None:
        return "uncertain"
    fusion = ctx.s2_out.results.get(key)
    if fusion and fusion.routing == "KEEPER":
        return "keeper"
    if fusion and fusion.routing == "REJECT":
        return "rejected"
    s3 = ctx.s3_results.get(key)
    if s3 and s3.is_keeper is True:
        return "keeper"
    if s3 and s3.is_keeper is False:
        return "rejected"
    return "uncertain"


def _stage_reached(key: str, ctx: _DecisionCtx) -> int:
    """Return the highest stage reached for a given photo."""
    if key in ctx.s3_results:
        return 3
    if ctx.s2_out and key in ctx.s2_out.results:
        return 2
    return 1


def _build_decision(path: Path, ctx: _DecisionCtx) -> PhotoDecision:
    """Assemble a single PhotoDecision from pipeline context."""
    key = str(path)
    label = _decide_label(path, ctx)
    stage = _stage_reached(key, ctx)
    s2_result = ctx.s2_out.results[key].stage2 if ctx.s2_out and key in ctx.s2_out.results else None
    return PhotoDecision(
        photo=_build_photo_meta(path),
        decision=label,
        stage1=ctx.s1_out.results.get(key),
        stage2=s2_result,
        stage3=ctx.s3_results.get(key),
        stage_reached=stage,
    )


def _build_all_decisions(ctx: _DecisionCtx) -> list[PhotoDecision]:
    """Build PhotoDecision objects for every photo."""
    return [_build_decision(p, ctx) for p in ctx.paths]


def _build_summary(decisions: list[PhotoDecision]) -> "SessionSummary":
    """Count decisions by label. NOTE: relies on Stage 4 mutating decisions in-place
    via `_mark_selected` BEFORE this is called, so 'select' counts post-curation."""
    from cull._pipeline.orchestrator import SessionSummary  # noqa: PLC0415

    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.decision] = counts.get(d.decision, 0) + 1
    return SessionSummary(
        keepers=counts.get("keeper", 0),
        rejected=counts.get("rejected", 0),
        duplicates=counts.get("duplicate", 0),
        uncertain=counts.get("uncertain", 0),
        selected=counts.get("select", 0),
    )
