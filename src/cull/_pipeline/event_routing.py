"""Event preset routing: representatives by VLM rating, or by cheap_prob rank without one."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cull.config import EVENT_NO_VLM_KEEPER_FRACTION, EVENT_NO_VLM_UNCERTAIN_FRACTION
from cull.models import DecisionLabel, Stage3Result
from cull.stage3.event_score import cheap_prob, event_score, rating_label, usable_rating

from cull._pipeline.stage1_runner import _Stage1Output
from cull._pipeline.stage2_runner import _Stage2Output


class EventRouteInput(BaseModel):
    """Stage outputs the event router reads, after stacks are resolved."""

    s1_out: _Stage1Output
    s2_out: _Stage2Output | None = None
    s3_results: dict[str, Stage3Result] = Field(default_factory=dict)


def collect_event_scores(route_in: EventRouteInput) -> dict[str, float]:
    """Return event_score for every Stage-2-scored photo, keyed by str(path)."""
    if route_in.s2_out is None:
        return {}
    return {
        key: event_score(fusion.stage2, route_in.s3_results.get(key))
        for key, fusion in route_in.s2_out.results.items()
        if fusion.stage2 is not None
    }


def _representatives(route_in: EventRouteInput) -> list[str]:
    """Return scored photos that passed Stage 1 and lead their stack (or stand alone)."""
    if route_in.s2_out is None:
        return []
    s1_out = route_in.s1_out
    return [
        key for key, fusion in route_in.s2_out.results.items()
        if fusion.stage2 is not None
        and key not in s1_out.duplicate_paths
        and (key not in s1_out.results or s1_out.results[key].is_pass)
    ]


def label_by_rank(keys_best_first: list[str]) -> dict[str, DecisionLabel]:
    """Label the top 20% keeper, the next 25% uncertain, the rest rejected."""
    total = len(keys_best_first)
    keeper_count = round(total * EVENT_NO_VLM_KEEPER_FRACTION)
    uncertain_end = keeper_count + round(total * EVENT_NO_VLM_UNCERTAIN_FRACTION)
    labels: dict[str, DecisionLabel] = {}
    for index, key in enumerate(keys_best_first):
        if index < keeper_count:
            labels[key] = "keeper"
        elif index < uncertain_end:
            labels[key] = "uncertain"
        else:
            labels[key] = "rejected"
    return labels


def _cheap_labels(route_in: EventRouteInput, reps: list[str]) -> dict[str, DecisionLabel]:
    """Rank all representatives by cheap_prob, so the cut is shoot-relative."""
    results = route_in.s2_out.results if route_in.s2_out is not None else {}
    ranked = sorted(reps, key=lambda key: (-cheap_prob(results[key].stage2), key))
    return label_by_rank(ranked)


def route_event(route_in: EventRouteInput) -> dict[str, DecisionLabel]:
    """Return the event label for every representative.

    A rated representative routes by its rating. One without a rating
    (--no-vlm, or a parse error after retries) routes by its cheap_prob rank.
    Non-representatives are absent: they stay duplicates.
    """
    reps = _representatives(route_in)
    cheap = _cheap_labels(route_in, reps)
    labels: dict[str, DecisionLabel] = {}
    for key in reps:
        rating = usable_rating(route_in.s3_results.get(key))
        labels[key] = rating_label(rating) if rating is not None else cheap[key]
    return labels
