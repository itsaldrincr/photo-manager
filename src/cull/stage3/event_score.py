"""Event preset scoring: VLM rating first, a cheap Stage 2 logistic as tiebreak."""

from __future__ import annotations

import math
from typing import NamedTuple

from cull.config import (
    EVENT_CHEAP_TIEBREAK_WEIGHT,
    EVENT_KEEPER_RATING,
    EVENT_UNCERTAIN_RATING,
)
from cull.models import DecisionLabel, Stage2Result, Stage3Result


class _Term(NamedTuple):
    """One standardised logistic term: coef * (x - mean) / std."""

    coef: float
    mean: float
    std: float


# Logistic regression fitted on 488 AI-labelled singles-mixer photos
# (cross-validated keeper AUC 0.65). A missing value counts as 0 before
# standardising, because that is how the fit saw it.
CHEAP_TERMS: dict[str, _Term] = {
    "topiq": _Term(0.242, 0.4915, 0.0749),
    "clipiqa": _Term(0.047, 0.6062, 0.0535),
    "laion_aesthetic": _Term(0.131, 0.4607, 0.0274),
    "composition": _Term(-0.063, 1.2097, 0.3717),
    "valence": _Term(0.727, 0.1619, 0.3286),
    "arousal": _Term(0.118, 0.0529, 0.1318),
    "has_face": _Term(-0.53, 0.4406, 0.4965),
    "eye_sharpness_left": _Term(-0.08, 1085.8869, 1528.8111),
    "is_eyes_closed": _Term(-0.347, 0.1107, 0.3137),
}
CHEAP_INTERCEPT: float = -1.143


def _or_zero(value: float | None) -> float:
    """Return value, or 0.0 when it is missing."""
    return 0.0 if value is None else float(value)


def _portrait_features(stage2: Stage2Result) -> dict[str, float]:
    """Return the face terms; every one is 0 when no face was scored."""
    portrait = stage2.portrait
    if portrait is None:
        return {"valence": 0.0, "arousal": 0.0, "has_face": 0.0,
                "eye_sharpness_left": 0.0, "is_eyes_closed": 0.0}
    return {
        "valence": _or_zero(portrait.valence),
        "arousal": _or_zero(portrait.arousal),
        "has_face": 1.0 if portrait.valence is not None else 0.0,
        "eye_sharpness_left": _or_zero(portrait.eye_sharpness_left),
        "is_eyes_closed": 1.0 if portrait.is_eyes_closed else 0.0,
    }


def cheap_features(stage2: Stage2Result) -> dict[str, float]:
    """Return the raw (unstandardised) feature values the cheap model reads."""
    composition = stage2.composition.composite if stage2.composition is not None else None
    return {
        "topiq": stage2.topiq,
        "clipiqa": stage2.clipiqa,
        "laion_aesthetic": stage2.laion_aesthetic,
        "composition": _or_zero(composition),
        **_portrait_features(stage2),
    }


def cheap_prob(stage2: Stage2Result) -> float:
    """Return the cheap-model keeper probability in (0, 1)."""
    features = cheap_features(stage2)
    logit = CHEAP_INTERCEPT + sum(
        term.coef * (features[name] - term.mean) / term.std
        for name, term in CHEAP_TERMS.items()
    )
    return 1.0 / (1.0 + math.exp(-logit))


def usable_rating(stage3: Stage3Result | None) -> int | None:
    """Return the VLM rating, or None when there is no usable one."""
    if stage3 is None or stage3.is_parse_error:
        return None
    return stage3.rating


def event_score(stage2: Stage2Result, stage3: Stage3Result | None) -> float:
    """Return rating + 0.01 * cheap_prob, or cheap_prob alone without a rating."""
    prob = cheap_prob(stage2)
    rating = usable_rating(stage3)
    if rating is None:
        return prob
    return rating + EVENT_CHEAP_TIEBREAK_WEIGHT * prob


def rating_label(rating: int) -> DecisionLabel:
    """Return keeper for 5, uncertain (human review) for 4, rejected for 3 or less."""
    if rating >= EVENT_KEEPER_RATING:
        return "keeper"
    if rating >= EVENT_UNCERTAIN_RATING:
        return "uncertain"
    return "rejected"


def rating_verdict(rating: int | None) -> bool | None:
    """Return the keep verdict a rating implies; None for the review band or no rating."""
    if rating is None:
        return None
    label = rating_label(rating)
    return None if label == "uncertain" else label == "keeper"
