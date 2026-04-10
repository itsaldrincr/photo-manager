"""Tests for taste-uncertainty queue ordering in the TUI."""

from __future__ import annotations

from pathlib import Path

from cull.models import (
    BlurScores,
    BurstInfo,
    ExposureScores,
    PhotoMeta,
    PhotoDecision,
    Stage1Result,
    Stage2Result,
    TasteScore,
)
from cull.tui.app import _sort_by_uncertainty

TASTE_PROBABILITIES: list[float] = [0.1, 0.49, 0.51, 0.95]


def _make_stage1(idx: int) -> Stage1Result:
    """Build a minimal Stage1Result for test use."""
    return Stage1Result(
        photo_path=Path(f"/tmp/photo_{idx}.jpg"),
        blur=BlurScores(tenengrad=1.0, fft_ratio=1.0, blur_tier=1),
        exposure=ExposureScores(
            dr_score=1.0,
            clipping_highlight=0.0,
            clipping_shadow=0.0,
            midtone_pct=0.5,
            color_cast_score=0.0,
        ),
        noise_score=0.0,
    )


def _make_stage2(probability: float) -> Stage2Result:
    """Build a Stage2Result with a TasteScore at the given probability."""
    taste = TasteScore(
        probability=probability,
        label_count_at_score=10,
        weight_applied=0.15,
        model_version="logreg-v10",
    )
    return Stage2Result(
        photo_path=Path("/tmp/placeholder.jpg"),
        topiq=0.5,
        laion_aesthetic=0.5,
        clipiqa=0.5,
        composite=0.5,
        taste=taste,
    )


def _make_decision(idx: int, probability: float) -> PhotoDecision:
    """Build a PhotoDecision with TasteScore at the given probability."""
    meta = PhotoMeta(path=Path(f"/tmp/photo_{idx}.jpg"), filename=f"photo_{idx}.jpg")
    return PhotoDecision(
        photo=meta,
        decision="uncertain",
        stage1=_make_stage1(idx),
        stage2=_make_stage2(probability),
    )


def _build_decisions() -> list[PhotoDecision]:
    """Build a list of decisions with varying taste probabilities."""
    return [_make_decision(i, p) for i, p in enumerate(TASTE_PROBABILITIES)]


def test_sort_by_uncertainty_orders_by_abs_p_minus_half() -> None:
    """Queue is ordered by |p - 0.5| descending when TasteScore is present."""
    decisions = _build_decisions()
    indices = list(range(len(decisions)))
    sorted_indices = _sort_by_uncertainty(decisions, indices)

    uncertainties = [abs(decisions[i].stage2.taste.probability - 0.5) for i in sorted_indices]
    assert uncertainties == sorted(uncertainties, reverse=True)


def test_sort_preserves_order_without_taste_scores() -> None:
    """Queue keeps original order when no decision has TasteScore."""
    decisions = [_make_decision(i, 0.5) for i in range(3)]
    for d in decisions:
        d.stage2.taste = None
    indices = [0, 1, 2]
    result = _sort_by_uncertainty(decisions, indices)
    assert result == indices


def test_sort_highest_uncertainty_is_first() -> None:
    """The most uncertain photo (p closest to 0.5) appears last; most certain first."""
    decisions = _build_decisions()
    indices = list(range(len(decisions)))
    sorted_indices = _sort_by_uncertainty(decisions, indices)

    first_prob = decisions[sorted_indices[0]].stage2.taste.probability
    last_prob = decisions[sorted_indices[-1]].stage2.taste.probability

    assert abs(first_prob - 0.5) >= abs(last_prob - 0.5)
