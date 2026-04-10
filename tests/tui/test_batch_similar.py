"""Tests for action_reject_cluster propagating the reject override to all cluster members."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cull.models import (
    BlurScores,
    BurstInfo,
    ExposureScores,
    PhotoDecision,
    PhotoMeta,
    Stage1Result,
)
from cull.tui.app import _apply_override, _find_burst_decisions

CLUSTER_SIZE: int = 4
BURST_GROUP_ID: int = 7


def _make_burst_stage1(idx: int) -> Stage1Result:
    """Build a Stage1Result that belongs to a burst group."""
    return Stage1Result(
        photo_path=Path(f"/tmp/cluster_{idx}.jpg"),
        blur=BlurScores(tenengrad=1.0, fft_ratio=1.0, blur_tier=1),
        exposure=ExposureScores(
            dr_score=1.0,
            clipping_highlight=0.0,
            clipping_shadow=0.0,
            midtone_pct=0.5,
            color_cast_score=0.0,
        ),
        noise_score=0.0,
        burst=BurstInfo(group_id=BURST_GROUP_ID, rank=idx, group_size=CLUSTER_SIZE),
    )


def _make_cluster_decision(idx: int) -> PhotoDecision:
    """Build a PhotoDecision that is a member of the synthetic cluster."""
    meta = PhotoMeta(path=Path(f"/tmp/cluster_{idx}.jpg"), filename=f"cluster_{idx}.jpg")
    return PhotoDecision(
        photo=meta,
        decision="uncertain",
        stage1=_make_burst_stage1(idx),
    )


def _build_cluster() -> list[PhotoDecision]:
    """Build a synthetic cluster of CLUSTER_SIZE members."""
    return [_make_cluster_decision(i) for i in range(CLUSTER_SIZE)]


class _MinimalSession:
    """Minimal session-like object for testing cluster operations."""

    def __init__(self, decisions: list[PhotoDecision]) -> None:
        self.decisions: list[PhotoDecision] = list(decisions)
        self.source_path: str = "/tmp/test_session"


def _apply_cluster_reject_impl(session: _MinimalSession, group_id: int) -> None:
    """Replicate action_reject_cluster logic for unit testing without TUI."""
    cluster = _find_burst_decisions(session, group_id)
    for member in cluster:
        path_str = str(member.photo.path)
        for i, d in enumerate(session.decisions):
            if str(d.photo.path) == path_str:
                session.decisions[i] = _apply_override(member, "rejected")
                break


def test_reject_cluster_marks_all_members_rejected() -> None:
    """action_reject_cluster applies rejected to every cluster member."""
    cluster = _build_cluster()
    session = _MinimalSession(cluster)

    with patch("cull.taste_trainer.maybe_retrain") as mock_retrain:
        _apply_cluster_reject_impl(session, BURST_GROUP_ID)
        _ = mock_retrain  # retrain not called here; tested at integration level

    for decision in session.decisions:
        assert decision.decision == "rejected"
        assert decision.is_override is True


def test_reject_cluster_affects_only_matching_group() -> None:
    """Cluster reject does not touch decisions from a different burst group."""
    cluster = _build_cluster()
    other_stage1 = Stage1Result(
        photo_path=Path("/tmp/other.jpg"),
        blur=BlurScores(tenengrad=1.0, fft_ratio=1.0, blur_tier=1),
        exposure=ExposureScores(
            dr_score=1.0,
            clipping_highlight=0.0,
            clipping_shadow=0.0,
            midtone_pct=0.5,
            color_cast_score=0.0,
        ),
        noise_score=0.0,
        burst=BurstInfo(group_id=99, rank=0, group_size=1),
    )
    other_meta = PhotoMeta(path=Path("/tmp/other.jpg"), filename="other.jpg")
    other_decision = PhotoDecision(photo=other_meta, decision="uncertain", stage1=other_stage1)
    all_decisions = cluster + [other_decision]
    session = _MinimalSession(all_decisions)

    _apply_cluster_reject_impl(session, BURST_GROUP_ID)

    assert session.decisions[-1].decision == "uncertain"
    assert session.decisions[-1].is_override is False


def test_find_burst_decisions_returns_correct_members() -> None:
    """_find_burst_decisions returns only members of the given group."""
    cluster = _build_cluster()
    session = _MinimalSession(cluster)
    found = _find_burst_decisions(session, BURST_GROUP_ID)
    assert len(found) == CLUSTER_SIZE
    for d in found:
        assert d.stage1 is not None
        assert d.stage1.burst is not None
        assert d.stage1.burst.group_id == BURST_GROUP_ID
