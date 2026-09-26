"""Wording of the one-line 'why' for each kind of AI decision."""

from __future__ import annotations

from pathlib import Path

from cull.models import BurstInfo, Stage3Result
from cull.tui.why_line import WhyContext, build_why_text

from tests.tui._helpers import PhotoSpec, make_decision


def test_burst_loser_names_the_sharper_frame(tmp_path: Path) -> None:
    """Stage 1 burst losers say which frame won."""
    decision = make_decision(tmp_path, PhotoSpec(
        name="DSCF0002.JPG", label="rejected", burst=BurstInfo(group_id=1, rank=1, group_size=2),
    ))
    text = build_why_text(WhyContext(decision=decision, ai_label="rejected", burst_winner="DSCF0123.JPG"))
    assert text == "REJECT · Stage 1: burst loser (sharper frame DSCF0123.JPG)"


def test_blurry_reject_says_blurry(tmp_path: Path) -> None:
    """A Stage 1 blur reject says so in words."""
    decision = make_decision(tmp_path, PhotoSpec(name="a.jpg", label="rejected"))
    decision.stage1.is_pass = False
    decision.stage1.reject_reason = "blur"
    assert build_why_text(WhyContext(decision=decision, ai_label="rejected")) == "REJECT · Stage 1: blurry"


def test_uncertain_shows_threshold_and_vlm_flags(tmp_path: Path) -> None:
    """Uncertain photos show the composite against the keep threshold, plus VLM flags."""
    decision = make_decision(tmp_path, PhotoSpec(name="a.jpg", composite=0.88))
    decision.stage3 = Stage3Result(photo_path=decision.photo.path, flags=["eyes_closed"])
    text = build_why_text(WhyContext(decision=decision, ai_label="uncertain"))
    assert text == "UNCERTAIN · composite 0.88 (keep ≥0.94) · VLM: eyes_closed"


def test_keeper_shows_composite(tmp_path: Path) -> None:
    """Keepers show their composite."""
    decision = make_decision(tmp_path, PhotoSpec(name="a.jpg", label="keeper", composite=0.95))
    assert build_why_text(WhyContext(decision=decision, ai_label="keeper")) == "KEEP · composite 0.95"


def test_user_override_is_appended(tmp_path: Path) -> None:
    """When the user disagreed, the line keeps the AI's call and adds theirs."""
    decision = make_decision(tmp_path, PhotoSpec(name="a.jpg", label="rejected", composite=0.95))
    text = build_why_text(WhyContext(decision=decision, ai_label="keeper"))
    assert text == "KEEP · composite 0.95  → you: REJECT"
