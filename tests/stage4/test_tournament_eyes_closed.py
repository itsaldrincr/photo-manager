"""Stage 4 tournament must pass the Stage 2 eyes-closed verdict into the VLM prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from cull.config import CullConfig
from cull.stage2.portrait import PortraitResult
from cull.stage4.curator_prompt import CuratorTiebreakResult
from cull.stage4.tournament import TournamentContext, TournamentInput, run


def test_match_context_carries_stage2_eyes_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A blinking photo's match context says eyes_closed; its open-eyed rival's does not."""
    blinking, open_eyed = tmp_path / "a.jpg", tmp_path / "b.jpg"
    contexts: dict[str, bool] = {}

    def _compare(call_in: object) -> CuratorTiebreakResult:
        tb = call_in.tiebreak_input  # type: ignore[attr-defined]
        contexts[tb.photo_a.name] = tb.context.eyes_closed
        contexts[tb.photo_b.name] = tb.context_b.eyes_closed
        return CuratorTiebreakResult(winner=tb.photo_a, reason="mock", confidence=1.0)

    monkeypatch.setattr("cull.stage4.tournament.compare_photos", _compare)
    ctx = TournamentContext(
        s1_results={},
        composite_scores={str(blinking): 0.9, str(open_eyed): 0.5},
        portraits={str(blinking): PortraitResult(face_count=1, eyes_closed=True)},
    )
    run(TournamentInput(candidates=[blinking, open_eyed], config=CullConfig()), ctx)
    assert contexts == {"a.jpg": True, "b.jpg": False}
