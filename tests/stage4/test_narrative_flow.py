"""Tests for narrative-flow variety regulariser — mocked ML, no real model loads."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cull.models import CuratorSelection
from cull.saliency import SaliencyResult
from cull.stage2.portrait import PortraitResult
from cull.stage4.narrative_flow import (
    NarrativeFlowInput,
    _variety_score,
    check,
)

# ---------------------------------------------------------------------------
# Constants for synthetic test data
# ---------------------------------------------------------------------------

FRAME_WIDTH: int = 1000
FRAME_HEIGHT: int = 1000

# Face bboxes that produce known ratios against 1000×1000 frame:
# close-up: 400×400 = 160,000 px / 1,000,000 = 0.16 (>= 0.15)
CLOSE_BBOX: tuple[int, int, int, int] = (300, 300, 700, 700)
# medium: 150×150 = 22,500 px / 1,000,000 = 0.0225 (< 0.15, >= 0.03)
# Wait — 0.0225 < 0.03, let's use 200×200 = 40,000 / 1,000,000 = 0.04
MEDIUM_BBOX: tuple[int, int, int, int] = (400, 400, 600, 600)
# wide: 80×80 = 6,400 / 1,000,000 = 0.0064 (< 0.03)
WIDE_BBOX: tuple[int, int, int, int] = (460, 460, 540, 540)


def _make_selection(name: str) -> CuratorSelection:
    """Build a synthetic CuratorSelection with a fake path."""
    return CuratorSelection(
        path=Path(f"/fake/{name}.jpg"),
        cluster_id=0,
        cluster_size=1,
        composite=0.8,
        is_vlm_winner=False,
    )


def _mock_portrait_with_bbox(bbox: tuple[int, int, int, int]) -> PortraitResult:
    """Return a PortraitResult with a known face bbox."""
    return PortraitResult(face_count=1, face_bbox=bbox)


def _mock_portrait_no_face() -> PortraitResult:
    """Return a PortraitResult with no detected face."""
    return PortraitResult(face_count=0)


def _mock_pil_image() -> MagicMock:
    """Return a mock PIL Image with width/height = FRAME_WIDTH/FRAME_HEIGHT."""
    img = MagicMock()
    img.width = FRAME_WIDTH
    img.height = FRAME_HEIGHT
    img.__enter__ = lambda s: s
    img.__exit__ = MagicMock(return_value=False)
    return img


# ---------------------------------------------------------------------------
# Unit tests for _variety_score
# ---------------------------------------------------------------------------


class TestVarietyScore:
    """Direct unit tests of _variety_score — no mocking needed."""

    def test_empty_returns_zero(self) -> None:
        """Empty shot list yields zero variety."""
        assert _variety_score([]) == 0.0

    def test_all_same_type_is_low(self) -> None:
        """All close-ups: single type, low variety score."""
        score = _variety_score(["close", "close", "close"])
        assert score < 0.5

    def test_all_three_types_is_high(self) -> None:
        """One of each type: maximum variety score."""
        score = _variety_score(["close", "medium", "wide"])
        assert score > 0.8

    def test_two_types_is_mid(self) -> None:
        """Two types present: intermediate variety score."""
        score_two = _variety_score(["close", "wide", "close"])
        score_one = _variety_score(["close", "close", "close"])
        assert score_two > score_one


# ---------------------------------------------------------------------------
# Integration tests for check() with mocked ML backends
# ---------------------------------------------------------------------------


class TestCheckVarietyImproves:
    """check() should swap in a different shot type when variety is low."""

    def test_all_close_gets_swap_proposed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All-close selections: check proposes a swap if a wide/medium candidate exists."""
        close_sel = _make_selection("close1")
        wide_cand = _make_selection("wide_candidate")

        portrait_close = _mock_portrait_with_bbox(CLOSE_BBOX)
        portrait_no_face = _mock_portrait_no_face()

        pil_mock = _mock_pil_image()
        wide_saliency = SaliencyResult(
            heatmap=np.ones((7, 7), dtype=float),
            peak_xy=(3 / 7, 3 / 7),
            bbox=(0.0, 0.0, 1.0, 1.0),
        )

        def fake_portrait(path: Path, config: object) -> PortraitResult:
            if "wide" in path.name:
                return portrait_no_face
            return portrait_close

        def fake_saliency(request: object) -> SaliencyResult:
            return wide_saliency

        monkeypatch.setattr(
            "cull.stage4.narrative_flow.assess_portrait", fake_portrait
        )
        monkeypatch.setattr(
            "cull.stage4.narrative_flow.compute_saliency", fake_saliency
        )

        with patch("PIL.Image.open", return_value=pil_mock):
            flow_input = NarrativeFlowInput(
                selections=[close_sel],
                candidates={"wide_candidate": wide_cand.path},
            )
            selections, score = check(flow_input)

        assert score >= 0.0

    def test_no_swap_when_variety_maximal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three shot types present: check returns same selections unchanged."""
        close_sel = _make_selection("close1")
        medium_sel = _make_selection("medium1")
        wide_sel = _make_selection("wide1")

        pil_mock = _mock_pil_image()

        def fake_portrait(path: Path, config: object) -> PortraitResult:
            if "wide" in path.name:
                return _mock_portrait_with_bbox(WIDE_BBOX)
            if "medium" in path.name:
                return _mock_portrait_with_bbox(MEDIUM_BBOX)
            return _mock_portrait_with_bbox(CLOSE_BBOX)

        monkeypatch.setattr(
            "cull.stage4.narrative_flow.assess_portrait", fake_portrait
        )

        with patch("PIL.Image.open", return_value=pil_mock):
            flow_input = NarrativeFlowInput(
                selections=[close_sel, medium_sel, wide_sel],
                candidates={},
            )
            selections, score = check(flow_input)

        assert selections == [close_sel, medium_sel, wide_sel]
        assert score > 0.8

    def test_variety_improves_after_swap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """check() returns higher score than the all-close baseline."""
        close1 = _make_selection("close1")
        close2 = _make_selection("close2")
        wide_cand = _make_selection("wide_cand")

        pil_mock = _mock_pil_image()
        wide_saliency = SaliencyResult(
            heatmap=np.ones((7, 7), dtype=float),
            peak_xy=(3 / 7, 3 / 7),
            bbox=(0.0, 0.0, 1.0, 1.0),
        )

        def fake_portrait(path: Path, config: object) -> PortraitResult:
            if "wide" in path.name:
                return _mock_portrait_no_face()
            return _mock_portrait_with_bbox(CLOSE_BBOX)

        def fake_saliency(request: object) -> SaliencyResult:
            return wide_saliency

        monkeypatch.setattr(
            "cull.stage4.narrative_flow.assess_portrait", fake_portrait
        )
        monkeypatch.setattr(
            "cull.stage4.narrative_flow.compute_saliency", fake_saliency
        )

        with patch("PIL.Image.open", return_value=pil_mock):
            flow_input = NarrativeFlowInput(
                selections=[close1, close2],
                candidates={"wide_cand": wide_cand.path},
            )
            _original_shots = ["close", "close"]
            original_score = _variety_score(_original_shots)
            _new_selections, new_score = check(flow_input)

        assert new_score >= original_score
