"""Burst comparison view for selecting the best frame in a burst group."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from cull.models import BurstInfo, DecisionLabel, PhotoDecision

logger = logging.getLogger(__name__)

MAX_BURST_FRAMES: int = 9


class BurstFrame(BaseModel):
    """Display data for a single frame in a burst group."""

    filename: str
    path: Path
    sharpness: float
    is_current_winner: bool = False
    index: int = 0


class BurstGroup(BaseModel):
    """All frames in a burst group with their display data."""

    group_id: int
    frames: list[BurstFrame] = Field(default_factory=list)
    selected_index: int = 0


class BurstResult(BaseModel):
    """Result of a burst selection: winner path and duplicate paths."""

    winner: Path
    duplicates: list[Path] = Field(default_factory=list)
    is_confirmed: bool = False


def _extract_sharpness(decision: PhotoDecision) -> float:
    """Extract sharpness score from a decision's stage data."""
    if decision.stage1 and decision.stage1.blur.subject_sharpness is not None:
        return decision.stage1.blur.subject_sharpness
    if decision.stage1:
        return decision.stage1.blur.tenengrad
    return 0.0


def build_burst_group(decisions: list[PhotoDecision], group_id: int) -> BurstGroup:
    """Build a BurstGroup from decisions sharing a burst group ID."""
    frames: list[BurstFrame] = []
    for idx, dec in enumerate(decisions):
        is_winner = dec.stage1 is not None and dec.stage1.burst is not None and dec.stage1.burst.is_burst_winner
        frame = BurstFrame(
            filename=dec.photo.filename,
            path=dec.photo.path,
            sharpness=_extract_sharpness(dec),
            is_current_winner=is_winner,
            index=idx,
        )
        frames.append(frame)
    return BurstGroup(group_id=group_id, frames=frames)


def _format_frame_cell(frame: BurstFrame) -> str:
    """Format a single frame for display in the burst grid."""
    star = " *" if frame.is_current_winner else ""
    return f"[{frame.index + 1}] {frame.filename}\nSharp: {frame.sharpness:.2f}{star}"


def _render_burst_grid(group: BurstGroup) -> str:
    """Render the full burst grid as a text string."""
    header = f"BURST GROUP #{group.group_id} -- {len(group.frames)} frames"
    lines = [header, ""]
    for frame in group.frames:
        marker = " << SELECTED" if frame.index == group.selected_index else ""
        lines.append(f"  {_format_frame_cell(frame)}{marker}")
        lines.append("")
    lines.append("[1-9] Select winner   [Enter] Confirm   [Esc] Back")
    return "\n".join(lines)


def build_burst_result(group: BurstGroup) -> BurstResult:
    """Build a BurstResult from the current selection."""
    winner = group.frames[group.selected_index].path
    duplicates = [f.path for f in group.frames if f.index != group.selected_index]
    return BurstResult(winner=winner, duplicates=duplicates, is_confirmed=True)


class BurstView(Screen):
    """Modal screen for burst group comparison and winner selection."""

    BINDINGS = [
        Binding("escape", "cancel", "Back"),
        Binding("enter", "confirm", "Confirm"),
        Binding("1", "select_1", "Frame 1", show=False),
        Binding("2", "select_2", "Frame 2", show=False),
        Binding("3", "select_3", "Frame 3", show=False),
        Binding("4", "select_4", "Frame 4", show=False),
        Binding("5", "select_5", "Frame 5", show=False),
        Binding("6", "select_6", "Frame 6", show=False),
        Binding("7", "select_7", "Frame 7", show=False),
        Binding("8", "select_8", "Frame 8", show=False),
        Binding("9", "select_9", "Frame 9", show=False),
    ]

    def __init__(self, group: BurstGroup) -> None:
        super().__init__()
        self._group = group
        self._result: BurstResult | None = None

    def compose(self) -> ComposeResult:
        """Build the burst view layout."""
        yield Header()
        yield Static(_render_burst_grid(self._group), id="burst-grid")
        yield Footer()

    @property
    def result(self) -> BurstResult | None:
        """Return the burst selection result, if confirmed."""
        return self._result

    def _select_frame(self, index: int) -> None:
        """Select a frame by zero-based index if in range."""
        if index >= len(self._group.frames):
            return
        self._group.selected_index = index
        grid = self.query_one("#burst-grid", Static)
        grid.update(_render_burst_grid(self._group))

    def action_cancel(self) -> None:
        """Return to main view without changes."""
        self.dismiss(None)

    def action_confirm(self) -> None:
        """Confirm selection and return result."""
        self._result = build_burst_result(self._group)
        self.dismiss(self._result)

    def action_select_1(self) -> None:
        """Select frame 1."""
        self._select_frame(0)

    def action_select_2(self) -> None:
        """Select frame 2."""
        self._select_frame(1)

    def action_select_3(self) -> None:
        """Select frame 3."""
        self._select_frame(2)

    def action_select_4(self) -> None:
        """Select frame 4."""
        self._select_frame(3)

    def action_select_5(self) -> None:
        """Select frame 5."""
        self._select_frame(4)

    def action_select_6(self) -> None:
        """Select frame 6."""
        self._select_frame(5)

    def action_select_7(self) -> None:
        """Select frame 7."""
        self._select_frame(6)

    def action_select_8(self) -> None:
        """Select frame 8."""
        self._select_frame(7)

    def action_select_9(self) -> None:
        """Select frame 9."""
        self._select_frame(8)
