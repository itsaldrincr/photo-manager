"""One-row filmstrip of queue neighbours with decision markers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from cull.models import DecisionLabel
from cull.tui.images import ImageCells
from cull.tui.why_line import DECISION_COLOURS, DECISION_WORDS

FILMSTRIP_SLOTS: int = 7
FILMSTRIP_CENTRE: int = FILMSTRIP_SLOTS // 2


class StripItem(BaseModel):
    """One thumbnail: the photo, its current label, and whether it is the cursor."""

    source: Path
    filename: str
    label: DecisionLabel
    is_current: bool = False


def decision_marker(label: DecisionLabel) -> Text:
    """Return a coloured '● KEEP'-style marker for a label."""
    return Text(f"● {DECISION_WORDS[label]}", style=DECISION_COLOURS[label], no_wrap=True, overflow="ellipsis")


class FilmstripCell(Vertical):
    """A thumbnail over its decision marker."""

    DEFAULT_CSS = """
    FilmstripCell {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }
    FilmstripCell.current {
        background: $accent 40%;
    }
    FilmstripCell > ImageCells {
        height: 1fr;
    }
    FilmstripCell > Static {
        height: 1;
        content-align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        yield ImageCells()
        yield Static("")

    def set_item(self, item: StripItem | None) -> None:
        """Show ``item``, or an empty slot past either end of the queue."""
        self.set_class(item is not None and item.is_current, "current")
        self.query_one(ImageCells).show(item.source if item else None)
        self.query_one(Static).update(decision_marker(item.label) if item else "", layout=False)


class Filmstrip(Horizontal):
    """Seven neighbouring photos, the current one in the middle and highlighted."""

    DEFAULT_CSS = """
    Filmstrip {
        height: 7;
        width: 1fr;
    }
    Filmstrip.hidden {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        for _ in range(FILMSTRIP_SLOTS):
            yield FilmstripCell()

    def update_items(self, items: list[StripItem | None]) -> None:
        """Fill the slots left to right; ``items`` has FILMSTRIP_SLOTS entries."""
        for cell, item in zip(self.query(FilmstripCell), items):
            cell.set_item(item)

    def thumbnail_boxes(self) -> list[ImageCells]:
        """Return the thumbnail widgets, left to right."""
        return list(self.query(ImageCells))
