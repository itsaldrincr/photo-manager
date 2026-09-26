"""Side-by-side compare mode for a burst stack (2-4 frames on screen)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Static

from cull.models import DecisionLabel, PhotoDecision
from cull.tui.filmstrip import decision_marker
from cull.tui.images import ImageCells

MAX_COMPARE_TILES: int = 4


class StackFrame(BaseModel):
    """One frame of a stack as compare mode shows it."""

    index: int
    source: Path
    filename: str
    label: DecisionLabel
    sharpness: float
    is_ai_pick: bool


class CompareSource(BaseModel):
    """How compare mode reads the stack and acts on it (the app owns the state)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    group_id: int
    load_frames: Callable[[], list[StackFrame]]
    pick: Callable[[StackFrame], None]
    undo: Callable[[], None]
    start_path: Path | None = None


def frame_sharpness(decision: PhotoDecision) -> float:
    """Return the Stage 1 subject sharpness, else Tenengrad, else 0."""
    if decision.stage1 is None:
        return 0.0
    blur = decision.stage1.blur
    return blur.subject_sharpness if blur.subject_sharpness is not None else blur.tenengrad


def window_start(cursor: int, total: int) -> int:
    """Return the first visible frame so the cursor stays on screen."""
    visible = min(MAX_COMPARE_TILES, total)
    return max(0, min(cursor - visible // 2, total - visible))


def tile_caption(frame: StackFrame) -> Text:
    """Return 'DSCF0001  ● KEEP  sharp 0.84  ★ AI pick'."""
    caption = Text(no_wrap=True, overflow="ellipsis")
    caption.append(f"{frame.filename}  ")
    caption.append_text(decision_marker(frame.label))
    caption.append(f"  sharp {frame.sharpness:.2f}")
    if frame.is_ai_pick:
        caption.append("  ★ AI pick", style="bold yellow")
    return caption


class CompareTile(Vertical):
    """One frame and its caption; the cursor tile gets a bright border."""

    DEFAULT_CSS = """
    CompareTile {
        width: 1fr;
        height: 1fr;
        border: round $panel-lighten-2;
    }
    CompareTile.cursor {
        border: heavy $accent;
    }
    CompareTile > ImageCells {
        height: 1fr;
    }
    CompareTile > Static {
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield ImageCells()
        yield Static("")

    def set_frame(self, frame: StackFrame, is_cursor: bool) -> None:
        """Show one frame; highlight it when the cursor is on it."""
        self.set_class(is_cursor, "cursor")
        self.query_one(ImageCells).show(frame.source)
        self.query_one(Static).update(tile_caption(frame), layout=False)


class CompareView(Screen):
    """Compare a stack's frames; ``p`` picks the cursor frame, the rest become rejects."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("left", "move(-1)", "Prev frame"),
        Binding("right", "move(1)", "Next frame"),
        Binding("p,k", "pick", "Pick"),
        Binding("u,ctrl+z", "undo", "Undo"),
    ]

    DEFAULT_CSS = """
    CompareView > #compare-title {
        height: 1;
        padding: 0 1;
        background: $boost;
    }
    CompareView > #compare-tiles {
        height: 1fr;
    }
    """

    def __init__(self, source: CompareSource) -> None:
        super().__init__()
        self._source = source
        self._frames = source.load_frames()
        self._cursor = self._initial_cursor()

    def _initial_cursor(self) -> int:
        """Start on the photo compare mode was opened from."""
        for position, frame in enumerate(self._frames):
            if frame.source == self._source.start_path:
                return position
        return 0

    @property
    def cursor(self) -> int:
        """Return the cursor position within the stack."""
        return self._cursor

    @property
    def frames(self) -> list[StackFrame]:
        """Return the frames as last loaded."""
        return self._frames

    def compose(self) -> ComposeResult:
        yield Static("", id="compare-title")
        with Horizontal(id="compare-tiles"):
            for _ in range(min(MAX_COMPARE_TILES, len(self._frames))):
                yield CompareTile()
        yield Footer()

    def on_mount(self) -> None:
        """Fill tiles once they have a size."""
        self.call_after_refresh(self._render_tiles)

    def _render_tiles(self) -> None:
        """Show the visible window of frames and the title line."""
        start = window_start(self._cursor, len(self._frames))
        for offset, tile in enumerate(self.query(CompareTile)):
            position = start + offset
            tile.set_frame(self._frames[position], position == self._cursor)
        self.query_one("#compare-title", Static).update(
            f"STACK #{self._source.group_id} · frame {self._cursor + 1}/{len(self._frames)}"
            "  ←/→ move · p pick (others rejected) · u undo · esc back"
        )

    def _reload(self) -> None:
        """Re-read labels after the app changed them."""
        self._frames = self._source.load_frames()
        self._render_tiles()

    def action_move(self, step: int) -> None:
        """Move the cursor within the stack."""
        self._cursor = max(0, min(len(self._frames) - 1, self._cursor + step))
        self._render_tiles()

    def action_pick(self) -> None:
        """Pick the cursor frame; the app rejects the others as one undoable step."""
        self._source.pick(self._frames[self._cursor])
        self._reload()

    def action_undo(self) -> None:
        """Undo the last action (usually the last pick)."""
        self._source.undo()
        self._reload()

    def action_back(self) -> None:
        """Return to the main view."""
        self.dismiss(None)
