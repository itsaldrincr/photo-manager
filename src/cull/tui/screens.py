"""Modal overlays: key help and quit-without-saving confirmation."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

HELP_ROWS: tuple[tuple[str, str], ...] = (
    ("p / k", "keep (pick), then next"),
    ("x / r", "reject, then next"),
    ("c", "select (curate), then next"),
    ("d", "duplicate, then next"),
    ("u / ctrl+z", "undo (also bulk, stack and cluster actions)"),
    ("← →  , .  < >", "previous / next photo"),
    ("home / end", "first / last photo in queue"),
    ("tab, 1-5", "next queue; uncertain, rejected, duplicates, keepers, selected"),
    ("b", "compare the burst stack side by side"),
    ("X", "reject the whole burst stack"),
    ("K / R", "keep / reject every photo in this queue"),
    ("A", "accept confident VLM verdicts"),
    ("f", "filmstrip on/off"),
    ("z", "face close-ups on/off"),
    ("s", "score details on/off"),
    ("e", "explain this photo (VLM)"),
    ("?", "this help"),
    ("q", "save and quit"),
    ("Q", "quit without saving (asks first)"),
)


def help_text() -> str:
    """Return the key table as aligned lines."""
    width = max(len(keys) for keys, _ in HELP_ROWS)
    lines = [f"  {keys.ljust(width)}   {meaning}" for keys, meaning in HELP_ROWS]
    return "Keys\n\n" + "\n".join(lines) + "\n\n  esc / ? to close"


class HelpScreen(ModalScreen[None]):
    """Key reference overlay."""

    BINDINGS = [
        Binding("escape,question_mark,q", "dismiss_help", "Close"),
    ]

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    HelpScreen > Vertical {
        width: auto;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $panel;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(help_text())

    def action_dismiss_help(self) -> None:
        """Close the overlay."""
        self.dismiss(None)


class ConfirmQuitScreen(ModalScreen[bool]):
    """Asks before quitting without saving; dismisses True to quit."""

    BINDINGS = [
        Binding("y,Q", "answer(True)", "Quit"),
        Binding("n,escape", "answer(False)", "Cancel"),
    ]

    DEFAULT_CSS = """
    ConfirmQuitScreen {
        align: center middle;
    }
    ConfirmQuitScreen > Vertical {
        width: auto;
        height: auto;
        padding: 1 2;
        border: round $error;
        background: $panel;
    }
    """

    def __init__(self, pending_changes: int) -> None:
        super().__init__()
        self._pending_changes = pending_changes

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                f"Quit without saving?\n\n{self._pending_changes} decision(s) this session: "
                "no photos will move and the report will not change.\n\n[y] quit   [n / esc] keep reviewing"
            )

    def action_answer(self, should_quit: bool) -> None:
        """Return the answer to the app."""
        self.dismiss(should_quit)
