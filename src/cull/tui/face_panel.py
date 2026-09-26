"""Side panel of face close-ups with traffic-light markers (eyes, sharpness)."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from cull.models import PhotoDecision
from cull.tui.faces import MAX_FACES, FaceReading, FaceReport
from cull.tui.images import ImageCells

LIGHT: str = "●"
EYE_COLOURS: dict[str, str] = {"open": "green", "squint": "yellow", "closed": "red"}
SHARPNESS_COLOURS: dict[str, str] = {"good": "green", "soft": "yellow", "blurry": "red"}


def face_markers(face: FaceReading) -> Text:
    """Return 'eyes ● open  sharp ● good' with each light coloured."""
    text = Text(no_wrap=True, overflow="ellipsis")
    text.append("eyes ")
    text.append(LIGHT, style=EYE_COLOURS[face.eyes])
    text.append(f" {face.eyes}  sharp ")
    text.append(LIGHT, style=SHARPNESS_COLOURS[face.sharpness_level])
    text.append(f" {face.sharpness_level}")
    return text


def report_summary(decision: PhotoDecision) -> str:
    """Return what the session report already says about the main face."""
    portrait = decision.stage2.portrait if decision.stage2 is not None else None
    if portrait is None:
        return "report: no portrait data"
    eyes = "closed" if portrait.is_eyes_closed else ("squinting" if portrait.is_squinting else "open")
    return f"report: eyes {eyes}"


class FaceCard(Vertical):
    """One face crop over its markers."""

    DEFAULT_CSS = """
    FaceCard {
        height: 1fr;
        border: round $panel-lighten-2;
    }
    FaceCard > ImageCells {
        height: 1fr;
    }
    FaceCard > Static {
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield ImageCells()
        yield Static("")

    def set_face(self, face: FaceReading | None) -> None:
        """Show one face, or hide the card."""
        self.display = face is not None
        self.query_one(ImageCells).show(face.crop.path if face else None)
        self.query_one(Static).update(face_markers(face) if face else "", layout=False)


class FacePanel(Vertical):
    """Docked beside the photo; hidden until toggled with ``z``."""

    DEFAULT_CSS = """
    FacePanel {
        width: 34;
        height: 1fr;
        display: none;
        border-left: solid $panel-lighten-2;
    }
    FacePanel.visible {
        display: block;
    }
    FacePanel > #face-status {
        height: 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="face-status")
        for _ in range(MAX_FACES):
            yield FaceCard()

    @property
    def is_visible(self) -> bool:
        """Return True when the panel is shown."""
        return self.has_class("visible")

    def toggle(self) -> None:
        """Show or hide the panel."""
        self.toggle_class("visible")

    def show_loading(self, decision: PhotoDecision) -> None:
        """Clear old faces and say detection is running."""
        self._set_status(f"{decision.photo.filename}: finding faces…\n{report_summary(decision)}")
        self._set_faces([])

    def show_report(self, report: FaceReport, decision: PhotoDecision) -> None:
        """Show a finished report."""
        if report.error is not None:
            headline = f"faces: {report.error}"
        else:
            headline = f"{len(report.faces)} face(s)" if report.faces else "no faces found"
        self._set_status(f"{headline}\n{report_summary(decision)}")
        self._set_faces(report.faces)

    def _set_status(self, message: str) -> None:
        """Update the two-line header."""
        self.query_one("#face-status", Static).update(message, layout=False)

    def _set_faces(self, faces: list[FaceReading]) -> None:
        """Fill cards in order; unused cards hide."""
        for index, card in enumerate(self.query(FaceCard)):
            card.set_face(faces[index] if index < len(faces) else None)
