"""Filmstrip and face-panel behaviour in the running app."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from textual.widgets import Static

from cull.config import CullConfig
from cull.tui import faces
from cull.tui.app import AppInput, CullApp
from cull.tui.face_panel import FacePanel
from cull.tui.faces import FaceReport
from cull.tui.filmstrip import Filmstrip, FilmstripCell
from cull.tui.images import ImageCells
from cull.tui.photo_view import PhotoView

from tests.tui._helpers import make_session, numbered, run, wait_until

APP_SIZE: tuple[int, int] = (140, 45)


def _app(tmp_path: Path, count: int) -> CullApp:
    """Build a review app over ``count`` uncertain photos."""
    return CullApp(AppInput(session=make_session(tmp_path, numbered(count)), config=CullConfig()))


def _slot_sources(app: CullApp) -> list[str | None]:
    """Return each filmstrip slot's filename, or None for an empty slot."""
    return [cell.query_one(ImageCells).source.name if cell.query_one(ImageCells).source else None
            for cell in app.query(FilmstripCell)]


def test_filmstrip_centres_current_photo_among_neighbours(tmp_path: Path) -> None:
    """Seven slots: the cursor in the middle (highlighted), neighbours either side."""
    async def body() -> None:
        app = _app(tmp_path, 6)
        async with app.run_test(size=APP_SIZE) as pilot:
            await wait_until(lambda: app.query_one(PhotoView).shown is not None)
            assert _slot_sources(app) == [None, None, None, "p00.jpg", "p01.jpg", "p02.jpg", "p03.jpg"]
            cells = list(app.query(FilmstripCell))
            assert [c.has_class("current") for c in cells] == [False, False, False, True, False, False, False]
            await pilot.press("x")
            assert _slot_sources(app)[2:5] == ["p00.jpg", "p01.jpg", "p02.jpg"]
            assert "REJECT" in str(cells[2].query_one(Static).render())

    run(body)


def test_filmstrip_toggles_with_f(tmp_path: Path) -> None:
    """f hides and shows the filmstrip."""
    async def body() -> None:
        app = _app(tmp_path, 3)
        async with app.run_test(size=APP_SIZE) as pilot:
            await wait_until(lambda: app.query_one(PhotoView).shown is not None)
            await pilot.press("f")
            assert app.query_one(Filmstrip).has_class("hidden")
            await pilot.press("f")
            assert not app.query_one(Filmstrip).has_class("hidden")

    run(body)


def test_face_detection_never_blocks_navigation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the detector stuck, z still opens the panel and arrows still move."""
    gate = threading.Event()

    def slow_analyse(source: Path) -> FaceReport:
        gate.wait(5)
        return FaceReport(source=source, error="stub")

    monkeypatch.setattr(faces, "analyse_faces", slow_analyse)

    async def body() -> None:
        app = _app(tmp_path, 3)
        async with app.run_test(size=APP_SIZE) as pilot:
            await wait_until(lambda: app.query_one(PhotoView).shown is not None)
            await pilot.press("z")
            panel = app.query_one(FacePanel)
            assert panel.is_visible
            assert "finding faces" in str(panel.query_one("#face-status", Static).render())
            await pilot.press("right", "right")
            assert app._photo_index == 2
            gate.set()
            await wait_until(lambda: "stub" in str(panel.query_one("#face-status", Static).render()))
            assert app._photo_index == 2

    run(body)
