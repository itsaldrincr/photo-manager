"""Pilot-driven resize robustness tests for the CULL review TUI.

Exercises real Textual layout/resize plumbing via ``App.run_test`` so that
size-math bugs (zero/negative dimensions, layout overflow) are caught the
same way a live terminal would trigger them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Static

from cull.config import CullConfig
from cull.tui import images
from cull.tui.app import (
    AppInput,
    CullApp,
    MIN_TERMINAL_COLS,
    MIN_TERMINAL_ROWS,
    TOO_SMALL_BANNER_ID,
)
from cull.tui.filmstrip import Filmstrip
from cull.tui.photo_view import PhotoView

from tests.tui._helpers import make_session, numbered, run

SETTLE_SECONDS: float = 0.3
STORM_SIZES: tuple[tuple[int, int], ...] = (
    (20, 5), (35, 10), (50, 15), (65, 5), (80, 10),
    (20, 15), (35, 5), (50, 10), (65, 15), (80, 5),
    (20, 10), (35, 15), (50, 5), (65, 10), (80, 15),
    (100, 30),
)


def _app(tmp_path: Path) -> CullApp:
    """Build a one-photo review app."""
    return CullApp(AppInput(session=make_session(tmp_path, numbered(1)), config=CullConfig()))


def test_mount_and_resize_storm_no_crash(tmp_path: Path) -> None:
    """A mount at (0,0)-ish plus a rapid resize storm raises no exception."""
    async def body() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await asyncio.sleep(SETTLE_SECONDS)
            sizes = [(80, 24), (40, 12), (1, 1), (0, 0), (200, 50), (10, 3), (80, 24)]
            for width, height in sizes:
                await pilot.resize_terminal(width, height)
            for width, height in STORM_SIZES:
                await pilot.resize_terminal(width, height)
            await pilot.pause(images.RESIZE_DEBOUNCE_SECONDS + 0.05)

    run(body)


def test_tiny_terminal_shows_placeholder_and_hides_photo(tmp_path: Path) -> None:
    """Dropping below the minimum size shows the placeholder and hides every image."""
    async def body() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(20, 6)
            await pilot.pause()
            banner = app.query_one(f"#{TOO_SMALL_BANNER_ID}", Static)
            assert banner.display is True
            assert app.query_one(PhotoView).display is False
            assert app.query_one(Filmstrip).display is False
            assert app.query_one("#main-row").display is False

    run(body)


def test_recovery_to_normal_size_restores_layout(tmp_path: Path) -> None:
    """Resizing back up above the minimum hides the placeholder again."""
    async def body() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(20, 6)
            await pilot.pause()
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            banner = app.query_one(f"#{TOO_SMALL_BANNER_ID}", Static)
            assert banner.display is False
            assert app.query_one(PhotoView).display is True

    run(body)


def test_exact_minimum_size_does_not_trigger_placeholder(tmp_path: Path) -> None:
    """The minimum size itself is treated as usable, not too-small."""
    async def body() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(MIN_TERMINAL_COLS, MIN_TERMINAL_ROWS)
            await pilot.pause()
            banner = app.query_one(f"#{TOO_SMALL_BANNER_ID}", Static)
            assert banner.display is False

    run(body)
