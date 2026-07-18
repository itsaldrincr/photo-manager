"""Pilot-driven resize robustness tests for the CULL review TUI.

Exercises real Textual layout/resize plumbing via ``App.run_test`` so that
size-math bugs (zero/negative dimensions, layout overflow) and Kitty APC
traffic during resize storms are caught the same way a live terminal would
trigger them. No real terminal or pytest-asyncio plugin is required: async
test bodies are driven with a plain ``asyncio.run`` wrapper.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Awaitable, Callable

import pytest
from PIL import Image
from textual.app import App, ComposeResult
from textual.widgets import Static

from cull.config import CullConfig
from cull.models import PhotoDecision, PhotoMeta
from cull._pipeline.orchestrator import SessionResult, SessionSummary
from cull.tui import photo_view as pv
from cull.tui.app import (
    AppInput,
    CullApp,
    MIN_TERMINAL_COLS,
    MIN_TERMINAL_ROWS,
    TOO_SMALL_BANNER_ID,
)

IMAGE_SIZE: tuple[int, int] = (400, 300)
SETTLE_SECONDS: float = 0.3
STORM_SIZES: tuple[tuple[int, int], ...] = (
    (20, 5), (35, 10), (50, 15), (65, 5), (80, 10),
    (20, 15), (35, 5), (50, 10), (65, 15), (80, 5),
    (20, 10), (35, 15), (50, 5), (65, 10), (80, 15),
    (100, 30),
)


def _make_jpeg_bytes() -> bytes:
    """Return tiny solid-color JPEG bytes for use as a photo fixture."""
    img = Image.new("RGB", IMAGE_SIZE, color=(10, 20, 30))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_session(tmp_path: Path) -> SessionResult:
    """Build a one-photo session for TUI resize tests."""
    photo_path = tmp_path / "photo1.jpg"
    photo_path.write_bytes(_make_jpeg_bytes())
    decision = PhotoDecision(
        photo=PhotoMeta(path=photo_path, filename="photo1.jpg"), decision="uncertain",
    )
    return SessionResult(
        source_path=str(tmp_path), total_photos=1, summary=SessionSummary(), decisions=[decision],
    )


@pytest.fixture(autouse=True)
def _reset_module_caches() -> Iterator[None]:
    """Clear photo_view's module-level caches so tests don't leak identity state."""
    pv._upload_identity_cache.clear()
    pv._png_cache.clear()
    yield
    pv._upload_identity_cache.clear()
    pv._png_cache.clear()


class _PhotoViewHarness(App):
    """Minimal app hosting a bare PhotoView, for widget-level upload tests."""

    def compose(self) -> ComposeResult:
        yield pv.PhotoView()


def _run(body: Callable[[], Awaitable[None]]) -> None:
    """Run an async test body synchronously (no pytest-asyncio dependency)."""
    asyncio.run(body())


def test_mount_and_resize_storm_no_crash(tmp_path: Path) -> None:
    """A mount at (0,0)-ish plus a rapid resize storm raises no exception."""
    async def body() -> None:
        session = _make_session(tmp_path)
        app = CullApp(AppInput(session=session, config=CullConfig()))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await asyncio.sleep(SETTLE_SECONDS)
            sizes = [(80, 24), (40, 12), (1, 1), (0, 0), (200, 50), (10, 3), (80, 24)]
            for width, height in sizes:
                await pilot.resize_terminal(width, height)
            for width, height in STORM_SIZES:
                await pilot.resize_terminal(width, height)
            await pilot.pause(pv.RESIZE_DEBOUNCE_SECONDS + 0.05)

    _run(body)


def test_resize_storm_coalesces_to_single_settled_emit(tmp_path: Path) -> None:
    """A resize storm produces zero writes mid-storm and one put after settling."""
    async def body() -> None:
        calls: list[str] = []
        original = pv._write_raw
        pv._write_raw = lambda data: calls.append(data)  # type: ignore[assignment]
        try:
            session = _make_session(tmp_path)
            app = CullApp(AppInput(session=session, config=CullConfig()))
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                await asyncio.sleep(SETTLE_SECONDS)
                calls.clear()
                for width, height in STORM_SIZES:
                    await pilot.resize_terminal(width, height)
                mid_storm_write_count = len(calls)
                await pilot.pause(pv.RESIZE_DEBOUNCE_SECONDS + 0.05)
                joined = "".join(calls)
        finally:
            pv._write_raw = original
        assert mid_storm_write_count == 0
        assert joined.count("a=p,i=") == 1
        assert joined.count("a=T,i=") == 0

    _run(body)


def test_tiny_terminal_shows_placeholder_and_hides_photo(tmp_path: Path) -> None:
    """Dropping below the minimum size shows the placeholder and hides content."""
    async def body() -> None:
        session = _make_session(tmp_path)
        app = CullApp(AppInput(session=session, config=CullConfig()))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(20, 6)
            await pilot.pause()
            banner = app.query_one(f"#{TOO_SMALL_BANNER_ID}", Static)
            assert banner.display is True
            assert app.query_one(pv.PhotoView).display is False

    _run(body)


def test_recovery_to_normal_size_restores_layout(tmp_path: Path) -> None:
    """Resizing back up above the minimum hides the placeholder again."""
    async def body() -> None:
        session = _make_session(tmp_path)
        app = CullApp(AppInput(session=session, config=CullConfig()))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(20, 6)
            await pilot.pause()
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            banner = app.query_one(f"#{TOO_SMALL_BANNER_ID}", Static)
            assert banner.display is False
            assert app.query_one(pv.PhotoView).display is True

    _run(body)


def test_exact_minimum_size_does_not_trigger_placeholder(tmp_path: Path) -> None:
    """The minimum size itself is treated as usable, not too-small."""
    async def body() -> None:
        session = _make_session(tmp_path)
        app = CullApp(AppInput(session=session, config=CullConfig()))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(MIN_TERMINAL_COLS, MIN_TERMINAL_ROWS)
            await pilot.pause()
            banner = app.query_one(f"#{TOO_SMALL_BANNER_ID}", Static)
            assert banner.display is False

    _run(body)


def test_repeat_display_of_identical_content_uploads_once(tmp_path: Path) -> None:
    """Displaying identical PNG content twice reuses the upload; no re-upload APC."""
    async def body() -> None:
        calls: list[str] = []
        original = pv._write_raw
        pv._write_raw = lambda data: calls.append(data)  # type: ignore[assignment]
        try:
            app = _PhotoViewHarness()
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                photo_view = app.query_one(pv.PhotoView)
                viewport = pv.ViewportSize(cols=40, rows=20)
                image_bytes = _make_jpeg_bytes()
                request_a = pv.RenderRequest(
                    image_id="photo-a", image_bytes=image_bytes, viewport=viewport,
                )
                photo_view.display_photo(request_a)
                await asyncio.sleep(SETTLE_SECONDS)
                request_b = pv.RenderRequest(
                    image_id="photo-b", image_bytes=image_bytes, viewport=viewport,
                )
                photo_view.display_photo(request_b)
                await asyncio.sleep(SETTLE_SECONDS)
                joined = "".join(calls)
        finally:
            pv._write_raw = original
        assert joined.count("a=T,i=") == 1

    _run(body)
