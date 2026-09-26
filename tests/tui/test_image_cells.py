"""ImageCells + ImageService: what reaches the terminal, and what the widget paints."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from cull.tui import images, kitty, previews
from cull.tui.kitty import CellSize
from cull.tui.photo_view import PhotoView

from tests.tui._helpers import JpegSpec, run, wait_until, write_jpeg

HARNESS_SIZE: tuple[int, int] = (80, 24)


class Harness(App):
    """A bare PhotoView with its own image service."""

    def __init__(self) -> None:
        super().__init__()
        self.images = images.ImageService(self)

    def compose(self) -> ComposeResult:
        yield PhotoView()

    def on_mount(self) -> None:
        self.images.start()

    def on_unmount(self) -> None:
        self.images.stop()


@pytest.fixture(autouse=True)
def fixed_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend every cell is 10x20 px so fits are predictable."""
    monkeypatch.setattr(kitty, "query_cell_size", lambda: CellSize(width=10.0, height=20.0))


def _row_text(view: PhotoView, y: int) -> str:
    """Return the text of one rendered row."""
    return "".join(segment.text for segment in view.render_line(y))


def test_upload_is_store_only_then_virtual_placement(tmp_path: Path, terminal) -> None:
    """A shown photo is transmitted with a=t and placed virtually; no a=T, no clear-all."""
    source = write_jpeg(JpegSpec(path=tmp_path / "a.jpg"))

    async def body() -> None:
        app = Harness()
        async with app.run_test(size=HARNESS_SIZE) as pilot:
            view = app.query_one(PhotoView)
            view.show(source)
            await wait_until(lambda: view.shown is not None)
            await pilot.pause()

    run(body)
    assert "a=t,t=f" in terminal.joined
    assert "a=p,U=1" in terminal.joined
    assert terminal.joined.index("a=t,") < terminal.joined.index("a=p,U=1")
    assert "a=T" not in terminal.joined
    assert "d=a" not in terminal.joined


@pytest.fixture(params=[((3000, 2000), (72, 24)), ((2000, 3000), (32, 24))], ids=["landscape", "portrait"])
def fit_case(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Path, tuple[int, int]]:
    """A photo on disk and the cell box it should fit in an 80x24 view."""
    size, expected = request.param
    return write_jpeg(JpegSpec(path=tmp_path / "a.jpg", size=size)), expected


def test_placement_fits_aspect_and_is_centred(fit_case, terminal) -> None:
    """80x24 cells at 10x20 px is 800x480 px: 3:2 fits 72x24, 2:3 fits 32x24, centred."""
    source, expected = fit_case

    async def body() -> None:
        app = Harness()
        async with app.run_test(size=HARNESS_SIZE) as pilot:
            view = app.query_one(PhotoView)
            view.show(source)
            await wait_until(lambda: view.shown is not None)
            await pilot.pause()
            cols, rows = expected
            assert (view.shown.box.cols, view.shown.box.rows) == expected
            assert f"c={cols},r={rows}" in terminal.joined
            row = _row_text(view, 12)
            left = (80 - cols) // 2
            assert row[:left] == " " * left
            assert row[left] == kitty.PLACEHOLDER
            assert row.count(kitty.PLACEHOLDER) == cols

    run(body)


def test_old_image_stays_until_new_one_is_ready(tmp_path: Path, terminal, monkeypatch: pytest.MonkeyPatch) -> None:
    """Navigating never blanks the view: the old image is painted until the new one is placed."""
    first = write_jpeg(JpegSpec(path=tmp_path / "a.jpg"))
    second = write_jpeg(JpegSpec(path=tmp_path / "b.jpg", colour=(200, 0, 0)))
    gate = threading.Event()
    real_render = previews.render_preview

    def gated_render(spec: previews.PreviewSpec) -> previews.Preview:
        if spec.source == second:
            gate.wait(5)
        return real_render(spec)

    monkeypatch.setattr(previews, "render_preview", gated_render)

    async def body() -> None:
        app = Harness()
        async with app.run_test(size=HARNESS_SIZE) as pilot:
            view = app.query_one(PhotoView)
            view.show(first)
            await wait_until(lambda: view.shown is not None)
            first_id = view.shown.image_id
            terminal.clear()
            view.show(second)
            await pilot.pause(0.2)
            assert view.shown.image_id == first_id
            assert kitty.PLACEHOLDER in _row_text(view, 12)
            assert "a=d" not in terminal.joined
            gate.set()
            await wait_until(lambda: view.shown.image_id != first_id)

    run(body)


def test_identical_content_uploads_once(tmp_path: Path, terminal) -> None:
    """Going back to a photo re-uses its upload: one transmit per image."""
    first = write_jpeg(JpegSpec(path=tmp_path / "a.jpg"))
    second = write_jpeg(JpegSpec(path=tmp_path / "b.jpg"))

    async def body() -> None:
        app = Harness()
        async with app.run_test(size=HARNESS_SIZE):
            view = app.query_one(PhotoView)
            view.show(first)
            await wait_until(lambda: view.shown is not None)
            first_id = view.shown.image_id
            view.show(second)
            await wait_until(lambda: view.shown.image_id != first_id)
            view.show(first)
            assert view.shown.image_id == first_id

    run(body)
    assert terminal.joined.count("a=t,") == 2


def test_eviction_frees_old_images_but_not_the_one_on_screen(tmp_path: Path, terminal, monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the upload cap, least-recently-used images are deleted with a=d,d=I."""
    monkeypatch.setattr(images, "MAX_UPLOADED_IMAGES", 2)
    sources = [write_jpeg(JpegSpec(path=tmp_path / f"{i}.jpg", colour=(i * 40, 0, 0))) for i in range(4)]

    async def body() -> None:
        app = Harness()
        async with app.run_test(size=HARNESS_SIZE):
            view = app.query_one(PhotoView)
            for source in sources:
                shown_before = view.shown
                view.show(source)
                await wait_until(lambda: view.shown is not None and view.shown != shown_before)
            on_screen = view.shown.image_id
            assert f"a=d,d=I,i={on_screen}," not in terminal.joined

    run(body)
    assert terminal.joined.count("a=d,d=I") >= 2


def test_resize_storm_writes_nothing_until_settled(tmp_path: Path, terminal) -> None:
    """Resizes are debounced: no terminal traffic mid-storm, one fresh upload after."""
    source = write_jpeg(JpegSpec(path=tmp_path / "a.jpg"))

    async def body() -> None:
        app = Harness()
        async with app.run_test(size=HARNESS_SIZE) as pilot:
            view = app.query_one(PhotoView)
            view.show(source)
            await wait_until(lambda: view.shown is not None)
            terminal.clear()
            for width, height in ((20, 5), (50, 15), (65, 10), (80, 5), (100, 30)):
                await pilot.resize_terminal(width, height)
            assert terminal.writes == []
            await wait_until(lambda: view.shown is not None and view.shown.box.rows == 30)

    run(body)
    assert terminal.joined.count("a=t,") == 1
