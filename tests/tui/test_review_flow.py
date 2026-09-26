"""Pilot tests for review flow: keys, auto-advance, undo, compare mode, confirmations."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from cull.config import CullConfig
from cull.models import BurstInfo, ExplainResult
from cull.tui import kitty
from cull.tui.app import AppInput, CullApp
from cull.tui.compare_view import CompareView
from cull.tui.explain_modal import ExplainPanel
from cull.tui.photo_view import PhotoView
from cull.tui.screens import ConfirmQuitScreen, HelpScreen

from tests.tui._helpers import PhotoSpec, make_session, numbered, run, wait_until

APP_SIZE: tuple[int, int] = (120, 40)


def _app(tmp_path: Path, photos: list[PhotoSpec]) -> CullApp:
    """Build a review app over photos written to ``tmp_path``."""
    return CullApp(AppInput(session=make_session(tmp_path, photos), config=CullConfig()))


async def _ready(app: CullApp) -> PhotoView:
    """Wait until the first photo is on screen."""
    view = app.query_one(PhotoView)
    await wait_until(lambda: view.shown is not None)
    return view


def _labels(app: CullApp) -> list[str]:
    """Return every decision label in session order."""
    return [d.decision for d in app._session.decisions]


def _burst_photos() -> list[PhotoSpec]:
    """Three frames of burst 7 (the last is the AI pick) plus two singles."""
    burst = [
        PhotoSpec(name=f"b{i}.jpg", burst=BurstInfo(group_id=7, rank=i, group_size=3, is_burst_winner=i == 2))
        for i in range(3)
    ]
    return burst + [PhotoSpec(name="s0.jpg"), PhotoSpec(name="s1.jpg")]


def test_navigation_keys_and_aliases(tmp_path: Path) -> None:
    """Arrows, . , > < and home/end all move the cursor."""
    async def body() -> None:
        app = _app(tmp_path, numbered(6))
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await pilot.press("right", "full_stop", "greater_than_sign")
            assert app._photo_index == 3
            await pilot.press("left", "comma", "less_than_sign")
            assert app._photo_index == 0
            await pilot.press("end")
            assert app._photo_index == 5
            await pilot.press("home")
            assert app._photo_index == 0

    run(body)


def test_decisions_auto_advance_with_pro_and_legacy_keys(tmp_path: Path) -> None:
    """p/k keep, x/r reject, c select: each labels the photo and moves to the next."""
    async def body() -> None:
        app = _app(tmp_path, numbered(6))
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await pilot.press("p", "x", "k", "r", "c")
            assert _labels(app)[:5] == ["keeper", "rejected", "keeper", "rejected", "select"]
            assert app._photo_index == 5
            await pilot.press("x")
            assert app._photo_index == 5
            assert _labels(app)[5] == "rejected"

    run(body)


def test_decided_photo_keeps_its_place_in_the_queue(tmp_path: Path) -> None:
    """After deciding, going back shows the same photo with its new label."""
    async def body() -> None:
        app = _app(tmp_path, numbered(3))
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await pilot.press("x", "left")
            assert app._current_decision().photo.filename == "p00.jpg"
            assert app._current_decision().decision == "rejected"

    run(body)


def test_same_label_is_not_logged_but_advances(tmp_path: Path, override_log) -> None:
    """Confirming the AI's label writes no override entry and counts as reviewed."""
    async def body() -> None:
        app = _app(tmp_path, numbered(3, label="keeper"))
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await pilot.press("p")
            app.flush_log_jobs()
            assert app._photo_index == 1
            assert override_log.entries == []
            assert len(app._ledger.reviewed) == 1

    run(body)


def test_undo_reverts_label_log_and_retrain_counter(tmp_path: Path, override_log) -> None:
    """u and ctrl+z undo a decision, its log entry, and its retrain bump."""
    async def body() -> None:
        app = _app(tmp_path, numbered(3))
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await pilot.press("x", "p")
            app.flush_log_jobs()
            assert len(override_log.entries) == 2
            assert override_log.retrain_counter == 2
            await pilot.press("u")
            app.flush_log_jobs()
            assert _labels(app)[1] == "uncertain"
            assert app._photo_index == 1
            await pilot.press("ctrl+z")
            app.flush_log_jobs()
            assert _labels(app) == ["uncertain"] * 3
            assert override_log.entries == []
            assert override_log.retrain_counter == 0

    run(body)


def test_undo_reverts_a_whole_bulk_action(tmp_path: Path, override_log) -> None:
    """R rejects the queue as one step; one undo restores every photo and log entry."""
    async def body() -> None:
        app = _app(tmp_path, numbered(5))
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await pilot.press("R")
            app.flush_log_jobs()
            assert _labels(app) == ["rejected"] * 5
            assert len(override_log.entries) == 5
            await pilot.press("u")
            app.flush_log_jobs()
            assert _labels(app) == ["uncertain"] * 5
            assert override_log.entries == []

    run(body)


def test_stack_reject_key_is_distinct_from_bulk_reject(tmp_path: Path) -> None:
    """X rejects only the current burst stack (one undo step); R would take the queue."""
    async def body() -> None:
        app = _app(tmp_path, _burst_photos())
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await pilot.press("X")
            assert _labels(app) == ["rejected"] * 3 + ["uncertain"] * 2
            await pilot.press("u")
            assert _labels(app) == ["uncertain"] * 5

    run(body)


def test_uncertain_queue_shows_most_uncertain_first(tmp_path: Path) -> None:
    """The queue opens on the photo whose taste p is nearest 0.5."""
    photos = [PhotoSpec(name=f"t{i}.jpg", taste=p) for i, p in enumerate((0.1, 0.95, 0.52, 0.3))]

    async def body() -> None:
        app = _app(tmp_path, photos)
        async with app.run_test(size=APP_SIZE):
            await _ready(app)
            names = [app._session.decisions[i].photo.filename for i in app._queue_indices]
            assert names == ["t2.jpg", "t3.jpg", "t0.jpg", "t1.jpg"]

    run(body)


def test_compare_mode_pick_rejects_others_and_is_undoable(tmp_path: Path) -> None:
    """b opens the stack; p on a frame keeps it, rejects the rest; u undoes; esc returns."""
    async def body() -> None:
        app = _app(tmp_path, _burst_photos())
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await pilot.press("b")
            await wait_until(lambda: isinstance(app.screen, CompareView))
            compare = app.screen
            assert len(compare.frames) == 3
            assert [f.is_ai_pick for f in compare.frames] == [False, False, True]
            await pilot.press("right", "p")
            assert _labels(app)[:3] == ["rejected", "keeper", "rejected"]
            await pilot.press("u")
            assert _labels(app)[:3] == ["uncertain"] * 3
            await pilot.press("escape")
            await wait_until(lambda: not isinstance(app.screen, CompareView))
            assert kitty.PLACEHOLDER in "".join(s.text for s in app.query_one(PhotoView).render_line(10))

    run(body)


def test_compare_without_a_stack_stays_put(tmp_path: Path) -> None:
    """b on a photo with no burst just notifies."""
    async def body() -> None:
        app = _app(tmp_path, numbered(2))
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await pilot.press("b")
            assert not isinstance(app.screen, CompareView)

    run(body)


def test_quit_without_saving_asks_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Q opens a confirmation; n keeps reviewing; Q then y quits."""
    exits: list[bool] = []

    async def body() -> None:
        app = _app(tmp_path, numbered(2))
        monkeypatch.setattr(app, "exit", lambda *args, **kwargs: exits.append(True))
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await pilot.press("Q")
            assert isinstance(app.screen, ConfirmQuitScreen)
            await pilot.press("n")
            assert not isinstance(app.screen, ConfirmQuitScreen)
            assert exits == []
            await pilot.press("Q", "y")
            assert exits == [True]

    run(body)


def test_help_overlay_hides_the_photo_until_closed(tmp_path: Path) -> None:
    """? shows the key help; image cells stop painting under it and come back after."""
    async def body() -> None:
        app = _app(tmp_path, numbered(2))
        async with app.run_test(size=APP_SIZE) as pilot:
            view = await _ready(app)
            await pilot.press("question_mark")
            assert isinstance(app.screen, HelpScreen)
            assert all(kitty.PLACEHOLDER not in "".join(s.text for s in view.render_line(y)) for y in range(view.size.height))
            await pilot.press("escape")
            assert not isinstance(app.screen, HelpScreen)
            assert any(kitty.PLACEHOLDER in "".join(s.text for s in view.render_line(y)) for y in range(view.size.height))

    run(body)


def test_explain_moved_to_e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """e still runs the VLM explain panel (? is now help)."""
    monkeypatch.setattr(
        "cull.tui.app.fetch_explanation_result",
        lambda request: ExplainResult(photo_path=request.image_path, explanation="fine"),
    )

    async def body() -> None:
        app = _app(tmp_path, numbered(2))
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await pilot.press("e")
            await wait_until(lambda: app.query_one(ExplainPanel).has_class("visible"))

    run(body)


def test_why_line_states_decision_and_reason(tmp_path: Path) -> None:
    """The why line names the AI decision and the score that routed it."""
    async def body() -> None:
        app = _app(tmp_path, [PhotoSpec(name="a.jpg", composite=0.88)])
        async with app.run_test(size=APP_SIZE):
            await _ready(app)
            text = str(app.query_one("#why-line").render())
            assert "UNCERTAIN · composite 0.88 (keep ≥0.94)" in text

    run(body)


def test_keypresses_do_no_disk_io_on_the_ui_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Once paths are warmed, navigating and deciding neither stats nor reads photos."""
    ui_reads: list[str] = []
    real_read_bytes = Path.read_bytes

    def tracking_read_bytes(path: Path) -> bytes:
        if threading.current_thread() is threading.main_thread():
            ui_reads.append(path.name)
        return real_read_bytes(path)

    async def body() -> None:
        app = _app(tmp_path, numbered(8))
        async with app.run_test(size=APP_SIZE) as pilot:
            await _ready(app)
            await wait_until(lambda: all(app._paths.cached(d) for d in app._session.decisions))

            def no_stat(*_args: object) -> None:
                raise AssertionError("path resolved on a keypress")

            monkeypatch.setattr("cull.tui.paths.resolve_on_disk", no_stat)
            monkeypatch.setattr(Path, "read_bytes", tracking_read_bytes)
            await pilot.press("right", "right", "x", "p", "left")

    run(body)
    assert ui_reads == []
