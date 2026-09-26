"""Isolation for TUI tests: no real caches, override log, taste profile, or terminal writes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from cull.models import OverrideEntry


class TerminalCapture:
    """Collects escape data the app would have queued for the terminal."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    def __call__(self, _app: object, data: str) -> None:
        self.writes.append(data)

    @property
    def joined(self) -> str:
        """Return everything written, in order."""
        return "".join(self.writes)

    def clear(self) -> None:
        """Forget earlier writes."""
        self.writes.clear()


class OverrideLogCapture:
    """Stands in for the on-disk override log; the retrain counter lives in a temp file."""

    def __init__(self, profile_path: Path) -> None:
        self.entries: list[OverrideEntry] = []
        self.profile_path = profile_path

    def log(self, entry: OverrideEntry) -> None:
        self.entries.append(entry)

    def remove(self, entries: list[OverrideEntry]) -> int:
        doomed = {id(e) for e in entries}
        before = len(self.entries)
        self.entries = [e for e in self.entries if id(e) not in doomed]
        return before - len(self.entries)

    @property
    def retrain_counter(self) -> int:
        """Return the persisted retrain counter (0 when never written)."""
        counter = self.profile_path.with_suffix(self.profile_path.suffix + ".counter")
        return int(counter.read_text(encoding="utf-8")) if counter.exists() else 0


@pytest.fixture(autouse=True)
def terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[TerminalCapture]:
    """Capture terminal writes instead of queueing APC bytes for real stdout."""
    capture = TerminalCapture()
    monkeypatch.setattr("cull.tui.images.terminal_write", capture)
    monkeypatch.setenv("CULL_TUI_TRANSMIT", "file")
    yield capture


@pytest.fixture(autouse=True)
def preview_cache(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Point the preview cache at a per-test temp directory."""
    cache_dir = tmp_path_factory.mktemp("previews")
    monkeypatch.setattr("cull.tui.previews.PREVIEW_CACHE_DIR", cache_dir)
    return cache_dir


@pytest.fixture(autouse=True)
def override_log(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> OverrideLogCapture:
    """Keep TUI decisions out of ~/.cull: capture log writes; counter and profile go to temp."""
    capture = OverrideLogCapture(tmp_path_factory.mktemp("taste") / "taste_profile.joblib")
    monkeypatch.setattr("cull.tui.app.log_override", capture.log)
    monkeypatch.setattr("cull.tui.app.remove_overrides", capture.remove)
    monkeypatch.setattr("cull.tui.app.TASTE_PROFILE_PATH", capture.profile_path)
    monkeypatch.setattr("cull.taste_trainer.load_overrides", lambda: list(capture.entries))
    return capture
