"""Capture-time gating of duplicate and burst edges (moment window)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from cull.stage1 import duplicate
from cull.stage1.burst import split_by_moment_window

IMAGE_DIR = Path("/shoot")
T0 = datetime(2026, 9, 20, 18, 0, 0)
LINKED_SIM: float = 0.85
NEAR_IDENTICAL_SIM: float = 0.97


class _FakeCnn:
    """imagededup CNN stand-in with fixed encodings and duplicate map."""

    def __init__(self, names: list[str], cnn_map: dict[str, list[str]]) -> None:
        self._names = names
        self._cnn_map = cnn_map

    def encode_images(self, image_dir: str, recursive: bool) -> dict[str, np.ndarray]:
        """Return one dummy vector per name."""
        return {name: np.ones(4) for name in self._names}

    def find_duplicates(self, **_kwargs: object) -> dict:
        """Return the configured CNN duplicate map."""
        return self._cnn_map


def _groups(monkeypatch: pytest.MonkeyPatch, setup: dict) -> list[set[str]]:
    """Run find_duplicates with mocked models, times, and DINOv2 similarity."""
    names: list[str] = setup["names"]
    monkeypatch.setattr(duplicate, "_load_cnn", lambda: _FakeCnn(names, setup.get("cnn", {})))
    monkeypatch.setattr(duplicate, "read_exif_capture_time", lambda path: setup["times"][path.name])
    monkeypatch.setattr(duplicate, "select_device", lambda: "cpu")
    monkeypatch.setattr(duplicate, "_embed_dinov2_batch", MagicMock(return_value=np.zeros((len(names), 2))))
    monkeypatch.setattr(duplicate, "_cosine_similarity_matrix", lambda vectors: np.array(setup["sim"]))
    result = duplicate.find_duplicates(IMAGE_DIR)
    return [{p.name for p in g.paths} for g in result.duplicate_groups]


def _chain_sim(size: int) -> list[list[float]]:
    """Return a similarity matrix where every pair sits at LINKED_SIM."""
    sim = np.full((size, size), LINKED_SIM)
    np.fill_diagonal(sim, 1.0)
    return sim.tolist()


def test_time_window_blocks_chaining_across_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Similar photos two minutes apart form separate moments, not one chain."""
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    offsets = [0, 5, 120, 126]
    times = {n: T0 + timedelta(seconds=s) for n, s in zip(names, offsets)}
    groups = _groups(monkeypatch, {"names": names, "times": times, "sim": _chain_sim(4)})
    assert sorted(groups, key=sorted) == [{"a.jpg", "b.jpg"}, {"c.jpg", "d.jpg"}]


def test_cnn_pair_outside_window_does_not_link(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CNN near-identical pair with capture times minutes apart stays unlinked."""
    names = ["a.jpg", "b.jpg"]
    times = {"a.jpg": T0, "b.jpg": T0 + timedelta(minutes=3)}
    sim = [[1.0, 0.1], [0.1, 1.0]]
    setup = {"names": names, "times": times, "sim": sim, "cnn": {"a.jpg": ["b.jpg"]}}
    assert _groups(monkeypatch, setup) == []


def test_missing_timestamp_links_only_near_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without capture times, a 0.85 pair stays apart and a 0.97 pair links."""
    names = ["a.jpg", "b.jpg", "c.jpg"]
    times = {"a.jpg": None, "b.jpg": T0, "c.jpg": None}
    sim = [
        [1.0, LINKED_SIM, NEAR_IDENTICAL_SIM],
        [LINKED_SIM, 1.0, LINKED_SIM],
        [NEAR_IDENTICAL_SIM, LINKED_SIM, 1.0],
    ]
    groups = _groups(monkeypatch, {"names": names, "times": times, "sim": sim})
    assert groups == [{"a.jpg", "c.jpg"}]


def test_missing_timestamp_keeps_cnn_near_identical_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CNN pair (>= 0.98) links even when one photo has no capture time."""
    names = ["a.jpg", "b.jpg"]
    times = {"a.jpg": None, "b.jpg": T0}
    sim = [[1.0, 0.1], [0.1, 1.0]]
    setup = {"names": names, "times": times, "sim": sim, "cnn": {"a.jpg": ["b.jpg"]}}
    assert _groups(monkeypatch, setup) == [{"a.jpg", "b.jpg"}]


def test_long_burst_splits_at_the_moment_window() -> None:
    """A 1 s-cadence burst lasting 15 s splits into runs no longer than the window."""
    frames = [Path(f"/shoot/f{i:02d}.jpg") for i in range(16)]
    times = {p: T0 + timedelta(seconds=i) for i, p in enumerate(frames)}
    runs = split_by_moment_window(frames, times)
    assert runs == [frames[:11], frames[11:]]
