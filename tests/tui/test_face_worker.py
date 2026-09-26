"""The face worker runs detached from the tty, and its line protocol."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cull.tui import face_worker, faces
from cull.tui.faces import FaceReport, FaceWorkerClient


class _FakeProcess:
    """Stands in for a running worker."""

    def poll(self) -> None:
        return None


def test_worker_process_cannot_touch_the_tty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Own session, pipes for stdin/stdout, stderr to a log file, native logs quietened."""
    launched: dict[str, object] = {}

    def fake_popen(args: list[str], **kwargs: object) -> _FakeProcess:
        launched.update(kwargs, args=args)
        return _FakeProcess()

    monkeypatch.setattr(faces, "NATIVE_LOG_PATH", tmp_path / "native.log")
    monkeypatch.setattr(faces.subprocess, "Popen", fake_popen)
    FaceWorkerClient()._ensure_started()

    assert launched["start_new_session"] is True
    assert launched["stdin"] == subprocess.PIPE and launched["stdout"] == subprocess.PIPE
    assert Path(launched["stderr"].name) == tmp_path / "native.log"
    assert launched["env"]["GLOG_minloglevel"] == "2"
    assert launched["env"]["TF_CPP_MIN_LOG_LEVEL"] == "3"
    assert launched["args"][-1] == "cull.tui.face_worker"


def test_worker_answers_one_line_per_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A request line yields one FaceReport JSON line; a failure is reported, not raised."""
    def boom(source: Path) -> FaceReport:
        raise RuntimeError("model missing")

    monkeypatch.setattr(faces, "analyse_faces", boom)
    request = json.dumps({"source": str(tmp_path / "a.jpg"), "cache_dir": str(tmp_path)})
    reply = FaceReport.model_validate_json(face_worker.handle_line(request))
    assert reply.error == "model missing"
    assert reply.source == tmp_path / "a.jpg"


def test_brightness_normalisation_lifts_dark_frames() -> None:
    """CLAHE on luminance brightens a dark, low-contrast frame before detection."""
    rng = np.random.default_rng(0)
    dark = (rng.random((64, 64, 3)) * 30).astype(np.uint8)
    assert faces.normalise_brightness(dark).mean() > dark.mean() * 1.5


def test_tiles_cover_the_frame_at_several_scales() -> None:
    """Whole frame first, then overlapping grids that reach every edge."""
    tiles = faces.tiles_for((1000, 1500, 3))
    assert tiles[0] == (0, 0, 1500, 1000)
    assert len(tiles) == sum(grid * grid for grid in faces.TILE_GRIDS)
    for tile in tiles:
        assert tile.left >= 0 and tile.top >= 0
        assert tile.left + tile.width <= 1500 and tile.top + tile.height <= 1000
    assert max(t.left + t.width for t in tiles[1:5]) == 1500


def test_same_face_from_two_tiles_counts_once() -> None:
    """Overlapping detections of one face dedupe by box overlap; distinct faces do not."""
    face = [faces._Point(0.10, 0.10), faces._Point(0.20, 0.25)]
    shifted = [faces._Point(0.11, 0.10), faces._Point(0.21, 0.25)]
    elsewhere = [faces._Point(0.60, 0.60), faces._Point(0.70, 0.75)]
    assert faces._overlap(face, shifted) > faces.DUPLICATE_FACE_IOU
    assert faces._overlap(face, elsewhere) == 0.0
