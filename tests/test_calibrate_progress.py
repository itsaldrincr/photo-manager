"""Tests for the calibration progress reporter and its wiring into run_calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cull.calibrate import CalibrationRequest, run_calibration
from cull.calibrate_progress import (
    NullProgress,
    PhaseStart,
    RichProgress,
    calibration_progress,
)

CORPUS_PHOTO_COUNT: int = 3
TEST_IMAGE_SIZE: tuple[int, int] = (32, 32)


class _RecordingProgress:
    """Test double that records all reporter events in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def start_phase(self, phase: PhaseStart) -> None:
        self.events.append(("start", phase))

    def advance(self) -> None:
        self.events.append(("advance", None))

    def end_phase(self) -> None:
        self.events.append(("end", None))


def _make_corpus(tmp_path: Path) -> Path:
    """Build a tiny corpus with manifest.json + N JPEGs at tmp_path; return tmp_path."""
    files = []
    for i in range(CORPUS_PHOTO_COUNT):
        path = tmp_path / f"img_{i:02d}.JPG"
        Image.new("RGB", TEST_IMAGE_SIZE, (i * 4, i * 4, i * 4)).save(path, "JPEG")
        files.append({
            "name": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "category": "real",
        })
    manifest = {"generated_at": "test", "count": CORPUS_PHOTO_COUNT, "files": files}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_null_progress_is_noop() -> None:
    """NullProgress methods return None and never raise."""
    null = NullProgress()
    null.start_phase(PhaseStart(label="x", total=10))
    null.advance()
    null.end_phase()


def test_calibration_progress_factory_yields_null_when_disabled() -> None:
    """calibration_progress(use_rich=False) yields a NullProgress."""
    with calibration_progress(use_rich=False) as progress:
        assert isinstance(progress, NullProgress)


def test_calibration_progress_factory_yields_rich_when_enabled() -> None:
    """calibration_progress(use_rich=True) yields a RichProgress."""
    with calibration_progress(use_rich=True) as progress:
        assert isinstance(progress, RichProgress)


def _fake_p1_scores() -> dict[str, dict[str, float]]:
    """Return canned per-photo p1 score map keyed by synthetic corpus filenames."""
    return {
        f"img_{i:02d}.JPG": {"topiq": 0.5, "laion_aesthetic": 0.4, "clipiqa": 0.6}
        for i in range(CORPUS_PHOTO_COUNT)
    }


def test_run_calibration_emits_progress_events_in_order(tmp_path: Path) -> None:
    """run_calibration emits start/end for p1, then start/advance×N/end for p4lite."""
    corpus = _make_corpus(tmp_path)
    recorder = _RecordingProgress()
    with patch("cull.calibrate._score_p1", return_value=_fake_p1_scores()), \
         patch("cull.calibrate._score_one_aesthetic", return_value=0.42), \
         patch("cull.calibrate._resolve_fixtures_dir", return_value=tmp_path):
        run_calibration(CalibrationRequest(corpus_dir=corpus), recorder)
    kinds = [e[0] for e in recorder.events]
    expected = ["start", "end", "start"] + ["advance"] * CORPUS_PHOTO_COUNT + ["end"]
    assert kinds == expected
