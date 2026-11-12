"""PhotoMeta.exif_datetime carries the capture time Stage 1 read."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from cull._pipeline.decision_assembly import _build_decision, _DecisionCtx
from cull._pipeline.stage1_runner import _Stage1Output
from cull.models import BlurScores, ExposureScores, Stage1Result

_CAPTURED = datetime(2026, 9, 26, 13, 20, 14)


def _stage1(path: Path, capture_time: datetime | None) -> Stage1Result:
    return Stage1Result(
        photo_path=path,
        blur=BlurScores(tenengrad=100.0, fft_ratio=0.5, blur_tier=1),
        exposure=ExposureScores(
            dr_score=0.8, clipping_highlight=0.01, clipping_shadow=0.05, midtone_pct=0.6, color_cast_score=0.02
        ),
        noise_score=0.1,
        capture_time=capture_time,
    )


def _decision_for(tmp_path: Path, capture_time: datetime | None):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    s1_out = _Stage1Output(results={str(photo): _stage1(photo, capture_time)})
    return _build_decision(photo, _DecisionCtx(paths=[photo], s1_out=s1_out))


def test_exif_datetime_is_the_stage1_capture_time(tmp_path: Path) -> None:
    assert _decision_for(tmp_path, _CAPTURED).photo.exif_datetime == _CAPTURED


def test_exif_datetime_stays_none_when_stage1_read_no_capture_time(tmp_path: Path) -> None:
    assert _decision_for(tmp_path, None).photo.exif_datetime is None
