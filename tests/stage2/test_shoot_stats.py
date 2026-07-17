"""Tests for cull.stage2.shoot_stats — outlier detection and scene boundary maths."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from cull.config import BURST_GAP_DEFAULT_SECONDS
from cull.models import (
    BlurScores,
    ExifSummary,
    ExposureScores,
    Stage1Result,
    Stage2Result,
)
from cull.stage2.shoot_stats import ShootStatsInput, compute

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INLIER_COUNT: int = 9
OUTLIER_INDEX: int = 5
TOTAL_PHOTOS: int = INLIER_COUNT + 1
INLIER_LAB_VALUE: float = 50.0
OUTLIER_LAB_VALUE: float = 200.0
INLIER_DR_SCORE: float = 0.6
OUTLIER_DR_SCORE: float = 0.9
INLIER_EXIF_ISO: int = 200
INLIER_EXIF_SHUTTER: str = "1/200"
INLIER_EXIF_APERTURE: str = "f/2.8"
INLIER_EXIF_FOCAL: float = 50.0
OUTLIER_EXIF_ISO: int = 6400
OUTLIER_EXIF_SHUTTER: str = "1/30"
OUTLIER_EXIF_APERTURE: str = "f/16"
OUTLIER_EXIF_FOCAL: float = 200.0
OUTLIER_THRESHOLD: float = 0.5
INLIER_MAX_SCORE: float = 0.2
SCENE_GAP_INDEX: int = 6
SCENE_GAP_MULTIPLIER: float = 5.0
BASE_TIMESTAMP: datetime = datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc)
BASE_BLUR_TENENGRAD: float = 100.0
BASE_FFT_RATIO: float = 0.5
BASE_NOISE_SCORE: float = 0.1


# ---------------------------------------------------------------------------
# Mock helpers (no real ML)
# ---------------------------------------------------------------------------


class _MockStage2(BaseModel):
    """Stage2Result-shaped object carrying a synthetic palette_lab attribute."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    photo_path: Path
    palette_lab: list[float]


class _MockExif(BaseModel):
    """Synthetic EXIF payload exposed via getattr in shoot_stats."""

    iso: int
    shutter: str
    aperture: str
    focal_length_mm: float


class _MockStage1(BaseModel):
    """Stage1Result-shaped object with exif + capture_time attributes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    photo_path: Path
    exposure: ExposureScores
    blur: BlurScores
    noise_score: float
    exif: _MockExif
    capture_time: datetime
    is_pass: bool = True


def _make_inlier_blur() -> BlurScores:
    """Build a baseline BlurScores instance shared by every photo in the test."""
    return BlurScores(
        tenengrad=BASE_BLUR_TENENGRAD,
        fft_ratio=BASE_FFT_RATIO,
        blur_tier=1,
    )


def _make_inlier_exposure(dr_score: float) -> ExposureScores:
    """Build an ExposureScores with the given dynamic-range scalar."""
    return ExposureScores(
        dr_score=dr_score,
        clipping_highlight=0.0,
        clipping_shadow=0.0,
        midtone_pct=0.5,
        color_cast_score=0.0,
    )


def _make_inlier_exif() -> _MockExif:
    """Return the EXIF payload shared by every inlier photo."""
    return _MockExif(
        iso=INLIER_EXIF_ISO,
        shutter=INLIER_EXIF_SHUTTER,
        aperture=INLIER_EXIF_APERTURE,
        focal_length_mm=INLIER_EXIF_FOCAL,
    )


def _make_outlier_exif() -> _MockExif:
    """Return the engineered outlier EXIF payload."""
    return _MockExif(
        iso=OUTLIER_EXIF_ISO,
        shutter=OUTLIER_EXIF_SHUTTER,
        aperture=OUTLIER_EXIF_APERTURE,
        focal_length_mm=OUTLIER_EXIF_FOCAL,
    )


def _capture_for_index(index: int) -> datetime:
    """Return capture_time for index — burst-rate cadence with one large gap."""
    if index < SCENE_GAP_INDEX:
        return BASE_TIMESTAMP + timedelta(seconds=index * BURST_GAP_DEFAULT_SECONDS)
    return BASE_TIMESTAMP + timedelta(
        seconds=index * BURST_GAP_DEFAULT_SECONDS
        + SCENE_GAP_MULTIPLIER * BURST_GAP_DEFAULT_SECONDS * SCENE_GAP_INDEX
    )


def _build_inlier_pair(index: int, tmp_path: Path) -> tuple[Any, Any]:
    """Build a (stage1, stage2) pair for an inlier photo at the given index."""
    path = tmp_path / f"photo_{index:03d}.jpg"
    s2 = _MockStage2(
        photo_path=path,
        palette_lab=[INLIER_LAB_VALUE, INLIER_LAB_VALUE, INLIER_LAB_VALUE],
    )
    s1 = _MockStage1(
        photo_path=path,
        exposure=_make_inlier_exposure(INLIER_DR_SCORE),
        blur=_make_inlier_blur(),
        noise_score=BASE_NOISE_SCORE,
        exif=_make_inlier_exif(),
        capture_time=_capture_for_index(index),
    )
    return s1, s2


def _build_outlier_pair(index: int, tmp_path: Path) -> tuple[Any, Any]:
    """Build a (stage1, stage2) pair for the engineered outlier photo."""
    path = tmp_path / f"photo_{index:03d}.jpg"
    s2 = _MockStage2(
        photo_path=path,
        palette_lab=[OUTLIER_LAB_VALUE, OUTLIER_LAB_VALUE, OUTLIER_LAB_VALUE],
    )
    s1 = _MockStage1(
        photo_path=path,
        exposure=_make_inlier_exposure(OUTLIER_DR_SCORE),
        blur=_make_inlier_blur(),
        noise_score=BASE_NOISE_SCORE,
        exif=_make_outlier_exif(),
        capture_time=_capture_for_index(index),
    )
    return s1, s2


def _build_corpus(tmp_path: Path) -> ShootStatsInput:
    """Synthesise 10 photos: 9 inliers and one engineered outlier at OUTLIER_INDEX."""
    stage1_results: list[Any] = []
    stage2_results: list[Any] = []
    for i in range(TOTAL_PHOTOS):
        if i == OUTLIER_INDEX:
            s1, s2 = _build_outlier_pair(i, tmp_path)
        else:
            s1, s2 = _build_inlier_pair(i, tmp_path)
        stage1_results.append(s1)
        stage2_results.append(s2)
    return ShootStatsInput(stage1_results=stage1_results, stage2_results=stage2_results)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _outlier_path_str(input_in: ShootStatsInput) -> str:
    """Return the str(photo_path) of the engineered outlier in the corpus."""
    return str(input_in.stage2_results[OUTLIER_INDEX].photo_path)


def test_outlier_scores_exceed_threshold(tmp_path: Path) -> None:
    """The engineered outlier must score > 0.5 on palette / exposure / EXIF axes."""
    input_in = _build_corpus(tmp_path)
    scores = compute(input_in)
    outlier = scores[_outlier_path_str(input_in)]
    assert outlier.palette_outlier_score > OUTLIER_THRESHOLD, outlier.palette_outlier_score
    assert outlier.exposure_drift_score > OUTLIER_THRESHOLD, outlier.exposure_drift_score
    assert outlier.exif_anomaly_score > OUTLIER_THRESHOLD, outlier.exif_anomaly_score


def test_inliers_score_below_threshold(tmp_path: Path) -> None:
    """Every inlier (non-outlier index) must score below the inlier max on all axes."""
    input_in = _build_corpus(tmp_path)
    scores = compute(input_in)
    for i in range(TOTAL_PHOTOS):
        if i == OUTLIER_INDEX:
            continue
        score = scores[str(input_in.stage2_results[i].photo_path)]
        assert score.palette_outlier_score < INLIER_MAX_SCORE, (i, score.palette_outlier_score)
        assert score.exposure_drift_score < INLIER_MAX_SCORE, (i, score.exposure_drift_score)
        assert score.exif_anomaly_score < INLIER_MAX_SCORE, (i, score.exif_anomaly_score)


def test_scene_boundary_detected_at_burst_gap(tmp_path: Path) -> None:
    """A capture-time gap larger than the burst threshold must produce a scene boundary."""
    input_in = _build_corpus(tmp_path)
    scores = compute(input_in)
    boundary_path = str(input_in.stage2_results[SCENE_GAP_INDEX].photo_path)
    assert scores[boundary_path].scene_start_bonus > 0.0


# ---------------------------------------------------------------------------
# Real-model wiring tests — proves the signals fire on the actual production
# Stage1Result / Stage2Result models, not just on duck-typed mocks that were
# already shaped like the fields the reducer expected (which is exactly how
# these three signals stayed dead: mocks passed, real models had no such
# fields, and getattr() fell back silently).
# ---------------------------------------------------------------------------


def _real_inlier_pair(index: int, tmp_path: Path) -> tuple[Stage1Result, Stage2Result]:
    """Build a real Stage1Result/Stage2Result inlier pair (production model types)."""
    path = tmp_path / f"real_{index:03d}.jpg"
    s1 = Stage1Result(
        photo_path=path,
        blur=_make_inlier_blur(),
        exposure=_make_inlier_exposure(INLIER_DR_SCORE),
        noise_score=BASE_NOISE_SCORE,
        exif=ExifSummary(
            iso=INLIER_EXIF_ISO,
            shutter=INLIER_EXIF_SHUTTER,
            aperture=INLIER_EXIF_APERTURE,
            focal_length_mm=INLIER_EXIF_FOCAL,
        ),
        capture_time=_capture_for_index(index),
    )
    s2 = Stage2Result(
        photo_path=path,
        topiq=0.5,
        laion_aesthetic=0.5,
        clipiqa=0.5,
        composite=0.5,
        palette_lab=(INLIER_LAB_VALUE, INLIER_LAB_VALUE, INLIER_LAB_VALUE),
    )
    return s1, s2


def _real_outlier_pair(index: int, tmp_path: Path) -> tuple[Stage1Result, Stage2Result]:
    """Build a real Stage1Result/Stage2Result outlier pair (production model types)."""
    path = tmp_path / f"real_{index:03d}.jpg"
    s1 = Stage1Result(
        photo_path=path,
        blur=_make_inlier_blur(),
        exposure=_make_inlier_exposure(OUTLIER_DR_SCORE),
        noise_score=BASE_NOISE_SCORE,
        exif=ExifSummary(
            iso=OUTLIER_EXIF_ISO,
            shutter=OUTLIER_EXIF_SHUTTER,
            aperture=OUTLIER_EXIF_APERTURE,
            focal_length_mm=OUTLIER_EXIF_FOCAL,
        ),
        capture_time=_capture_for_index(index),
    )
    s2 = Stage2Result(
        photo_path=path,
        topiq=0.5,
        laion_aesthetic=0.5,
        clipiqa=0.5,
        composite=0.5,
        palette_lab=(OUTLIER_LAB_VALUE, OUTLIER_LAB_VALUE, OUTLIER_LAB_VALUE),
    )
    return s1, s2


def _build_real_model_corpus(tmp_path: Path) -> ShootStatsInput:
    """Synthesise a corpus using real Stage1Result/Stage2Result (not duck-typed mocks)."""
    stage1_results: list[Stage1Result] = []
    stage2_results: list[Stage2Result] = []
    for i in range(TOTAL_PHOTOS):
        s1, s2 = _real_outlier_pair(i, tmp_path) if i == OUTLIER_INDEX else _real_inlier_pair(i, tmp_path)
        stage1_results.append(s1)
        stage2_results.append(s2)
    return ShootStatsInput(stage1_results=stage1_results, stage2_results=stage2_results)


def test_real_models_palette_outlier_fires(tmp_path: Path) -> None:
    """A real Stage2Result carrying palette_lab must produce a firing palette_outlier_score."""
    input_in = _build_real_model_corpus(tmp_path)
    scores = compute(input_in)
    outlier_path = str(input_in.stage2_results[OUTLIER_INDEX].photo_path)
    assert scores[outlier_path].palette_outlier_score > OUTLIER_THRESHOLD


def test_real_models_exif_anomaly_fires(tmp_path: Path) -> None:
    """A real Stage1Result carrying exif must produce a firing exif_anomaly_score."""
    input_in = _build_real_model_corpus(tmp_path)
    scores = compute(input_in)
    outlier_path = str(input_in.stage2_results[OUTLIER_INDEX].photo_path)
    assert scores[outlier_path].exif_anomaly_score > OUTLIER_THRESHOLD


def test_real_models_scene_start_bonus_fires(tmp_path: Path) -> None:
    """A real Stage1Result carrying capture_time must produce a scene boundary."""
    input_in = _build_real_model_corpus(tmp_path)
    scores = compute(input_in)
    boundary_path = str(input_in.stage2_results[SCENE_GAP_INDEX].photo_path)
    assert scores[boundary_path].scene_start_bonus > 0.0


def test_real_models_without_exif_degrade_gracefully(tmp_path: Path) -> None:
    """A photo missing EXIF (exif=None, capture_time=None) must not raise or crash."""
    path = tmp_path / "no_exif.jpg"
    s1 = Stage1Result(
        photo_path=path,
        blur=_make_inlier_blur(),
        exposure=_make_inlier_exposure(INLIER_DR_SCORE),
        noise_score=BASE_NOISE_SCORE,
    )
    s2 = Stage2Result(
        photo_path=path, topiq=0.5, laion_aesthetic=0.5, clipiqa=0.5, composite=0.5,
    )
    scores = compute(ShootStatsInput(stage1_results=[s1], stage2_results=[s2]))
    score = scores[str(path)]
    assert score.exif_anomaly_score == 0.0
    assert score.scene_id == 0
