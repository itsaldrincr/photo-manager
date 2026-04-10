"""End-to-end fast pipeline tests with mocked scorers.

All scoring goes through mock_scorers + mock_musiq_scorers — never pyiqa.
Corpus is PIL-generated 64x64 gradient JPEGs written to tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from cull.config import GENRE_WEIGHTS, CullConfig
from cull.pipeline import _PipelineRunInput, SessionResult
from cull_fast.cli_hook import run_fast_pipeline

ALL_PRESETS = (
    "general",
    "wedding",
    "documentary",
    "wildlife",
    "landscape",
    "street",
    "holiday",
)

GRADIENT_IMAGE_SIZE: int = 64
CORPUS_FIVE_COUNT: int = 5
CORPUS_THREE_COUNT: int = 3
CORPUS_TWO_COUNT: int = 2
COLOR_CHANNEL_RANGE: int = 256
COLOR_SHIFT_THIRD: int = 85
COLOR_SHIFT_TWO_THIRDS: int = 170


def _write_gradient_jpeg(path: Path, index: int) -> None:
    """Write a 64x64 gradient JPEG to path, varied by index."""
    img = Image.new("RGB", (GRADIENT_IMAGE_SIZE, GRADIENT_IMAGE_SIZE))
    pixels = img.load()
    for row in range(GRADIENT_IMAGE_SIZE):
        for col in range(GRADIENT_IMAGE_SIZE):
            value = (row * index + col) % COLOR_CHANNEL_RANGE
            pixels[col, row] = (
                value,
                (value + COLOR_SHIFT_THIRD) % COLOR_CHANNEL_RANGE,
                (value + COLOR_SHIFT_TWO_THIRDS) % COLOR_CHANNEL_RANGE,
            )
    img.save(path, "JPEG")


def _build_corpus(tmp_path: Path, count: int) -> None:
    """Write `count` gradient JPEG files into tmp_path."""
    for idx in range(count):
        _write_gradient_jpeg(tmp_path / f"photo_{idx:03d}.jpg", idx + 1)


def _make_config(preset: str = "general", is_portrait: bool = False) -> CullConfig:
    """Build a CullConfig with stage 2 only (no VLM) to keep tests fast."""
    return CullConfig(preset=preset, is_portrait=is_portrait, stages=[1, 2])


def _make_run_input(config: CullConfig, source_path: Path) -> _PipelineRunInput:
    """Build _PipelineRunInput from config and source path."""
    return _PipelineRunInput(config=config, source_path=source_path)


def test_run_fast_pipeline_produces_session_result(
    tmp_path: Path,
    mock_scorers: None,
    mock_musiq_scorers: None,
) -> None:
    """run_fast_pipeline returns a SessionResult with one decision per photo."""
    _build_corpus(tmp_path, CORPUS_FIVE_COUNT)
    config = _make_config()
    run_in = _make_run_input(config, tmp_path)
    result = run_fast_pipeline(run_in)
    assert isinstance(result, SessionResult)
    assert len(result.decisions) == CORPUS_FIVE_COUNT


def test_unload_musiq_called_before_s3(
    tmp_path: Path,
    mock_scorers: None,
    mock_musiq_scorers: None,
) -> None:
    """unload_musiq must be called before _run_s3 when STAGE_VLM is active."""
    _build_corpus(tmp_path, CORPUS_THREE_COUNT)
    config = CullConfig(preset="general", is_portrait=False, stages=[1, 2, 3])
    run_in = _make_run_input(config, tmp_path)
    calls: list[str] = []
    unload_mock = MagicMock(side_effect=lambda: calls.append("unload"))
    s3_mock = MagicMock(side_effect=lambda _in: calls.append("s3") or {})

    with (
        patch("cull_fast.pipeline_fast.unload_musiq", unload_mock),
        patch("cull_fast.cli_hook._run_s3", s3_mock),
    ):
        run_fast_pipeline(run_in)

    assert "unload" in calls
    assert "s3" in calls
    assert calls.index("unload") < calls.index("s3")


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_fast_mode_runs_all_presets(
    preset: str,
    tmp_path: Path,
    mock_scorers: None,
    mock_musiq_scorers: None,
) -> None:
    """Fast pipeline returns a valid SessionResult for every shipped preset."""
    _build_corpus(tmp_path, CORPUS_THREE_COUNT)
    config = _make_config(preset=preset)
    run_in = _make_run_input(config, tmp_path)
    result = run_fast_pipeline(run_in)
    assert isinstance(result, SessionResult)
    assert result.preset == preset


def test_fast_mode_respects_is_portrait(
    tmp_path: Path,
    mock_scorers: None,
    mock_musiq_scorers: None,
) -> None:
    """_run_portrait_if_needed is called at least once per photo when is_portrait=True."""
    _build_corpus(tmp_path, CORPUS_TWO_COUNT)
    config = _make_config(is_portrait=True)
    run_in = _make_run_input(config, tmp_path)
    portrait_mock = MagicMock()

    with patch("cull_fast.pipeline_fast._run_portrait_if_needed", portrait_mock):
        run_fast_pipeline(run_in)

    assert portrait_mock.call_count >= CORPUS_TWO_COUNT
