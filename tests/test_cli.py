"""Tests for CLI entry point and disk utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cull.cli import _build_config, _cull_pipeline_command
from cull.cli_pipeline import PostPipelineInput, _post_pipeline
from cull.config import CullConfig
from cull._pipeline.orchestrator import SessionResult, SessionSummary
from cull.disk import scan_jpegs


# ---------------------------------------------------------------------------
# scan_jpegs tests
# ---------------------------------------------------------------------------


def test_scan_jpegs_finds_all_jpeg_variants(tmp_path: Path) -> None:
    """scan_jpegs finds .jpg, .jpeg, .JPG, .JPEG files and excludes .png."""
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.jpeg").write_bytes(b"")
    (tmp_path / "c.JPG").write_bytes(b"")
    (tmp_path / "d.JPEG").write_bytes(b"")
    (tmp_path / "e.png").write_bytes(b"")
    result = scan_jpegs(tmp_path)
    names = {p.name for p in result}
    assert {"a.jpg", "b.jpeg", "c.JPG", "d.JPEG"} == names


def test_scan_jpegs_recursive(tmp_path: Path) -> None:
    """scan_jpegs finds JPEG files in nested subdirectories."""
    subdir = tmp_path / "sub" / "deep"
    subdir.mkdir(parents=True)
    (subdir / "nested.jpg").write_bytes(b"")
    (tmp_path / "top.jpeg").write_bytes(b"")
    result = scan_jpegs(tmp_path)
    names = {p.name for p in result}
    assert {"nested.jpg", "top.jpeg"} == names


# ---------------------------------------------------------------------------
# CLI flag tests
# ---------------------------------------------------------------------------


def test_dry_run_sets_is_dry_run(tmp_path: Path) -> None:
    """--dry-run flag propagates as is_dry_run=True in CullConfig."""
    runner = CliRunner()
    captured: list = []

    def fake_standard_pipeline(kwargs, config):  # type: ignore[no-untyped-def]
        captured.append(config)
        return None

    with patch("cull.cli._run_standard_pipeline", side_effect=fake_standard_pipeline):
        result = runner.invoke(_cull_pipeline_command, ["--dry-run", str(tmp_path)])
    assert result.exit_code == 0
    assert captured[0].is_dry_run is True


def test_no_vlm_removes_stage_3() -> None:
    """--no-vlm removes stage 3 from the stages list."""
    kwargs = {
        "stage": (),
        "no_vlm": True,
        "threshold": 0.65,
        "burst_gap": 0.5,
        "preset": "general",
        "model": "qwen3-vl-4b-q8",
        "portrait": False,
        "dry_run": False,
    }
    config = _build_config(kwargs)
    assert 3 not in config.stages


def test_help_exits_zero() -> None:
    """--help prints usage and exits with code 0."""
    runner = CliRunner()
    result = runner.invoke(_cull_pipeline_command, ["--help"])
    assert result.exit_code == 0
    assert "usage" in result.output.lower()


def test_review_after_requests_post_pipeline_handoff(tmp_path: Path) -> None:
    """--review-after should set explicit review handoff on post-pipeline input."""
    runner = CliRunner()
    captured: list[object] = []

    def fake_standard_pipeline(kwargs, config):  # type: ignore[no-untyped-def]
        captured.append(kwargs["review_after"])
        return None

    with patch("cull.cli._run_standard_pipeline", side_effect=fake_standard_pipeline):
        result = runner.invoke(_cull_pipeline_command, ["--review-after", str(tmp_path)])
    assert result.exit_code == 0
    assert captured == [True]


def test_post_pipeline_launches_review_only_when_requested() -> None:
    """Review handoff should be explicit rather than inferred from uncertain count."""
    result = SessionResult(
        source_path="/tmp/photos",
        total_photos=10,
        summary=SessionSummary(uncertain=5),
    )
    config = CullConfig()
    post_in = PostPipelineInput(
        result=result,
        config=config,
        report_flag=False,
        review_after=False,
    )
    with (
        patch("cull.cli_pipeline._show_results"),
        patch("cull.cli_pipeline._move_files"),
        patch("cull.cli_pipeline._launch_review_after") as launch_review,
    ):
        _post_pipeline(post_in)
    launch_review.assert_not_called()
