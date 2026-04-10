"""CLI hook tests for `cull --fast`.

Verifies the Pattern B wiring in `cull.cli`:
- `cull_fast` is NOT imported at `cull.cli` module load time (lazy import).
- `--fast` combined with any subcommand flag (e.g. `--search`) is refused.
- `--fast` dispatches to `cull_fast.cli_hook.run_fast_pipeline` exactly once.

Authored for inspection. Tests are NOT executed against real ML models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from cull.cli import main
from cull.pipeline import SessionResult

GRADIENT_SIZE: int = 64
JPEG_PIXEL_R: int = 100
JPEG_PIXEL_G: int = 150
JPEG_PIXEL_B: int = 200


def _write_synthetic_jpeg(path: Path) -> None:
    """Write a single 64x64 solid JPEG to path."""
    img = Image.new("RGB", (GRADIENT_SIZE, GRADIENT_SIZE), (JPEG_PIXEL_R, JPEG_PIXEL_G, JPEG_PIXEL_B))
    img.save(path, "JPEG")


def test_lazy_import() -> None:
    """Importing cull.cli must NOT pull cull_fast into sys.modules."""
    sys.modules.pop("cull_fast", None)
    sys.modules.pop("cull_fast.cli_hook", None)
    import cull.cli  # noqa: F401, PLC0415

    assert "cull_fast" not in sys.modules
    assert "cull_fast.cli_hook" not in sys.modules


def test_fast_search_conflict(tmp_path: Path) -> None:
    """`--fast --search` must exit non-zero with a conflict message."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["--fast", "--search", "test query", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


def test_fast_dispatches_to_cli_hook(
    tmp_path: Path,
    mock_scorers: None,
    mock_musiq_scorers: None,
) -> None:
    """`--fast SOURCE` must call run_fast_pipeline exactly once."""
    _write_synthetic_jpeg(tmp_path / "img_001.jpg")
    fake_result = MagicMock(spec=SessionResult)
    with patch("cull_fast.cli_hook.run_fast_pipeline", return_value=fake_result) as mock_run:
        with patch("cull.cli._post_pipeline") as mock_post:
            runner = CliRunner()
            runner.invoke(main, ["--fast", str(tmp_path)])
    assert mock_run.call_count == 1
    assert mock_post.call_count == 1
