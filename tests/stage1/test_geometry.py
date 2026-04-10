"""Tests for cull.stage1.geometry — synthetic tilted-line ground truth."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

from cull.stage1.geometry import GeometryResult, assess_geometry

# ---------------------------------------------------------------------------
# Synthetic image constants
# ---------------------------------------------------------------------------

CANVAS_SIZE: int = 512
LINE_COUNT: int = 8
LINE_LENGTH: int = 400
LINE_THICKNESS: int = 3
LINE_COLOR: int = 0
BACKGROUND_COLOR: int = 255
LINE_SPACING: int = 40
TILT_GROUND_TRUTH_DEG: float = 5.0
TILT_TOLERANCE_DEG: float = 1.0


# ---------------------------------------------------------------------------
# Helpers (≤2 params, ≤20 LOC)
# ---------------------------------------------------------------------------


def _draw_tilted_line(draw: ImageDraw.ImageDraw, y_offset: int) -> None:
    """Draw a single tilted line at y_offset on the canvas."""
    radians = math.radians(TILT_GROUND_TRUTH_DEG)
    dx = LINE_LENGTH
    dy = int(LINE_LENGTH * math.tan(radians))
    x_start = (CANVAS_SIZE - LINE_LENGTH) // 2
    y_start = y_offset
    draw.line(
        ((x_start, y_start), (x_start + dx, y_start + dy)),
        fill=LINE_COLOR,
        width=LINE_THICKNESS,
    )


def _make_tilted_image(out_path: Path) -> None:
    """Write a grayscale PNG containing LINE_COUNT parallel tilted lines."""
    img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    base_y = (CANVAS_SIZE - LINE_COUNT * LINE_SPACING) // 2
    for index in range(LINE_COUNT):
        _draw_tilted_line(draw, base_y + index * LINE_SPACING)
    img.save(out_path)


def _make_blank_image(out_path: Path) -> None:
    """Write a blank grayscale PNG with no detectable lines."""
    img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), BACKGROUND_COLOR)
    img.save(out_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tilted_lines_recover_ground_truth(tmp_path: Path) -> None:
    """assess_geometry must recover the drawn 5-degree tilt within 1 degree."""
    image_path = tmp_path / "tilted.png"
    _make_tilted_image(image_path)

    result = assess_geometry(image_path)

    assert isinstance(result, GeometryResult)
    assert result.scores.has_horizon is True
    assert abs(result.scores.tilt_degrees - TILT_GROUND_TRUTH_DEG) <= TILT_TOLERANCE_DEG


def test_blank_image_returns_zero_confidence(tmp_path: Path) -> None:
    """An empty/blank image must yield confidence=0 and has_horizon=False."""
    image_path = tmp_path / "blank.png"
    _make_blank_image(image_path)

    result = assess_geometry(image_path)

    assert result.scores.confidence == 0.0
    assert result.scores.has_horizon is False
    assert result.scores.has_verticals is False
    assert result.scores.tilt_degrees == 0.0
    assert result.scores.keystone_degrees == 0.0


def test_missing_file_returns_empty_scores(tmp_path: Path) -> None:
    """A non-existent path must not raise — returns empty scores."""
    missing = tmp_path / "does_not_exist.png"

    result = assess_geometry(missing)

    assert result.scores.confidence == 0.0
    assert result.scores.has_horizon is False
