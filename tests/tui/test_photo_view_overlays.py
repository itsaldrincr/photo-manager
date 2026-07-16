"""Unit tests for PIL overlay drawing in photo_view."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from cull.models import CropProposal, GeometryScore
from cull.tui.photo_view import (
    RenderRequest,
    ViewportSize,
    _draw_crop_box,
    _draw_horizon,
    _prepare_png_for_request,
)

IMAGE_WIDTH: int = 200
IMAGE_HEIGHT: int = 150
VIEWPORT_COLS: int = 25
VIEWPORT_ROWS: int = 10

CROP_TOP: int = 20
CROP_LEFT: int = 30
CROP_BOTTOM: int = 120
CROP_RIGHT: int = 170


def _make_black_png() -> bytes:
    """Create a small solid-black PNG in memory."""
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_geometry(has_horizon: bool = True) -> GeometryScore:
    """Build a GeometryScore fixture."""
    return GeometryScore(
        tilt_degrees=5.0,
        keystone_degrees=0.0,
        confidence=0.9,
        has_horizon=has_horizon,
        has_verticals=False,
    )


def _make_crop() -> CropProposal:
    """Build a CropProposal fixture."""
    return CropProposal(
        top=CROP_TOP,
        left=CROP_LEFT,
        bottom=CROP_BOTTOM,
        right=CROP_RIGHT,
        source="smartcrop",
    )


def _render_and_open(geometry: GeometryScore | None, crop: CropProposal | None) -> Image.Image:
    """Build a RenderRequest, run the real overlay pipeline, return decoded PIL image."""
    png_bytes = _make_black_png()
    viewport = ViewportSize(cols=VIEWPORT_COLS, rows=VIEWPORT_ROWS)
    request = RenderRequest(
        image_id="test_img",
        image_bytes=png_bytes,
        viewport=viewport,
        geometry=geometry,
        crop=crop,
    )
    result_bytes = _prepare_png_for_request(request)
    return Image.open(BytesIO(result_bytes))


def test_horizon_overlay_draws_non_black_pixels() -> None:
    """Horizon line overlay produces non-black pixels on a black background."""
    geometry = _make_geometry(has_horizon=True)
    img = _render_and_open(geometry, None)
    pixels = list(img.getdata())
    has_non_black = any(p != (0, 0, 0) for p in pixels)
    assert has_non_black, "Expected horizon overlay pixels but found none"


def test_horizon_overlay_skipped_when_no_horizon() -> None:
    """Horizon overlay must NOT draw when has_horizon=False."""
    geometry = _make_geometry(has_horizon=False)
    img = _render_and_open(geometry, None)
    pixels = list(img.getdata())
    all_black = all(p == (0, 0, 0) for p in pixels)
    assert all_black, "No pixels should be drawn when has_horizon=False"


def test_crop_overlay_draws_non_black_pixels() -> None:
    """Crop box overlay produces non-black pixels on a black background."""
    crop = _make_crop()
    img = _render_and_open(None, crop)
    pixels = list(img.getdata())
    has_non_black = any(p != (0, 0, 0) for p in pixels)
    assert has_non_black, "Expected crop box overlay pixels but found none"


def test_render_request_optional_fields_default_none() -> None:
    """RenderRequest geometry and crop default to None."""
    viewport = ViewportSize(cols=VIEWPORT_COLS, rows=VIEWPORT_ROWS)
    req = RenderRequest(image_id="x", image_bytes=b"", viewport=viewport)
    assert req.geometry is None
    assert req.crop is None


def test_draw_horizon_helper_marks_pixels_directly() -> None:
    """_draw_horizon marks non-black pixels on a fresh black image."""
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))
    geometry = _make_geometry(has_horizon=True)
    _draw_horizon(img, geometry)
    pixels = list(img.getdata())
    has_non_black = any(p != (0, 0, 0) for p in pixels)
    assert has_non_black


def test_draw_crop_box_helper_marks_border_pixels() -> None:
    """_draw_crop_box marks pixels on the top edge of the bounding box."""
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))
    crop = _make_crop()
    _draw_crop_box(img, crop)
    top_row_pixels = [img.getpixel((x, CROP_TOP)) for x in range(CROP_LEFT, CROP_RIGHT + 1)]
    has_non_black = any(p != (0, 0, 0) for p in top_row_pixels)
    assert has_non_black, "Top edge of crop box should have non-black pixels"
