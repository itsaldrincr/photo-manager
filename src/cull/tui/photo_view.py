"""Main photo widget, plus PIL overlay helpers for horizon and crop proposals.

Display goes through ``ImageCells`` (kitty Unicode placeholders); see
``cull.tui.images``. Requires a kitty-graphics terminal (Kitty, Ghostty,
WezTerm) with Unicode-placeholder support.
"""

from __future__ import annotations

import logging
import math
from io import BytesIO

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict

from cull.config import (
    OVERLAY_CROP_COLOR,
    OVERLAY_HORIZON_COLOR,
    OVERLAY_LABEL_OFFSET_PX,
    OVERLAY_LINE_THICKNESS,
)
from cull.models import CropProposal, GeometryScore
from cull.tui.images import ImageCells, ShownImage

logger = logging.getLogger(__name__)

OVERLAY_TARGET_LONG_EDGE: int = 1024
PNG_COMPRESS_LEVEL: int = 1


class ViewportSize(BaseModel):
    """Terminal viewport dimensions in columns and rows."""

    cols: int
    rows: int


class RenderRequest(BaseModel):
    """Input bundle for rendering a photo with optional overlays."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_id: str
    image_bytes: bytes
    viewport: ViewportSize
    geometry: GeometryScore | None = None
    crop: CropProposal | None = None


def _pil_from_bytes(image_bytes: bytes) -> Image.Image:
    """Decode upright (EXIF transpose) and reduce via JPEG draft + thumbnail."""
    img = Image.open(BytesIO(image_bytes))
    target = (OVERLAY_TARGET_LONG_EDGE, OVERLAY_TARGET_LONG_EDGE)
    img.draft("RGB", target)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail(target, Image.Resampling.LANCZOS)
    return img


def _pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    """Encode a PIL image as PNG with fast compression and return the bytes."""
    buf = BytesIO()
    pil_img.save(buf, format="PNG", compress_level=PNG_COMPRESS_LEVEL, optimize=False)
    return buf.getvalue()


def _draw_horizon(img: Image.Image, geometry: GeometryScore) -> None:
    """Draw a horizon line and tilt label onto a PIL image in place."""
    from PIL import ImageDraw  # noqa: PLC0415

    if not geometry.has_horizon:
        return
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx, cy = w // 2, h // 2
    rad = math.radians(geometry.tilt_degrees)
    dx = int(cx * math.cos(rad))
    dy = int(cx * math.sin(rad))
    x0, y0 = cx - dx, cy - dy
    x1, y1 = cx + dx, cy + dy
    draw.line([(x0, y0), (x1, y1)], fill=OVERLAY_HORIZON_COLOR, width=OVERLAY_LINE_THICKNESS)
    label = f"{geometry.tilt_degrees:+.1f}°"
    draw.text((x0 + OVERLAY_LABEL_OFFSET_PX, y0 - OVERLAY_LABEL_OFFSET_PX - 10), label, fill=OVERLAY_HORIZON_COLOR)


def _draw_crop_box(img: Image.Image, crop: CropProposal) -> None:
    """Draw a crop bounding box onto a PIL image in place."""
    from PIL import ImageDraw  # noqa: PLC0415

    draw = ImageDraw.Draw(img)
    coords = [(crop.left, crop.top), (crop.right, crop.bottom)]
    draw.rectangle(coords, outline=OVERLAY_CROP_COLOR, width=OVERLAY_LINE_THICKNESS)


def _apply_overlays(img: Image.Image, request: RenderRequest) -> None:
    """Apply horizon and crop overlays onto the PIL image in place."""
    if request.geometry is not None:
        _draw_horizon(img, request.geometry)
    if request.crop is not None:
        _draw_crop_box(img, request.crop)


def _prepare_png_for_request(request: RenderRequest) -> bytes:
    """Decode, overlay, PNG-encode; return fresh PNG bytes."""
    pil_img = _pil_from_bytes(request.image_bytes)
    _apply_overlays(pil_img, request)
    return _pil_to_png_bytes(pil_img)


class PhotoView(ImageCells):
    """The large photo; logs keypress-to-placed latency when CULL_TUI_DEBUG is set."""

    DEFAULT_CSS = """
    PhotoView {
        height: 1fr;
        width: 1fr;
    }
    """

    def on_image_shown(self, shown: ShownImage) -> None:
        """Record latency once the frame holding the new placeholder cells has painted."""
        timer = getattr(self.app, "latency", None)
        if timer is None:
            return
        label = f"image={shown.image_id} {self.source.name if self.source else ''}"
        self.call_after_refresh(timer.finish, label)
