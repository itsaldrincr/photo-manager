"""Shared PIL decode for model input."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def open_rgb_upright(path: Path) -> Image.Image:
    """Decode path as RGB with its EXIF Orientation applied.

    cv2.imread honours EXIF orientation and PIL does not, so without this the
    PIL-fed models see portrait-orientation frames sideways while the cv2
    stages see them upright.
    """
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")
