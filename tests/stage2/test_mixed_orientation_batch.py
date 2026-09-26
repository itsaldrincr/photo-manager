"""A Stage 2 batch mixing portrait and landscape shots must still stack.

Regression: once EXIF orientation was applied, an upright 853x1280 portrait
next to a 1280x853 landscape made torch.cat fail and crashed a 488-photo run.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from cull.config import SHARED_DECODE_PIXEL_PX
from cull.pipeline import _DualLoadInput, _load_dual_pil_batch, _load_tensor_batch

ORIENTATION_TAG: int = 0x0112
ROTATE_90_CW: int = 6


def _jpeg(path: Path, size: tuple[int, int]) -> Path:
    Image.new("RGB", size, color=(40, 90, 160)).save(str(path), format="JPEG")
    return path


def _portrait_jpeg(path: Path) -> Path:
    """Landscape pixels tagged to display rotated, as a camera writes a vertical shot."""
    exif = Image.Exif()
    exif[ORIENTATION_TAG] = ROTATE_90_CW
    Image.new("RGB", (3000, 2000), color=(40, 90, 160)).save(str(path), format="JPEG", exif=exif.tobytes())
    return path


def test_portrait_and_landscape_share_one_iqa_batch(tmp_path: Path) -> None:
    landscape = _jpeg(tmp_path / "land.jpg", (3000, 2000))
    portrait = _portrait_jpeg(tmp_path / "port.jpg")
    batch = _load_dual_pil_batch(_DualLoadInput(paths=[landscape, portrait], device="cpu"))
    assert batch.pil_1280[1].height > batch.pil_1280[1].width
    assert batch.tensor_1280.shape[0] == 2
    assert batch.tensor_1280.shape[-1] == SHARED_DECODE_PIXEL_PX


def test_other_aspect_ratio_is_resized_into_the_batch(tmp_path: Path) -> None:
    three_two = _jpeg(tmp_path / "a.jpg", (3000, 2000))
    four_three = _jpeg(tmp_path / "b.jpg", (2400, 1800))
    batch = _load_dual_pil_batch(_DualLoadInput(paths=[three_two, four_three], device="cpu"))
    assert batch.tensor_1280.shape[0] == 2


def test_fast_path_tensor_batch_mixes_orientations(tmp_path: Path) -> None:
    landscape = _jpeg(tmp_path / "land.jpg", (3000, 2000))
    portrait = _portrait_jpeg(tmp_path / "port.jpg")
    tensor, pils = _load_tensor_batch([landscape, portrait])
    assert tensor.shape[0] == 2
    assert pils[1].height > pils[1].width
