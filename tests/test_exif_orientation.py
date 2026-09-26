"""PIL model-input decodes must apply EXIF Orientation, as cv2.imread does."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from cull import saliency, vlm_session
from cull._pipeline.stage2_scoring import _DualLoadInput, _load_dual_pil_batch, _load_tensor
from cull.stage1 import duplicate
from cull.stage3.vlm_scoring import load_image_b64

STORED_W: int = 64
STORED_H: int = 32
EXIF_ORIENTATION_TAG: int = 0x0112
ROTATE_90_CW: int = 6


@pytest.fixture()
def rotated_jpeg(tmp_path: Path) -> Path:
    """Write a landscape-stored JPEG whose EXIF says to display it rotated 90 degrees."""
    path = tmp_path / "portrait.jpg"
    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG] = ROTATE_90_CW
    Image.new("RGB", (STORED_W, STORED_H), (120, 80, 40)).save(path, exif=exif.tobytes())
    return path


def _is_upright(size: tuple[int, int]) -> bool:
    """Return True for the displayed (portrait) orientation of the fixture."""
    return size == (STORED_H, STORED_W)


def test_stage2_iqa_tensor_is_laid_landscape(rotated_jpeg: Path) -> None:
    """The TOPIQ/CLIP-IQA tensor is the one deliberate exception: portraits lie on
    their side so a mixed-orientation batch keeps one shape (see _landscape)."""
    tensor = _load_tensor(rotated_jpeg)
    assert tensor.shape[3] > tensor.shape[2]


def test_stage2_dual_pil_load_is_upright(rotated_jpeg: Path) -> None:
    """The shared 1280px decode feeding pixel consumers must be upright."""
    batch = _load_dual_pil_batch(_DualLoadInput(paths=[rotated_jpeg], device="cpu"))
    assert _is_upright(batch.pil_1280[0].size)


def test_saliency_load_is_upright(rotated_jpeg: Path) -> None:
    """The CLIP saliency decode must be upright."""
    img = saliency._load_image(rotated_jpeg, STORED_W)
    assert _is_upright(img.size)


def test_vlm_scoring_b64_is_upright(rotated_jpeg: Path) -> None:
    """The Stage 3 VLM JPEG payload must be upright."""
    decoded = Image.open(BytesIO(base64.b64decode(load_image_b64(rotated_jpeg))))
    assert _is_upright(decoded.size)


def test_vlm_session_resize_is_upright(rotated_jpeg: Path) -> None:
    """The in-memory VLM session image must be upright."""
    assert _is_upright(vlm_session._resize_image_for_vlm(rotated_jpeg).size)


def test_dinov2_duplicate_embed_input_is_upright(rotated_jpeg: Path, monkeypatch) -> None:
    """The DINOv2 duplicate pass must embed the upright frame."""
    seen: list[tuple[int, int]] = []

    def _processor(images, return_tensors):
        seen.extend(img.size for img in images)
        inputs = MagicMock()
        inputs.to.return_value = {}
        return inputs

    model = MagicMock(return_value=MagicMock(pooler_output=torch.zeros(1, 4)))
    monkeypatch.setattr(duplicate.dinov2_loader, "get_dinov2_processor", lambda: _processor)
    monkeypatch.setattr(duplicate.dinov2_loader, "get_dinov2_model", lambda: model)
    duplicate._embed_dinov2_batch(duplicate._DinoV2EmbedJob(paths=[rotated_jpeg], device="cpu"))
    assert seen and all(_is_upright(s) for s in seen)
