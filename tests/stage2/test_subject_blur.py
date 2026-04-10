"""Tests for cull.stage2.subject_blur — face / saliency / global crop fallbacks."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from pydantic import BaseModel

from cull.models import SubjectBlurScore
from cull.saliency import SaliencyResult
from cull.stage1.blur import compute_tenengrad
from cull.stage2 import subject_blur as subject_blur_module
from cull.stage2.portrait import PortraitResult
from cull.stage2.subject_blur import SubjectBlurInput, score_one

IMAGE_SIZE: int = 120
FACE_BBOX_X0: int = 10
FACE_BBOX_Y0: int = 10
FACE_BBOX_X1: int = 50
FACE_BBOX_Y1: int = 50
SHARP_BLOCK_VALUE: int = 250
SOFT_BLOCK_VALUE: int = 5
SALIENCY_FRAC_X0: float = 5 / 30
SALIENCY_FRAC_Y0: float = 5 / 30
SALIENCY_FRAC_X1: float = 15 / 30
SALIENCY_FRAC_Y1: float = 15 / 30


class _ImgCtx(BaseModel):
    """Bundle holding the test PIL image."""

    model_config = __import__("pydantic").ConfigDict(arbitrary_types_allowed=True)

    pil: Image.Image


def _make_synthetic_image() -> np.ndarray:
    """Build an image where the top-left region has high-contrast edges."""
    img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), SOFT_BLOCK_VALUE, dtype=np.uint8)
    for col_start in range(FACE_BBOX_X0, FACE_BBOX_X1, 4):
        img[FACE_BBOX_Y0:FACE_BBOX_Y1, col_start:col_start + 2] = SHARP_BLOCK_VALUE
    return img


@pytest.fixture()
def img_ctx() -> _ImgCtx:
    """Return a synthetic PIL image with a high-edge region in the face bbox area."""
    return _ImgCtx(pil=Image.fromarray(_make_synthetic_image()))


def _make_face_portrait() -> PortraitResult:
    """Return a PortraitResult with face_count=1 and the known face bbox."""
    return PortraitResult(
        face_count=1,
        face_bbox=(FACE_BBOX_X0, FACE_BBOX_Y0, FACE_BBOX_X1, FACE_BBOX_Y1),
    )


def _make_saliency() -> SaliencyResult:
    """Return a SaliencyResult with a fractional bbox over the same region."""
    heatmap = np.zeros((30, 30), dtype=np.float32)
    heatmap[5:15, 5:15] = 1.0
    return SaliencyResult(
        heatmap=heatmap,
        peak_xy=(10, 10),
        bbox=(SALIENCY_FRAC_X0, SALIENCY_FRAC_Y0, SALIENCY_FRAC_X1, SALIENCY_FRAC_Y1),
    )


def _full_image_tenengrad(pil: Image.Image) -> float:
    """Compute Tenengrad on the entire image as a baseline."""
    from cull.stage2.subject_blur import pil_to_gray
    gray = pil_to_gray(pil)
    return compute_tenengrad(gray)


def test_face_path_uses_face_bbox_crop(img_ctx: _ImgCtx) -> None:
    """Portrait with face bbox; crop score must differ from full-image score."""
    portrait = _make_face_portrait()
    sb_input = SubjectBlurInput(pil_1280=img_ctx.pil, portrait=portrait)
    score = score_one(sb_input)
    full_score = _full_image_tenengrad(img_ctx.pil)
    assert isinstance(score, SubjectBlurScore)
    assert score.subject_region_source == "face"
    assert score.has_subject is True
    assert score.tenengrad != pytest.approx(full_score, rel=1e-6)


def test_saliency_fallback_when_no_face(img_ctx: _ImgCtx) -> None:
    """No-face case: saliency supplied directly; source must be saliency_peak."""
    saliency = _make_saliency()
    sb_input = SubjectBlurInput(pil_1280=img_ctx.pil, portrait=None, saliency=saliency)
    score = score_one(sb_input)
    assert score.subject_region_source == "saliency_peak"
    assert score.has_subject is True


def test_global_fallback_when_no_face_no_saliency(img_ctx: _ImgCtx) -> None:
    """No portrait + no saliency → has_subject False, source global."""
    sb_input = SubjectBlurInput(pil_1280=img_ctx.pil, portrait=None, saliency=None)
    score = score_one(sb_input)
    assert score.subject_region_source == "global"
    assert score.has_subject is False
    assert score.tenengrad == pytest.approx(_full_image_tenengrad(img_ctx.pil), rel=1e-6)


def test_score_one_returns_subject_blur_score_type(img_ctx: _ImgCtx) -> None:
    """score_one always returns SubjectBlurScore regardless of fallback path."""
    sb_input = SubjectBlurInput(pil_1280=img_ctx.pil)
    assert isinstance(score_one(sb_input), SubjectBlurScore)


def test_subject_blur_module_exposes_public_api() -> None:
    """Public surface: SubjectBlurInput and score_one are importable from the module."""
    assert hasattr(subject_blur_module, "SubjectBlurInput")
    assert hasattr(subject_blur_module, "score_one")
