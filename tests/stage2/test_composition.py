"""Tests for cull.stage2.composition — saliency/topiq/smartcrop all mocked."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict

from cull.models import CompositionScore, CropProposal
from cull.saliency import SaliencyResult
from cull.stage2 import composition
from cull.stage2.composition import (
    SMARTCROP_DOWNSAMPLE_LONG_EDGE,
    CompositionInput,
    score_batch,
    score_one,
)

IMAGE_SIZE: int = 90
NATIVE_LARGE_LONG_EDGE: int = 4096
NATIVE_LARGE_SHORT_EDGE: int = 2730
DOWNSAMPLE_TOLERANCE: int = 2
HEATMAP_SIZE: int = 30
THIRDS_MIN_SCORE: float = 0.95
TOPIQ_IAA_STUB_SCORE: float = 0.42
FAKE_BOX_X: int = 10
FAKE_BOX_Y: int = 20
FAKE_BOX_WIDTH: int = 200
FAKE_BOX_HEIGHT: int = 150


class _CompCtx(BaseModel):
    """Bundle for one composition test: a pre-loaded PIL image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pil_1280: Image.Image


@pytest.fixture(autouse=True)
def _reset_metric_cache() -> None:
    """Clear the lazy topiq_iaa cache between tests."""
    composition._TOPIQ_IAA_METRIC.clear()


@pytest.fixture()
def comp_ctx(monkeypatch: pytest.MonkeyPatch) -> _CompCtx:
    """Build a composition test context with patched topiq + smartcrop."""
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 64, 32))
    monkeypatch.setattr(
        composition,
        "_get_topiq_iaa_metric",
        lambda _device=None: (lambda _img: TOPIQ_IAA_STUB_SCORE),
    )
    monkeypatch.setattr(composition, "_try_smartcrop", lambda _image: None)
    return _CompCtx(pil_1280=img)


def _make_peak_heatmap() -> np.ndarray:
    """Build a zeroed heatmap with a hot cell at the (1/3, 1/3) intersection."""
    heatmap = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    peak_y = HEATMAP_SIZE // 3
    peak_x = HEATMAP_SIZE // 3
    heatmap[peak_y, peak_x] = 1.0
    return heatmap


def _peak_saliency() -> SaliencyResult:
    """Saliency with a single peak at the thirds intersection."""
    heatmap = _make_peak_heatmap()
    peak_y = HEATMAP_SIZE // 3
    peak_x = HEATMAP_SIZE // 3
    fx = peak_x / HEATMAP_SIZE
    fy = peak_y / HEATMAP_SIZE
    return SaliencyResult(
        heatmap=heatmap,
        peak_xy=(fx, fy),
        bbox=(fx, fy, fx + 1.0 / HEATMAP_SIZE, fy + 1.0 / HEATMAP_SIZE),
    )


def _uniform_saliency() -> SaliencyResult:
    """Saliency that is uniform across the frame."""
    heatmap = np.ones((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    return SaliencyResult(
        heatmap=heatmap,
        peak_xy=(0.5, 0.5),
        bbox=(0.0, 0.0, 1.0, 1.0),
    )


def test_thirds_alignment_high_when_peak_at_thirds(comp_ctx: _CompCtx) -> None:
    """Peak at the (w/3, h/3) intersection must score thirds_alignment ≥ 0.95."""
    bundle = CompositionInput(
        pil_1280=comp_ctx.pil_1280, saliency_result=_peak_saliency()
    )
    score, crop = score_one(bundle)
    assert isinstance(score, CompositionScore)
    assert isinstance(crop, CropProposal)
    assert score.thirds_alignment >= THIRDS_MIN_SCORE


def test_uniform_saliency_uses_saliency_thirds_fallback(comp_ctx: _CompCtx) -> None:
    """When smartcrop returns nothing, the crop source must be saliency_thirds."""
    bundle = CompositionInput(
        pil_1280=comp_ctx.pil_1280, saliency_result=_uniform_saliency()
    )
    score, crop = score_one(bundle)
    assert crop.source == "saliency_thirds"
    assert score.topiq_iaa == TOPIQ_IAA_STUB_SCORE


def test_score_batch_preserves_order(comp_ctx: _CompCtx) -> None:
    """score_batch must return one pair per input in matching order."""
    bundles = [
        CompositionInput(
            pil_1280=comp_ctx.pil_1280, saliency_result=_peak_saliency()
        ),
        CompositionInput(
            pil_1280=comp_ctx.pil_1280, saliency_result=_uniform_saliency()
        ),
    ]
    results = score_batch(bundles)
    assert len(results) == 2
    assert results[0][0].thirds_alignment >= THIRDS_MIN_SCORE
    assert results[1][1].source == "saliency_thirds"


class _SmartcropProbe:
    """Recorder for smartcrop.SmartCrop().crop calls."""

    def __init__(self) -> None:
        self.received_image: Image.Image | None = None
        self.received_dims: tuple[int, int] | None = None

    def crop(self, image: Image.Image, width: int, height: int) -> dict:
        self.received_image = image
        self.received_dims = (width, height)
        return {
            "top_crop": {
                "x": FAKE_BOX_X, "y": FAKE_BOX_Y,
                "width": FAKE_BOX_WIDTH, "height": FAKE_BOX_HEIGHT,
            }
        }


def _install_fake_smartcrop(
    monkeypatch: pytest.MonkeyPatch, probe: _SmartcropProbe
) -> None:
    """Inject a fake `smartcrop` module whose SmartCrop returns the probe."""
    fake_module = ModuleType("smartcrop")
    fake_module.SmartCrop = lambda: probe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "smartcrop", fake_module)


def _make_large_pil() -> Image.Image:
    """Build a synthetic large (4096x2730) PIL image in-memory."""
    return Image.new(
        "RGB", (NATIVE_LARGE_LONG_EDGE, NATIVE_LARGE_SHORT_EDGE), color=(50, 100, 150)
    )


def test_smartcrop_receives_downsampled_and_bbox_scaled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """smartcrop must see a 512px image; the returned bbox is scaled to native dims."""
    monkeypatch.setattr(
        composition,
        "_get_topiq_iaa_metric",
        lambda _device=None: (lambda _img: TOPIQ_IAA_STUB_SCORE),
    )
    probe = _SmartcropProbe()
    _install_fake_smartcrop(monkeypatch, probe)
    bundle = CompositionInput(
        pil_1280=_make_large_pil(), saliency_result=_uniform_saliency()
    )
    _, crop = score_one(bundle)
    assert probe.received_image is not None
    received_long_edge = max(probe.received_image.size)
    assert (
        abs(received_long_edge - SMARTCROP_DOWNSAMPLE_LONG_EDGE)
        <= DOWNSAMPLE_TOLERANCE
    )
    assert crop is not None
    assert crop.source == "smartcrop"
    expected_scale = NATIVE_LARGE_LONG_EDGE / float(SMARTCROP_DOWNSAMPLE_LONG_EDGE)
    expected_left = int(round(FAKE_BOX_X * expected_scale))
    expected_right = int(round((FAKE_BOX_X + FAKE_BOX_WIDTH) * expected_scale))
    assert crop.left == expected_left
    assert crop.right == expected_right
    assert crop.right <= NATIVE_LARGE_LONG_EDGE
    assert crop.bottom <= NATIVE_LARGE_SHORT_EDGE


def test_skip_crop_bypasses_smartcrop_call(
    comp_ctx: _CompCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """skip_crop=True must NOT invoke _try_smartcrop / _build_crop at all."""
    smartcrop_spy = MagicMock(return_value=None)
    monkeypatch.setattr(composition, "_try_smartcrop", smartcrop_spy)
    build_crop_spy = MagicMock()
    monkeypatch.setattr(composition, "_build_crop", build_crop_spy)
    bundle = CompositionInput(
        pil_1280=comp_ctx.pil_1280,
        saliency_result=_peak_saliency(),
        skip_crop=True,
    )
    score, crop = score_one(bundle)
    assert score.thirds_alignment >= THIRDS_MIN_SCORE
    assert crop is None
    assert smartcrop_spy.call_count == 0
    assert build_crop_spy.call_count == 0


def test_score_batch_skip_crop_bypasses_smartcrop(
    comp_ctx: _CompCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In score_batch, photos marked skip_crop must not be dispatched to smartcrop."""
    build_crop_spy = MagicMock()
    monkeypatch.setattr(composition, "_build_crop", build_crop_spy)
    bundles = [
        CompositionInput(
            pil_1280=comp_ctx.pil_1280,
            saliency_result=_peak_saliency(),
            skip_crop=True,
        ),
    ]
    results = score_batch(bundles)
    assert len(results) == 1
    assert results[0][1] is None
    assert build_crop_spy.call_count == 0
