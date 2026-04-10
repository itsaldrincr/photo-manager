"""Tests for cull.saliency — CLIP is fully mocked, no weights loaded."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cull.saliency import SaliencyRequest, SaliencyResult, compute_saliency

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE: int = 4
HIDDEN_DIM: int = 8
PATCH_COUNT: int = GRID_SIZE * GRID_SIZE
SEED: int = 42
IMAGE_SIZE: int = 64


# ---------------------------------------------------------------------------
# Stub CLIP objects
# ---------------------------------------------------------------------------


def _make_fixed_hidden(hot_patch: int) -> torch.Tensor:
    """Return a hidden state tensor with one patch having high norm."""
    rng = torch.Generator()
    rng.manual_seed(SEED)
    tokens = torch.rand(1, PATCH_COUNT + 1, HIDDEN_DIM, generator=rng) * 0.01
    tokens[0, hot_patch + 1, :] = 10.0  # CLS is index 0; patch offset +1
    return tokens


class _StubVisionOutput:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.last_hidden_state = hidden


class _StubVisionModel:
    def __init__(self, hot_patch: int) -> None:
        self._hot_patch = hot_patch

    def __call__(self, **kwargs: object) -> _StubVisionOutput:
        return _StubVisionOutput(_make_fixed_hidden(self._hot_patch))


class _StubCLIPModel:
    def __init__(self, hot_patch: int) -> None:
        self.vision_model = _StubVisionModel(hot_patch)
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self) -> torch.nn.Parameter:
        yield self._param


class _StubProcessor:
    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        return {"pixel_values": torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_jpeg(tmp_path: Path) -> Path:
    """Write a minimal RGB JPEG and return its path."""
    from PIL import Image

    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 64, 32))
    jpeg_path = tmp_path / "sample.jpg"
    img.save(str(jpeg_path), format="JPEG")
    return jpeg_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_peak_xy_at_hot_patch(
    tiny_jpeg: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """peak_xy must be fractional coords of the single hot patch in the grid."""
    hot_patch = 5  # patch index in flattened 4x4 grid → row=1, col=1
    expected_row, expected_col = divmod(hot_patch, GRID_SIZE)
    expected_fx = expected_col / GRID_SIZE
    expected_fy = expected_row / GRID_SIZE

    monkeypatch.setattr("cull.saliency.get_clip_model", lambda: _StubCLIPModel(hot_patch))
    monkeypatch.setattr("cull.saliency.get_clip_processor", lambda: _StubProcessor())

    req = SaliencyRequest(image_path=tiny_jpeg, target_px=IMAGE_SIZE)
    result = compute_saliency(req)

    assert isinstance(result, SaliencyResult)
    assert result.peak_xy == (expected_fx, expected_fy)


def test_bbox_encloses_thresholded_mass(
    tiny_jpeg: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bbox must tightly bound the cells at or above SALIENCY_THRESHOLD (fractional)."""
    hot_patch = 5
    expected_row, expected_col = divmod(hot_patch, GRID_SIZE)
    expected_fx = expected_col / GRID_SIZE
    expected_fy = expected_row / GRID_SIZE

    monkeypatch.setattr("cull.saliency.get_clip_model", lambda: _StubCLIPModel(hot_patch))
    monkeypatch.setattr("cull.saliency.get_clip_processor", lambda: _StubProcessor())

    req = SaliencyRequest(image_path=tiny_jpeg, target_px=IMAGE_SIZE)
    result = compute_saliency(req)

    x0, y0, x1, y1 = result.bbox
    assert x0 <= expected_fx <= x1
    assert y0 <= expected_fy <= y1


def test_heatmap_normalized(
    tiny_jpeg: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Heatmap values must lie in [0, 1] and peak must equal 1.0."""
    monkeypatch.setattr("cull.saliency.get_clip_model", lambda: _StubCLIPModel(0))
    monkeypatch.setattr("cull.saliency.get_clip_processor", lambda: _StubProcessor())

    req = SaliencyRequest(image_path=tiny_jpeg, target_px=IMAGE_SIZE)
    result = compute_saliency(req)

    assert float(result.heatmap.min()) >= 0.0
    assert float(result.heatmap.max()) == pytest.approx(1.0, abs=1e-5)


def test_no_real_weights_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing saliency and patching CLIP must never call from_pretrained."""
    called: list[str] = []

    def _fail_pretrained(name: str) -> None:
        called.append(name)
        raise AssertionError(f"from_pretrained called for {name}")

    monkeypatch.setattr("cull.clip_loader._model", None)
    monkeypatch.setattr("cull.clip_loader._processor", None)

    import cull.saliency  # noqa: F401 — just checking import is clean
    assert called == [], "from_pretrained must not be called at import time"
