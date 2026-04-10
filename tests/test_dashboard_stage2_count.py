"""Test that _run_stage2_loop counter increments once per batch via shared CLIP forward.

Exercises the batched pipeline at N=16 (2 full batches at STAGE2_BATCH_SIZE=8).
Uses mock scorers so no real ML models are loaded (no MPS pressure).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn

from cull import clip_loader
from cull.config import CullConfig, STAGE2_BATCH_SIZE
from cull.pipeline import _Stage2LoopInput, _run_stage2_loop
from cull.stage2.fusion import IqaScores
from cull.stage2 import aesthetic
from tests._golden_helpers import _MinimalDashboard

logger = logging.getLogger(__name__)

PHOTO_COUNT: int = 16
CLIP_HIDDEN_DIM: int = 1024
PROJECTION_DIM: int = 768
TOTAL_TOKENS: int = 257


class _StubInputs:
    """Stand-in for HuggingFace BatchEncoding with a `to(device)` method."""

    def __init__(self, pixel_values: torch.Tensor) -> None:
        self.pixel_values = pixel_values

    def to(self, _device: str) -> "_StubInputs":
        return self

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.pixel_values


class _StubProcessor:
    """CLIPProcessor stub returning deterministic pixel values."""

    def __call__(self, images: list[object], return_tensors: str) -> _StubInputs:
        return _StubInputs(torch.zeros(len(images), 3, 224, 224))


class _RecordingVisionModel:
    """Vision model stub that records its call count across batches."""

    def __init__(self) -> None:
        self.call_count: int = 0

    def __call__(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.call_count += 1
        batch = pixel_values.shape[0]
        return SimpleNamespace(
            last_hidden_state=torch.randn(batch, TOTAL_TOKENS, CLIP_HIDDEN_DIM),
            pooler_output=torch.randn(batch, CLIP_HIDDEN_DIM),
        )


class _StubClipModel:
    """CLIPModel stub with a recording vision_model and a linear projection."""

    def __init__(self) -> None:
        self.vision_model = _RecordingVisionModel()
        self.visual_projection = nn.Linear(CLIP_HIDDEN_DIM, PROJECTION_DIM)


@pytest.fixture()
def stub_clip(monkeypatch: pytest.MonkeyPatch) -> _StubClipModel:
    """Patch clip_loader singletons + aesthetic head with deterministic stubs."""
    clip = _StubClipModel()
    monkeypatch.setattr(clip_loader, "get_clip_model", lambda: clip)
    monkeypatch.setattr(clip_loader, "get_clip_processor", lambda: _StubProcessor())
    head = aesthetic.AestheticHead(layers=nn.Linear(PROJECTION_DIM, 1))
    monkeypatch.setattr(
        "cull._pipeline.stage2_runner._get_head", lambda _device: head
    )
    monkeypatch.setattr(
        "cull._pipeline.stage2_runner._prewarm_stage2_models",
        lambda loop_in, dashboard: None,
    )
    monkeypatch.setattr(
        "cull._pipeline.stage2_runner._build_iqa_pyiqa_only",
        lambda batch_input, device: [
            IqaScores(
                photo_path=path,
                topiq=0.1 + (int(path.stem.split("_")[-1]) * 0.01),
                laion_aesthetic=0.0,
                clipiqa=0.2 + (int(path.stem.split("_")[-1]) * 0.01),
                exposure=0.5,
            )
            for path in batch_input.photo_paths
        ],
    )
    monkeypatch.setattr(
        "cull._pipeline.stage2_runner._apply_composition_to_scores",
        lambda apply_in: None,
    )
    monkeypatch.setattr(
        "cull._pipeline.stage2_runner._apply_taste_to_scores",
        lambda iqa_list, paths: None,
    )
    monkeypatch.setattr(
        "cull._pipeline.stage2_runner._apply_subject_blur_to_scores",
        lambda apply_in: {},
    )
    return clip


def _make_jpegs(tmp_path: Path) -> list[Path]:
    """Create a deterministic 16-photo JPEG set without external corpora."""
    paths: list[Path] = []
    for idx in range(PHOTO_COUNT):
        path = tmp_path / f"img_{idx:02d}.jpg"
        Image.new("RGB", (16, 16), color=(idx, idx, idx)).save(path, format="JPEG")
        paths.append(path)
    return paths


def _build_loop_input(
    photo_paths: list[Path],
) -> tuple[_Stage2LoopInput, _MinimalDashboard]:
    """Construct a _Stage2LoopInput and fake dashboard for the given photos."""
    dashboard = _MinimalDashboard()
    config = CullConfig(is_portrait=False)
    loop_in = _Stage2LoopInput(survivors=photo_paths, config=config)
    return loop_in, dashboard


def _check_topiq_uniqueness(dashboard: _MinimalDashboard) -> None:
    """Assert all topiq scores are distinct — shared score means alignment bug."""
    topiq_scores = [fusion.stage2.topiq for _, fusion in dashboard.stage2_calls]
    assert len(topiq_scores) == len(set(topiq_scores)), (
        f"Duplicate topiq scores detected — alignment bug. Scores: {topiq_scores}"
    )


def test_batched_pipeline_counter_and_alignment(
    tmp_path: Path, mock_scorers: None, stub_clip: _StubClipModel
) -> None:
    """Verify vision_model is called once per batch, counter counts per photo."""
    assert PHOTO_COUNT == STAGE2_BATCH_SIZE * 2, "Test requires exactly 2 full batches"

    photo_paths = _make_jpegs(tmp_path)
    loop_in, dashboard = _build_loop_input(photo_paths)
    photo_paths = loop_in.survivors

    _run_stage2_loop(loop_in, dashboard)

    expected_batch_count = PHOTO_COUNT // STAGE2_BATCH_SIZE
    assert stub_clip.vision_model.call_count == expected_batch_count, (
        f"Forward pass counter: expected {expected_batch_count} calls "
        f"(once per batch), got {stub_clip.vision_model.call_count}"
    )
    assert dashboard._s2.done == PHOTO_COUNT, (
        f"Counter mismatch: expected {PHOTO_COUNT}, got {dashboard._s2.done}"
    )
    assert len(dashboard.stage2_calls) == PHOTO_COUNT, (
        f"Call count mismatch: expected {PHOTO_COUNT}, got {len(dashboard.stage2_calls)}"
    )

    for i, (recorded_path, _) in enumerate(dashboard.stage2_calls):
        assert recorded_path == photo_paths[i], (
            f"Order mismatch at index {i}: expected {photo_paths[i]}, got {recorded_path}"
        )

    _check_topiq_uniqueness(dashboard)
