"""Mixed landscape/portrait batches are scored per size group, in input order."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from cull.stage2 import composition
from cull.stage2.composition_topiq import TOPIQ_IAA_DEFAULT, score_topiq_iaa_batch
from cull.stage2.size_groups import group_indices_by_size, score_in_size_groups

LANDSCAPE: tuple[int, int] = (48, 32)
PORTRAIT: tuple[int, int] = (32, 48)


def test_groups_keep_first_seen_order() -> None:
    """Indices are grouped by size; groups appear in first-seen order."""
    sizes = [LANDSCAPE, PORTRAIT, LANDSCAPE, PORTRAIT]
    assert group_indices_by_size(sizes) == [[0, 2], [1, 3]]


def test_scores_return_in_input_order() -> None:
    """Each item's score lands back at its own index."""
    items = [(LANDSCAPE, 1.0), (PORTRAIT, 2.0), (LANDSCAPE, 3.0)]
    assert score_in_size_groups(items, lambda group: [v * 10 for v in group]) == [10.0, 20.0, 30.0]


class _WidthMetric:
    """Stub pyiqa metric: scores each image by its tensor width / 100."""

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """Return one score per image in the (N,C,H,W) batch."""
        return torch.full((batch.shape[0], 1), batch.shape[-1] / 100.0)


def test_topiq_iaa_scores_mixed_orientations(monkeypatch: pytest.MonkeyPatch) -> None:
    """A landscape+portrait batch is scored, not replaced by the default score."""
    monkeypatch.setattr(composition, "_get_topiq_iaa_metric", lambda device: _WidthMetric())
    images = [Image.new("RGB", LANDSCAPE), Image.new("RGB", PORTRAIT)]
    scores = score_topiq_iaa_batch(images)
    assert scores == [pytest.approx(0.48), pytest.approx(0.32)]
    assert TOPIQ_IAA_DEFAULT not in scores
