"""Stage 2 IQA forwards at most STAGE2_IQA_SUB_BATCH_SIZE photos at once."""

from __future__ import annotations

import torch

import cull.stage2.iqa as iqa
from cull.config import STAGE2_IQA_SUB_BATCH_SIZE

BATCH_SIZE: int = 5


def test_metric_sees_only_sub_batches_and_scores_keep_order(monkeypatch) -> None:
    """Each forward is at most the sub-batch size; scores come back in photo order."""
    forward_sizes: list[int] = []

    def fake_get_metric(name: str, device: str):
        def metric(batch_tensor: torch.Tensor) -> torch.Tensor:
            forward_sizes.append(batch_tensor.shape[0])
            return batch_tensor[:, 0, 0, 0].reshape(-1, 1)

        return metric

    monkeypatch.setattr(iqa, "_get_metric", fake_get_metric)
    batch = torch.arange(BATCH_SIZE, dtype=torch.float32).reshape(-1, 1, 1, 1).expand(-1, 3, 4, 4)

    scores = iqa.score_topiq_batch(batch, device="cpu")

    assert scores == [float(i) for i in range(BATCH_SIZE)]
    assert max(forward_sizes) <= STAGE2_IQA_SUB_BATCH_SIZE
    assert sum(forward_sizes) == BATCH_SIZE
