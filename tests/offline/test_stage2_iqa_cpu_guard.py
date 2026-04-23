"""Regression tests for Stage 2 IQA thread handling."""

from __future__ import annotations

import torch

import cull.stage2.iqa as iqa


def _fake_scores(batch_tensor: torch.Tensor) -> torch.Tensor:
    """Return one scalar score per batch item."""
    return torch.full((batch_tensor.shape[0], 1), 0.5)


def test_cpu_device_keeps_default_torch_threads(
    monkeypatch,
) -> None:
    """CPU scoring should not override PyTorch's default thread count."""
    calls: list[tuple[str, str | int]] = []

    def fake_set_num_threads(count: int) -> None:
        calls.append(("set_num_threads", count))

    def fake_get_metric(name: str, device: str):
        calls.append(("get_metric", device))

        def metric(batch_tensor: torch.Tensor) -> torch.Tensor:
            calls.append(("metric", device))
            return _fake_scores(batch_tensor)

        return metric

    monkeypatch.setattr(iqa.torch, "set_num_threads", fake_set_num_threads)
    monkeypatch.setattr(iqa, "_get_metric", fake_get_metric)

    scores = iqa.score_topiq_batch(torch.zeros((2, 3, 8, 8)), device="cpu")

    assert scores == [0.5, 0.5]
    assert not any(name == "set_num_threads" for name, _ in calls)


def test_mps_fallback_keeps_default_torch_threads(monkeypatch) -> None:
    """MPS fallback should not override PyTorch's default thread count."""
    calls: list[tuple[str, str | int]] = []

    def fake_set_num_threads(count: int) -> None:
        calls.append(("set_num_threads", count))

    def fake_get_metric(name: str, device: str):
        calls.append(("get_metric", device))

        def metric(batch_tensor: torch.Tensor) -> torch.Tensor:
            calls.append(("metric", device))
            if device == "mps":
                raise RuntimeError("simulated mps failure")
            return _fake_scores(batch_tensor)

        return metric

    monkeypatch.setattr(iqa.torch, "set_num_threads", fake_set_num_threads)
    monkeypatch.setattr(iqa, "_get_metric", fake_get_metric)

    request = iqa._BatchScoreRequest(
        name=iqa.TOPIQ_METRIC_NAME,
        batch_tensor=torch.zeros((1, 3, 8, 8)),
        device="mps",
    )

    scores = iqa._score_batch_with_fallback(request)

    assert scores == [0.5]
    assert not any(name == "set_num_threads" for name, _ in calls)
