"""Composition TOPIQ IAA — pyiqa metric cache, warmup, score, unload."""

# Torch/pyiqa weights are pinned via TORCH_HOME, set in cull.env_bootstrap.
# No loader kwargs are needed here — pyiqa resolves from $TORCH_HOME/hub/pyiqa.

from __future__ import annotations

import logging

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor as tv_to_tensor

from cull.io_silence import _silence_stdio
from cull.stage2.iqa import CPU_FALLBACK, select_device

logger = logging.getLogger(__name__)

# Neutral fallback score when the pyiqa topiq_iaa metric is unavailable or fails.
TOPIQ_IAA_DEFAULT: float = 0.5
TOPIQ_IAA_METRIC_NAME: str = "topiq_iaa"

_TOPIQ_IAA_METRIC: dict[str, object] = {}


def warmup_topiq_iaa(device: str) -> None:
    """Pre-load the topiq_iaa metric singleton for the given device."""
    _get_topiq_iaa_metric(device)


def _get_topiq_iaa_metric(device: str = CPU_FALLBACK) -> object | None:
    """Return cached pyiqa topiq_iaa metric for device, loading lazily on first call."""
    cache_key = f"{TOPIQ_IAA_METRIC_NAME}:{device}"
    if cache_key in _TOPIQ_IAA_METRIC:
        return _TOPIQ_IAA_METRIC[cache_key]
    try:
        import pyiqa  # noqa: PLC0415

        logger.info("Loading pyiqa metric '%s' on device '%s'", TOPIQ_IAA_METRIC_NAME, device)
        with _silence_stdio():
            metric = pyiqa.create_metric(TOPIQ_IAA_METRIC_NAME, device=device)
    except Exception as exc:  # noqa: BLE001
        logger.warning("topiq_iaa metric unavailable: %s", exc)
        metric = None
    _TOPIQ_IAA_METRIC[cache_key] = metric
    return metric


def _score_topiq_iaa(image: Image.Image) -> float:
    """Run topiq_iaa on a PIL image, returning the default if unavailable."""
    from cull.stage2 import composition  # noqa: PLC0415 — lazy to preserve monkeypatch seam

    metric = composition._get_topiq_iaa_metric(select_device())
    if metric is None:
        return TOPIQ_IAA_DEFAULT
    try:
        score = float(metric(image))
    except Exception as exc:  # noqa: BLE001
        logger.warning("topiq_iaa scoring failed: %s", exc)
        return TOPIQ_IAA_DEFAULT
    return score


def _images_to_tensor_batch(images: list[Image.Image]) -> torch.Tensor:
    """Stack PIL images into a single (N,C,H,W) float tensor in [0,1]."""
    tensors = [tv_to_tensor(img).unsqueeze(0) for img in images]
    return torch.cat(tensors, dim=0)


def _normalize_batch_scores(scores: object, batch_size: int) -> list[float]:
    """Normalize a metric's raw batch output into a per-image score list."""
    if isinstance(scores, torch.Tensor):
        return torch.atleast_1d(scores.detach().cpu().squeeze(-1)).tolist()
    return [float(scores)] * batch_size


def score_topiq_iaa_batch(images: list[Image.Image]) -> list[float]:
    """Run topiq_iaa on a batch of PIL images in one forward pass.

    Replaces the historical per-image metric() call (one pyiqa forward per
    photo) with a single (N,C,H,W) batched forward, mirroring
    iqa.py's topiq_nr/clipiqa batching.
    """
    if not images:
        return []
    from cull.stage2 import composition  # noqa: PLC0415 — lazy to preserve monkeypatch seam

    device = select_device()
    metric = composition._get_topiq_iaa_metric(device)
    if metric is None:
        return [TOPIQ_IAA_DEFAULT] * len(images)
    try:
        batch_tensor = _images_to_tensor_batch(images).to(device)
        with _silence_stdio(), torch.no_grad():
            scores = metric(batch_tensor)
        return _normalize_batch_scores(scores, len(images))
    except Exception as exc:  # noqa: BLE001
        logger.warning("topiq_iaa batch scoring failed: %s", exc)
        return [TOPIQ_IAA_DEFAULT] * len(images)


def unload_topiq_iaa() -> None:
    """Release the cached topiq_iaa metric and free its memory."""
    _TOPIQ_IAA_METRIC.clear()
    logger.info("topiq_iaa metric unloaded")
