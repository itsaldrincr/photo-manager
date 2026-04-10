"""Composition TOPIQ IAA — pyiqa metric cache, warmup, score, unload."""

# Torch/pyiqa weights are pinned via TORCH_HOME, set in cull.env_bootstrap.
# No loader kwargs are needed here — pyiqa resolves from $TORCH_HOME/hub/pyiqa.

from __future__ import annotations

import logging

from PIL import Image

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


def unload_topiq_iaa() -> None:
    """Release the cached topiq_iaa metric and free its memory."""
    _TOPIQ_IAA_METRIC.clear()
    logger.info("topiq_iaa metric unloaded")
