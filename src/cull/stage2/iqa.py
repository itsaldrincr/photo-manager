"""IQA scoring via pyiqa — TOPIQ-NR and CLIP-IQA+ metrics."""

# Torch/pyiqa cache is pinned via TORCH_HOME, set in cull.env_bootstrap.
# No loader kwargs are needed here — pyiqa resolves weights from torch.hub
# under $TORCH_HOME/hub/pyiqa, which env_bootstrap points at the offline cache.

from __future__ import annotations

import logging

import torch
from pydantic import BaseModel, ConfigDict

from cull.io_silence import _silence_stdio

logger = logging.getLogger(__name__)


class _BatchScoreRequest(BaseModel):
    """Request parameters for scoring a batch of tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    batch_tensor: torch.Tensor
    device: str


# Module-level singleton registry for lazy-loaded metrics.
_METRICS: dict[str, object] = {}

TOPIQ_METRIC_NAME: str = "topiq_nr"
CLIPIQA_METRIC_NAME: str = "clipiqa+"

_MODEL_LABELS: dict[str, str] = {
    TOPIQ_METRIC_NAME: "TOPIQ-NR",
    CLIPIQA_METRIC_NAME: "CLIP-IQA+",
}


def select_device() -> str:
    """Return 'mps' if available, else 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_metric(name: str, device: str) -> object:
    """Return cached pyiqa metric, loading on first call."""
    key = f"{name}:{device}"
    if key not in _METRICS:
        import pyiqa  # noqa: PLC0415

        logger.info("Loading pyiqa metric '%s' on device '%s'", name, device)
        with _silence_stdio():
            _METRICS[key] = pyiqa.create_metric(name, device=device)
    return _METRICS[key]


def warmup_metrics(device: str) -> None:
    """Pre-load both IQA metrics onto the given device."""
    _get_metric(TOPIQ_METRIC_NAME, device)
    _get_metric(CLIPIQA_METRIC_NAME, device)


CPU_FALLBACK: str = "cpu"


def unload_metrics() -> None:
    """Release all cached IQA models and free memory."""
    _METRICS.clear()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    logger.info("IQA metrics unloaded")


def _score_batch_with_fallback(request: _BatchScoreRequest) -> list[float]:
    """Score batch of tensors, falling back to CPU if MPS fails."""
    metric = _get_metric(request.name, request.device)
    try:
        with _silence_stdio(), torch.no_grad():
            scores = metric(request.batch_tensor.to(request.device))
            return scores.detach().cpu().squeeze(-1).tolist()
    except RuntimeError:
        if request.device == CPU_FALLBACK:
            raise
        logger.warning("%s batch failed on %s, falling back to CPU", request.name, request.device)
        cpu_metric = _get_metric(request.name, CPU_FALLBACK)
        with _silence_stdio(), torch.no_grad():
            scores = cpu_metric(request.batch_tensor.to(CPU_FALLBACK))
            return scores.detach().cpu().squeeze(-1).tolist()


def score_topiq_batch(tensor_batch: torch.Tensor, device: str) -> list[float]:
    """Score a batch of tensors with TOPIQ-NR; returns list of values in [0, 1]."""
    request = _BatchScoreRequest(name=TOPIQ_METRIC_NAME, batch_tensor=tensor_batch, device=device)
    return _score_batch_with_fallback(request)


def score_clipiqa_batch(tensor_batch: torch.Tensor, device: str) -> list[float]:
    """Score a batch of tensors with CLIP-IQA+; returns list of values in [0, 1]."""
    request = _BatchScoreRequest(name=CLIPIQA_METRIC_NAME, batch_tensor=tensor_batch, device=device)
    return _score_batch_with_fallback(request)
