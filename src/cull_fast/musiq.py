"""MUSIQ scoring for fast mode: pyiqa('musiq') + pyiqa('musiq-ava').

MUSIQ's 14-layer ViT attention is vulnerable to fp16 softmax overflow on MPS.
Ships fp32-only until a regression harness validates fp16 parity. Tracked
upstream as pyiqa#190 for the MPS fallback requirement.

Outputs normalized to [0, 1] before writeback into IqaScores fields.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

MUSIQ_KONIQ_METRIC_NAME: str = "musiq"
MUSIQ_AVA_METRIC_NAME: str = "musiq-ava"
MUSIQ_KONIQ_MAX_MOS: float = 100.0
MUSIQ_AVA_MIN_MOS: float = 1.0
MUSIQ_AVA_MAX_MOS: float = 10.0
MUSIQ_USE_FP16: bool = False
MPS_FALLBACK_ENV_VAR: str = "PYTORCH_ENABLE_MPS_FALLBACK"
MPS_FALLBACK_ENABLED: str = "1"

os.environ.setdefault(MPS_FALLBACK_ENV_VAR, MPS_FALLBACK_ENABLED)

logger = logging.getLogger(__name__)

_MUSIQ_METRICS: dict[str, object] = {}


class MusiQScorePair(BaseModel):
    """Normalized MUSIQ score pair for a single photo."""

    photo_path: Path
    technical: float
    aesthetic: float


class _MusiQBatchRequest(BaseModel):
    """Parameters for scoring a batch of tensors with both MUSIQ metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tensor_batch: torch.Tensor
    photo_paths: list[Path]
    device: str


class _MetricRunRequest(BaseModel):
    """Parameters for running one pyiqa metric on a tensor batch."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    metric: object
    batch: torch.Tensor
    device: str


def _normalize_musiq_koniq(raw: float) -> float:
    """Normalize a KonIQ MOS score (0–100 scale) to [0, 1]."""
    normalized = raw / MUSIQ_KONIQ_MAX_MOS
    return max(0.0, min(1.0, normalized))


def _normalize_musiq_ava(raw: float) -> float:
    """Normalize an AVA MOS score (1–10 scale) to [0, 1]."""
    normalized = (raw - MUSIQ_AVA_MIN_MOS) / (MUSIQ_AVA_MAX_MOS - MUSIQ_AVA_MIN_MOS)
    return max(0.0, min(1.0, normalized))


def _get_musiq_metric(name: str) -> object:
    """Return cached pyiqa metric, lazy-loading on first call."""
    from cull.stage2.iqa import select_device  # noqa: PLC0415

    device = select_device()
    key = f"{name}:{device}"
    if key not in _MUSIQ_METRICS:
        import pyiqa  # noqa: PLC0415

        logger.info("Loading pyiqa metric '%s' on device '%s'", name, device)
        _MUSIQ_METRICS[key] = pyiqa.create_metric(name, device=device)
    return _MUSIQ_METRICS[key]


def _run_metric_on_batch(run_req: _MetricRunRequest) -> list[float]:
    """Run a single pyiqa metric on a tensor batch; returns list of floats."""
    with torch.no_grad():
        scores = run_req.metric(run_req.batch.to(run_req.device))  # type: ignore[operator]
    flat = scores.detach().cpu().squeeze(-1)
    if flat.ndim != 1:
        raise RuntimeError(
            f"pyiqa metric returned tensor of shape {tuple(flat.shape)}; "
            f"expected 1-D after squeeze(-1)"
        )
    return flat.tolist()


def score_musiq_batch(req: _MusiQBatchRequest) -> list[MusiQScorePair]:
    """Score a batch with musiq + musiq-ava; returns normalized MusiQScorePair list."""
    if req.tensor_batch.shape[0] == 0:
        return []

    koniq_metric = _get_musiq_metric(MUSIQ_KONIQ_METRIC_NAME)
    ava_metric = _get_musiq_metric(MUSIQ_AVA_METRIC_NAME)

    koniq_scores = _run_metric_on_batch(
        _MetricRunRequest(metric=koniq_metric, batch=req.tensor_batch, device=req.device)
    )
    ava_scores = _run_metric_on_batch(
        _MetricRunRequest(metric=ava_metric, batch=req.tensor_batch, device=req.device)
    )

    return [
        MusiQScorePair(
            photo_path=path,
            technical=_normalize_musiq_koniq(tech),
            aesthetic=_normalize_musiq_ava(aes),
        )
        for path, tech, aes in zip(req.photo_paths, koniq_scores, ava_scores)
    ]


def unload_musiq() -> None:
    """Release all cached MUSIQ models and free MPS memory if available."""
    _MUSIQ_METRICS.clear()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    logger.info("MUSIQ metrics unloaded")
