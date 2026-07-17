"""Shared singleton loader for the DINOv2-small model and image processor."""

from __future__ import annotations

import gc
import logging

from cull.config import DINOV2_MODEL_ID, ModelCacheConfig
from cull.io_silence import _silence_stdio
from cull.stage2.iqa import select_device

logger = logging.getLogger(__name__)

_CACHE: ModelCacheConfig = ModelCacheConfig.from_env()

_model: object = None
_processor: object = None


def get_dinov2_model() -> object:
    """Return the cached DINOv2 model singleton, loading it on first call."""
    global _model
    if _model is not None:
        return _model
    from transformers import AutoModel  # noqa: PLC0415

    device = select_device()
    logger.info("Loading DINOv2 model '%s' on device '%s'", DINOV2_MODEL_ID, device)
    with _silence_stdio():
        _model = AutoModel.from_pretrained(
            DINOV2_MODEL_ID,
            cache_dir=str(_CACHE.hf_home / "hub"),
            local_files_only=True,
        ).to(device).eval()
    return _model


def get_dinov2_processor() -> object:
    """Return the cached DINOv2 image processor singleton, loading it on first call."""
    global _processor
    if _processor is not None:
        return _processor
    from transformers import AutoImageProcessor  # noqa: PLC0415

    logger.info("Loading DINOv2 processor '%s'", DINOV2_MODEL_ID)
    with _silence_stdio():
        _processor = AutoImageProcessor.from_pretrained(
            DINOV2_MODEL_ID,
            cache_dir=str(_CACHE.hf_home / "hub"),
            local_files_only=True,
            use_fast=True,
        )
    return _processor


def unload() -> None:
    """Reset the DINOv2 model and processor singletons and free memory."""
    global _model, _processor
    _model = None
    _processor = None
    gc.collect()
