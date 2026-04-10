"""Shared singleton loader for the CLIP model and processor."""

from __future__ import annotations

import gc
import logging

from cull.config import CLIP_MODEL_ID, ModelCacheConfig
from cull.io_silence import _silence_stdio
from cull.stage2.iqa import select_device

logger = logging.getLogger(__name__)

_CACHE: ModelCacheConfig = ModelCacheConfig.from_env()

_model: object = None
_processor: object = None


def get_clip_model() -> object:
    """Return the cached CLIPModel singleton, loading it on first call."""
    global _model
    if _model is not None:
        return _model
    from transformers import CLIPModel  # noqa: PLC0415

    device = select_device()
    logger.info("Loading CLIP model '%s' on device '%s'", CLIP_MODEL_ID, device)
    with _silence_stdio():
        _model = CLIPModel.from_pretrained(
            CLIP_MODEL_ID,
            cache_dir=str(_CACHE.hf_home / "hub"),
            local_files_only=True,
        ).to(device)
    return _model


def get_clip_processor() -> object:
    """Return the cached CLIPProcessor singleton, loading it on first call."""
    global _processor
    if _processor is not None:
        return _processor
    from transformers import CLIPProcessor  # noqa: PLC0415

    logger.info("Loading CLIP processor '%s'", CLIP_MODEL_ID)
    with _silence_stdio():
        _processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_ID,
            cache_dir=str(_CACHE.hf_home / "hub"),
            local_files_only=True,
        )
    return _processor


def unload() -> None:
    """Reset the CLIP model and processor singletons and free memory."""
    global _model, _processor
    _model = None
    _processor = None
    gc.collect()
