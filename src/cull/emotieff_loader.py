"""Shared singleton loader for the EmotiEffLib ONNX emotion recognizer."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any

from cull.config import EMOTIEFF_MODEL_NAME, EMOTIEFF_ONNX_FILENAME, ModelCacheConfig
from cull.model_cache import ConfigError

logger = logging.getLogger(__name__)

_CACHE: ModelCacheConfig = ModelCacheConfig.from_env()

# EmotiEffLib hardcodes its download cache to ~/.emotiefflib and has no API
# to point at an arbitrary path. Pre-seeding a symlink there from our own
# offline cache root keeps the library's internal isfile() check satisfied,
# so it never attempts a network fetch.
_EMOTIEFFLIB_HOME_DIR: Path = Path.home() / ".emotiefflib"

_recognizer: Any | None = None


def _resolve_onnx_path(cache: ModelCacheConfig) -> Path:
    """Return the cached EmotiEffLib ONNX path or raise ConfigError."""
    path = cache.emotieff_dir / EMOTIEFF_ONNX_FILENAME
    if not path.exists():
        raise ConfigError(
            f"{EMOTIEFF_ONNX_FILENAME} not found at {path}. "
            "Run 'cull setup --allow-network' to populate the model cache."
        )
    return path


def _ensure_emotiefflib_cache_link(onnx_path: Path) -> None:
    """Symlink the cached ONNX file into EmotiEffLib's own ~/.emotiefflib cache dir."""
    _EMOTIEFFLIB_HOME_DIR.mkdir(parents=True, exist_ok=True)
    link = _EMOTIEFFLIB_HOME_DIR / EMOTIEFF_ONNX_FILENAME
    if link.is_symlink() and not link.exists():
        link.unlink()
    if link.exists():
        return
    os.symlink(onnx_path, link)


def get_emotieff_recognizer() -> Any:
    """Return the cached EmotiEffLib ONNX recognizer singleton, loading it on first call."""
    global _recognizer
    if _recognizer is not None:
        return _recognizer
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer  # noqa: PLC0415

    onnx_path = _resolve_onnx_path(_CACHE)
    _ensure_emotiefflib_cache_link(onnx_path)
    logger.info("Loading EmotiEffLib ONNX recognizer '%s'", EMOTIEFF_MODEL_NAME)
    _recognizer = EmotiEffLibRecognizer(engine="onnx", model_name=EMOTIEFF_MODEL_NAME)
    return _recognizer


def unload() -> None:
    """Reset the EmotiEffLib recognizer singleton and free memory."""
    global _recognizer
    _recognizer = None
    gc.collect()
