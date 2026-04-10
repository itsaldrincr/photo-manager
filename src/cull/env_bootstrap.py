"""Set ML framework env vars before any model imports. MUST be imported first.

Must be imported before any ``cull.*`` module that transitively imports
``torch``, ``transformers``, ``huggingface_hub``, ``mediapipe``, or
``deepface``. The module body has no side effects — env mutation only
happens when one of the ``apply_*`` functions is called explicitly.
"""

from __future__ import annotations

import os

from cull.config import ModelCacheConfig

OFFLINE_FLAG: str = "1"
HF_HUB_SUBDIR: str = "hub"


def _write_path_env(cache: ModelCacheConfig) -> None:
    """Write the cache-path env vars from a ModelCacheConfig."""
    os.environ["HF_HOME"] = str(cache.hf_home)
    os.environ["HF_HUB_CACHE"] = str(cache.hf_home / HF_HUB_SUBDIR)
    os.environ["TORCH_HOME"] = str(cache.torch_home)
    os.environ["DEEPFACE_HOME"] = str(cache.deepface_home)


def apply_offline_env(cache: ModelCacheConfig) -> None:
    """Pin cache paths and force offline mode for HF/Transformers."""
    _write_path_env(cache)
    os.environ["HF_HUB_OFFLINE"] = OFFLINE_FLAG
    os.environ["TRANSFORMERS_OFFLINE"] = OFFLINE_FLAG


def apply_online_env(cache: ModelCacheConfig) -> None:
    """Pin cache paths and clear offline flags so HF can fetch."""
    _write_path_env(cache)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)


def bootstrap_default() -> ModelCacheConfig:
    """Resolve cache from env, apply offline pins, return the config."""
    cache = ModelCacheConfig.from_env()
    apply_offline_env(cache)
    return cache
