"""Tests for cull.env_bootstrap: offline env vars + side-effect-free module body."""

from __future__ import annotations

import ast
import os
from pathlib import Path

from cull.config import ModelCacheConfig
from cull.env_bootstrap import (
    apply_offline_env,
    apply_online_env,
    bootstrap_default,
)

OFFLINE_VAR: str = "HF_HUB_OFFLINE"
TRANSFORMERS_VAR: str = "TRANSFORMERS_OFFLINE"
CACHE_OVERRIDE_VAR: str = "PHOTO_MANAGER_CACHE"
PATH_VARS: tuple[str, ...] = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "TORCH_HOME",
    "DEEPFACE_HOME",
)
ENV_BOOTSTRAP_PATH: Path = (
    Path(__file__).resolve().parents[2] / "src" / "cull" / "env_bootstrap.py"
)


def _isolate_env(monkeypatch, tmp_path: Path) -> None:
    """Wipe bootstrap env vars and pin PHOTO_MANAGER_CACHE to a tmp dir."""
    for var in (OFFLINE_VAR, TRANSFORMERS_VAR, CACHE_OVERRIDE_VAR, *PATH_VARS):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(CACHE_OVERRIDE_VAR, str(tmp_path))


def test_apply_offline_env_sets_hf_hub_offline(monkeypatch, tmp_path: Path) -> None:
    """apply_offline_env must set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1."""
    _isolate_env(monkeypatch, tmp_path)
    cache = ModelCacheConfig.from_env()
    apply_offline_env(cache)
    assert os.environ[OFFLINE_VAR] == "1"
    assert os.environ[TRANSFORMERS_VAR] == "1"


def test_apply_online_env_pops_offline_flags(monkeypatch, tmp_path: Path) -> None:
    """apply_online_env must remove HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE."""
    _isolate_env(monkeypatch, tmp_path)
    cache = ModelCacheConfig.from_env()
    apply_offline_env(cache)
    apply_online_env(cache)
    assert OFFLINE_VAR not in os.environ
    assert TRANSFORMERS_VAR not in os.environ


def test_bootstrap_default_returns_config(monkeypatch, tmp_path: Path) -> None:
    """bootstrap_default must return a ModelCacheConfig and pin HF_HUB_OFFLINE=1."""
    _isolate_env(monkeypatch, tmp_path)
    result = bootstrap_default()
    assert isinstance(result, ModelCacheConfig)
    assert os.environ[OFFLINE_VAR] == "1"


def _module_body_has_no_top_level_calls() -> bool:
    """Walk env_bootstrap.py AST and confirm no top-level call expressions exist."""
    tree = ast.parse(ENV_BOOTSTRAP_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return False
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is not None and isinstance(node.value, ast.Call):
                return False
    return True


def test_module_import_has_no_side_effects() -> None:
    """env_bootstrap.py must not mutate os.environ at import time."""
    assert _module_body_has_no_top_level_calls()
