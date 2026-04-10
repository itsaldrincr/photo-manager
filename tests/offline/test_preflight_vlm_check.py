"""Tests for VLM preflight integration inside run_setup."""

from __future__ import annotations

from pathlib import Path

import pytest

from cull import setup_command
from cull.config import MODEL_MANIFEST, VLM_DEFAULT_ALIAS, ModelCacheConfig
from cull.model_cache import BootstrapStatus, ConfigError
from cull.setup_command import SetupRequest, run_setup
from cull.vlm_registry import VLMResolutionError


def _build_cache(tmp_path: Path) -> ModelCacheConfig:
    """Return a ModelCacheConfig rooted at an empty tmp_path."""
    return ModelCacheConfig(
        root=tmp_path,
        hf_home=tmp_path / "hf",
        torch_home=tmp_path / "torch",
        deepface_home=tmp_path / "deepface",
        mediapipe_dir=tmp_path / "mediapipe",
    )


def _stub_ok_preflight(monkeypatch, cache_root: Path) -> None:
    """Stub run_preflight to always return missing on first call, ok on second."""
    missing = BootstrapStatus(
        state="missing",
        missing=list(MODEL_MANIFEST.keys()),
        cache_root=cache_root,
    )
    ok = BootstrapStatus(state="ok", missing=[], corrupt=[], cache_root=cache_root)
    responses = iter([missing, ok])
    monkeypatch.setattr(setup_command, "run_preflight", lambda c: next(responses))


def _install_noop_fetchers(monkeypatch) -> None:
    """Replace all three fetchers with no-ops."""
    monkeypatch.setattr(setup_command, "fetch_hf_repo", lambda entry, cache: None)
    monkeypatch.setattr(setup_command, "fetch_url_file", lambda entry, cache: None)
    monkeypatch.setattr(setup_command, "warm_pyiqa_metrics", lambda: None)


def test_run_setup_passes_vlm_preflight(tmp_path: Path, monkeypatch) -> None:
    """run_setup completes successfully when VLM preflight is a no-op."""
    cache = _build_cache(tmp_path)
    _install_noop_fetchers(monkeypatch)
    _stub_ok_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(setup_command, "write_bootstrap_marker", lambda c: None)
    monkeypatch.setattr(
        "cull.setup_command.vlm_registry.run_vlm_preflight",
        lambda alias: None,
    )
    result = run_setup(SetupRequest(cache=cache, allow_network=True))
    assert result.status.state == "ok"


def _raise_vlm_resolution_error(alias: str) -> None:
    """Stub that always raises VLMResolutionError."""
    raise VLMResolutionError("missing")


def test_run_setup_fails_on_missing_vlm_alias(tmp_path: Path, monkeypatch) -> None:
    """run_setup raises ConfigError naming VLM_DEFAULT_ALIAS when VLM preflight fails."""
    cache = _build_cache(tmp_path)
    _install_noop_fetchers(monkeypatch)
    _stub_ok_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(setup_command, "write_bootstrap_marker", lambda c: None)
    monkeypatch.setattr(
        "cull.setup_command.vlm_registry.run_vlm_preflight",
        _raise_vlm_resolution_error,
    )
    with pytest.raises(ConfigError) as exc_info:
        run_setup(SetupRequest(cache=cache, allow_network=True))
    assert VLM_DEFAULT_ALIAS in str(exc_info.value)
