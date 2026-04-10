# stdlib
import json
from pathlib import Path

# third-party
import pytest

# local
from cull.vlm_registry import (
    VLMResolutionError,
    VLMRegistry,
    discover_vlms,
    resolve_alias,
    run_vlm_preflight,
)


def _make_vlm_dir(root: Path, name: str, has_vision_config: bool = True) -> Path:
    """Create a fake VLM directory under root with config.json."""
    vlm_dir = root / name
    vlm_dir.mkdir(parents=True, exist_ok=True)
    config: dict = {"model_type": "qwen3_vl"}
    if has_vision_config:
        config["vision_config"] = {}
    (vlm_dir / "config.json").write_text(json.dumps(config))
    return vlm_dir


def test_discover_vlms_empty_root(tmp_path: Path) -> None:
    registry = discover_vlms(tmp_path)
    assert registry.entries == []
    assert registry.by_alias == {}


def test_discover_vlms_ignores_dir_without_config_json(tmp_path: Path) -> None:
    no_config_dir = tmp_path / "NoConfig-Model"
    no_config_dir.mkdir()
    registry = discover_vlms(tmp_path)
    assert registry.entries == []


def test_discover_vlms_ignores_config_without_vision_config_key(tmp_path: Path) -> None:
    _make_vlm_dir(tmp_path, "LlamaModel-7B", has_vision_config=False)
    registry = discover_vlms(tmp_path)
    assert registry.entries == []


def test_discover_vlms_accepts_vision_config_present(tmp_path: Path) -> None:
    _make_vlm_dir(tmp_path, "Qwen3-VL-4B-Instruct-MLX-8bit")
    registry = discover_vlms(tmp_path)
    assert len(registry.entries) == 1
    assert registry.entries[0].display_name == "Qwen3-VL-4B-Instruct-MLX-8bit"


def test_discover_vlms_auto_slug_collision(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _make_vlm_dir(tmp_path, "My Model")
    _make_vlm_dir(tmp_path, "My-Model")
    import logging
    with caplog.at_level(logging.WARNING, logger="cull.vlm_registry"):
        registry = discover_vlms(tmp_path)
    assert len(registry.entries) == 1
    assert "collision" in caplog.text.lower()


def test_resolve_alias_via_override_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_vlm_dir(tmp_path, "Qwen3-VL-4B-Instruct-MLX-8bit")
    monkeypatch.setattr(
        "cull.vlm_registry.VLM_ALIASES",
        {"qwen3-vl-4b": "Qwen3-VL-4B-Instruct-MLX-8bit"},
    )
    registry = discover_vlms(tmp_path)
    entry = resolve_alias("qwen3-vl-4b", registry)
    assert entry.alias == "qwen3-vl-4b"
    assert entry.display_name == "Qwen3-VL-4B-Instruct-MLX-8bit"


def test_resolve_alias_via_auto_slug_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_vlm_dir(tmp_path, "Qwen3-VL-4B")
    monkeypatch.setattr("cull.vlm_registry.VLM_ALIASES", {})
    registry = discover_vlms(tmp_path)
    entry = resolve_alias("qwen3-vl-4b", registry)
    assert entry.display_name == "Qwen3-VL-4B"


def test_resolve_alias_unknown_raises_VLMResolutionError(tmp_path: Path) -> None:
    registry = discover_vlms(tmp_path)
    with pytest.raises(VLMResolutionError, match="not found"):
        resolve_alias("nonexistent-model", registry)


def test_run_vlm_preflight_missing_directory_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr("cull.vlm_registry.VLM_MODELS_ROOT", missing)
    monkeypatch.setattr("cull.vlm_registry.VLM_ALIASES", {})
    monkeypatch.setattr("cull.vlm_registry.VLM_DEFAULT_ALIAS", "any-alias")
    with pytest.raises(VLMResolutionError):
        run_vlm_preflight("any-alias")


def test_run_vlm_preflight_missing_config_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vlm_dir = tmp_path / "SomeModel"
    vlm_dir.mkdir()
    # No config.json — directory is not discovered
    monkeypatch.setattr("cull.vlm_registry.VLM_MODELS_ROOT", tmp_path)
    monkeypatch.setattr("cull.vlm_registry.VLM_ALIASES", {})
    monkeypatch.setattr("cull.vlm_registry.VLM_DEFAULT_ALIAS", "somemodel")
    with pytest.raises(VLMResolutionError):
        run_vlm_preflight("somemodel")


def test_run_vlm_preflight_config_without_vision_config_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_vlm_dir(tmp_path, "SomeModel", has_vision_config=False)
    monkeypatch.setattr("cull.vlm_registry.VLM_MODELS_ROOT", tmp_path)
    monkeypatch.setattr("cull.vlm_registry.VLM_ALIASES", {})
    monkeypatch.setattr("cull.vlm_registry.VLM_DEFAULT_ALIAS", "somemodel")
    with pytest.raises(VLMResolutionError):
        run_vlm_preflight("somemodel")


def test_run_vlm_preflight_missing_weights_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_vlm_dir(tmp_path, "SomeModel", has_vision_config=True)
    # No weights file added
    monkeypatch.setattr("cull.vlm_registry.VLM_MODELS_ROOT", tmp_path)
    monkeypatch.setattr("cull.vlm_registry.VLM_ALIASES", {"somemodel": "SomeModel"})
    monkeypatch.setattr("cull.vlm_registry.VLM_DEFAULT_ALIAS", "somemodel")
    with pytest.raises(VLMResolutionError, match="No weights"):
        run_vlm_preflight("somemodel")


def test_run_vlm_preflight_valid_structure_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vlm_dir = _make_vlm_dir(tmp_path, "SomeModel", has_vision_config=True)
    (vlm_dir / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr("cull.vlm_registry.VLM_MODELS_ROOT", tmp_path)
    monkeypatch.setattr("cull.vlm_registry.VLM_ALIASES", {"somemodel": "SomeModel"})
    monkeypatch.setattr("cull.vlm_registry.VLM_DEFAULT_ALIAS", "somemodel")
    run_vlm_preflight("somemodel")  # must not raise
