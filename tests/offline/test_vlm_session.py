# stdlib
import json
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

# third-party
import pytest
from PIL import Image

# local
from cull.vlm_registry import VLMEntry
from cull.vlm_session import VlmGenerateInput, VlmLoadError, VlmSession


def _make_fake_vlm_dir(root: Path) -> Path:
    """Create a minimal fake VLM directory with config.json."""
    vlm_dir = root / "TestModel"
    vlm_dir.mkdir(parents=True, exist_ok=True)
    config = {"vision_config": {}, "model_type": "qwen3_vl"}
    (vlm_dir / "config.json").write_text(json.dumps(config))
    return vlm_dir


def _make_fake_mlx_vlm(sentinel_model: object, sentinel_processor: object) -> types.ModuleType:
    """Build a fake mlx_vlm module with load + generate + apply_chat_template."""
    fake = types.ModuleType("mlx_vlm")
    fake.load = lambda path: (sentinel_model, sentinel_processor)
    fake.apply_chat_template = lambda processor, config, prompt, num_images: prompt
    fake.generate = MagicMock(return_value=SimpleNamespace(text=""))
    return fake


def _patch_fake_mlx_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake mlx.core module so session.unload() never touches real MLX."""
    fake_core = types.ModuleType("mlx.core")
    fake_core.clear_cache = lambda: None
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core
    monkeypatch.setitem(__import__("sys").modules, "mlx", fake_mlx)
    monkeypatch.setitem(__import__("sys").modules, "mlx.core", fake_core)


def _patch_registry(monkeypatch: pytest.MonkeyPatch, vlm_dir: Path) -> None:
    """Monkeypatch resolve_alias in vlm_session so 'test-alias' resolves directly."""
    monkeypatch.setattr(
        "cull.vlm_session.resolve_alias",
        lambda alias: VLMEntry(alias=alias, directory=vlm_dir, display_name=vlm_dir.name),
    )


def test_vlm_session_cm_calls_load_and_unload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vlm_dir = _make_fake_vlm_dir(tmp_path)
    _patch_registry(monkeypatch, vlm_dir)

    sentinel_model = object()
    sentinel_processor = object()
    fake = _make_fake_mlx_vlm(sentinel_model, sentinel_processor)
    _patch_fake_mlx_core(monkeypatch)
    monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", fake)

    from cull.vlm_session import vlm_session

    with vlm_session("test-alias") as session:
        assert isinstance(session, VlmSession)
        assert session.model is sentinel_model
        assert session.processor is sentinel_processor

    assert session.model is None
    assert session.processor is None
    assert session.config is None


def test_vlm_session_load_failure_raises_VlmLoadError(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vlm_dir = _make_fake_vlm_dir(tmp_path)
    _patch_registry(monkeypatch, vlm_dir)

    fake = types.ModuleType("mlx_vlm")
    fake.load = MagicMock(side_effect=RuntimeError("boom"))
    _patch_fake_mlx_core(monkeypatch)
    monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", fake)

    from cull.vlm_session import vlm_session

    with pytest.raises(VlmLoadError) as exc_info:
        with vlm_session("test-alias"):
            pass

    msg = str(exc_info.value)
    assert "test-alias" in msg
    assert str(vlm_dir) in msg


def test_vlm_session_generate_passes_images_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vlm_dir = _make_fake_vlm_dir(tmp_path)
    _patch_registry(monkeypatch, vlm_dir)

    recorded: dict = {}

    def fake_generate(model, processor, prompt, **kwargs):  # type: ignore[no-untyped-def]
        recorded.update(kwargs)
        return SimpleNamespace(text="ok")

    sentinel_model = object()
    sentinel_processor = object()
    fake = types.ModuleType("mlx_vlm")
    fake.load = lambda path: (sentinel_model, sentinel_processor)
    fake.apply_chat_template = lambda processor, config, prompt, num_images: prompt
    fake.generate = fake_generate
    _patch_fake_mlx_core(monkeypatch)
    monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", fake)

    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(p1, "JPEG")
    Image.new("RGB", (8, 8), color=(4, 5, 6)).save(p2, "JPEG")

    from cull.vlm_session import vlm_session

    with vlm_session("test-alias") as session:
        session.generate(VlmGenerateInput(prompt="describe", images=[p1, p2]))

    assert "image" in recorded
    assert len(recorded["image"]) == 2
