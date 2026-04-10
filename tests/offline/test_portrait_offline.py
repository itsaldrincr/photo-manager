"""Tests for cull.stage2.portrait: face_landmarker resolver + DeepFace fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from cull.config import FACE_LANDMARKER_FILENAME, ModelCacheConfig
from cull.model_cache import ConfigError
from cull.stage2 import portrait
from cull.stage2.portrait import _resolve_face_landmarker_path, detect_expression


def _build_cache(tmp_path: Path) -> ModelCacheConfig:
    """Return a ModelCacheConfig rooted at an empty tmp_path."""
    return ModelCacheConfig(
        root=tmp_path,
        hf_home=tmp_path / "hf",
        torch_home=tmp_path / "torch",
        deepface_home=tmp_path / "deepface",
        mediapipe_dir=tmp_path / "mediapipe",
    )


def test_resolve_face_landmarker_path_raises_on_missing(tmp_path: Path) -> None:
    """_resolve_face_landmarker_path must raise ConfigError mentioning 'cull setup'."""
    cache = _build_cache(tmp_path)
    with pytest.raises(ConfigError) as excinfo:
        _resolve_face_landmarker_path(cache)
    assert "cull setup" in str(excinfo.value)


def test_resolve_face_landmarker_path_returns_existing(tmp_path: Path) -> None:
    """_resolve_face_landmarker_path must return the path when the .task file exists."""
    cache = _build_cache(tmp_path)
    cache.mediapipe_dir.mkdir(parents=True, exist_ok=True)
    expected = cache.mediapipe_dir / FACE_LANDMARKER_FILENAME
    expected.write_bytes(b"fake face landmarker task payload")
    result = _resolve_face_landmarker_path(cache)
    assert result == expected


class _FakeDeepFace:
    """Stand-in for the deepface.DeepFace module whose analyze always raises."""

    @staticmethod
    def analyze(path: str, **kwargs: object) -> list:
        """Simulate a DeepFace analysis failure."""
        raise RuntimeError("simulated deepface failure")


def test_detect_expression_handles_deepface_failure(monkeypatch) -> None:
    """detect_expression must swallow DeepFace errors and return an empty string."""
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    fake_module = types.ModuleType("deepface")
    fake_module.DeepFace = _FakeDeepFace  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepface", fake_module)
    result = detect_expression(Path("/tmp/does-not-exist.jpg"))
    assert result == ""


def test_detect_expression_returns_dominant_emotion(monkeypatch) -> None:
    """detect_expression must return the dominant_emotion value from DeepFace."""
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    class _HappyDeepFace:
        @staticmethod
        def analyze(path: str, **kwargs: object) -> list:
            """Return a single-face analysis result with dominant_emotion=happy."""
            return [{"dominant_emotion": "happy"}]

    fake_module = types.ModuleType("deepface")
    fake_module.DeepFace = _HappyDeepFace  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepface", fake_module)
    assert detect_expression(Path("/tmp/any.jpg")) == "happy"
