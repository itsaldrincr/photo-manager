"""Tests that cull.clip_loader passes cache_dir + local_files_only=True to transformers."""

from __future__ import annotations

import pytest

from cull import clip_loader


@pytest.fixture(autouse=True)
def _reset_clip_singletons():
    """Clear the clip_loader module singletons before and after each test."""
    clip_loader.unload()
    yield
    clip_loader.unload()


class _LoaderRecorder:
    """Captures from_pretrained kwargs for offline-kwargs assertions."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def __call__(self, model_id: str, **kwargs) -> "_LoaderRecorder":
        """Record kwargs and return self so chained .to(device) is a no-op."""
        self.kwargs = kwargs
        return self

    def to(self, device: str) -> "_LoaderRecorder":
        """No-op device-move so the loader path in get_clip_model succeeds."""
        return self


def test_get_clip_model_passes_cache_kwargs(monkeypatch) -> None:
    """get_clip_model must forward cache_dir + local_files_only=True to CLIPModel."""
    import transformers  # noqa: PLC0415

    recorder = _LoaderRecorder()
    monkeypatch.setattr(
        transformers.CLIPModel, "from_pretrained",
        classmethod(lambda cls, model_id, **kw: recorder(model_id, **kw)),
    )
    monkeypatch.setattr(clip_loader, "select_device", lambda: "cpu")
    clip_loader.get_clip_model()
    expected_cache = str(clip_loader._CACHE.hf_home / "hub")
    assert recorder.kwargs.get("cache_dir") == expected_cache
    assert recorder.kwargs.get("local_files_only") is True


def test_get_clip_processor_passes_cache_kwargs(monkeypatch) -> None:
    """get_clip_processor must forward cache_dir + local_files_only=True to CLIPProcessor."""
    import transformers  # noqa: PLC0415

    recorder = _LoaderRecorder()
    monkeypatch.setattr(
        transformers.CLIPProcessor, "from_pretrained",
        classmethod(lambda cls, model_id, **kw: recorder(model_id, **kw)),
    )
    clip_loader.get_clip_processor()
    expected_cache = str(clip_loader._CACHE.hf_home / "hub")
    assert recorder.kwargs.get("cache_dir") == expected_cache
    assert recorder.kwargs.get("local_files_only") is True
