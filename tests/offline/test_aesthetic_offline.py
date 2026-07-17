"""Tests that cull.stage2.aesthetic resolves model weights from the cache root."""

from __future__ import annotations

import pytest

from cull import clip_loader
from cull.config import ModelCacheConfig
from cull.stage2 import aesthetic


@pytest.fixture(autouse=True)
def _reset_aesthetic_singletons():
    """Clear the aesthetic head and shared CLIP singletons before and after each test."""
    aesthetic.unload_predictor()
    clip_loader.unload()
    yield
    aesthetic.unload_predictor()
    clip_loader.unload()


class _ProcessorRecorder:
    """Captures kwargs from CLIPProcessor.from_pretrained."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def __call__(self, model_id: str, **kwargs) -> "_ProcessorRecorder":
        """Record kwargs and return self."""
        self.kwargs = kwargs
        return self


def _make_fake_cache(tmp_path) -> ModelCacheConfig:
    """Build a ModelCacheConfig rooted at a pytest tmp_path."""
    return ModelCacheConfig(
        root=tmp_path,
        hf_home=tmp_path / "hf",
        torch_home=tmp_path / "torch",
        emotieff_dir=tmp_path / "emotieff",
        mediapipe_dir=tmp_path / "mediapipe",
    )


def test_aesthetic_head_locates_weights_under_cache_root(tmp_path, monkeypatch) -> None:
    """_locate_head_weights must resolve the safetensors file under the configured cache root."""
    fake_cache = _make_fake_cache(tmp_path)
    monkeypatch.setattr(aesthetic, "_CACHE", fake_cache)
    flat = aesthetic.AESTHETIC_MODEL_ID.replace("/", "--")
    snapshot_dir = fake_cache.hf_home / "hub" / f"models--{flat}" / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / aesthetic.AESTHETIC_HEAD_FILENAME).write_bytes(b"")

    weight_path = aesthetic._locate_head_weights()

    expected_cache = str(fake_cache.hf_home / "hub")
    assert weight_path.startswith(expected_cache)


def test_aesthetic_processor_passes_cache_kwargs(monkeypatch) -> None:
    """clip_loader.get_clip_processor (used for aesthetic embeddings) forwards cache kwargs."""
    proc_rec = _ProcessorRecorder()
    import transformers  # noqa: PLC0415

    monkeypatch.setattr(
        transformers.CLIPProcessor,
        "from_pretrained",
        classmethod(lambda cls, mid, **kw: proc_rec(mid, **kw)),
    )

    clip_loader.get_clip_processor()

    expected_cache = str(clip_loader._CACHE.hf_home / "hub")
    assert proc_rec.kwargs.get("cache_dir") == expected_cache
    assert proc_rec.kwargs.get("local_files_only") is True
