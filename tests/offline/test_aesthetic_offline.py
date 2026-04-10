"""Tests that cull.stage2.aesthetic passes cache kwargs to head + processor loads."""

from __future__ import annotations

import pytest

from cull.stage2 import aesthetic


@pytest.fixture(autouse=True)
def _reset_aesthetic_singletons():
    """Clear the aesthetic module head/processor caches before and after each test."""
    aesthetic.unload_predictor()
    yield
    aesthetic.unload_predictor()


class _HeadRecorder:
    """Captures kwargs from AestheticsPredictorV2Linear.from_pretrained."""

    def __init__(self) -> None:
        self.kwargs: dict = {}
        self.layers = _FakeLayers()

    def __call__(self, model_id: str, **kwargs) -> "_HeadRecorder":
        """Record kwargs and return self so .layers access works downstream."""
        self.kwargs = kwargs
        return self


class _FakeLayers:
    """Stand-in for torch.nn.Module with a .to(device) method."""

    def to(self, device: str) -> "_FakeLayers":
        """No-op device move for the head-extraction path."""
        return self

    def eval(self) -> "_FakeLayers":
        """No-op eval switch for the head-extraction path."""
        return self


class _ProcessorRecorder:
    """Captures kwargs from CLIPProcessor.from_pretrained."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def __call__(self, model_id: str, **kwargs) -> "_ProcessorRecorder":
        """Record kwargs and return self."""
        self.kwargs = kwargs
        return self


def _install_head_stub(monkeypatch, recorder: _HeadRecorder) -> None:
    """Monkeypatch AestheticsPredictorV2Linear.from_pretrained to the recorder."""
    import aesthetics_predictor  # noqa: PLC0415

    monkeypatch.setattr(
        aesthetics_predictor.AestheticsPredictorV2Linear,
        "from_pretrained",
        classmethod(lambda cls, mid, **kw: recorder(mid, **kw)),
    )


def _install_processor_stub(monkeypatch, recorder: _ProcessorRecorder) -> None:
    """Monkeypatch CLIPProcessor.from_pretrained to the recorder."""
    import transformers  # noqa: PLC0415

    monkeypatch.setattr(
        transformers.CLIPProcessor,
        "from_pretrained",
        classmethod(lambda cls, mid, **kw: recorder(mid, **kw)),
    )


def test_aesthetic_head_passes_cache_kwargs(monkeypatch) -> None:
    """_extract_head must forward cache_dir + local_files_only=True to the linear head."""
    head_rec = _HeadRecorder()
    proc_rec = _ProcessorRecorder()
    _install_head_stub(monkeypatch, head_rec)
    _install_processor_stub(monkeypatch, proc_rec)
    aesthetic._get_head("cpu")
    expected_cache = str(aesthetic._CACHE.hf_home / "hub")
    assert head_rec.kwargs.get("cache_dir") == expected_cache
    assert head_rec.kwargs.get("local_files_only") is True


def test_aesthetic_processor_passes_cache_kwargs(monkeypatch) -> None:
    """_get_head must forward cache_dir + local_files_only=True to CLIPProcessor."""
    head_rec = _HeadRecorder()
    proc_rec = _ProcessorRecorder()
    _install_head_stub(monkeypatch, head_rec)
    _install_processor_stub(monkeypatch, proc_rec)
    aesthetic._get_head("cpu")
    expected_cache = str(aesthetic._CACHE.hf_home / "hub")
    assert proc_rec.kwargs.get("cache_dir") == expected_cache
    assert proc_rec.kwargs.get("local_files_only") is True
