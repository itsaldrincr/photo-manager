"""Tests for cull.search semantic search module (CLIP mocked)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cull.config import (
    CLIP_MODEL_ID,
    EMBEDDING_CACHE_FILENAME,
    EMBEDDING_INDEX_FILENAME,
)
from cull.models import SearchRequest
from cull.search import (
    _RankInput,
    _cache_valid,
    _load_cache,
    _rank_by_similarity,
    search_by_similarity,
    search_by_text,
)


# ---------------------------------------------------------------------------
# Stub CLIP objects
# ---------------------------------------------------------------------------


class StubCLIP:
    """Deterministic CLIP model stub returning unit vectors."""

    def get_text_features(self, **kwargs: object) -> torch.Tensor:  # type: ignore[override]
        return torch.tensor([[1.0, 0.0, 0.0]])

    def get_image_features(self, **kwargs: object) -> torch.Tensor:  # type: ignore[override]
        return torch.tensor([[1.0, 0.0, 0.0]])

    def to(self, device: str) -> "StubCLIP":
        return self


class StubProcessor:
    """Deterministic CLIP processor stub returning fixed tensors."""

    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.zeros(1, 1, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 224, 224),
        }


def _make_jpeg(path: Path) -> Path:
    """Write a minimal valid JPEG using Pillow and return its path."""
    from PIL import Image

    img = Image.new("RGB", (10, 10))
    img.save(str(path), format="JPEG")
    return path


def _write_valid_cache(source: Path, paths: list[Path]) -> None:
    """Write a valid cache pair (.npy + .json) for the given paths."""
    n = len(paths)
    embeddings = np.ones((n, 3), dtype=np.float32)
    np.save(str(source / EMBEDDING_CACHE_FILENAME), embeddings)
    index = {
        "model_id": CLIP_MODEL_ID,
        "paths": [str(p) for p in paths],
        "built_at": "2026-01-01T00:00:00+00:00",
    }
    (source / EMBEDDING_INDEX_FILENAME).write_text(json.dumps(index))


# ---------------------------------------------------------------------------
# Test 1: cosine similarity ranking (pure numpy)
# ---------------------------------------------------------------------------


def test_cosine_similarity_ranking() -> None:
    """_rank_by_similarity returns results ordered by descending cosine score."""
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    paths = ["a.jpg", "b.jpg", "c.jpg"]
    rank_in = _RankInput(query_vec=query, embeddings=embeddings, cached_paths=paths, top_k=3)
    results = _rank_by_similarity(rank_in)
    assert results[0].rank == 1
    assert results[0].path == Path("a.jpg")
    assert results[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert results[1].rank == 2
    assert results[2].rank == 3
    assert results[1].similarity == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 2: search_by_text returns top_k ranked results
# ---------------------------------------------------------------------------


def test_search_by_text_returns_top_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """search_by_text returns exactly top_k results with ascending rank order."""
    jpegs = [_make_jpeg(tmp_path / f"img{i}.jpg") for i in range(5)]
    _write_valid_cache(tmp_path, jpegs)

    monkeypatch.setattr("cull.search.get_clip_model", lambda: StubCLIP())
    monkeypatch.setattr("cull.search.get_clip_processor", lambda: StubProcessor())

    req = SearchRequest(query_text="golden hour", source=tmp_path, top_k=3)
    results = search_by_text(req)
    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Test 3: search_by_similarity puts reference image first
# ---------------------------------------------------------------------------


def test_search_by_similarity_includes_reference_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reference image should rank #1 with similarity ~1.0."""
    ref = _make_jpeg(tmp_path / "ref.jpg")
    other = _make_jpeg(tmp_path / "other.jpg")
    _write_valid_cache(tmp_path, [ref, other])

    monkeypatch.setattr("cull.search.get_clip_model", lambda: StubCLIP())
    monkeypatch.setattr("cull.search.get_clip_processor", lambda: StubProcessor())

    req = SearchRequest(reference_path=ref, source=tmp_path, top_k=2)
    results = search_by_similarity(req)
    assert len(results) == 2
    assert results[0].rank == 1
    assert results[0].similarity == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 4: cache is reused when JPEGs are unchanged
# ---------------------------------------------------------------------------


def test_cache_reused_when_jpegs_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When disk cache is valid, _build_embeddings is not called on second search."""
    jpegs = [_make_jpeg(tmp_path / f"img{i}.jpg") for i in range(2)]
    _write_valid_cache(tmp_path, jpegs)

    build_count = {"n": 0}
    original_build = __import__("cull.search", fromlist=["_build_embeddings"])._build_embeddings

    def counting_build(paths: list[Path], device: str) -> np.ndarray:
        build_count["n"] += 1
        return original_build(paths, device)

    monkeypatch.setattr("cull.search._build_embeddings", counting_build)
    monkeypatch.setattr("cull.search.get_clip_model", lambda: StubCLIP())
    monkeypatch.setattr("cull.search.get_clip_processor", lambda: StubProcessor())

    req = SearchRequest(query_text="forest", source=tmp_path, top_k=2)
    search_by_text(req)
    assert build_count["n"] == 0, "_build_embeddings should not run when cache is valid"


# ---------------------------------------------------------------------------
# Test 5: cache is invalidated when a new JPEG is added
# ---------------------------------------------------------------------------


def test_cache_invalidated_when_jpeg_added(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding a new JPEG after cache is written triggers a cache rebuild."""
    original = _make_jpeg(tmp_path / "img0.jpg")
    _write_valid_cache(tmp_path, [original])

    new_jpeg = _make_jpeg(tmp_path / "img_new.jpg")

    call_count = {"n": 0}

    def counting_get_model() -> object:
        call_count["n"] += 1
        return StubCLIP()

    monkeypatch.setattr("cull.search.get_clip_model", counting_get_model)
    monkeypatch.setattr("cull.search.get_clip_processor", lambda: StubProcessor())
    monkeypatch.setattr("cull.search.select_device", lambda: "cpu")

    assert not _cache_valid(tmp_path, [original, new_jpeg])
    req = SearchRequest(query_text="test", source=tmp_path, top_k=1)
    search_by_text(req)
    assert call_count["n"] > 0, "get_clip_model should be called to rebuild cache"


# ---------------------------------------------------------------------------
# Test 6: cache is invalidated when model_id differs
# ---------------------------------------------------------------------------


def test_cache_invalidated_on_model_id_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache index with a different model_id triggers a rebuild."""
    jpeg = _make_jpeg(tmp_path / "img.jpg")
    emb = np.ones((1, 3), dtype=np.float32)
    np.save(str(tmp_path / EMBEDDING_CACHE_FILENAME), emb)
    stale_index = {
        "model_id": "old/model",
        "paths": [str(jpeg)],
        "built_at": "2026-01-01T00:00:00+00:00",
    }
    (tmp_path / EMBEDDING_INDEX_FILENAME).write_text(json.dumps(stale_index))

    assert not _cache_valid(tmp_path, [jpeg])

    call_count = {"n": 0}

    def counting_get_model() -> object:
        call_count["n"] += 1
        return StubCLIP()

    monkeypatch.setattr("cull.search.get_clip_model", counting_get_model)
    monkeypatch.setattr("cull.search.get_clip_processor", lambda: StubProcessor())
    monkeypatch.setattr("cull.search.select_device", lambda: "cpu")

    req = SearchRequest(query_text="sunset", source=tmp_path, top_k=1)
    search_by_text(req)
    assert call_count["n"] > 0, "get_clip_model should be called when model_id mismatches"


# ---------------------------------------------------------------------------
# Test 7: ValueError raised when both text and reference are set
# ---------------------------------------------------------------------------


def test_search_text_rejects_conflicting_inputs(tmp_path: Path) -> None:
    """search_text raises ValueError when both query_text and reference_path are set."""
    from cull.search import search_text

    req = SearchRequest(
        query_text="hello",
        reference_path=tmp_path / "img.jpg",
        source=tmp_path,
        top_k=5,
    )
    with pytest.raises(ValueError):
        search_text(req)
