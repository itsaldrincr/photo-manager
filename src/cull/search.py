"""Semantic image search via CLIP embeddings.

NOTE: First use triggers an ~890 MB one-time download of CLIP-L weights.
Subsequent calls reuse the cull.clip_loader singleton.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from cull.clip_loader import get_clip_model, get_clip_processor
from cull.config import (
    CLIP_MODEL_ID,
    EMBEDDING_CACHE_FILENAME,
    EMBEDDING_INDEX_FILENAME,
    JPEG_EXTENSIONS,
    SEARCH_TOP_K_DEFAULT,
)
from cull.io_silence import _silence_stdio
from cull.models import SearchRequest, SearchResult
from cull.stage2.iqa import select_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic bundles for multi-param helpers
# ---------------------------------------------------------------------------


class _EmbedContext(BaseModel):
    """Context bundle for _embed_single."""

    model: object
    processor: object
    device: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class _RankInput(BaseModel):
    """Input bundle for _rank_by_similarity."""

    query_vec: object
    embeddings: object
    cached_paths: list[str]
    top_k: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _to_feature_tensor(features: object) -> object:
    """Return a raw feature tensor from either HF output objects or direct tensors."""
    return getattr(features, "pooler_output", features)


def _scan_jpegs(source: Path) -> list[Path]:
    """Return sorted list of all JPEG files found recursively under source."""
    found: list[Path] = []
    for ext in JPEG_EXTENSIONS:
        found.extend(source.rglob(f"*{ext}"))
        found.extend(source.rglob(f"*{ext.upper()}"))
    return sorted(set(found))


def _cache_valid(source: Path, paths: list[Path]) -> bool:
    """Return True if both cache files exist, model matches, and mtimes are valid."""
    emb_path = source / EMBEDDING_CACHE_FILENAME
    idx_path = source / EMBEDDING_INDEX_FILENAME
    if not emb_path.exists() or not idx_path.exists():
        return False
    index: dict = json.loads(idx_path.read_text())
    if index.get("model_id") != CLIP_MODEL_ID:
        return False
    if set(index.get("paths", [])) != {str(p) for p in paths}:
        return False
    if not paths:
        return True
    max_jpeg_mtime = max(p.stat().st_mtime for p in paths)
    cache_mtime = idx_path.stat().st_mtime
    return max_jpeg_mtime <= cache_mtime


def _load_cache(source: Path) -> tuple[np.ndarray, list[str]]:
    """Load and return (embeddings, paths) from the on-disk cache."""
    emb_path = source / EMBEDDING_CACHE_FILENAME
    idx_path = source / EMBEDDING_INDEX_FILENAME
    embeddings: np.ndarray = np.load(str(emb_path))
    index: dict = json.loads(idx_path.read_text())
    return embeddings, index["paths"]


def _embed_single(path: Path, ctx: _EmbedContext) -> np.ndarray:
    """Return the L2-normalized CLIP image embedding for a single JPEG."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    image = Image.open(path).convert("RGB")
    with _silence_stdio():
        inputs = ctx.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(ctx.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = _to_feature_tensor(ctx.model.get_image_features(**inputs))
    vec: np.ndarray = features.cpu().numpy().squeeze()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _build_embeddings(paths: list[Path], device: str) -> np.ndarray:
    """Build L2-normalized CLIP embeddings matrix of shape (N, D) for all paths."""
    ctx = _EmbedContext(model=get_clip_model(), processor=get_clip_processor(), device=device)
    vectors = [_embed_single(p, ctx) for p in paths]
    return np.stack(vectors, axis=0).astype(np.float32)


def _save_cache(source: Path, embeddings: np.ndarray, paths: list[Path]) -> None:
    """Write embeddings .npy and index .json files to source directory."""
    emb_path = source / EMBEDDING_CACHE_FILENAME
    idx_path = source / EMBEDDING_INDEX_FILENAME
    np.save(str(emb_path), embeddings)
    index = {
        "model_id": CLIP_MODEL_ID,
        "paths": [str(p) for p in paths],
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    idx_path.write_text(json.dumps(index))
    logger.info("Saved embedding cache: %d images → %s", len(paths), emb_path)


def _cache_dimensions_match(embeddings: np.ndarray, cached_paths: list[str]) -> bool:
    """Return True if embeddings row count matches the number of cached paths."""
    return len(embeddings) == len(cached_paths)


def _load_or_build_cache(source: Path) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings, paths) from cache if valid, else build and save."""
    paths = _scan_jpegs(source)
    if _cache_valid(source, paths):
        logger.debug("Using existing embedding cache for %s", source)
        embeddings, cached_paths = _load_cache(source)
        if _cache_dimensions_match(embeddings, cached_paths):
            return embeddings, cached_paths
        logger.warning("Cache dimension mismatch for %s — rebuilding", source)
    logger.info("Building embedding cache for %d images in %s", len(paths), source)
    embeddings = _build_embeddings(paths, select_device())
    _save_cache(source, embeddings, paths)
    return embeddings, [str(p) for p in paths]


def _encode_text(query: str, device: str) -> np.ndarray:
    """Return L2-normalized CLIP text embedding of shape (D,) for the query."""
    import torch  # noqa: PLC0415

    model = get_clip_model()
    processor = get_clip_processor()
    with _silence_stdio():
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = _to_feature_tensor(model.get_text_features(**inputs))
    vec: np.ndarray = features.cpu().numpy().squeeze()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _encode_reference(path: Path, device: str) -> np.ndarray:
    """Return L2-normalized CLIP image embedding of shape (D,) for a single image."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    model = get_clip_model()
    processor = get_clip_processor()
    image = Image.open(path).convert("RGB")
    with _silence_stdio():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = _to_feature_tensor(model.get_image_features(**inputs))
    vec: np.ndarray = features.cpu().numpy().squeeze()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _rank_by_similarity(rank_in: _RankInput) -> list[SearchResult]:
    """Compute cosine scores, argsort descending, return top_k SearchResults."""
    scores: np.ndarray = rank_in.embeddings @ rank_in.query_vec
    ranked_indices = np.argsort(scores)[::-1][:rank_in.top_k]
    return [
        SearchResult(
            path=Path(rank_in.cached_paths[i]),
            similarity=float(scores[i]),
            rank=rank + 1,
        )
        for rank, i in enumerate(ranked_indices)
    ]


def search_text(req: SearchRequest) -> list[SearchResult]:
    """Search the source folder by text query; returns top-k ranked results."""
    if req.query_text is None:
        raise ValueError("SearchRequest.query_text must be set for text search")
    if req.reference_path is not None:
        raise ValueError("Provide either query_text or reference_path, not both")
    embeddings, cached_paths = _load_or_build_cache(req.source)
    device = select_device()
    query_vec = _encode_text(req.query_text, device)
    return _rank_by_similarity(_RankInput(query_vec=query_vec, embeddings=embeddings, cached_paths=cached_paths, top_k=req.top_k))


def search_similar(req: SearchRequest) -> list[SearchResult]:
    """Search the source folder by image similarity; returns top-k ranked results."""
    if req.reference_path is None:
        raise ValueError("SearchRequest.reference_path must be set for similarity search")
    if req.query_text is not None:
        raise ValueError("Provide either query_text or reference_path, not both")
    embeddings, cached_paths = _load_or_build_cache(req.source)
    device = select_device()
    query_vec = _encode_reference(req.reference_path, device)
    return _rank_by_similarity(_RankInput(query_vec=query_vec, embeddings=embeddings, cached_paths=cached_paths, top_k=req.top_k))


def search_by_text(req: SearchRequest) -> list[SearchResult]:
    """Search source folder by text query using CLIP."""
    return search_text(req)


def search_by_similarity(req: SearchRequest) -> list[SearchResult]:
    """Search source folder by image similarity using CLIP."""
    return search_similar(req)
