"""Tests for MMR-based diversity selection.

Synthesises a 20x768 embedding matrix in 3 tight clusters and verifies
that MMR with target_count=3 selects exactly one representative per cluster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cull.stage4.diversity import MmrContext, MmrInput, select

EMBEDDING_DIM: int = 768
IMAGES_PER_CLUSTER: int = 7
CLUSTER_NOISE_SCALE: float = 0.01
CLUSTER_COUNT: int = 3
TOTAL_IMAGES: int = 20
RANDOM_SEED: int = 42
LAMBDA_QUALITY: float = 0.5


def _make_cluster_center(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Return a single L2-normalised random unit vector."""
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _make_cluster_embeddings(
    center: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return IMAGES_PER_CLUSTER L2-normalised embeddings near center."""
    noise = rng.standard_normal((IMAGES_PER_CLUSTER, center.shape[0])).astype(np.float32)
    vecs = center + CLUSTER_NOISE_SCALE * noise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _build_synthetic_dataset() -> tuple[np.ndarray, list[Path], dict[str, int]]:
    """Return (embeddings, paths, path_to_row) for a 3-cluster synthetic set."""
    rng = np.random.default_rng(RANDOM_SEED)
    centers = [_make_cluster_center(rng, EMBEDDING_DIM) for _ in range(CLUSTER_COUNT)]
    cluster_blocks = [_make_cluster_embeddings(c, rng) for c in centers]
    # Fill to TOTAL_IMAGES with extra random images from cluster 0
    extra_count = TOTAL_IMAGES - CLUSTER_COUNT * IMAGES_PER_CLUSTER
    extra = _make_cluster_embeddings(centers[0], rng)[:extra_count]
    all_embeddings = np.vstack(cluster_blocks + [extra])
    paths = [Path(f"/img/photo_{i:03d}.jpg") for i in range(len(all_embeddings))]
    path_to_row = {str(p): i for i, p in enumerate(paths)}
    return all_embeddings, paths, path_to_row


def _cluster_label(path_index: int) -> int:
    """Return 0/1/2 cluster label for a given path index (by construction)."""
    if path_index < IMAGES_PER_CLUSTER:
        return 0
    if path_index < 2 * IMAGES_PER_CLUSTER:
        return 1
    if path_index < 3 * IMAGES_PER_CLUSTER:
        return 2
    return 0  # extra images belong to cluster 0


def test_mmr_selects_one_per_cluster() -> None:
    """MMR with target_count=3 must pick one representative from each cluster."""
    embeddings, paths, path_to_row = _build_synthetic_dataset()
    scores = {str(p): 1.0 for p in paths}
    mmr_input = MmrInput(candidates=paths, scores=scores)
    context = MmrContext(
        embeddings=embeddings,
        path_to_row=path_to_row,
        lambda_quality=LAMBDA_QUALITY,
        target_count=CLUSTER_COUNT,
    )

    selected = select(mmr_input, context)

    assert len(selected) == CLUSTER_COUNT, f"Expected {CLUSTER_COUNT}, got {len(selected)}"
    selected_cluster_labels = {
        _cluster_label(path_to_row[str(p)]) for p in selected
    }
    assert selected_cluster_labels == {0, 1, 2}, (
        f"Expected one representative from each cluster; got labels {selected_cluster_labels}"
    )


def test_mmr_empty_candidates_returns_empty() -> None:
    """MMR with no candidates returns an empty list without error."""
    embeddings = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
    context = MmrContext(
        embeddings=embeddings,
        path_to_row={},
        lambda_quality=LAMBDA_QUALITY,
        target_count=CLUSTER_COUNT,
    )
    mmr_input = MmrInput(candidates=[], scores={})
    result = select(mmr_input, context)
    assert result == []
