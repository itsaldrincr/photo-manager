"""Similarity-based clustering of keeper images using MobileNetV3 embeddings."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.cluster.hierarchy import fcluster, linkage

logger = logging.getLogger(__name__)

MIN_PATHS_FOR_LINKAGE: int = 2


class ClusterInput(BaseModel):
    """Input bundle for cluster_by_similarity."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    encodings: dict[str, Any]
    paths: list[Path]
    threshold: float


def _build_embedding_matrix(cluster_in: ClusterInput) -> tuple[list[Path], np.ndarray]:
    """Extract valid paths and stack their embeddings into a matrix."""
    valid_paths: list[Path] = []
    vectors: list[Any] = []
    for path in cluster_in.paths:
        key = str(path)
        if key not in cluster_in.encodings:
            logger.warning("Encoding missing for path %s — skipping", key)
            continue
        valid_paths.append(path)
        vectors.append(np.asarray(cluster_in.encodings[key], dtype=np.float32))
    matrix = np.stack(vectors, axis=0) if vectors else np.empty((0, 0), dtype=np.float32)
    return valid_paths, matrix


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of the embedding matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _group_paths_by_label(
    paths: list[Path], labels: np.ndarray
) -> list[list[Path]]:
    """Group paths by integer cluster label, preserving input order."""
    groups: dict[int, list[Path]] = defaultdict(list)
    for path, label in zip(paths, labels):
        groups[label].append(path)
    return [groups[label] for label in sorted(groups)]


def cluster_by_similarity(cluster_in: ClusterInput) -> list[list[Path]]:
    """Cluster paths by cosine similarity of their MobileNetV3 embeddings."""
    if not cluster_in.paths:
        return []
    if len(cluster_in.paths) == 1:
        return [[cluster_in.paths[0]]]
    valid_paths, matrix = _build_embedding_matrix(cluster_in)
    if len(valid_paths) == 0:
        return []
    if len(valid_paths) == 1:
        return [[valid_paths[0]]]
    # scipy cosine metric normalizes internally; manual L2-normalization is redundant.
    linkage_matrix = linkage(matrix, method="average", metric="cosine")
    labels = fcluster(linkage_matrix, t=cluster_in.threshold, criterion="distance")
    return _group_paths_by_label(valid_paths, labels)
