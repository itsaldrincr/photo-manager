"""Near-duplicate detection: imagededup CNN + a DINOv2 second tier.

The CNN pass (imagededup MobileNetV3) runs first and its embeddings are kept
in ``DuplicateResult.encodings`` for Stage 4 clustering, whose per-preset
``CLUSTER_THRESHOLD`` values are calibrated for MobileNetV3's embedding
space. A DINOv2-small pass then runs over the same candidate set and adds
duplicate pairs the CNN misses — chiefly cropped near-duplicates, where the
CNN's measured recall is 0.50 vs DINOv2's 1.00 (see
benchmarks/runs/dedupe_eval_report.md). DINOv2 pairs are merged with the
CNN's groups by connected components; DINOv2 embeddings themselves are never
exposed downstream, so Stage 4's calibration is unaffected.

Both passes link a pair only when the two EXIF capture times fall within
DUPLICATE_TIME_WINDOW_SECONDS. A pair missing a capture time links only when
it is near-identical.
"""

from __future__ import annotations

import gc
import logging
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from cull import dinov2_loader
from cull.config import (
    BLUR_CNN_SIMILARITY_EXACT,
    DINOV2_DUPLICATE_SIMILARITY,
    DINOV2_EMBED_BATCH_SIZE,
    DINOV2_NEAR_IDENTICAL_SIMILARITY,
)
from cull.image_io import open_rgb_upright
from cull.router import CURATED_DIR, REVIEW_DIR
from cull.stage1.burst import read_exif_capture_time
from cull.stage1.representatives import is_within_moment_window
from cull.stage2.iqa import select_device

logger = logging.getLogger(__name__)

CaptureTimes = dict[str, datetime | None]
ScoredPair = tuple[str, str, float]

_CNN_INSTANCE: object | None = None
_IMPORT_FAILED: bool = False

# Prior pipeline output dirs must never re-enter duplicate detection — canonical
# names live in cull.router (single source of truth).
_EXCLUDED_DUPLICATE_DIRS: frozenset[str] = frozenset({REVIEW_DIR, CURATED_DIR})


class DuplicateGroup(BaseModel):
    """A group of duplicate images with similarity scores."""

    paths: list[Path]
    similarities: list[float] = Field(default_factory=list)


class DuplicateResult(BaseModel):
    """Output of duplicate detection across an image directory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    duplicate_groups: list[DuplicateGroup] = Field(default_factory=list)
    encodings: dict[str, Any] = Field(default_factory=dict)


class _GroupMergeInput(BaseModel):
    """Inputs for merging CNN duplicate groups with DINOv2-detected pairs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    duplicates_map: dict
    dinov2_pairs: list[tuple[str, str]]
    image_dir: Path


def _groups_to_adjacency(duplicates_map: dict) -> dict[str, set[str]]:
    """Convert an imagededup name->dupes map into an undirected adjacency dict."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for name, dupes in duplicates_map.items():
        for dupe in dupes:
            adjacency[name].add(dupe)
            adjacency[dupe].add(name)
    return adjacency


def _add_pair_edges(adjacency: dict[str, set[str]], pairs: list[tuple[str, str]]) -> None:
    """Mutate adjacency in place, adding an edge for each DINOv2-detected pair."""
    for name_a, name_b in pairs:
        adjacency[name_a].add(name_b)
        adjacency[name_b].add(name_a)


def _pair_times(pair: tuple[str, str], times: CaptureTimes) -> tuple[datetime, datetime] | None:
    """Return both capture times of a pair, or None when either is missing."""
    first, second = times.get(pair[0]), times.get(pair[1])
    if first is None or second is None:
        return None
    return first, second


def _is_linkable_cnn_pair(pair: tuple[str, str], times: CaptureTimes) -> bool:
    """Return True if a CNN pair may link; CNN pairs are near-identical already."""
    pair_times = _pair_times(pair, times)
    return pair_times is None or is_within_moment_window(*pair_times)


def _is_linkable_dinov2_pair(pair: ScoredPair, times: CaptureTimes) -> bool:
    """Return True if a DINOv2 pair may link given its capture times."""
    pair_times = _pair_times((pair[0], pair[1]), times)
    if pair_times is None:
        return pair[2] >= DINOV2_NEAR_IDENTICAL_SIMILARITY
    return is_within_moment_window(*pair_times)


def _gate_cnn_map(duplicates_map: dict, times: CaptureTimes) -> dict[str, list[str]]:
    """Drop CNN duplicate entries whose pair fails the capture-time gate."""
    return {
        name: [dupe for dupe in dupes if _is_linkable_cnn_pair((name, dupe), times)]
        for name, dupes in duplicates_map.items()
    }


def _read_capture_times(image_dir: Path, names: list[str]) -> CaptureTimes:
    """Return the EXIF capture time of every candidate, keyed by relative name."""
    return {name: read_exif_capture_time(image_dir / name) for name in names}


class _BfsState(BaseModel):
    """Mutable traversal state shared across connected-component BFS calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    adjacency: dict[str, set[str]]
    visited: set[str] = Field(default_factory=set)


def _bfs_component(start: str, state: _BfsState) -> list[str]:
    """Breadth-first traversal returning one connected component from start."""
    queue = deque([start])
    component: list[str] = []
    state.visited.add(start)
    while queue:
        node = queue.popleft()
        component.append(node)
        for neighbor in state.adjacency[node]:
            if neighbor not in state.visited:
                state.visited.add(neighbor)
                queue.append(neighbor)
    return component


def _connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    """Return every connected component (size >= 2) in the adjacency graph."""
    state = _BfsState(adjacency=adjacency)
    components: list[list[str]] = []
    for start in sorted(adjacency):
        if start in state.visited:
            continue
        component = _bfs_component(start, state)
        if len(component) > 1:
            components.append(component)
    return components


def _build_duplicate_groups(merge_in: _GroupMergeInput) -> list[DuplicateGroup]:
    """Merge CNN duplicate groups and DINOv2 pairs into DuplicateGroup objects.

    Component members are sorted for a deterministic, reproducible ordering —
    neither source produces a quality-ranked "keeper" today, so ties are
    broken alphabetically rather than by arbitrary dict-iteration order.
    """
    adjacency = _groups_to_adjacency(merge_in.duplicates_map)
    _add_pair_edges(adjacency, merge_in.dinov2_pairs)
    components = _connected_components(adjacency)
    return [
        DuplicateGroup(paths=[merge_in.image_dir / name for name in sorted(component)])
        for component in components
    ]


def _build_encodings_map(raw_encodings: dict, image_dir: Path) -> dict[str, Any]:
    """Build encodings map keyed by full str(path)."""
    return {str(image_dir / name): vec for name, vec in raw_encodings.items()}


def _load_cnn() -> object | None:
    """Lazy-load imagededup CNN, returning None if unavailable."""
    global _CNN_INSTANCE, _IMPORT_FAILED  # noqa: PLW0603
    if _IMPORT_FAILED:
        return None
    if _CNN_INSTANCE is not None:
        return _CNN_INSTANCE
    try:
        from imagededup.methods import CNN  # noqa: PLC0415

        _CNN_INSTANCE = CNN()
        return _CNN_INSTANCE
    except ImportError:
        _IMPORT_FAILED = True
        logger.warning("imagededup not available — skipping CNN duplicate detection")
        return None


def _unload_cnn() -> None:
    """Drop the cached CNN encoder to free memory after Stage 1 completes."""
    global _CNN_INSTANCE  # noqa: PLW0603
    _CNN_INSTANCE = None
    gc.collect()


def _unload_dinov2() -> None:
    """Drop the cached DINOv2 model and processor to free memory after Stage 1 completes."""
    dinov2_loader.unload()


def unload_models() -> None:
    """Free both the CNN and DINOv2 duplicate-detection models."""
    _unload_cnn()
    _unload_dinov2()


def _is_pipeline_output_name(relative_name: str) -> bool:
    """Return True if a relative encoding key lives under a _review/_curated subtree."""
    return any(part in _EXCLUDED_DUPLICATE_DIRS for part in Path(relative_name).parts)


def _exclude_pipeline_output(raw_encodings: dict) -> dict:
    """Drop CNN encodings for images already routed into prior pipeline output dirs."""
    return {
        name: vector
        for name, vector in raw_encodings.items()
        if not _is_pipeline_output_name(name)
    }


def _run_cnn_encoding(cnn: object, image_dir: Path) -> tuple[dict, dict]:
    """Run CNN encode and find_duplicates; return (raw_encodings, duplicates_map)."""
    from cull.io_silence import _silence_stdio  # noqa: PLC0415

    threshold = BLUR_CNN_SIMILARITY_EXACT
    with _silence_stdio():
        raw_encodings = cnn.encode_images(image_dir=str(image_dir), recursive=True)  # type: ignore[attr-defined]
        raw_encodings = _exclude_pipeline_output(raw_encodings)
        duplicates_map = cnn.find_duplicates(  # type: ignore[attr-defined]
            encoding_map=raw_encodings,
            min_similarity_threshold=threshold,
            scores=False,
        )
    return raw_encodings, duplicates_map


class _DinoV2EmbedJob(BaseModel):
    """Inputs for batch-embedding candidate images with DINOv2-small."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    paths: list[Path]
    device: str


def _dinov2_pixel_values(path: Path) -> torch.Tensor:
    """Decode one photo and return its (1,3,224,224) DINOv2 input.

    One photo at a time keeps a single full-resolution frame in memory
    instead of a batch of them; the processor's output is bitwise equal
    either way.
    """
    processor = dinov2_loader.get_dinov2_processor()
    return processor(images=[open_rgb_upright(path)], return_tensors="pt")["pixel_values"]  # type: ignore[operator]


def _embed_dinov2_batch(job: _DinoV2EmbedJob) -> np.ndarray:
    """Batch-embed images with DINOv2-small, returning pooled CLS embeddings."""
    model = dinov2_loader.get_dinov2_model()
    vectors: list[np.ndarray] = []
    for start in range(0, len(job.paths), DINOV2_EMBED_BATCH_SIZE):
        batch_paths = job.paths[start : start + DINOV2_EMBED_BATCH_SIZE]
        pixel_values = torch.cat([_dinov2_pixel_values(p) for p in batch_paths]).to(job.device)
        with torch.no_grad():
            output = model(pixel_values=pixel_values)  # type: ignore[operator]
        vectors.append(output.pooler_output.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine similarity matrix for a set of embedding vectors."""
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return norm @ norm.T


def _similarity_pairs_above_threshold(similarity: np.ndarray, names: list[str]) -> list[ScoredPair]:
    """Return (name, name, similarity) for pairs meeting the DINOv2 duplicate threshold."""
    pairs: list[ScoredPair] = []
    for i, j in zip(*np.triu_indices(len(names), k=1)):
        if similarity[i, j] >= DINOV2_DUPLICATE_SIMILARITY:
            pairs.append((names[i], names[j], float(similarity[i, j])))
    return pairs


class _DinoV2PassInput(BaseModel):
    """Inputs for the DINOv2 second-tier duplicate pass."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_dir: Path
    candidate_names: list[str]
    capture_times: CaptureTimes = Field(default_factory=dict)


def _find_dinov2_duplicate_pairs(pass_in: _DinoV2PassInput) -> list[tuple[str, str]]:
    """Run the DINOv2 embedding pass and return duplicate name pairs above threshold."""
    if len(pass_in.candidate_names) < 2:
        return []
    device = select_device()
    paths = [pass_in.image_dir / name for name in pass_in.candidate_names]
    embeddings = _embed_dinov2_batch(_DinoV2EmbedJob(paths=paths, device=device))
    similarity = _cosine_similarity_matrix(embeddings)
    scored = _similarity_pairs_above_threshold(similarity, pass_in.candidate_names)
    return [
        (pair[0], pair[1]) for pair in scored
        if _is_linkable_dinov2_pair(pair, pass_in.capture_times)
    ]


def _run_dinov2_pass(pass_in: _DinoV2PassInput) -> list[tuple[str, str]]:
    """Run the DINOv2 second-tier pass, returning [] on any failure."""
    logger.info(
        "Running DINOv2 duplicate detection (threshold=%.2f) on %d candidates",
        DINOV2_DUPLICATE_SIMILARITY,
        len(pass_in.candidate_names),
    )
    try:
        return _find_dinov2_duplicate_pairs(pass_in)
    except (RuntimeError, ValueError, OSError, ImportError, TypeError):
        logger.exception("DINOv2 duplicate pass failed for %s", pass_in.image_dir)
        return []


def find_duplicates(image_dir: Path) -> DuplicateResult:
    """Detect duplicate images in image_dir using the CNN pass plus a DINOv2 second tier."""
    from cull.io_silence import _silence_stdio  # noqa: PLC0415

    with _silence_stdio():
        cnn = _load_cnn()
    if cnn is None:
        return DuplicateResult()
    logger.info(
        "Running CNN duplicate detection (threshold=%.2f)",
        BLUR_CNN_SIMILARITY_EXACT,
    )
    try:
        raw_encodings, duplicates_map = _run_cnn_encoding(cnn, image_dir)
    except (RuntimeError, ValueError, OSError, TypeError):
        logger.exception("CNN duplicate detection failed for %s", image_dir)
        return DuplicateResult()
    names = list(raw_encodings.keys())
    times = _read_capture_times(image_dir, names)
    dinov2_pairs = _run_dinov2_pass(_DinoV2PassInput(image_dir=image_dir, candidate_names=names, capture_times=times))
    merge_in = _GroupMergeInput(
        duplicates_map=_gate_cnn_map(duplicates_map, times), dinov2_pairs=dinov2_pairs, image_dir=image_dir,
    )
    groups = _build_duplicate_groups(merge_in)
    encodings = _build_encodings_map(raw_encodings, image_dir)
    logger.info("Found %d duplicate groups (%d DINOv2-only pairs)", len(groups), len(dinov2_pairs))
    return DuplicateResult(duplicate_groups=groups, encodings=encodings)
