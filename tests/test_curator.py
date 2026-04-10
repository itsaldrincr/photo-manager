"""Tests for cull.stage4.curator and cull.stage4.cluster — synthetic embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from pydantic import BaseModel

from cull.config import CLUSTER_THRESHOLD, CullConfig
from cull.models import CurationResult, CuratorSelection
from cull.pipeline import SessionResult
from cull.stage4.cluster import ClusterInput, cluster_by_similarity
from cull.stage4.curator import CuratorInput, curate


@pytest.fixture(autouse=True)
def _stub_stage4_ml() -> Any:
    """Stub ML-heavy stage4 dependencies so tests never load real models."""
    with patch("cull.stage4.curator.run_tournament") as mock_tournament, \
         patch("cull.stage4.curator.narrative_check") as mock_narrative, \
         patch("cull.stage4.curator.diversity_select") as mock_diversity:
        mock_tournament.side_effect = lambda inp, ctx: list(inp.candidates)
        mock_narrative.side_effect = lambda flow_in: (flow_in.selections, 0.0)
        mock_diversity.side_effect = lambda mmr_in, ctx: list(mmr_in.candidates)
        yield

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM: int = 1280
SMALL_NOISE_SCALE: float = 0.001
LARGE_NOISE_SCALE: float = 1.0
SEED: int = 42
COMPOSITE_HIGH: float = 0.90
COMPOSITE_MID: float = 0.70
COMPOSITE_LOW: float = 0.50
CLUSTER_THRESHOLD_TEST: float = 0.15
EXPECTED_CLUSTER_SIZE: int = 3
MIN_CLUSTER_COUNT: int = 2
PHOTO_COUNT_SMALL: int = 5
PHOTO_COUNT_LARGE: int = 50
PHOTO_TARGET_MID: int = 30
PHOTO_TARGET_FILL: int = 10
SCORE_HIGH_DELTA: float = 0.6
SCORE_LOW_DELTA: float = 0.4
SCORE_STEP: float = 0.01
ELAPSED_STUB: float = 0.42
CURATE_TARGET_MIN: int = 2
CLUSTER_COUNT_MEDIUM: int = 10

# ---------------------------------------------------------------------------
# Parameter bundles
# ---------------------------------------------------------------------------


class _NearCopyParams(BaseModel):
    """Input bundle for _near_copies."""

    model_config = {"arbitrary_types_allowed": True}
    count: int
    rng: np.random.Generator


class _TwoClusterParams(BaseModel):
    """Input bundle for _build_two_cluster_input."""

    model_config = {"arbitrary_types_allowed": True}
    rng: np.random.Generator
    target: int


class _NClusterParams(BaseModel):
    """Input bundle for _build_n_cluster_input."""

    cluster_count: int
    target: int

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_config(preset: str = "general", target: int = 30) -> CullConfig:
    """Build a minimal CullConfig for testing."""
    return CullConfig(preset=preset, curate_target=target)  # type: ignore[arg-type]


def _make_base_vector(rng: np.random.Generator) -> np.ndarray:
    """Return a single random unit-normalised embedding vector."""
    vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm


def _near_copies(base: np.ndarray, params: _NearCopyParams) -> list[np.ndarray]:
    """Return count embeddings close to base (small noise)."""
    copies: list[np.ndarray] = []
    for _ in range(params.count):
        noise = params.rng.standard_normal(EMBED_DIM).astype(np.float32) * SMALL_NOISE_SCALE
        copies.append(base + noise)
    return copies


def _orthogonal_vector(rng: np.random.Generator) -> np.ndarray:
    """Return an embedding far from any typical vector (large random)."""
    vec = rng.standard_normal(EMBED_DIM).astype(np.float32) * LARGE_NOISE_SCALE
    return vec


def _make_paths(tmp_path: Path, names: list[str]) -> list[Path]:
    """Create empty files and return their Paths."""
    paths: list[Path] = []
    for name in names:
        p = tmp_path / name
        p.touch()
        paths.append(p)
    return paths


def _encodings_from_pairs(pairs: list[tuple[Path, np.ndarray]]) -> dict[str, np.ndarray]:
    """Build encodings dict from (path, embedding) pairs."""
    return {str(p): emb for p, emb in pairs}


# ---------------------------------------------------------------------------
# 1. Clustering correctness
# ---------------------------------------------------------------------------


def test_cluster_by_similarity_groups_similar_embeddings(tmp_path: Path) -> None:
    """Three near-identical embeddings and two orthogonal ones form separate clusters."""
    rng = np.random.default_rng(seed=SEED)
    base = _make_base_vector(rng)
    near = _near_copies(base, _NearCopyParams(count=EXPECTED_CLUSTER_SIZE, rng=rng))
    far_a = _orthogonal_vector(rng)
    far_b = _orthogonal_vector(rng)
    paths = _make_paths(tmp_path, ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"])
    encodings = _encodings_from_pairs(list(zip(paths, near + [far_a, far_b])))
    cluster_in = ClusterInput(encodings=encodings, paths=paths, threshold=0.15)
    clusters = cluster_by_similarity(cluster_in)
    sizes = sorted([len(c) for c in clusters], reverse=True)
    assert sizes[0] == EXPECTED_CLUSTER_SIZE, f"Largest cluster expected {EXPECTED_CLUSTER_SIZE}, got {sizes}"
    assert len(clusters) >= MIN_CLUSTER_COUNT, f"Expected at least {MIN_CLUSTER_COUNT} clusters, got {len(clusters)}"


def test_cluster_by_similarity_empty(tmp_path: Path) -> None:
    """Empty path list returns empty cluster list."""
    cluster_in = ClusterInput(encodings={}, paths=[], threshold=CLUSTER_THRESHOLD_TEST)
    assert cluster_by_similarity(cluster_in) == []


def test_cluster_by_similarity_single(tmp_path: Path) -> None:
    """Single path returns one cluster containing that path."""
    rng = np.random.default_rng(seed=SEED)
    paths = _make_paths(tmp_path, ["solo.jpg"])
    encodings = {str(paths[0]): _make_base_vector(rng)}
    cluster_in = ClusterInput(encodings=encodings, paths=paths, threshold=CLUSTER_THRESHOLD_TEST)
    result = cluster_by_similarity(cluster_in)
    assert len(result) == 1
    assert result[0] == [paths[0]]


def test_cluster_by_similarity_missing_encoding_skipped(tmp_path: Path) -> None:
    """Path with no encoding entry is silently skipped without crash."""
    rng = np.random.default_rng(seed=SEED)
    paths = _make_paths(tmp_path, ["known.jpg", "missing.jpg"])
    encodings = {str(paths[0]): _make_base_vector(rng)}
    cluster_in = ClusterInput(encodings=encodings, paths=paths, threshold=CLUSTER_THRESHOLD_TEST)
    result = cluster_by_similarity(cluster_in)
    assert len(result) == 1
    assert result[0] == [paths[0]]


# ---------------------------------------------------------------------------
# 2. Winner selection (mock VLM)
# ---------------------------------------------------------------------------


def _build_cluster_encodings_and_scores(
    cluster_a_paths: list[Path],
    cluster_b_paths: list[Path],
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Build encodings and scores for two clusters."""
    base_a = _make_base_vector(rng)
    base_b = _orthogonal_vector(rng)
    near_a = _near_copies(base_a, _NearCopyParams(count=EXPECTED_CLUSTER_SIZE, rng=rng))
    near_b = [base_b + v * SMALL_NOISE_SCALE for v in _near_copies(base_b, _NearCopyParams(count=EXPECTED_CLUSTER_SIZE, rng=rng))]
    all_paths = cluster_a_paths + cluster_b_paths
    all_vecs = near_a + near_b
    encodings = _encodings_from_pairs(list(zip(all_paths, all_vecs)))
    scores = {
        str(cluster_a_paths[0]): COMPOSITE_HIGH,
        str(cluster_a_paths[1]): COMPOSITE_MID,
        str(cluster_a_paths[2]): COMPOSITE_LOW,
        str(cluster_b_paths[0]): COMPOSITE_HIGH,
        str(cluster_b_paths[1]): COMPOSITE_MID,
        str(cluster_b_paths[2]): COMPOSITE_LOW,
    }
    return encodings, scores


def _build_two_cluster_input(
    tmp_path: Path,
    params: _TwoClusterParams,
) -> tuple[CuratorInput, list[Path], list[Path]]:
    """Build two clusters of 3 photos each with clear score differences."""
    cluster_a_paths = _make_paths(tmp_path, ["a1.jpg", "a2.jpg", "a3.jpg"])
    cluster_b_paths = _make_paths(tmp_path, ["b1.jpg", "b2.jpg", "b3.jpg"])
    encodings, scores = _build_cluster_encodings_and_scores(cluster_a_paths, cluster_b_paths, params.rng)
    config = _make_config(target=params.target)
    curator_in = CuratorInput(
        keepers=cluster_a_paths + cluster_b_paths,
        encodings=encodings,
        composite_scores=scores,
        config=config,
    )
    return curator_in, cluster_a_paths, cluster_b_paths


def test_curate_picks_highest_composite_per_cluster(tmp_path: Path) -> None:
    """Composite-sort path: no faces, no motion → highest composite per cluster wins."""
    rng = np.random.default_rng(seed=SEED)
    curator_in, cluster_a, cluster_b = _build_two_cluster_input(tmp_path, _TwoClusterParams(rng=rng, target=CURATE_TARGET_MIN))
    result = curate(curator_in)
    assert result.actual_count == 2
    selected_paths = {str(s.path) for s in result.selected}
    assert str(cluster_a[0]) in selected_paths
    assert str(cluster_b[0]) in selected_paths


# ---------------------------------------------------------------------------
# 3. Target filling
# ---------------------------------------------------------------------------


def _build_n_cluster_input(tmp_path: Path, params: _NClusterParams) -> CuratorInput:
    """Build n single-photo clusters (no real clustering needed)."""
    rng = np.random.default_rng(seed=SEED)
    paths: list[Path] = []
    encodings: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    for i in range(params.cluster_count):
        p = tmp_path / f"photo_{i:03d}.jpg"
        p.touch()
        vec = _orthogonal_vector(rng)
        paths.append(p)
        encodings[str(p)] = vec
        scores[str(p)] = float(i) / max(params.cluster_count - 1, 1)
    config = _make_config(target=params.target)
    return CuratorInput(
        keepers=paths,
        encodings=encodings,
        composite_scores=scores,
        config=config,
    )


def test_curate_keepers_less_than_target_returns_all(tmp_path: Path) -> None:
    """When keepers count is less than target, all keepers are returned."""
    curator_in = _build_n_cluster_input(tmp_path, _NClusterParams(cluster_count=PHOTO_COUNT_SMALL, target=PHOTO_TARGET_MID))
    result = curate(curator_in)
    assert result.actual_count == PHOTO_COUNT_SMALL
    assert result.is_enabled is True


def test_curate_clusters_equal_target(tmp_path: Path) -> None:
    """When cluster count equals target, exactly N winners are chosen."""
    cluster_count = CLUSTER_COUNT_MEDIUM
    curator_in = _build_n_cluster_input(tmp_path, _NClusterParams(cluster_count=cluster_count, target=cluster_count))
    result = curate(curator_in)
    assert result.actual_count == cluster_count


def test_curate_clusters_more_than_target(tmp_path: Path) -> None:
    """When clusters exceed target, top 30 by composite are kept."""
    target = PHOTO_TARGET_MID
    curator_in = _build_n_cluster_input(tmp_path, _NClusterParams(cluster_count=PHOTO_COUNT_LARGE, target=target))
    result = curate(curator_in)
    assert result.actual_count == target


def _build_mixed_cluster_input(tmp_path: Path) -> tuple[CuratorInput, int, int]:
    """Build input with one large cluster and many small ones."""
    rng = np.random.default_rng(seed=SEED)
    large_cluster_size = PHOTO_COUNT_SMALL
    small_cluster_count = PHOTO_COUNT_SMALL
    total_photos = large_cluster_size + small_cluster_count
    target = PHOTO_TARGET_FILL
    paths: list[Path] = []
    encodings: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    base_large = _make_base_vector(rng)
    large_vecs = _near_copies(base_large, _NearCopyParams(count=large_cluster_size, rng=rng))
    for i, vec in enumerate(large_vecs):
        p = tmp_path / f"large_{i}.jpg"
        p.touch()
        paths.append(p)
        encodings[str(p)] = vec
        scores[str(p)] = SCORE_HIGH_DELTA + i * SCORE_STEP
    for i in range(small_cluster_count):
        p = tmp_path / f"small_{i}.jpg"
        p.touch()
        far_vec = _orthogonal_vector(rng)
        paths.append(p)
        encodings[str(p)] = far_vec
        scores[str(p)] = SCORE_LOW_DELTA + i * SCORE_STEP
    config = _make_config(target=target)
    curator_in = CuratorInput(
        keepers=paths,
        encodings=encodings,
        composite_scores=scores,
        config=config,
    )
    return curator_in, target, total_photos


def test_curate_clusters_less_than_target_fills_from_largest(tmp_path: Path) -> None:
    """When clusters < target, fills from largest clusters until target reached."""
    curator_in, target, total_photos = _build_mixed_cluster_input(tmp_path)
    result = curate(curator_in)
    assert result.actual_count == min(target, total_photos)


def test_curate_zero_keepers_returns_empty_result() -> None:
    """Zero keepers returns empty CurationResult with is_enabled=True."""
    config = _make_config(target=30)
    curator_in = CuratorInput(
        keepers=[],
        encodings={},
        composite_scores={},
        config=config,
    )
    result = curate(curator_in)
    assert result.actual_count == 0
    assert result.is_enabled is True
    assert result.selected == []


# ---------------------------------------------------------------------------
# 4. Per-preset thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", list(CLUSTER_THRESHOLD.keys()))
def test_curate_uses_preset_threshold(tmp_path: Path, preset: str) -> None:
    """threshold_used in result matches CLUSTER_THRESHOLD for each preset."""
    rng = np.random.default_rng(seed=SEED)
    paths = _make_paths(tmp_path, ["p1.jpg", "p2.jpg"])
    vecs = [_orthogonal_vector(rng), _orthogonal_vector(rng)]
    encodings = _encodings_from_pairs(list(zip(paths, vecs)))
    scores = {str(paths[0]): 0.8, str(paths[1]): 0.6}
    config = CullConfig(preset=preset, curate_target=2)  # type: ignore[arg-type]
    curator_in = CuratorInput(
        keepers=paths,
        encodings=encodings,
        composite_scores=scores,
        config=config,
    )
    result = curate(curator_in)
    assert result.threshold_used == CLUSTER_THRESHOLD[preset]


# ---------------------------------------------------------------------------
# 5. Serialization
# ---------------------------------------------------------------------------


def _make_test_curation_result(tmp_path: Path) -> tuple[CurationResult, dict]:
    """Build a test CurationResult and its expected JSON data."""
    p = tmp_path / "test.jpg"
    p.touch()
    selection = CuratorSelection(
        path=p,
        cluster_id=0,
        cluster_size=EXPECTED_CLUSTER_SIZE,
        composite=COMPOSITE_HIGH,
        is_vlm_winner=True,
        reason="sharper focus",
    )
    curation = CurationResult(
        is_enabled=True,
        target_count=PHOTO_TARGET_FILL,
        actual_count=1,
        cluster_count=1,
        vlm_tiebreakers=1,
        threshold_used=CLUSTER_THRESHOLD_TEST,
        elapsed_seconds=ELAPSED_STUB,
        selected=[selection],
    )
    session = SessionResult(
        source_path=str(tmp_path),
        model="test-model",
        preset="general",
        curation=curation,
    )
    raw = session.model_dump_json()
    data = json.loads(raw)
    return curation, data


def test_curation_result_serializes_to_json(tmp_path: Path) -> None:
    """CurationResult with selections round-trips through model_dump_json."""
    curation, data = _make_test_curation_result(tmp_path)
    curation_data = data["curation"]
    assert curation_data["is_enabled"] is True
    assert curation_data["target_count"] == PHOTO_TARGET_FILL
    assert curation_data["actual_count"] == 1
    assert curation_data["cluster_count"] == 1
    assert curation_data["vlm_tiebreakers"] == 1
    assert curation_data["threshold_used"] == CLUSTER_THRESHOLD_TEST
    assert "elapsed_seconds" in curation_data
    assert len(curation_data["selected"]) == 1
    sel = curation_data["selected"][0]
    assert isinstance(sel["path"], str)
    assert sel["cluster_id"] == 0
    assert sel["cluster_size"] == EXPECTED_CLUSTER_SIZE
    assert sel["is_vlm_winner"] is True
    assert sel["reason"] == "sharper focus"
