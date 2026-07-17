"""Duplicate detection tests: CNN pipeline-output exclusion + DINOv2 second tier."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from cull.router import CURATED_DIR, REVIEW_DIR
from cull.stage1.duplicate import (
    DuplicateGroup,
    _add_pair_edges,
    _build_duplicate_groups,
    _connected_components,
    _cosine_similarity_matrix,
    _DinoV2EmbedJob,
    _DinoV2PassInput,
    _embed_dinov2_batch,
    _exclude_pipeline_output,
    _find_dinov2_duplicate_pairs,
    _GroupMergeInput,
    _groups_to_adjacency,
    _is_pipeline_output_name,
    _run_dinov2_pass,
    _similarity_pairs_above_threshold,
)


def test_is_pipeline_output_name_flags_review_subtree() -> None:
    """A relative name nested under _review must be flagged as pipeline output."""
    assert _is_pipeline_output_name(f"{REVIEW_DIR}/_rejected/old.jpg") is True


def test_is_pipeline_output_name_flags_curated_subtree() -> None:
    """A relative name nested under _curated must be flagged as pipeline output."""
    assert _is_pipeline_output_name(f"{CURATED_DIR}/_selects/old.jpg") is True


def test_is_pipeline_output_name_allows_fresh_photos() -> None:
    """A relative name outside _review/_curated must not be flagged."""
    assert _is_pipeline_output_name("session_a/new_photo.jpg") is False
    assert _is_pipeline_output_name("new_photo.jpg") is False


def test_exclude_pipeline_output_drops_review_and_curated_entries() -> None:
    """Filtering the raw encoding map must keep only fresh, un-routed photos."""
    raw_encodings = {
        "new_photo_01.jpg": [0.1, 0.2],
        f"{REVIEW_DIR}/_duplicates/old_duplicate.jpg": [0.3, 0.4],
        f"{CURATED_DIR}/_selects/old_select.jpg": [0.5, 0.6],
        "session_a/new_photo_02.jpg": [0.7, 0.8],
    }
    filtered = _exclude_pipeline_output(raw_encodings)
    assert filtered == {
        "new_photo_01.jpg": [0.1, 0.2],
        "session_a/new_photo_02.jpg": [0.7, 0.8],
    }


# ---------------------------------------------------------------------------
# DINOv2 second-tier — pure merge/similarity helpers
# ---------------------------------------------------------------------------


def test_cosine_similarity_matrix_identical_vectors_score_one() -> None:
    """Two identical vectors must have cosine similarity 1.0; orthogonal ones 0.0."""
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    similarity = _cosine_similarity_matrix(vectors)
    assert similarity[0, 1] == pytest.approx(1.0)
    assert similarity[0, 2] == pytest.approx(0.0)


def test_similarity_pairs_above_threshold_filters_correctly() -> None:
    """Only pairs at/above DINOV2_DUPLICATE_SIMILARITY must be returned."""
    similarity = np.array([[1.0, 0.9, 0.5], [0.9, 1.0, 0.1], [0.5, 0.1, 1.0]])
    names = ["a.jpg", "b.jpg", "c.jpg"]
    pairs = _similarity_pairs_above_threshold(similarity, names)
    assert pairs == [("a.jpg", "b.jpg")]


def test_groups_to_adjacency_builds_undirected_edges() -> None:
    """A CNN duplicates_map entry must produce edges in both directions."""
    adjacency = _groups_to_adjacency({"a.jpg": ["b.jpg"]})
    assert adjacency["a.jpg"] == {"b.jpg"}
    assert adjacency["b.jpg"] == {"a.jpg"}


def test_add_pair_edges_mutates_adjacency_in_place() -> None:
    """DINOv2 pairs must add undirected edges on top of the existing adjacency."""
    adjacency = _groups_to_adjacency({})
    _add_pair_edges(adjacency, [("x.jpg", "y.jpg")])
    assert adjacency["x.jpg"] == {"y.jpg"}
    assert adjacency["y.jpg"] == {"x.jpg"}


def test_connected_components_merges_cnn_and_dinov2_groups() -> None:
    """A DINOv2 edge bridging two separate CNN groups must merge them into one."""
    adjacency = _groups_to_adjacency({"a.jpg": ["b.jpg"], "c.jpg": ["d.jpg"]})
    _add_pair_edges(adjacency, [("b.jpg", "c.jpg")])
    components = _connected_components(adjacency)
    assert len(components) == 1
    assert set(components[0]) == {"a.jpg", "b.jpg", "c.jpg", "d.jpg"}


def test_build_duplicate_groups_sorts_members_deterministically() -> None:
    """Merged groups must render as DuplicateGroup with sorted, deterministic paths."""
    merge_in = _GroupMergeInput(
        duplicates_map={"z_base.jpg": ["m_reencode.jpg"]},
        dinov2_pairs=[("z_base.jpg", "a_crop.jpg")],
        image_dir=Path("/photos"),
    )
    groups = _build_duplicate_groups(merge_in)
    assert groups == [
        DuplicateGroup(
            paths=[
                Path("/photos/a_crop.jpg"),
                Path("/photos/m_reencode.jpg"),
                Path("/photos/z_base.jpg"),
            ]
        )
    ]


# ---------------------------------------------------------------------------
# DINOv2 second-tier — model pass (mocked processor/model, no network/weights)
# ---------------------------------------------------------------------------


def _make_tiny_jpeg(path: Path) -> None:
    """Write a minimal solid-color JPEG to path."""
    Image.new("RGB", (4, 4), color=(120, 60, 200)).save(path, "JPEG")


class _FakeBatchInputs:
    """Stand-in for a HF BatchFeature: .to(device) returns an empty kwargs mapping."""

    def to(self, device: str) -> dict:
        """Return an empty mapping — the mocked model ignores its kwargs."""
        return {}


class _FakeModelOutput:
    """Stand-in for a transformers model output exposing pooler_output."""

    def __init__(self, pooler_output: torch.Tensor) -> None:
        self.pooler_output = pooler_output


def test_embed_dinov2_batch_returns_pooled_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_embed_dinov2_batch must call the mocked processor/model and pool per image."""
    paths = [tmp_path / f"p{i}.jpg" for i in range(3)]
    for path in paths:
        _make_tiny_jpeg(path)
    fake_processor = MagicMock(return_value=_FakeBatchInputs())
    fake_pooler = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    fake_model = MagicMock(return_value=_FakeModelOutput(fake_pooler))
    monkeypatch.setattr(
        "cull.stage1.duplicate.dinov2_loader.get_dinov2_processor", lambda: fake_processor
    )
    monkeypatch.setattr(
        "cull.stage1.duplicate.dinov2_loader.get_dinov2_model", lambda: fake_model
    )

    embeddings = _embed_dinov2_batch(_DinoV2EmbedJob(paths=paths, device="cpu"))

    assert embeddings.shape == (3, 2)
    fake_model.assert_called_once()


def test_find_dinov2_duplicate_pairs_below_two_candidates_short_circuits() -> None:
    """A single candidate can never form a pair — no model call should occur."""
    pass_in = _DinoV2PassInput(image_dir=Path("/photos"), candidate_names=["only.jpg"])
    assert _find_dinov2_duplicate_pairs(pass_in) == []


def test_find_dinov2_duplicate_pairs_detects_near_identical_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two images whose mocked embeddings are near-identical must be paired."""
    names = ["base.jpg", "crop.jpg", "different.jpg"]
    for name in names:
        _make_tiny_jpeg(tmp_path / name)
    fake_processor = MagicMock(return_value=_FakeBatchInputs())
    fake_pooler = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    fake_model = MagicMock(return_value=_FakeModelOutput(fake_pooler))
    monkeypatch.setattr(
        "cull.stage1.duplicate.dinov2_loader.get_dinov2_processor", lambda: fake_processor
    )
    monkeypatch.setattr(
        "cull.stage1.duplicate.dinov2_loader.get_dinov2_model", lambda: fake_model
    )

    pass_in = _DinoV2PassInput(image_dir=tmp_path, candidate_names=names)
    pairs = _find_dinov2_duplicate_pairs(pass_in)

    assert pairs == [("base.jpg", "crop.jpg")]


def test_run_dinov2_pass_swallows_failures_and_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure inside the DINOv2 pass must degrade gracefully to no pairs."""
    monkeypatch.setattr(
        "cull.stage1.duplicate._find_dinov2_duplicate_pairs",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    result = _run_dinov2_pass(Path("/photos"), ["a.jpg", "b.jpg"])
    assert result == []
