"""Tests for cull.model_cache: hashing, preflight, markers, corruption detection."""

from __future__ import annotations

import hashlib
from pathlib import Path

from cull import config as cull_config
from cull.config import ModelCacheConfig
from cull.model_cache import (
    compute_sha256,
    read_bootstrap_marker,
    require_bootstrap_valid,
    run_preflight,
    write_bootstrap_marker,
)

SAMPLE_BYTES: bytes = b"photo-manager offline cache fixture"
BOGUS_HASH: str = "deadbeef" * 8


def _build_cache(tmp_path: Path) -> ModelCacheConfig:
    """Return a ModelCacheConfig rooted at an empty tmp_path."""
    return ModelCacheConfig(
        root=tmp_path,
        hf_home=tmp_path / "hf",
        torch_home=tmp_path / "torch",
        deepface_home=tmp_path / "deepface",
        mediapipe_dir=tmp_path / "mediapipe",
    )


def test_compute_sha256_matches_hashlib(tmp_path: Path) -> None:
    """compute_sha256 output must match stdlib hashlib.sha256 hex digest."""
    sample = tmp_path / "sample.bin"
    sample.write_bytes(SAMPLE_BYTES)
    assert compute_sha256(sample) == hashlib.sha256(SAMPLE_BYTES).hexdigest()


def test_run_preflight_empty_cache_returns_missing(tmp_path: Path) -> None:
    """run_preflight on an empty cache must flag state exactly 'missing'."""
    cache = _build_cache(tmp_path)
    status = run_preflight(cache)
    assert status.state == "missing"
    assert len(status.missing) == 4
    assert status.corrupt == []


def test_write_and_read_bootstrap_marker(tmp_path: Path) -> None:
    """write_bootstrap_marker + read_bootstrap_marker round-trip must return True."""
    cache = _build_cache(tmp_path)
    write_bootstrap_marker(cache)
    assert read_bootstrap_marker(cache) is True


def test_require_bootstrap_valid_empty_cache(tmp_path: Path) -> None:
    """require_bootstrap_valid must report is_ok=False on an empty cache."""
    cache = _build_cache(tmp_path)
    result = require_bootstrap_valid(cache)
    assert result.is_ok is False


def _seed_face_landmarker(cache: ModelCacheConfig) -> Path:
    """Create a fake face_landmarker.task file to simulate a cached asset."""
    cache.mediapipe_dir.mkdir(parents=True, exist_ok=True)
    path = cache.mediapipe_dir / "face_landmarker.task"
    path.write_bytes(SAMPLE_BYTES)
    return path


def _seed_hf_repo(cache: ModelCacheConfig, repo_id: str) -> None:
    flat = repo_id.replace("/", "--")
    repo_dir = cache.hf_home / "hub" / f"models--{flat}"
    blobs_dir = repo_dir / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    blob = blobs_dir / "abc123fakeblob"
    blob.write_bytes(b"fake model data")
    snapshots_dir = repo_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    link = snapshots_dir / "main"
    link.symlink_to(blob)


def _seed_deepface(cache: ModelCacheConfig) -> None:
    deepface_dir = cache.deepface_home / ".deepface" / "weights"
    deepface_dir.mkdir(parents=True, exist_ok=True)
    (deepface_dir / "facial_expression_model_weights.h5").write_bytes(SAMPLE_BYTES)


def test_hash_mismatch_is_corrupt(tmp_path: Path, monkeypatch) -> None:
    """A file with a pinned-but-mismatched sha256 must be flagged corrupt."""
    cache = _build_cache(tmp_path)
    _seed_face_landmarker(cache)
    _seed_hf_repo(cache, cull_config.CLIP_REPO_ID)
    _seed_hf_repo(cache, cull_config.AESTHETIC_REPO_ID)
    _seed_deepface(cache)
    face_entry = cull_config.MODEL_MANIFEST["face_landmarker"].model_copy(
        update={"sha256": BOGUS_HASH}
    )
    monkeypatch.setitem(cull_config.MODEL_MANIFEST, "face_landmarker", face_entry)
    status = run_preflight(cache)
    assert status.state == "corrupt"
    assert "face_landmarker" in status.corrupt
