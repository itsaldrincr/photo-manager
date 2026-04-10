"""Offline model cache preflight — hash verification, bootstrap marker, typed results."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from cull.config import (
    MODEL_MANIFEST,
    ModelCacheConfig,
    ModelManifestEntry,
    SUBDIR_DEEPFACE,
    SUBDIR_MEDIAPIPE,
)

BOOTSTRAP_MARKER_FILENAME: str = ".bootstrap-complete"
HASH_CHUNK_BYTES: int = 65536

_KIND_URL_FILE: str = "url_file"
_KIND_HF_REPO: str = "hf_repo"
_HF_HUB_SUBDIR: str = "hub"
_HF_SNAPSHOTS_SUBDIR: str = "snapshots"
_HF_BLOBS_SUBDIR: str = "blobs"
_DEEPFACE_WEIGHTS_SUBPATH: tuple[str, str] = (".deepface", "weights")


class BootstrapStatus(BaseModel):
    """Aggregate cache health snapshot for the offline model manifest."""

    state: Literal["ok", "missing", "corrupt", "incomplete"]
    missing: list[str] = []
    corrupt: list[str] = []
    cache_root: Path

    @property
    def is_ok(self) -> bool:
        """Return True when every manifest entry is present and verified."""
        return self.state == "ok"


class PreflightResult(BaseModel):
    """Status plus a human-readable explanation for CLI surfaces."""

    status: BootstrapStatus
    message: str

    @property
    def is_ok(self) -> bool:
        """Return True when the underlying preflight passed."""
        return self.status.is_ok


class ConfigError(Exception):
    """Raised on unexpected cache failures (unwritable dir, IO error)."""


def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file streamed in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_file_path(
    entry: ModelManifestEntry, cache: ModelCacheConfig
) -> Path | None:
    """Return on-disk path for a single-file manifest entry, or None for repos."""
    if entry.kind != _KIND_URL_FILE or entry.filename is None:
        return None
    if entry.subdir == SUBDIR_MEDIAPIPE:
        return cache.mediapipe_dir / entry.filename
    if entry.subdir == SUBDIR_DEEPFACE:
        return cache.deepface_home.joinpath(*_DEEPFACE_WEIGHTS_SUBPATH, entry.filename)
    raise ConfigError(f"unknown manifest subdir: {entry.subdir}")


def _hf_repo_dir(entry: ModelManifestEntry, cache: ModelCacheConfig) -> Path:
    """Return the HF cache layout directory for a repo entry."""
    flat = (entry.repo_id or "").replace("/", "--")
    return cache.hf_home / _HF_HUB_SUBDIR / f"models--{flat}"


def _hf_repo_is_valid(repo_dir: Path) -> bool:
    snapshots = repo_dir / _HF_SNAPSHOTS_SUBDIR
    if not snapshots.is_dir():
        return False
    blobs_dir = repo_dir / _HF_BLOBS_SUBDIR
    if not blobs_dir.is_dir():
        return False
    if any(b.name.endswith(".incomplete") for b in blobs_dir.iterdir()):
        return False
    for snap_file in snapshots.rglob("*"):
        if snap_file.is_symlink() and not snap_file.resolve().is_file():
            return False
    return True


def check_manifest_entry(
    entry: ModelManifestEntry, cache: ModelCacheConfig
) -> tuple[bool, bool]:
    """Return (exists, hash_ok) for a single manifest entry."""
    if entry.kind == _KIND_HF_REPO:
        repo_dir = _hf_repo_dir(entry, cache)
        return repo_dir.exists() and _hf_repo_is_valid(repo_dir), True
    file_path = _manifest_file_path(entry, cache)
    if file_path is None or not file_path.exists():
        return False, False
    if not entry.sha256:
        return True, True
    return True, compute_sha256(file_path) == entry.sha256


def run_preflight(cache: ModelCacheConfig) -> BootstrapStatus:
    """Iterate the model manifest and aggregate missing/corrupt entries."""
    missing: list[str] = []
    corrupt: list[str] = []
    for entry in MODEL_MANIFEST.values():
        exists, hash_ok = check_manifest_entry(entry, cache)
        if not exists:
            missing.append(entry.name)
            continue
        if not hash_ok:
            corrupt.append(entry.name)
    state = _resolve_state(missing, corrupt)
    return BootstrapStatus(
        state=state, missing=missing, corrupt=corrupt, cache_root=cache.root
    )


def _resolve_state(
    missing: list[str], corrupt: list[str]
) -> Literal["ok", "missing", "corrupt", "incomplete"]:
    """Map missing/corrupt lists to a single status state token."""
    if not missing and not corrupt:
        return "ok"
    if corrupt and missing:
        return "incomplete"
    if corrupt:
        return "corrupt"
    return "missing"


def write_bootstrap_marker(cache: ModelCacheConfig) -> None:
    """Write an ISO-8601 UTC timestamp to the bootstrap marker file."""
    try:
        cache.root.mkdir(parents=True, exist_ok=True)
        marker = cache.root / BOOTSTRAP_MARKER_FILENAME
        stamp = datetime.now(timezone.utc).isoformat()
        marker.write_text(stamp, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"unable to write bootstrap marker: {exc}") from exc


def read_bootstrap_marker(cache: ModelCacheConfig) -> bool:
    """Return True iff the bootstrap marker file exists in the cache root."""
    return (cache.root / BOOTSTRAP_MARKER_FILENAME).exists()


def require_bootstrap_valid(cache: ModelCacheConfig) -> PreflightResult:
    """Combine preflight + marker into a typed PreflightResult, never raises."""
    status = run_preflight(cache)
    has_marker = read_bootstrap_marker(cache)
    if status.is_ok and has_marker:
        return PreflightResult(status=status, message="ok")
    return PreflightResult(status=status, message=_describe(status, has_marker))


def _describe(status: BootstrapStatus, has_marker: bool) -> str:
    """Return a human-readable explanation of a non-OK preflight outcome."""
    if not has_marker and status.state == "missing":
        return "cache is empty — run 'cull setup --allow-network'"
    if status.corrupt:
        joined = ", ".join(status.corrupt)
        return f"cache has {len(status.corrupt)} corrupt entries: {joined}"
    if status.missing:
        joined = ", ".join(status.missing)
        return f"cache is missing entries: {joined}"
    return "cache marker absent — run 'cull setup --allow-network'"
