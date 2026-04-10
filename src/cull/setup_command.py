"""Offline model bootstrap — fetch HF repos, URL files, and warm pyiqa metrics."""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import click
from pydantic import BaseModel

from cull import vlm_registry
from cull.config import (
    MODEL_MANIFEST,
    VLM_DEFAULT_ALIAS,
    VLM_MODELS_ROOT,
    ModelCacheConfig,
    ModelManifestEntry,
)
from cull.env_bootstrap import apply_offline_env, apply_online_env
from cull.model_cache import (
    BootstrapStatus,
    ConfigError,
    compute_sha256,
    read_bootstrap_marker,
    run_preflight,
    write_bootstrap_marker,
)
from cull.vlm_registry import VLMResolutionError

_PYIQA_METRIC_NAMES: tuple[str, ...] = ("topiq_nr", "clipiqa+", "topiq_iaa")
_KIND_HF_REPO: str = "hf_repo"
_KIND_URL_FILE: str = "url_file"
_SUBDIR_MEDIAPIPE: str = "mediapipe"
_SUBDIR_DEEPFACE: str = "deepface"
_HF_HUB_SUBDIR: str = "hub"
_HF_IGNORE_PATTERNS: list[str] = ["*.bin", "*.msgpack", "*.h5", "flax_*"]
_PARTIAL_SUFFIX: str = ".part"
_HF_BLOBS_SUBDIR: str = "blobs"
_INCOMPLETE_SUFFIX: str = ".incomplete"
_DEEPFACE_WEIGHTS_SUBPATH: tuple[str, str] = (".deepface", "weights")


class SetupRequest(BaseModel):
    """Input bundle for offline model bootstrap."""

    model_config = {"arbitrary_types_allowed": True}
    cache: ModelCacheConfig
    allow_network: bool = False
    force: bool = False


class SetupResult(BaseModel):
    """Outcome of a bootstrap run with timings and per-entry status lists."""

    status: BootstrapStatus
    fetched: list[str] = []
    skipped: list[str] = []
    duration_seconds: float


def _cleanup_incomplete_blobs(repo_id: str, cache: ModelCacheConfig) -> None:
    flat = repo_id.replace("/", "--")
    blobs_dir = cache.hf_home / _HF_HUB_SUBDIR / f"models--{flat}" / _HF_BLOBS_SUBDIR
    if not blobs_dir.is_dir():
        return
    for blob in blobs_dir.iterdir():
        if blob.suffix == _INCOMPLETE_SUFFIX:
            blob.unlink(missing_ok=True)


def fetch_hf_repo(entry: ModelManifestEntry, cache: ModelCacheConfig) -> None:
    """Snapshot a HuggingFace repo into the cache hub directory."""
    from huggingface_hub import snapshot_download

    if not entry.repo_id:
        raise ConfigError(f"hf_repo entry {entry.name} has no repo_id")
    _cleanup_incomplete_blobs(entry.repo_id, cache)
    snapshot_download(
        repo_id=entry.repo_id,
        cache_dir=str(cache.hf_home / _HF_HUB_SUBDIR),
        ignore_patterns=_HF_IGNORE_PATTERNS,
    )


def _resolve_url_file_dest(
    entry: ModelManifestEntry, cache: ModelCacheConfig
) -> Path:
    """Return the destination path for a single-file manifest entry."""
    if not entry.filename:
        raise ConfigError(f"url_file entry {entry.name} has no filename")
    if entry.subdir == _SUBDIR_MEDIAPIPE:
        return cache.mediapipe_dir / entry.filename
    if entry.subdir == _SUBDIR_DEEPFACE:
        return cache.deepface_home.joinpath(*_DEEPFACE_WEIGHTS_SUBPATH, entry.filename)
    raise ConfigError(f"unknown manifest subdir: {entry.subdir}")


def fetch_url_file(entry: ModelManifestEntry, cache: ModelCacheConfig) -> None:
    """Download a URL-backed asset atomically and verify its SHA-256."""
    if not entry.url:
        raise ConfigError(f"url_file entry {entry.name} has no url")
    dest = _resolve_url_file_dest(entry, cache)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + _PARTIAL_SUFFIX)
    urllib.request.urlretrieve(entry.url, partial)
    if entry.sha256 and compute_sha256(partial) != entry.sha256:
        partial.unlink(missing_ok=True)
        raise ConfigError(f"sha256 mismatch for {entry.name}")
    partial.replace(dest)


def warm_pyiqa_metrics() -> None:
    """Instantiate the pinned pyiqa metrics once to populate TORCH_HOME."""
    import pyiqa

    for name in _PYIQA_METRIC_NAMES:
        pyiqa.create_metric(name, device="cpu")


def _dispatch_fetch(
    entry: ModelManifestEntry, cache: ModelCacheConfig
) -> None:
    """Route a manifest entry to the matching fetcher by kind."""
    if entry.kind == _KIND_HF_REPO:
        fetch_hf_repo(entry, cache)
        return
    if entry.kind == _KIND_URL_FILE:
        fetch_url_file(entry, cache)
        return
    raise ConfigError(f"unknown manifest kind: {entry.kind}")


class _SkipContext(BaseModel):
    """Bundle for the early-exit path so helpers take ≤2 params."""

    model_config = {"arbitrary_types_allowed": True}
    status: BootstrapStatus
    cache: ModelCacheConfig
    start: float


def _check_vlm_preflight() -> None:
    """Validate the default VLM alias resolves on disk; raise ConfigError on failure."""
    try:
        vlm_registry.run_vlm_preflight(VLM_DEFAULT_ALIAS)
    except VLMResolutionError as exc:
        raise ConfigError(
            f"VLM alias '{VLM_DEFAULT_ALIAS}' not found under {VLM_MODELS_ROOT}: {exc}"
        ) from exc


def run_setup(request: SetupRequest) -> SetupResult:
    """Orchestrate preflight, fetches, pyiqa warmup, marker write, offline re-pin."""
    start = time.monotonic()
    if request.allow_network:
        apply_online_env(request.cache)
    initial = run_preflight(request.cache)
    if initial.is_ok and not request.force and read_bootstrap_marker(request.cache):
        skip_ctx = _SkipContext(status=initial, cache=request.cache, start=start)
        return _build_skip_result(skip_ctx)
    if not request.allow_network:
        return _build_dry_result(initial, request.cache)
    fetched, skipped = _fetch_pending(initial, request)
    warm_pyiqa_metrics()
    final = run_preflight(request.cache)
    _check_vlm_preflight()
    if final.is_ok:
        write_bootstrap_marker(request.cache)
    apply_offline_env(request.cache)
    elapsed = time.monotonic() - start
    return SetupResult(
        status=final, fetched=fetched, skipped=skipped, duration_seconds=elapsed
    )


def _build_dry_result(
    status: BootstrapStatus, cache: ModelCacheConfig
) -> SetupResult:
    """Return an offline-only SetupResult without invoking any fetcher."""
    apply_offline_env(cache)
    return SetupResult(
        status=status, fetched=[], skipped=[e.name for e in MODEL_MANIFEST.values()],
        duration_seconds=0.0,
    )


def _build_skip_result(ctx: _SkipContext) -> SetupResult:
    """Apply offline env and assemble the all-skipped SetupResult."""
    apply_offline_env(ctx.cache)
    skipped = [entry.name for entry in MODEL_MANIFEST.values()]
    return SetupResult(
        status=ctx.status,
        fetched=[],
        skipped=skipped,
        duration_seconds=time.monotonic() - ctx.start,
    )


def _fetch_pending(
    initial: BootstrapStatus, request: SetupRequest
) -> tuple[list[str], list[str]]:
    """Fetch missing/corrupt entries (or all if forced); return per-entry lists."""
    pending = set(initial.missing) | set(initial.corrupt)
    fetched: list[str] = []
    skipped: list[str] = []
    for entry in MODEL_MANIFEST.values():
        if not request.force and entry.name not in pending:
            skipped.append(entry.name)
            continue
        _dispatch_fetch(entry, request.cache)
        fetched.append(entry.name)
    return fetched, skipped


def setup_command(allow_network: bool, force: bool) -> None:
    """Click-callable wrapper; prints a summary and exits non-zero on failure."""
    cache = ModelCacheConfig.from_env()
    request = SetupRequest(cache=cache, allow_network=allow_network, force=force)
    try:
        result = run_setup(request)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    click.echo(
        f"setup status={result.status.state} "
        f"fetched={len(result.fetched)} skipped={len(result.skipped)} "
        f"duration={result.duration_seconds:.2f}s"
    )
    if result.status.state != "ok":
        sys.exit(1)
