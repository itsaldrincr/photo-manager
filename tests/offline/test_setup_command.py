"""Tests for cull.setup_command: run_setup orchestration + fetcher dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from cull import setup_command
from cull.config import MODEL_MANIFEST, ModelCacheConfig
from cull.model_cache import BootstrapStatus, ConfigError
from cull.setup_command import (
    SetupRequest,
    fetch_hf_repo,
    fetch_url_file,
    run_setup,
)

SAMPLE_BYTES: bytes = b"photo-manager setup-command fixture"


def _build_cache(tmp_path: Path) -> ModelCacheConfig:
    """Return a ModelCacheConfig rooted at an empty tmp_path."""
    return ModelCacheConfig(
        root=tmp_path,
        hf_home=tmp_path / "hf",
        torch_home=tmp_path / "torch",
        deepface_home=tmp_path / "deepface",
        mediapipe_dir=tmp_path / "mediapipe",
    )


class _FetcherRecorder:
    """Mutable counter that tracks how many times each fetcher was invoked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def record(self, name: str) -> None:
        """Append a call identifier."""
        self.calls.append(name)


def _install_noop_fetchers(monkeypatch, recorder: _FetcherRecorder) -> None:
    """Replace all three fetchers with no-op recorders."""
    monkeypatch.setattr(
        setup_command, "fetch_hf_repo",
        lambda entry, cache: recorder.record(f"hf:{entry.name}"),
    )
    monkeypatch.setattr(
        setup_command, "fetch_url_file",
        lambda entry, cache: recorder.record(f"url:{entry.name}"),
    )
    monkeypatch.setattr(
        setup_command, "warm_pyiqa_metrics",
        lambda: recorder.record("pyiqa"),
    )


def _stub_two_phase_preflight(monkeypatch, cache_root: Path) -> None:
    """Preflight returns missing on first call, ok on second."""
    missing = BootstrapStatus(
        state="missing",
        missing=list(MODEL_MANIFEST.keys()),
        cache_root=cache_root,
    )
    ok = BootstrapStatus(state="ok", missing=[], corrupt=[], cache_root=cache_root)
    responses = iter([missing, ok])
    monkeypatch.setattr(setup_command, "run_preflight", lambda c: next(responses))


def test_run_setup_missing_cache_allow_network_calls_fetchers(
    tmp_path: Path, monkeypatch
) -> None:
    """run_setup with allow_network on a missing cache must call every fetcher."""
    cache = _build_cache(tmp_path)
    recorder = _FetcherRecorder()
    _install_noop_fetchers(monkeypatch, recorder)
    _stub_two_phase_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(setup_command, "write_bootstrap_marker", lambda c: None)
    result = run_setup(SetupRequest(cache=cache, allow_network=True))
    assert result.status.state == "ok"
    # 4 manifest entries + 1 explicit warm_pyiqa_metrics call = 5 recorder hits
    assert len(recorder.calls) == len(MODEL_MANIFEST) + 1
    assert "pyiqa" in recorder.calls


def test_run_setup_disallow_network_on_empty_cache_does_not_fetch(
    tmp_path: Path, monkeypatch
) -> None:
    """run_setup with allow_network=False must never invoke a fetcher."""
    cache = _build_cache(tmp_path)
    recorder = _FetcherRecorder()
    _install_noop_fetchers(monkeypatch, recorder)
    _stub_two_phase_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(setup_command, "write_bootstrap_marker", lambda c: None)
    result = run_setup(SetupRequest(cache=cache, allow_network=False))
    assert recorder.calls == []
    assert result.status.state != "ok"


def test_run_setup_idempotent_when_preflight_ok(
    tmp_path: Path, monkeypatch
) -> None:
    """When preflight reports ok + marker present, run_setup must return empty fetched list."""
    cache = _build_cache(tmp_path)
    recorder = _FetcherRecorder()
    _install_noop_fetchers(monkeypatch, recorder)
    ok_status = BootstrapStatus(state="ok", cache_root=tmp_path)
    monkeypatch.setattr(setup_command, "run_preflight", lambda c: ok_status)
    monkeypatch.setattr(setup_command, "read_bootstrap_marker", lambda c: True)
    result = run_setup(SetupRequest(cache=cache, allow_network=True))
    assert result.fetched == []
    assert recorder.calls == []


class _SnapshotRecorder:
    """Captures snapshot_download kwargs for assertion in the repo-fetch test."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def __call__(self, **kwargs) -> str:
        """Record the kwargs and return a placeholder path."""
        self.kwargs = kwargs
        return "/tmp/fake-snapshot"


def test_fetch_hf_repo_calls_snapshot_download(
    tmp_path: Path, monkeypatch
) -> None:
    """fetch_hf_repo must call huggingface_hub.snapshot_download with repo_id + cache_dir."""
    import huggingface_hub  # noqa: PLC0415

    recorder = _SnapshotRecorder()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", recorder)
    cache = _build_cache(tmp_path)
    entry = MODEL_MANIFEST["clip"]
    fetch_hf_repo(entry, cache)
    assert recorder.kwargs["repo_id"] == entry.repo_id
    assert "cache_dir" in recorder.kwargs


def _install_urlretrieve_stub(monkeypatch, source_path: Path) -> None:
    """Monkeypatch urllib.request.urlretrieve to copy a tmp source into dest."""
    import urllib.request  # noqa: PLC0415

    def _fake(url: str, dest: str) -> tuple[str, None]:
        Path(dest).write_bytes(source_path.read_bytes())
        return dest, None

    monkeypatch.setattr(urllib.request, "urlretrieve", _fake)


def test_fetch_url_file_verifies_hash(tmp_path: Path, monkeypatch) -> None:
    """fetch_url_file must accept matching sha256 and raise ConfigError on mismatch."""
    source = tmp_path / "upstream.task"
    source.write_bytes(SAMPLE_BYTES)
    _install_urlretrieve_stub(monkeypatch, source)
    cache = _build_cache(tmp_path)
    import hashlib  # noqa: PLC0415
    good_hash = hashlib.sha256(SAMPLE_BYTES).hexdigest()
    good_entry = MODEL_MANIFEST["face_landmarker"].model_copy(
        update={"sha256": good_hash}
    )
    fetch_url_file(good_entry, cache)
    bad_entry = MODEL_MANIFEST["face_landmarker"].model_copy(
        update={"sha256": "00" * 32}
    )
    with pytest.raises(ConfigError):
        fetch_url_file(bad_entry, cache)
