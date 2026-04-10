"""Tests for cull.cli bootstrap preflight gate in main() + setup subcommand."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cull import cli
from cull.model_cache import BootstrapStatus, PreflightResult


def _make_preflight(is_ok: bool, message: str = "") -> PreflightResult:
    """Return a minimal PreflightResult with controlled is_ok state."""
    state = "ok" if is_ok else "missing"
    status = BootstrapStatus(state=state, cache_root=Path("/tmp/fake-cache"))
    return PreflightResult(status=status, message=message or "cache missing")


def _gate_must_not_run(_cache) -> None:
    """Tripwire: invoking this means the bootstrap gate failed to short-circuit."""
    raise AssertionError("gate must not run")


def test_main_exits_nonzero_on_missing_cache(monkeypatch, capsys) -> None:
    """_enforce_bootstrap_gate must call sys.exit(1) when preflight is not ok."""
    monkeypatch.setattr(
        cli, "require_bootstrap_valid", lambda c: _make_preflight(False)
    )
    monkeypatch.setattr(sys, "argv", ["cull", "/tmp/fake-source"])
    with pytest.raises(SystemExit) as excinfo:
        cli._enforce_bootstrap_gate()
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "cull setup" in captured.err


def test_main_proceeds_when_preflight_ok(monkeypatch) -> None:
    """_enforce_bootstrap_gate must be a no-op when preflight reports ok."""
    monkeypatch.setattr(
        cli, "require_bootstrap_valid", lambda c: _make_preflight(True, "ok")
    )
    monkeypatch.setattr(sys, "argv", ["cull", "/tmp/fake-source"])
    cli._enforce_bootstrap_gate()


def test_setup_subcommand_bypasses_gate(monkeypatch) -> None:
    """Setup subcommand invocation must skip the preflight check entirely.

    The tripwire `_gate_must_not_run` raises AssertionError if invoked, so
    a clean return from `_enforce_bootstrap_gate()` IS the assertion."""
    monkeypatch.setattr(cli, "require_bootstrap_valid", _gate_must_not_run)
    monkeypatch.setattr(sys, "argv", ["cull", "setup"])
    cli._enforce_bootstrap_gate()
