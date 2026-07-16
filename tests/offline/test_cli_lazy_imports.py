"""Tests that importing cull.cli stays lightweight until dispatch time."""

from __future__ import annotations

import importlib
import sys

import pytest


PIPELINE_IMPORTS: tuple[str, ...] = (
    "cull.cli",
    "cull.cli_pipeline",
    "cull.cli_results",
    "cull.cli_review",
    "cull.cli_subcommands",
    "cull.pipeline",
    "cull._pipeline",
)


def test_cli_import_does_not_pull_pipeline_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing cull.cli must not eagerly import the heavy pipeline modules.

    Uses monkeypatch.delitem (not a raw sys.modules.pop) so the removed
    entries are restored after the test — a bare pop leaves cull._pipeline's
    already-imported submodules (orchestrator, stage2_runner, ...) orphaned
    from any later fresh reimport of the parent package, which silently
    breaks unrelated tests' `monkeypatch.setattr("cull._pipeline.X...")`
    calls for the rest of the session.
    """
    for name in PIPELINE_IMPORTS:
        monkeypatch.delitem(sys.modules, name, raising=False)

    importlib.import_module("cull.cli")

    for name in PIPELINE_IMPORTS[1:]:
        assert name not in sys.modules
