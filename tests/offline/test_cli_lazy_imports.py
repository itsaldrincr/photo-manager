"""Tests that importing cull.cli stays lightweight until dispatch time."""

from __future__ import annotations

import importlib
import sys


PIPELINE_IMPORTS: tuple[str, ...] = (
    "cull.cli",
    "cull.cli_pipeline",
    "cull.cli_results",
    "cull.cli_review",
    "cull.cli_subcommands",
    "cull.pipeline",
    "cull._pipeline",
)


def test_cli_import_does_not_pull_pipeline_stack() -> None:
    """Importing cull.cli must not eagerly import the heavy pipeline modules."""
    for name in PIPELINE_IMPORTS:
        sys.modules.pop(name, None)

    importlib.import_module("cull.cli")

    for name in PIPELINE_IMPORTS[1:]:
        assert name not in sys.modules
