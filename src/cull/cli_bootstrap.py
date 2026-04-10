"""Offline-cache bootstrap gate and setup subcommand for the cull CLI."""

from __future__ import annotations

import sys

import click

from cull.config import ModelCacheConfig
from cull.env_bootstrap import bootstrap_default
from cull.setup_command import setup_command as _run_setup_command

# Module-level side effect — REQUIRED. Sets HF_HOME / HF_HUB_CACHE / TORCH_HOME /
# DEEPFACE_HOME / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE before any cull.* import
# that transitively pulls torch / transformers / pyiqa / mediapipe / deepface.
# Importing this module is what bootstraps the offline cache env state.
_CACHE: ModelCacheConfig = bootstrap_default()

SETUP_SUBCOMMAND_TOKEN: str = "setup"


def _is_setup_invocation() -> bool:
    """Return True iff the first positional argument is the `setup` subcommand."""
    return len(sys.argv) > 1 and sys.argv[1] == SETUP_SUBCOMMAND_TOKEN


@click.command(SETUP_SUBCOMMAND_TOKEN)
@click.option("--allow-network", is_flag=True, default=False)
@click.option("--force", is_flag=True, default=False)
def setup(allow_network: bool, force: bool) -> None:
    """One-time bootstrap to fetch and verify offline model assets."""
    _run_setup_command(allow_network, force)
