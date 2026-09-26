"""Helpers for handing the review TUI off from cmux to Ghostty."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

HANDOFF_ENV_VAR: str = "CULL_REVIEW_HANDOFF"
HANDOFF_ENV_VALUE: str = "ghostty"
CMUX_ENV_VARS: tuple[str, str] = ("CMUX_WORKSPACE_ID", "CMUX_SURFACE_ID")
OPEN_BIN: str = "/usr/bin/open"
GHOSTTY_APP: str = "Ghostty"
SHELL_BIN: str = "/bin/zsh"
# `open -n -W` waits for the whole Ghostty instance, and on macOS an instance
# outlives its last window by default: without the first flag the parent
# blocks until Cmd+Q. The window opens maximised and restores no old windows.
GHOSTTY_CONFIG_ARGS: tuple[str, ...] = (
    "--quit-after-last-window-closed=true",
    "--maximize=true",
    "--window-save-state=never",
)
# LaunchServices starts Ghostty with a fresh environment, and a login shell
# only restores what the zsh profile sets, so settings exported in the
# launching shell are forwarded explicitly. Credential-like names stay out
# of argv, where any user could read them from `ps`.
FORWARDED_ENV_PREFIXES: tuple[str, ...] = ("CULL_", "PHOTO_MANAGER_")
SECRET_NAME_MARKERS: tuple[str, ...] = ("TOKEN", "SECRET", "KEY", "PASSWORD")


class ReviewHandoffInput(BaseModel):
    """Inputs required to launch a blocking Ghostty review session."""

    cwd: Path
    session_path: Path


class ReviewHandoffError(RuntimeError):
    """Raised when the Ghostty review handoff cannot be started."""


class ReviewHandoffUnavailable(ReviewHandoffError):
    """Raised before launch when Ghostty or the `cull` executable is missing."""


def resolve_cull_executable() -> str:
    """Return the installed ``cull`` executable path."""
    cull_bin = shutil.which("cull")
    if cull_bin:
        return cull_bin
    raise ReviewHandoffUnavailable("Could not find the `cull` executable in PATH.")


def is_ghostty_installed() -> bool:
    """Return True when LaunchServices can find the Ghostty app."""
    result = subprocess.run(
        [OPEN_BIN, "-Ra", GHOSTTY_APP], capture_output=True, text=True, check=False,
    )
    return result.returncode == 0


def ensure_handoff_available() -> None:
    """Raise ReviewHandoffUnavailable unless both Ghostty and `cull` can be found."""
    resolve_cull_executable()
    if not is_ghostty_installed():
        raise ReviewHandoffUnavailable("Ghostty is not installed.")


def is_cmux_session() -> bool:
    """Return True when running inside cmux."""
    return any(os.environ.get(name) for name in CMUX_ENV_VARS)


def is_handoff_child() -> bool:
    """Return True when already executing inside a Ghostty handoff child."""
    return os.environ.get(HANDOFF_ENV_VAR) == HANDOFF_ENV_VALUE


def should_handoff_review() -> bool:
    """Return True only for a top-level cmux process on macOS."""
    if sys.platform != "darwin":
        return False
    if is_handoff_child():
        return False
    return is_cmux_session()


def forwarded_env(environ: Mapping[str, str]) -> list[str]:
    """Return ``--env NAME=VALUE`` pairs for app settings to pass to the child."""
    args: list[str] = []
    for name in sorted(environ):
        if name == HANDOFF_ENV_VAR or not name.startswith(FORWARDED_ENV_PREFIXES):
            continue
        if any(marker in name.upper() for marker in SECRET_NAME_MARKERS):
            continue
        args.extend(["--env", f"{name}={environ[name]}"])
    return args


def build_ghostty_open_command(handoff_in: ReviewHandoffInput) -> list[str]:
    """Build the blocking Ghostty launch command for a serialized review session."""
    cull_bin = resolve_cull_executable()
    shell_command = " && ".join([
        f"cd {shlex.quote(str(handoff_in.cwd))}",
        "exec "
        + " ".join([
            shlex.quote(cull_bin),
            "--review-session",
            shlex.quote(str(handoff_in.session_path)),
        ]),
    ])
    return [
        OPEN_BIN,
        "-n",
        "-W",
        "-F",
        "-a",
        GHOSTTY_APP,
        "--env",
        f"{HANDOFF_ENV_VAR}={HANDOFF_ENV_VALUE}",
        *forwarded_env(os.environ),
        "--args",
        *GHOSTTY_CONFIG_ARGS,
        "-e",
        SHELL_BIN,
        "-lc",
        shell_command,
    ]


def launch_review_handoff(handoff_in: ReviewHandoffInput) -> None:
    """Launch Ghostty and block until the review session exits."""
    result = subprocess.run(
        build_ghostty_open_command(handoff_in),
        cwd=handoff_in.cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return
    stderr = result.stderr.strip() or "Ghostty launch failed."
    raise ReviewHandoffError(stderr)
