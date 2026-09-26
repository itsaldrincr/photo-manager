from __future__ import annotations

from pathlib import Path

import pytest

from cull.review_handoff import (
    HANDOFF_ENV_VALUE,
    HANDOFF_ENV_VAR,
    ReviewHandoffInput,
    ReviewHandoffUnavailable,
    build_ghostty_open_command,
    ensure_handoff_available,
    forwarded_env,
    should_handoff_review,
)


def test_should_handoff_review_requires_cmux_and_skips_child(
    monkeypatch,
) -> None:
    monkeypatch.setattr("cull.review_handoff.sys.platform", "darwin")
    monkeypatch.delenv("CMUX_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("CMUX_SURFACE_ID", raising=False)
    monkeypatch.delenv(HANDOFF_ENV_VAR, raising=False)

    assert should_handoff_review() is False

    monkeypatch.setenv("CMUX_WORKSPACE_ID", "workspace:1")
    assert should_handoff_review() is True

    monkeypatch.setenv(HANDOFF_ENV_VAR, HANDOFF_ENV_VALUE)
    assert should_handoff_review() is False


def test_build_ghostty_open_command_uses_waiting_open_and_review_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "cull.review_handoff.resolve_cull_executable",
        lambda: "/tmp/venv/bin/cull",
    )
    monkeypatch.setattr("cull.review_handoff.os.environ", {})
    handoff_in = ReviewHandoffInput(
        cwd=tmp_path,
        session_path=tmp_path / "review-session.json",
    )

    command = build_ghostty_open_command(handoff_in)

    assert command[:9] == [
        "/usr/bin/open",
        "-n",
        "-W",
        "-F",
        "-a",
        "Ghostty",
        "--env",
        f"{HANDOFF_ENV_VAR}={HANDOFF_ENV_VALUE}",
        "--args",
    ]
    assert command[-4:-1] == ["-e", "/bin/zsh", "-lc"]
    assert "--review-session" in command[-1]
    assert "review-session.json" in command[-1]
    assert "cd " in command[-1]
    assert HANDOFF_ENV_VAR not in command[-1]


def test_ghostty_quits_with_its_window_and_opens_maximised(monkeypatch, tmp_path: Path) -> None:
    """Config flags sit between --args and -e so Ghostty (not zsh) reads them."""
    monkeypatch.setattr("cull.review_handoff.resolve_cull_executable", lambda: "/tmp/cull")
    command = build_ghostty_open_command(ReviewHandoffInput(cwd=tmp_path, session_path=tmp_path / "s.json"))
    ghostty_args = command[command.index("--args") + 1:command.index("-e")]
    assert "--quit-after-last-window-closed=true" in ghostty_args
    assert "--maximize=true" in ghostty_args


def test_forwarded_env_passes_app_settings_but_not_credentials() -> None:
    """CULL_/PHOTO_MANAGER_ settings reach the child; token-like names do not."""
    environ = {
        "PHOTO_MANAGER_VLM_ROOT": "/models",
        "CULL_TUI_DEBUG": "1",
        "PHOTO_MANAGER_API_TOKEN": "hidden",
        HANDOFF_ENV_VAR: HANDOFF_ENV_VALUE,
        "HOME": "/Users/x",
    }
    assert forwarded_env(environ) == [
        "--env", "CULL_TUI_DEBUG=1",
        "--env", "PHOTO_MANAGER_VLM_ROOT=/models",
    ]


def test_missing_ghostty_is_reported_as_unavailable(monkeypatch) -> None:
    """A missing Ghostty raises the pre-launch error the CLI falls back on."""
    monkeypatch.setattr("cull.review_handoff.resolve_cull_executable", lambda: "/tmp/cull")
    monkeypatch.setattr("cull.review_handoff.is_ghostty_installed", lambda: False)
    with pytest.raises(ReviewHandoffUnavailable):
        ensure_handoff_available()
