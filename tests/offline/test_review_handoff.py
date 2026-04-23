from __future__ import annotations

from pathlib import Path

from cull.review_handoff import (
    HANDOFF_ENV_VALUE,
    HANDOFF_ENV_VAR,
    ReviewHandoffInput,
    build_ghostty_open_command,
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
    handoff_in = ReviewHandoffInput(
        cwd=tmp_path,
        session_path=tmp_path / "review-session.json",
    )

    command = build_ghostty_open_command(handoff_in)

    assert command[:6] == [
        "/usr/bin/open",
        "-n",
        "-W",
        "-a",
        "Ghostty",
        "--args",
    ]
    assert command[6:9] == ["-e", "/bin/zsh", "-lc"]
    assert "--review-session" in command[-1]
    assert "review-session.json" in command[-1]
    assert "cd " in command[-1]
