"""Stage 1 routing lists come out in path order, not worker completion order."""

from __future__ import annotations

from pathlib import Path

from cull._pipeline.stage1_runner import _sort_by_path, _Stage1Output

COMPLETION_ORDER: list[str] = ["c.jpg", "a.jpg", "d.jpg", "b.jpg"]


def test_routing_lists_are_sorted_by_path() -> None:
    """Survivors, rejects and failures are in path order whatever order workers finished."""
    paths = [Path("/shoot") / name for name in COMPLETION_ORDER]
    output = _Stage1Output(survivors=list(paths), rejected=list(reversed(paths)), failed_paths=list(paths))

    _sort_by_path(output)

    assert output.survivors == sorted(paths)
    assert output.rejected == sorted(paths)
    assert output.failed_paths == sorted(paths)
