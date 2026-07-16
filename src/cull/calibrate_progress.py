"""Rich progress reporter for `cull --calibrate`.

Two-phase reporter: an opaque spinner for Stage 2 batch scoring and a
determinate per-photo bar for the LAION aesthetic pass. The factory picks
RichProgress when the caller is on a TTY and NullProgress otherwise so
log capture and CI runs stay clean.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Protocol

from pydantic import BaseModel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

INDETERMINATE_PHASE_TOTAL: int = 1


class PhaseStart(BaseModel):
    """Inputs for starting a tracked phase."""

    label: str
    total: int | None


class CalibrationProgress(Protocol):
    """Reporter contract: start a phase, advance within it, end it."""

    def start_phase(self, phase: PhaseStart) -> None:
        """Begin a new tracked phase; closes any active one first."""

    def advance(self) -> None:
        """Advance the active phase by one unit."""

    def end_phase(self) -> None:
        """Mark the active phase complete."""


class NullProgress:
    """No-op reporter for non-TTY environments and tests."""

    def start_phase(self, phase: PhaseStart) -> None:
        """Discard the phase event."""
        return

    def advance(self) -> None:
        """Discard the advance event."""
        return

    def end_phase(self) -> None:
        """Discard the end event."""
        return


class RichProgress:
    """Rich-backed reporter; one active task at a time."""

    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._task_id: TaskID | None = None

    def start_phase(self, phase: PhaseStart) -> None:
        """Add a Rich task for the new phase and remember its id."""
        self._task_id = self._progress.add_task(phase.label, total=phase.total)

    def advance(self) -> None:
        """Increment the active Rich task by one."""
        if self._task_id is not None:
            self._progress.advance(self._task_id)

    def end_phase(self) -> None:
        """Flush the active Rich task to its completion target then clear it."""
        if self._task_id is None:
            return
        task = self._progress.tasks[self._task_id]
        target = task.total if task.total is not None else INDETERMINATE_PHASE_TOTAL
        self._progress.update(self._task_id, total=target, completed=target)
        self._task_id = None


def _build_rich_progress() -> Progress:
    """Construct the Rich Progress instance with our column layout."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )


@contextmanager
def calibration_progress(use_rich: bool) -> Iterator[CalibrationProgress]:
    """Yield a CalibrationProgress; manages Rich console lifecycle if use_rich."""
    if not use_rich:
        yield NullProgress()
        return
    progress = _build_rich_progress()
    with progress:
        yield RichProgress(progress)
