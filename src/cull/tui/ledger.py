"""Review decisions, user overrides, and compound undo for the TUI."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from cull.models import DecisionLabel, OverrideEntry, PhotoDecision
from cull.override_log import OverrideContext, build_override_entry

logger = logging.getLogger(__name__)

def _apply_override(decision: PhotoDecision, label: DecisionLabel) -> PhotoDecision:
    """Return a copy of the decision with the label overridden."""
    return decision.model_copy(update={
        "decision": label,
        "is_override": True,
        "override_from": decision.decision,
        "override_by": "user_tui",
    })


def ai_label_of(decision: PhotoDecision) -> DecisionLabel:
    """Return the label the pipeline chose, before any user override."""
    if decision.is_override and decision.override_from is not None:
        return decision.override_from
    return decision.decision


class LabelChange(BaseModel):
    """Set one decision (by index into the session) to a label."""

    index: int
    label: DecisionLabel


class ChangeBatch(BaseModel):
    """Label changes made by one user action, undone together."""

    changes: list[LabelChange]
    origin: str
    bumps_retrain: bool = False


class PriorState(BaseModel):
    """What one decision looked like before a change, for undo."""

    index: int
    decision: PhotoDecision
    override: DecisionLabel | None
    was_reviewed: bool


class UndoEntry(BaseModel):
    """One undoable user action and the override-log entries it wrote."""

    origin: str
    priors: list[PriorState] = Field(default_factory=list)
    logged: list[OverrideEntry] = Field(default_factory=list)
    retrain_bumps: int = 0
    reviewed_only: list[str] = Field(default_factory=list)


class ReviewLedger:
    """Owns decision labels during review; every mutation goes through here."""

    def __init__(self, decisions: list[PhotoDecision], session_source: str) -> None:
        self.decisions = decisions
        self.session_source = session_source
        self.overrides: dict[str, DecisionLabel] = {}
        self.reviewed: set[str] = set()
        self.ai_labels: dict[str, DecisionLabel] = {str(d.photo.path): ai_label_of(d) for d in decisions}
        self._undo: list[UndoEntry] = []

    @property
    def can_undo(self) -> bool:
        """Return True when there is an action to undo."""
        return bool(self._undo)

    def restore(self, overrides: dict[str, DecisionLabel]) -> None:
        """Re-apply overrides saved by an earlier, unsaved session."""
        index_by_path = {str(d.photo.path): i for i, d in enumerate(self.decisions)}
        for path_str, label in overrides.items():
            index = index_by_path.get(path_str)
            if index is None:
                continue
            self.overrides[path_str] = label
            self.decisions[index] = _apply_override(self.decisions[index], label)

    def apply(self, batch: ChangeBatch) -> UndoEntry | None:
        """Apply a batch; return its undo entry, or None if it changed nothing at all.

        A no-op (same label as now) marks the photo reviewed but writes no
        override-log entry, so confirming the AI never trains the taste model.
        """
        entry = UndoEntry(origin=batch.origin)
        for change in batch.changes:
            self._apply_one(change, entry)
        if not entry.priors and not entry.reviewed_only:
            return None
        if batch.bumps_retrain:
            entry.retrain_bumps = len(entry.logged)
        self._undo.append(entry)
        return entry

    def _apply_one(self, change: LabelChange, entry: UndoEntry) -> None:
        """Apply one change and record in ``entry`` what undo needs."""
        origin = entry.origin
        decision = self.decisions[change.index]
        path_str = str(decision.photo.path)
        was_reviewed = path_str in self.reviewed
        self.reviewed.add(path_str)
        if decision.decision == change.label:
            if not was_reviewed:
                entry.reviewed_only.append(path_str)
            return
        entry.priors.append(PriorState(
            index=change.index, decision=decision, override=self.overrides.get(path_str), was_reviewed=was_reviewed,
        ))
        logged = self._build_log_entry(decision, OverrideContext(
            new_label=change.label, session_source=self.session_source, origin=origin,
        ))
        if logged is not None:
            entry.logged.append(logged)
        self.overrides[path_str] = change.label
        self.decisions[change.index] = _apply_override(decision, change.label)

    def _build_log_entry(self, decision: PhotoDecision, ctx: OverrideContext) -> OverrideEntry | None:
        """Build the override-log entry; a malformed decision is logged, not fatal."""
        try:
            return build_override_entry(decision, ctx)
        except (AttributeError, ValueError) as exc:
            logger.warning("override entry build failed for %s: %s", decision.photo.filename, exc)
            return None

    def undo(self) -> UndoEntry | None:
        """Revert the most recent action and return it, or None if nothing to undo."""
        if not self._undo:
            return None
        entry = self._undo.pop()
        for prior in reversed(entry.priors):
            self._restore_prior(prior)
        for path_str in entry.reviewed_only:
            self.reviewed.discard(path_str)
        return entry

    def _restore_prior(self, prior: PriorState) -> None:
        """Put one decision, its override, and its reviewed mark back."""
        self.decisions[prior.index] = prior.decision
        path_str = str(prior.decision.photo.path)
        if prior.override is None:
            self.overrides.pop(path_str, None)
        else:
            self.overrides[path_str] = prior.override
        if not prior.was_reviewed:
            self.reviewed.discard(path_str)

    def label_counts(self) -> dict[str, int]:
        """Return how many decisions carry each label right now."""
        counts: dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.decision] = counts.get(decision.decision, 0) + 1
        return counts
