"""Textual TUI application for interactive photo culling review."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static

from cull.config import (
    CullConfig,
    TASTE_PROFILE_PATH,
    TUI_AUTOSAVE_BATCH_CONFIDENCE,
    TUI_AUTOSAVE_INTERVAL_SECONDS,
)
from cull.models import DecisionLabel, ExplainRequest, ExplainResult, OverrideEntry, PhotoDecision
from cull.override_log import OverrideContext, build_override_entry, log_override
from cull.pipeline import SessionResult
from cull._pipeline.decision_assembly import _build_summary
from cull.report import write_report
from cull.router import execute_moves
from cull.taste_trainer import TasteTrainerInput, maybe_retrain
from cull.tui.burst_view import BurstGroup, BurstResult, BurstView, build_burst_group
from cull.tui.explain_modal import ExplainPanel, fetch_explanation_result
from cull.tui.photo_view import PhotoView, PrecacheRequest, RenderRequest, ViewportSize, precache_images
from cull.tui.score_panel import ScorePanel

OVERRIDE_ORIGIN_SINGLE: str = "single"
OVERRIDE_ORIGIN_BURST: str = "burst"
OVERRIDE_ORIGIN_BULK: str = "bulk"
OVERRIDE_ORIGIN_AUTO: str = "auto_accept"

logger = logging.getLogger(__name__)

STATE_FILENAME: str = ".cull_tui_state.json"
SAVE_IN_PROGRESS_MESSAGE: str = "Saving review changes..."
SAVE_COMPLETE_MESSAGE: str = "Save complete. Exiting..."
SAVE_FAILED_PREFIX: str = "Save failed: "
SAVE_COMPLETE_DELAY_SECONDS: float = 0.25
QUEUE_UNCERTAIN: int = 0
QUEUE_REJECTED: int = 1
QUEUE_DUPLICATES: int = 2
QUEUE_KEEPERS: int = 3
QUEUE_SELECTED: int = 4
QUEUE_LABELS: dict[int, DecisionLabel] = {
    QUEUE_UNCERTAIN: "uncertain",
    QUEUE_REJECTED: "rejected",
    QUEUE_DUPLICATES: "duplicate",
    QUEUE_KEEPERS: "keeper",
    QUEUE_SELECTED: "select",
}
QUEUE_HOTKEY: dict[str, str] = {
    "uncertain": "1",
    "rejected": "2",
    "duplicate": "3",
    "keeper": "4",
    "select": "5",
}
QUEUE_CYCLE_ORDER: tuple[int, ...] = (
    QUEUE_SELECTED, QUEUE_KEEPERS, QUEUE_UNCERTAIN, QUEUE_REJECTED, QUEUE_DUPLICATES,
)
# Priority order for auto-fallback when the initially-chosen queue is empty.
# Curated selections win over keepers so --review after --curate lands on the top-N.
_FALLBACK_QUEUE_ORDER: tuple[int, ...] = (
    QUEUE_SELECTED, QUEUE_KEEPERS, QUEUE_UNCERTAIN, QUEUE_REJECTED, QUEUE_DUPLICATES,
)
_RECOVERY_DIRS: tuple[tuple[str, ...], ...] = (
    (),
    ("_curated", "_selects"),
    ("_review", "_uncertain"),
    ("_review", "_rejected"),
    ("_review", "_duplicates"),
)


class UndoEntry(BaseModel):
    """Record of a single decision change for undo support."""

    photo_path: str
    previous_label: DecisionLabel


class OverrideHookCtx(BaseModel):
    """Bundle for emitting one override-log hook from a TUI mutation site."""

    decision: PhotoDecision
    new_label: DecisionLabel
    origin: str


class TuiState(BaseModel):
    """Serializable TUI state for auto-save and crash recovery."""

    overrides: dict[str, DecisionLabel] = Field(default_factory=dict)
    current_index: int = 0
    current_queue: int = QUEUE_UNCERTAIN


class AppInput(BaseModel):
    """Input bundle for constructing CullApp."""

    session: SessionResult
    config: CullConfig


def _state_path(session: SessionResult) -> Path:
    """Return the path to the TUI state file."""
    return Path(session.source_path) / STATE_FILENAME


def _save_state(state: TuiState, path: Path) -> None:
    """Write TUI state to disk as JSON."""
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    logger.info("TUI state saved to %s", path)


def _load_state(path: Path) -> TuiState | None:
    """Load TUI state from disk if it exists."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TuiState.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Corrupt TUI state file, starting fresh")
        return None


def _filter_queue(decisions: list[PhotoDecision], label: DecisionLabel) -> list[int]:
    """Return indices of decisions matching the given label."""
    return [i for i, d in enumerate(decisions) if d.decision == label]


def _taste_uncertainty(decision: PhotoDecision) -> float:
    """Return taste uncertainty score: abs(probability - 0.5), or -1 if absent."""
    if decision.stage2 is None or decision.stage2.taste is None:
        return -1.0
    return abs(decision.stage2.taste.probability - 0.5)


def _sort_by_uncertainty(decisions: list[PhotoDecision], indices: list[int]) -> list[int]:
    """Sort indices by taste uncertainty descending if any decision has TasteScore."""
    has_taste = any(decisions[i].stage2 and decisions[i].stage2.taste for i in indices)
    if not has_taste:
        return indices
    return sorted(indices, key=lambda i: _taste_uncertainty(decisions[i]), reverse=True)


def _apply_override(decision: PhotoDecision, label: DecisionLabel) -> PhotoDecision:
    """Return a copy of the decision with the label overridden."""
    return decision.model_copy(update={
        "decision": label,
        "is_override": True,
        "override_from": decision.decision,
        "override_by": "user_tui",
    })


class InfoBarContext(BaseModel):
    """Context for building the info bar display text."""

    decision: PhotoDecision
    position: int
    total: int
    queue_label: str
    queue_counts: dict[str, int]


def _build_info_text(ctx: InfoBarContext) -> str:
    """Build the info bar text for the current photo with queue navigation hints."""
    pct = int((ctx.position + 1) / ctx.total * 100) if ctx.total > 0 else 0
    queue_badges = "  ".join(
        f"[{QUEUE_HOTKEY[label]}] {label}:{count}"
        for label, count in ctx.queue_counts.items()
    )
    return (
        f"{ctx.decision.photo.filename}    "
        f"{ctx.decision.decision.upper()}    "
        f"[{ctx.position + 1}/{ctx.total}]  {pct}%    "
        f"\u2502  queue: {ctx.queue_label}    {queue_badges}"
    )


def _find_burst_decisions(session: SessionResult, group_id: int) -> list[PhotoDecision]:
    """Find all decisions belonging to a burst group."""
    return [
        d for d in session.decisions
        if d.stage1 and d.stage1.burst and d.stage1.burst.group_id == group_id
    ]


class CullApp(App):
    """Main Textual application for interactive photo review."""

    TITLE = "CULL -- Review"

    BINDINGS = [
        Binding("k", "keep", "Keep"),
        Binding("r", "reject", "Reject"),
        Binding("d", "mark_duplicate", "Duplicate"),
        Binding("c", "curate", "Curate"),
        Binding("right,>,period", "next_photo", "Next", show=False),
        Binding("left,<,comma", "prev_photo", "Prev", show=False),
        Binding("u", "undo", "Undo"),
        Binding("s", "toggle_scores", "Scores"),
        Binding("b", "burst_view", "Burst"),
        Binding("q", "save_quit", "Save+Quit"),
        Binding("Q", "quit_no_save", "Quit (no save)", show=False),
        Binding("tab", "cycle_queue", "Next Queue"),
        Binding("1", "queue_1", "Uncertain", show=False),
        Binding("2", "queue_2", "Rejected", show=False),
        Binding("3", "queue_3", "Duplicates", show=False),
        Binding("4", "queue_4", "Keepers", show=False),
        Binding("5", "queue_5", "Selected", show=False),
        Binding("K", "bulk_keep", "Keep All", show=False),
        Binding("R", "bulk_reject", "Reject All", show=False),
        Binding("shift+r", "reject_cluster", "Reject Cluster", show=False),
        Binding("A", "auto_accept", "Auto-accept VLM", show=False),
        Binding("?,shift+slash", "explain", "Explain"),
    ]

    CSS = """
    #info-bar {
        height: 3;
        border: solid blue;
    }
    """

    def __init__(self, app_input: AppInput) -> None:
        super().__init__()
        self._session = app_input.session
        self._config = app_input.config
        self._overrides: dict[str, DecisionLabel] = {}
        self._undo_stack: list[UndoEntry] = []
        self._queue_index: int = QUEUE_UNCERTAIN
        self._photo_index: int = 0
        self._queue_indices: list[int] = []
        self._save_in_progress: bool = False
        self._status_message: str | None = None
        self._normalize_decision_destinations()
        self._restore_state()

    def _normalize_decision_destinations(self) -> None:
        """Walk decisions and set `destination` to the current on-disk path.

        Session reports written before router.process_single_move started
        persisting post-move destinations leave ``destination=None`` even
        though the file has been physically moved.  We probe each decision's
        original source, any existing destination, and the routed destination
        to find where the file actually lives right now, then pin it.
        """
        from cull.router import route_photo  # noqa: PLC0415

        for decision in self._session.decisions:
            if decision.destination is not None and decision.destination.exists():
                continue
            if decision.photo.path.exists():
                decision.destination = None
                continue
            routed = route_photo(decision, self._config)
            if routed.exists():
                decision.destination = routed

    def _restore_state(self) -> None:
        """Restore state from disk if available."""
        state = _load_state(_state_path(self._session))
        if state is None:
            return
        self._overrides = dict(state.overrides)
        self._photo_index = state.current_index
        self._queue_index = state.current_queue
        self._apply_restored_overrides()

    def _apply_restored_overrides(self) -> None:
        """Apply restored overrides to session decisions."""
        for path_str, label in self._overrides.items():
            for i, d in enumerate(self._session.decisions):
                if str(d.photo.path) == path_str:
                    self._session.decisions[i] = _apply_override(d, label)
                    break

    def compose(self) -> ComposeResult:
        """Build the main screen layout."""
        yield Header()
        yield PhotoView()
        yield ExplainPanel()
        yield Static("", id="info-bar")
        yield ScorePanel()
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the display after mounting; fall back to first non-empty queue."""
        from cull.tui.photo_view import _dlog  # noqa: PLC0415

        _dlog(f"CullApp.on_mount: initial queue_index={self._queue_index}")
        self._rebuild_queue()
        _dlog(f"CullApp.on_mount: after rebuild len={len(self._queue_indices)}")
        if not self._queue_indices:
            self._fallback_to_non_empty_queue()
            _dlog(f"CullApp.on_mount: after fallback queue_index={self._queue_index} len={len(self._queue_indices)}")
        self._display_current()
        self._schedule_autosave()

    def _fallback_to_non_empty_queue(self) -> None:
        """Switch to the first queue in _FALLBACK_QUEUE_ORDER that has decisions."""
        for queue_index in _FALLBACK_QUEUE_ORDER:
            if queue_index == self._queue_index:
                continue
            self._queue_index = queue_index
            self._rebuild_queue()
            if self._queue_indices:
                return

    def _schedule_autosave(self) -> None:
        """Schedule periodic auto-save."""
        self.set_interval(TUI_AUTOSAVE_INTERVAL_SECONDS, self._autosave)

    def _autosave(self) -> None:
        """Save current state to disk."""
        state = TuiState(
            overrides=dict(self._overrides),
            current_index=self._photo_index,
            current_queue=self._queue_index,
        )
        _save_state(state, _state_path(self._session))

    def _rebuild_queue(self) -> None:
        """Rebuild the current queue indices from decisions, sorted by taste uncertainty."""
        label = QUEUE_LABELS[self._queue_index]
        raw_indices = _filter_queue(self._session.decisions, label)
        self._queue_indices = _sort_by_uncertainty(self._session.decisions, raw_indices)
        self._photo_index = min(self._photo_index, max(0, len(self._queue_indices) - 1))

    def _current_decision(self) -> PhotoDecision | None:
        """Return the current photo decision, or None if queue is empty."""
        if not self._queue_indices:
            return None
        idx = self._queue_indices[self._photo_index]
        return self._session.decisions[idx]

    def _display_current(self) -> None:
        """Update all widgets with the current photo."""
        from cull.tui.photo_view import _dlog  # noqa: PLC0415

        decision = self._current_decision()
        _dlog(f"_display_current: decision={None if decision is None else decision.photo.path.name}")
        if decision is None:
            self._show_empty_queue()
            return
        self._sync_explain_panel(decision)
        self._show_photo(decision)
        self._show_info(decision)

    def _sync_explain_panel(self, decision: PhotoDecision) -> None:
        """Hide a stale explain panel when the current photo changes."""
        panel = self.query_one(ExplainPanel)
        current_path = str(self._resolve_decision_path(decision))
        if panel.photo_path is not None and panel.photo_path != current_path:
            panel.hide_panel()

    def _show_empty_queue(self) -> None:
        """Display empty queue message."""
        self._update_info_bar(None)

    def _update_info_bar(self, decision: PhotoDecision | None) -> None:
        """Render the info bar, preferring any active status message."""
        info_bar = self.query_one("#info-bar", Static)
        if self._status_message is not None:
            info_bar.update(self._status_message)
            return
        if decision is None:
            info_bar.update("Queue is empty")
            return
        counts = {
            QUEUE_LABELS[qi]: len(_filter_queue(self._session.decisions, QUEUE_LABELS[qi]))
            for qi in QUEUE_CYCLE_ORDER
        }
        ctx = InfoBarContext(
            decision=decision,
            position=self._photo_index,
            total=len(self._queue_indices),
            queue_label=QUEUE_LABELS[self._queue_index],
            queue_counts=counts,
        )
        info_bar.update(_build_info_text(ctx))

    def _resolve_decision_path(self, decision: PhotoDecision) -> Path:
        """Return the current on-disk path for a decision, routing through moves."""
        from cull.router import route_photo  # noqa: PLC0415

        if decision.destination is not None and decision.destination.exists():
            return decision.destination
        if decision.photo.path.exists():
            return decision.photo.path
        routed = route_photo(decision, self._config)
        if routed.exists():
            return routed
        recovered = self._recover_missing_path(decision)
        if recovered is not None:
            decision.destination = recovered
            return recovered
        return decision.photo.path

    def _recover_missing_path(self, decision: PhotoDecision) -> Path | None:
        """Search known review/curation folders for a moved photo by filename."""
        source = decision.photo.path
        for parts in _RECOVERY_DIRS:
            candidate = source.parent.joinpath(*parts, source.name)
            if candidate.exists():
                return candidate
        return None

    def _show_photo(self, decision: PhotoDecision) -> None:
        """Load and display the current photo."""
        from cull.tui.photo_view import _dlog  # noqa: PLC0415

        _dlog(f"_show_photo: entry path={decision.photo.path.name}")
        try:
            photo_view = self.query_one(PhotoView)
        except Exception as exc:  # noqa: BLE001
            _dlog(f"_show_photo: query_one(PhotoView) failed: {type(exc).__name__}: {exc}")
            return
        _dlog(f"_show_photo: photo_view.size=({photo_view.size.width},{photo_view.size.height})")
        actual_path = self._resolve_decision_path(decision)
        _dlog(f"_show_photo: resolved to {actual_path}")
        try:
            image_bytes = actual_path.read_bytes()
        except OSError as exc:
            _dlog(f"_show_photo: read_bytes failed: {exc}")
            logger.warning("Cannot read %s", actual_path)
            photo_view.clear_terminal_image()
            self.query_one("#info-bar", Static).update(f"Missing photo: {actual_path.name}")
            return
        viewport = ViewportSize(cols=photo_view.size.width, rows=photo_view.size.height)
        req = RenderRequest(image_id=str(actual_path), image_bytes=image_bytes, viewport=viewport)
        _dlog(f"_show_photo: calling display_photo bytes={len(image_bytes)}")
        photo_view.display_photo(req)
        self._trigger_precache()

    def _show_info(self, decision: PhotoDecision) -> None:
        """Update the info bar with current photo details + queue navigation hints."""
        self._update_info_bar(decision)
        score_panel = self.query_one(ScorePanel)
        score_panel.show_scores(decision)

    def _trigger_precache(self) -> None:
        """Pre-cache nearby photos in a background worker so navigation stays snappy."""
        if not self._queue_indices:
            return
        photo_view = self.query_one(PhotoView)
        cols, rows = photo_view.size.width, photo_view.size.height
        if cols <= 1 or rows <= 1:
            return
        resolved = [
            self._resolve_decision_path(self._session.decisions[i])
            for i in self._queue_indices
        ]
        request = PrecacheRequest(paths=resolved, current_index=self._photo_index)
        viewport = ViewportSize(cols=cols, rows=rows)
        self.run_worker(
            lambda: precache_images(request, viewport),
            thread=True,
            exclusive=True,
            group="precache",
        )

    def _emit_override_log(self, ctx: OverrideHookCtx) -> OverrideEntry | None:
        """Build and persist one override-log entry; warn-only on write failure."""
        log_ctx = OverrideContext(
            new_label=ctx.new_label,
            session_source=str(self._session.source_path),
            origin=ctx.origin,
        )
        try:
            entry = build_override_entry(ctx.decision, log_ctx)
            log_override(entry)
            return entry
        except (OSError, AttributeError, ValueError) as exc:
            self.log.warning("override log write failed: %s", exc)
            return None

    def _trigger_retrain(self, entry: OverrideEntry) -> None:
        """Invoke maybe_retrain after an override is written to disk."""
        trainer_ctx = TasteTrainerInput(overrides=[entry], profile_path=TASTE_PROFILE_PATH)
        try:
            maybe_retrain(trainer_ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("taste retrain skipped: %s", exc)

    def _stage_move(self, label: DecisionLabel) -> None:
        """Stage a decision override for the current photo."""
        decision = self._current_decision()
        if decision is None:
            return
        path_str = str(decision.photo.path)
        undo = UndoEntry(photo_path=path_str, previous_label=decision.decision)
        self._undo_stack.append(undo)
        self._overrides[path_str] = label
        idx = self._queue_indices[self._photo_index]
        self._session.decisions[idx] = _apply_override(decision, label)
        hook_ctx = OverrideHookCtx(decision=decision, new_label=label, origin=OVERRIDE_ORIGIN_SINGLE)
        entry = self._emit_override_log(hook_ctx)
        if entry is not None:
            self._trigger_retrain(entry)
        self._rebuild_queue()
        self._display_current()

    def _switch_queue(self, queue_index: int) -> None:
        """Switch to a different review queue."""
        self._queue_index = queue_index
        self._photo_index = 0
        self._rebuild_queue()
        self._display_current()

    def action_keep(self) -> None:
        """Mark current photo as keeper."""
        self._stage_move("keeper")

    def action_reject(self) -> None:
        """Mark current photo as rejected."""
        self._stage_move("rejected")

    def action_mark_duplicate(self) -> None:
        """Mark current photo as duplicate."""
        self._stage_move("duplicate")

    def action_curate(self) -> None:
        """Promote current photo to the curated select queue."""
        self._stage_move("select")

    def action_next_photo(self) -> None:
        """Navigate to the next photo in the queue."""
        if self._photo_index < len(self._queue_indices) - 1:
            self._photo_index += 1
            self._display_current()

    def action_prev_photo(self) -> None:
        """Navigate to the previous photo in the queue."""
        if self._photo_index > 0:
            self._photo_index -= 1
            self._display_current()

    def action_undo(self) -> None:
        """Undo the last decision override."""
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        self._overrides.pop(entry.photo_path, None)
        self._apply_undo(entry)
        self._rebuild_queue()
        self._display_current()

    def _apply_undo(self, entry: UndoEntry) -> None:
        """Restore a single decision to its previous label."""
        for i, d in enumerate(self._session.decisions):
            if str(d.photo.path) == entry.photo_path:
                self._session.decisions[i] = _apply_override(d, entry.previous_label)
                break

    def action_toggle_scores(self) -> None:
        """Toggle the score detail panel."""
        self.query_one(ScorePanel).toggle_visible()

    def action_burst_view(self) -> None:
        """Open burst comparison view for the current photo."""
        decision = self._current_decision()
        if decision is None or decision.stage1 is None or decision.stage1.burst is None:
            return
        group_id = decision.stage1.burst.group_id
        burst_decs = _find_burst_decisions(self._session, group_id)
        group = build_burst_group(burst_decs, group_id)
        burst_screen = BurstView(group)
        self.push_screen(burst_screen, self._on_burst_complete)

    def _on_burst_complete(self, result: object) -> None:
        """Handle burst view completion and apply results."""
        if not isinstance(result, BurstResult):
            return
        if not result.is_confirmed:
            return
        self._apply_burst_result_winner(result.winner)
        self._apply_burst_result_dupes(result.duplicates)
        self._rebuild_queue()
        self._display_current()

    def _apply_burst_result_winner(self, winner: Path) -> None:
        """Apply keeper label to the burst winner."""
        for i, d in enumerate(self._session.decisions):
            if d.photo.path == winner:
                self._session.decisions[i] = _apply_override(d, "keeper")
                self._overrides[str(winner)] = "keeper"
                self._emit_override_log(OverrideHookCtx(decision=d, new_label="keeper", origin=OVERRIDE_ORIGIN_BURST))
                break

    def _apply_burst_result_dupes(self, duplicates: list[Path]) -> None:
        """Apply duplicate label to burst losers."""
        for dup_path in duplicates:
            for i, d in enumerate(self._session.decisions):
                if d.photo.path == dup_path:
                    self._session.decisions[i] = _apply_override(d, "duplicate")
                    self._overrides[str(dup_path)] = "duplicate"
                    self._emit_override_log(OverrideHookCtx(decision=d, new_label="duplicate", origin=OVERRIDE_ORIGIN_BURST))
                    break

    def action_save_quit(self) -> None:
        """Show a save banner, then persist moves/report after the next refresh."""
        if self._save_in_progress:
            return
        self._save_in_progress = True
        self._status_message = SAVE_IN_PROGRESS_MESSAGE
        self._update_info_bar(self._current_decision())
        self.call_after_refresh(self._commit_save_and_exit)

    def _commit_save_and_exit(self) -> None:
        """Persist pending review changes after the save banner has painted."""
        try:
            execute_moves(self._session.decisions, self._config)
            self._session.summary = _build_summary(self._session.decisions)
            write_report(self._session, overwrite=True)
            state_path = _state_path(self._session)
            if state_path.exists():
                state_path.unlink()
        except OSError as exc:
            self._save_in_progress = False
            self._status_message = f"{SAVE_FAILED_PREFIX}{exc}"
            self._update_info_bar(self._current_decision())
            self.log.warning("review save failed: %s", exc)
            return
        self._status_message = SAVE_COMPLETE_MESSAGE
        self._update_info_bar(self._current_decision())
        self.log.info("review save complete")
        self.set_timer(SAVE_COMPLETE_DELAY_SECONDS, self.exit)

    def action_quit_no_save(self) -> None:
        """Quit without saving (should prompt for confirmation)."""
        self.exit()

    def action_queue_1(self) -> None:
        """Switch to uncertain queue."""
        self._switch_queue(QUEUE_UNCERTAIN)

    def action_queue_2(self) -> None:
        """Switch to rejected queue."""
        self._switch_queue(QUEUE_REJECTED)

    def action_queue_3(self) -> None:
        """Switch to duplicates queue."""
        self._switch_queue(QUEUE_DUPLICATES)

    def action_queue_4(self) -> None:
        """Switch to keepers queue."""
        self._switch_queue(QUEUE_KEEPERS)

    def action_queue_5(self) -> None:
        """Switch to curated-selected queue."""
        self._switch_queue(QUEUE_SELECTED)

    def action_cycle_queue(self) -> None:
        """Switch to the next non-empty queue in QUEUE_CYCLE_ORDER."""
        current_pos = (
            QUEUE_CYCLE_ORDER.index(self._queue_index)
            if self._queue_index in QUEUE_CYCLE_ORDER
            else -1
        )
        for offset in range(1, len(QUEUE_CYCLE_ORDER) + 1):
            candidate = QUEUE_CYCLE_ORDER[(current_pos + offset) % len(QUEUE_CYCLE_ORDER)]
            raw = _filter_queue(self._session.decisions, QUEUE_LABELS[candidate])
            if raw:
                self._switch_queue(candidate)
                return

    def action_bulk_keep(self) -> None:
        """Keep all remaining photos in the current queue."""
        self._bulk_apply("keeper")

    def action_bulk_reject(self) -> None:
        """Reject all remaining photos in the current queue."""
        self._bulk_apply("rejected")

    def action_reject_cluster(self) -> None:
        """Reject every member of the current photo's cluster."""
        decision = self._current_decision()
        if decision is None or decision.stage1 is None or decision.stage1.burst is None:
            self._stage_move("rejected")
            return
        group_id = decision.stage1.burst.group_id
        cluster = _find_burst_decisions(self._session, group_id)
        for member in cluster:
            self._apply_cluster_reject(member)
        self._rebuild_queue()
        self._display_current()

    def _apply_cluster_reject(self, decision: PhotoDecision) -> None:
        """Apply rejected override to one cluster member and trigger retrain."""
        path_str = str(decision.photo.path)
        self._overrides[path_str] = "rejected"
        for i, d in enumerate(self._session.decisions):
            if str(d.photo.path) == path_str:
                self._session.decisions[i] = _apply_override(decision, "rejected")
                break
        hook_ctx = OverrideHookCtx(decision=decision, new_label="rejected", origin=OVERRIDE_ORIGIN_BULK)
        entry = self._emit_override_log(hook_ctx)
        if entry is not None:
            self._trigger_retrain(entry)

    def _bulk_apply(self, label: DecisionLabel) -> None:
        """Apply a label to all photos in the current queue."""
        for idx in self._queue_indices:
            decision = self._session.decisions[idx]
            path_str = str(decision.photo.path)
            self._overrides[path_str] = label
            self._session.decisions[idx] = _apply_override(decision, label)
            self._emit_override_log(OverrideHookCtx(decision=decision, new_label=label, origin=OVERRIDE_ORIGIN_BULK))
        self._rebuild_queue()
        self._display_current()

    def action_auto_accept(self) -> None:
        """Auto-accept VLM recommendations with confidence above threshold."""
        for i, d in enumerate(self._session.decisions):
            if d.decision != "uncertain" or d.stage3 is None:
                continue
            if d.stage3.confidence <= TUI_AUTOSAVE_BATCH_CONFIDENCE:
                continue
            label = self._vlm_label(d)
            path_str = str(d.photo.path)
            self._overrides[path_str] = label
            self._session.decisions[i] = _apply_override(d, label)
            self._emit_override_log(OverrideHookCtx(decision=d, new_label=label, origin=OVERRIDE_ORIGIN_AUTO))
        self._rebuild_queue()
        self._display_current()

    def _vlm_label(self, decision: PhotoDecision) -> DecisionLabel:
        """Determine label from VLM recommendation."""
        if decision.stage3 and decision.stage3.is_keeper:
            return "keeper"
        return "rejected"

    def _build_explain_request(self, decision: PhotoDecision) -> ExplainRequest:
        """Build an ExplainRequest from a current photo decision."""
        composite = decision.stage2.composite if decision.stage2 is not None else None
        return ExplainRequest(
            image_path=self._resolve_decision_path(decision),
            stage1_result=decision.stage1,
            stage2_composite=composite,
            stage3_result=decision.stage3,
            model=self._config.model,
        )

    def action_explain(self) -> None:
        """Show the VLM explanation in a docked panel below the photo."""
        decision = self._current_decision()
        if decision is None:
            return
        request = self._build_explain_request(decision)
        panel = self.query_one(ExplainPanel)
        panel.show_loading(str(request.image_path))
        self.run_worker(
            lambda: self._fetch_explain_panel(request),
            thread=True,
            exclusive=True,
            group="explain",
        )

    def _fetch_explain_panel(self, request: ExplainRequest) -> None:
        """Fetch explain result off-thread and update the docked panel."""
        result = fetch_explanation_result(request)
        self.call_from_thread(self._show_explain_result, result)

    def _show_explain_result(self, result: ExplainResult) -> None:
        """Render one explain result into the docked panel."""
        decision = self._current_decision()
        if decision is None or str(result.photo_path) != str(self._resolve_decision_path(decision)):
            return
        self.query_one(ExplainPanel).show_result(result)
