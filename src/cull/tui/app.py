"""Textual TUI application for interactive photo culling review."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from pydantic import BaseModel, Field
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from cull.config import (
    CullConfig,
    TASTE_PROFILE_PATH,
    TUI_AUTOSAVE_BATCH_CONFIDENCE,
    TUI_AUTOSAVE_INTERVAL_SECONDS,
)
from cull.models import DecisionLabel, ExplainRequest, ExplainResult, OverrideEntry, PhotoDecision
from cull.override_log import log_override, remove_overrides
from cull.pipeline import SessionResult
from cull._pipeline.decision_assembly import _build_summary
from cull.report import write_report
from cull.router import execute_moves
from cull.taste_trainer import TasteTrainerInput, maybe_retrain, unwind_retrain_counter
from cull.tui.compare_view import CompareSource, CompareView, StackFrame, frame_sharpness
from cull.tui.explain_modal import ExplainPanel, fetch_explanation_result
from cull.tui.face_panel import FacePanel
from cull.tui.faces import FaceReport, FaceRequest, FaceService
from cull.tui.filmstrip import FILMSTRIP_CENTRE, FILMSTRIP_SLOTS, Filmstrip, StripItem
from cull.tui.images import ImageCells, ImageRequest, ImageService, PrefetchPlan
from cull.tui.ledger import ChangeBatch, LabelChange, ReviewLedger, UndoEntry, _apply_override
from cull.tui.paths import PathCache, resolve_all
from cull.tui.photo_view import PhotoView
from cull.tui.previews import PREVIEW_CACHE_MAX_BYTES, prune_cache
from cull.tui.score_panel import ScorePanel
from cull.tui.screens import ConfirmQuitScreen, HelpScreen
from cull.tui.timing import LatencyTimer, debug_log
from cull.tui.why_line import WhyContext, build_why_line

__all__ = ["AppInput", "CullApp", "_apply_override"]

OVERRIDE_ORIGIN_SINGLE: str = "single"
OVERRIDE_ORIGIN_BURST: str = "burst"
OVERRIDE_ORIGIN_BULK: str = "bulk"
OVERRIDE_ORIGIN_AUTO: str = "auto_accept"

logger = logging.getLogger(__name__)

STATE_FILENAME: str = ".cull_tui_state.json"
STATE_BACKUP_SUFFIX: str = ".bak"
STATE_TEMP_SUFFIX: str = ".tmp"
SAVE_IN_PROGRESS_MESSAGE: str = "Saving review changes..."
SAVE_COMPLETE_MESSAGE: str = "Save complete. Exiting..."
SAVE_FAILED_PREFIX: str = "Save failed: "
SAVE_COMPLETE_DELAY_SECONDS: float = 0.25
MIN_TERMINAL_COLS: int = 40
MIN_TERMINAL_ROWS: int = 12
TOO_SMALL_BANNER_ID: str = "too-small-banner"
LOOKAHEAD_UPLOADS: int = 2
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


class TuiState(BaseModel):
    """Serializable TUI state for auto-save and crash recovery."""

    overrides: dict[str, DecisionLabel] = Field(default_factory=dict)
    current_index: int = 0
    current_queue: int = QUEUE_UNCERTAIN
    reviewed: list[str] = Field(default_factory=list)


class AppInput(BaseModel):
    """Input bundle for constructing CullApp."""

    session: SessionResult
    config: CullConfig


def _state_path(session: SessionResult) -> Path:
    """Return the path to the TUI state file."""
    return Path(session.source_path) / STATE_FILENAME


def _state_backup_path(path: Path) -> Path:
    """Return the sibling .bak path for a state file."""
    return path.with_name(path.name + STATE_BACKUP_SUFFIX)


def _parse_state_file(path: Path) -> TuiState | None:
    """Parse a state file into TuiState, or None if missing/corrupt."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TuiState.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        return None


def _rotate_state_backup(path: Path) -> None:
    """Copy a currently-valid state file to its .bak sibling before overwrite."""
    if _parse_state_file(path) is not None:
        shutil.copy2(path, _state_backup_path(path))


def _save_state(state: TuiState, path: Path) -> None:
    """Write TUI state to disk atomically, rotating any valid prior state to .bak."""
    tmp_path = path.with_name(path.name + STATE_TEMP_SUFFIX)
    tmp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    _rotate_state_backup(path)
    os.replace(tmp_path, path)
    logger.info("TUI state saved to %s", path)


def _load_state(path: Path) -> TuiState | None:
    """Load TUI state from disk, recovering from .bak if the primary is corrupt."""
    state = _parse_state_file(path)
    if state is not None or not path.exists():
        return state
    backup_state = _parse_state_file(_state_backup_path(path))
    if backup_state is not None:
        logger.warning("Corrupt TUI state file %s; recovered from backup", path)
        return backup_state
    logger.warning("Corrupt TUI state file %s and no valid backup; starting fresh", path)
    return None


def _filter_queue(decisions: list[PhotoDecision], label: DecisionLabel) -> list[int]:
    """Return indices of decisions matching the given label."""
    return [i for i, d in enumerate(decisions) if d.decision == label]


def _taste_uncertainty(decision: PhotoDecision) -> float:
    """Return |p - 0.5| (0 = most uncertain), or infinity when there is no taste score."""
    if decision.stage2 is None or decision.stage2.taste is None:
        return math.inf
    return abs(decision.stage2.taste.probability - 0.5)


def _sort_by_uncertainty(decisions: list[PhotoDecision], indices: list[int]) -> list[int]:
    """Sort indices most-uncertain first (p nearest 0.5) when any decision has a taste score.

    The sort is stable, so untrained taste (every p = 0.5) keeps capture order.
    """
    has_taste = any(decisions[i].stage2 and decisions[i].stage2.taste for i in indices)
    if not has_taste:
        return indices
    return sorted(indices, key=lambda i: _taste_uncertainty(decisions[i]))


class InfoBarContext(BaseModel):
    """Context for building the info bar display text."""

    decision: PhotoDecision
    position: int
    total: int
    queue_label: str
    queue_counts: dict[str, int]
    reviewed: int = 0


def _build_info_text(ctx: InfoBarContext) -> str:
    """Build the info bar: file, position, queue, per-queue counts, reviewed progress."""
    reviewed_pct = int(ctx.reviewed / ctx.total * 100) if ctx.total > 0 else 0
    queue_badges = "  ".join(
        f"[{QUEUE_HOTKEY[label]}] {label}:{count}"
        for label, count in ctx.queue_counts.items()
    )
    return (
        f"{ctx.decision.photo.filename}   "
        f"{ctx.position + 1}/{ctx.total}   "
        f"reviewed {ctx.reviewed}/{ctx.total} ({reviewed_pct}%)   "
        f"│ queue: {ctx.queue_label}   {queue_badges}"
    )


def _too_small_message(cols: int, rows: int) -> str:
    """Build the placeholder message shown when the terminal is below the minimum size."""
    return (
        f"Terminal too small ({cols}x{rows}).\n"
        f"Resize to at least {MIN_TERMINAL_COLS}x{MIN_TERMINAL_ROWS}."
    )


def _find_burst_decisions(session: SessionResult, group_id: int) -> list[PhotoDecision]:
    """Find all decisions belonging to a burst group."""
    return [
        d for d in session.decisions
        if d.stage1 and d.stage1.burst and d.stage1.burst.group_id == group_id
    ]


def _burst_group_id(decision: PhotoDecision) -> int | None:
    """Return the decision's burst group id, if it is in a burst."""
    if decision.stage1 is None or decision.stage1.burst is None:
        return None
    return decision.stage1.burst.group_id


def _burst_winner_names(decisions: list[PhotoDecision]) -> dict[int, str]:
    """Map burst group id to the filename Stage 1 picked as the sharpest frame."""
    winners: dict[int, str] = {}
    for decision in decisions:
        burst = decision.stage1.burst if decision.stage1 else None
        if burst is not None and burst.is_burst_winner:
            winners[burst.group_id] = decision.photo.filename
    return winners


def _nearest_first(centre: int, total: int) -> list[int]:
    """Return queue positions ordered by distance from ``centre`` (centre first)."""
    order = [centre] if 0 <= centre < total else []
    for distance in range(1, total):
        order.extend(p for p in (centre + distance, centre - distance) if 0 <= p < total)
    return order


class ReviewScreen(Screen):
    """Main review screen; stops painting images while another screen covers it."""

    def on_screen_suspend(self) -> None:
        """A modal or compare screen is on top: hide our image cells."""
        for cells in self.query(ImageCells):
            cells.set_suspended(True)

    def on_screen_resume(self) -> None:
        """We are on top again: paint image cells."""
        for cells in self.query(ImageCells):
            cells.set_suspended(False)


class CullApp(App):
    """Main Textual application for interactive photo review."""

    TITLE = "CULL -- Review"

    BINDINGS = [
        Binding("p,k", "keep", "Keep"),
        Binding("x,r", "reject", "Reject"),
        Binding("c", "curate", "Select"),
        Binding("d", "mark_duplicate", "Dup", show=False),
        Binding("u,ctrl+z", "undo", "Undo"),
        Binding("right,greater_than_sign,full_stop", "next_photo", "Next", show=False),
        Binding("left,less_than_sign,comma", "prev_photo", "Prev", show=False),
        Binding("home", "first_photo", "First", show=False),
        Binding("end", "last_photo", "Last", show=False),
        Binding("b", "compare", "Compare"),
        Binding("f", "toggle_filmstrip", "Film"),
        Binding("z", "toggle_faces", "Faces"),
        Binding("s", "toggle_scores", "Scores"),
        Binding("e", "explain", "Explain"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "save_quit", "Save+Quit"),
        Binding("Q", "quit_no_save", "Quit (no save)", show=False),
        Binding("tab", "cycle_queue", "Queue"),
        Binding("1", "queue_1", "Uncertain", show=False),
        Binding("2", "queue_2", "Rejected", show=False),
        Binding("3", "queue_3", "Duplicates", show=False),
        Binding("4", "queue_4", "Keepers", show=False),
        Binding("5", "queue_5", "Selected", show=False),
        Binding("K", "bulk_keep", "Keep All", show=False),
        Binding("R", "bulk_reject", "Reject All", show=False),
        Binding("X", "reject_cluster", "Reject Stack", show=False),
        Binding("A", "auto_accept", "Auto-accept VLM", show=False),
    ]

    CSS = f"""
    #info-bar {{
        height: 1;
        padding: 0 1;
        background: $boost;
    }}

    #why-line {{
        height: 1;
        padding: 0 1;
    }}

    #main-row {{
        height: 1fr;
    }}

    Filmstrip.hidden {{
        display: none;
    }}

    #{TOO_SMALL_BANNER_ID} {{
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        background: $panel;
        display: none;
    }}
    """

    def __init__(self, app_input: AppInput) -> None:
        super().__init__()
        self._session = app_input.session
        self._config = app_input.config
        self._ledger = ReviewLedger(self._session.decisions, str(self._session.source_path))
        self._paths = PathCache(self._config)
        self._burst_winners = _burst_winner_names(self._session.decisions)
        self.images = ImageService(self)
        self.faces = FaceService()
        self.latency = LatencyTimer()
        self._log_jobs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cull-override-log")
        self._queue_index: int = QUEUE_UNCERTAIN
        self._photo_index: int = 0
        self._queue_indices: list[int] = []
        self._save_in_progress: bool = False
        self._status_message: str | None = None
        self._normalize_decision_destinations()
        self._restore_state()

    @property
    def _overrides(self) -> dict[str, DecisionLabel]:
        """Return the user's label overrides keyed by source path."""
        return self._ledger.overrides

    def get_default_screen(self) -> Screen:
        """Use a screen that hides images while covered."""
        return ReviewScreen(id="_default")

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
        self._ledger.restore(state.overrides)
        self._ledger.reviewed.update(state.reviewed)
        self._photo_index = state.current_index
        self._queue_index = state.current_queue

    def compose(self) -> ComposeResult:
        """Build the main screen layout."""
        yield Header()
        yield Static("", id="info-bar")
        yield Static("", id="why-line")
        with Horizontal(id="main-row"):
            yield PhotoView()
            yield FacePanel()
        yield Filmstrip()
        yield ExplainPanel()
        yield ScorePanel()
        yield Static("", id=TOO_SMALL_BANNER_ID)
        yield Footer()

    def on_resize(self, event: events.Resize) -> None:
        """Show a placeholder when the terminal drops below the minimum usable size."""
        try:
            self._update_too_small_state(event.size.width, event.size.height)
        except NoMatches:
            self.log.debug("on_resize: DOM not ready yet")

    def _update_too_small_state(self, cols: int, rows: int) -> None:
        """Toggle the too-small placeholder based on current terminal dimensions."""
        is_too_small = cols < MIN_TERMINAL_COLS or rows < MIN_TERMINAL_ROWS
        self._set_too_small_visible(is_too_small)
        if is_too_small:
            banner = self.query_one(f"#{TOO_SMALL_BANNER_ID}", Static)
            banner.update(_too_small_message(cols, rows))

    def _set_too_small_visible(self, is_too_small: bool) -> None:
        """Swap every image-bearing and status widget for the too-small placeholder."""
        self.query_one(f"#{TOO_SMALL_BANNER_ID}", Static).display = is_too_small
        for selector in ("#main-row", "#info-bar", "#why-line", "PhotoView", "Filmstrip"):
            self.query_one(selector).display = not is_too_small

    def on_mount(self) -> None:
        """Start background services, then show the first photo."""
        self.images.start()
        self.faces.start()
        self.run_worker(self._warm_paths, thread=True, group="paths", exit_on_error=False)
        self.run_worker(self._prune_previews, thread=True, group="prune", exit_on_error=False)
        self._rebuild_queue()
        if not self._queue_indices:
            self._fallback_to_non_empty_queue()
        debug_log(f"on_mount: queue={self._queue_index} len={len(self._queue_indices)}")
        self.call_after_refresh(self._display_current)
        self._schedule_autosave()

    def on_unmount(self) -> None:
        """Free terminal images and let pending override-log writes finish."""
        self.images.stop()
        self.faces.stop()
        self._log_jobs.shutdown(wait=True)

    def _prune_previews(self) -> None:
        """Worker: keep the on-disk preview cache under its size budget."""
        try:
            removed = prune_cache(PREVIEW_CACHE_MAX_BYTES)
        except OSError as exc:
            logger.warning("preview cache prune failed: %s", exc)
            return
        debug_log(f"preview cache pruned {removed} file(s)")

    def _warm_paths(self) -> None:
        """Worker: resolve every photo path once, off the UI thread."""
        resolved = resolve_all(self._session.decisions, self._config)
        self.call_from_thread(self._on_paths_warmed, resolved)

    def _on_paths_warmed(self, resolved: dict[str, Path]) -> None:
        """Adopt background-resolved paths and widen the prefetch plan."""
        self._paths.merge(resolved)
        self._prefetch()

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
            reviewed=sorted(self._ledger.reviewed),
        )
        _save_state(state, _state_path(self._session))

    def _rebuild_queue(self) -> None:
        """Snapshot the current queue, most uncertain first.

        The snapshot stays fixed while reviewing: a photo you just decided
        keeps its place (with its new marker), so auto-advance and going
        back behave predictably, as in pro culling tools.
        """
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
        decision = self._current_decision()
        if decision is None:
            self._show_empty_queue()
            return
        source = self._paths.resolve(decision)
        self._sync_explain_panel(source)
        self.query_one(PhotoView).show(source)
        self._show_info(decision)
        self._update_filmstrip()
        self._update_faces(decision)
        self._prefetch()

    def _sync_explain_panel(self, source: Path) -> None:
        """Hide a stale explain panel when the current photo changes."""
        panel = self.query_one(ExplainPanel)
        if panel.photo_path is not None and panel.photo_path != str(source):
            panel.hide_panel()

    def _show_empty_queue(self) -> None:
        """Display empty queue message."""
        self.query_one(PhotoView).show(None)
        self.query_one("#why-line", Static).update("", layout=False)
        self._update_info_bar(None)

    def _update_info_bar(self, decision: PhotoDecision | None) -> None:
        """Render the info bar, preferring any active status message."""
        info_bar = self.query_one("#info-bar", Static)
        if self._status_message is not None:
            info_bar.update(self._status_message, layout=False)
            return
        if decision is None:
            info_bar.update("Queue is empty", layout=False)
            return
        info_bar.update(_build_info_text(self._info_context(decision)), layout=False)

    def _info_context(self, decision: PhotoDecision) -> InfoBarContext:
        """Gather position, per-queue counts, and reviewed progress for the info bar."""
        label_counts = self._ledger.label_counts()
        reviewed = sum(
            1 for i in self._queue_indices
            if str(self._session.decisions[i].photo.path) in self._ledger.reviewed
        )
        return InfoBarContext(
            decision=decision,
            position=self._photo_index,
            total=len(self._queue_indices),
            queue_label=QUEUE_LABELS[self._queue_index],
            queue_counts={QUEUE_LABELS[qi]: label_counts.get(QUEUE_LABELS[qi], 0) for qi in QUEUE_CYCLE_ORDER},
            reviewed=reviewed,
        )

    def _show_info(self, decision: PhotoDecision) -> None:
        """Update the info bar, the why line, and the score panel."""
        self._update_info_bar(decision)
        self.query_one("#why-line", Static).update(build_why_line(self._why_context(decision)), layout=False)
        self.query_one(ScorePanel).show_scores(decision)

    def _why_context(self, decision: PhotoDecision) -> WhyContext:
        """Build the why-line input: the AI's own label and the burst winner's name."""
        group_id = _burst_group_id(decision)
        return WhyContext(
            decision=decision,
            ai_label=self._ledger.ai_labels.get(str(decision.photo.path), decision.decision),
            burst_winner=self._burst_winners.get(group_id) if group_id is not None else None,
        )

    def _resolve_decision_path(self, decision: PhotoDecision) -> Path:
        """Return the current on-disk path for a decision (cached after first lookup)."""
        return self._paths.resolve(decision)

    def _queue_source(self, position: int) -> Path | None:
        """Return the cached path at a queue position, without touching the disk."""
        return self._paths.cached(self._session.decisions[self._queue_indices[position]])

    def _update_filmstrip(self) -> None:
        """Show the queue neighbours around the cursor in the filmstrip."""
        strip = self.query_one(Filmstrip)
        if strip.has_class("hidden"):
            return
        items: list[StripItem | None] = []
        for slot in range(FILMSTRIP_SLOTS):
            position = self._photo_index + slot - FILMSTRIP_CENTRE
            items.append(self._strip_item(position) if 0 <= position < len(self._queue_indices) else None)
        strip.update_items(items)

    def _strip_item(self, position: int) -> StripItem:
        """Build one filmstrip slot for a queue position."""
        decision = self._session.decisions[self._queue_indices[position]]
        return StripItem(
            source=self._paths.resolve(decision),
            filename=decision.photo.filename,
            label=decision.decision,
            is_current=position == self._photo_index,
        )

    def _prefetch(self) -> None:
        """Queue previews nearest-first; upload the next/previous photos ahead of time."""
        photo_view = self.query_one(PhotoView)
        box = photo_view.cell_box()
        if box.cols < 2 or box.rows < 2 or not self._queue_indices:
            return
        requests: list[ImageRequest] = []
        for position in _nearest_first(self._photo_index, len(self._queue_indices)):
            source = self._queue_source(position)
            if source is not None:
                requests.append(ImageRequest(source=source, box=box))
        thumbs = self._thumbnail_requests()
        near = requests[:1 + 2 * LOOKAHEAD_UPLOADS]
        self.images.prefetch(PrefetchPlan(
            urgent=near + thumbs,
            upload_ahead=near[1:] + thumbs,
            background=requests[len(near):],
        ))

    def _thumbnail_requests(self) -> list[ImageRequest]:
        """Return requests for the filmstrip thumbnails currently in view."""
        strip = self.query_one(Filmstrip)
        if strip.has_class("hidden"):
            return []
        return [
            ImageRequest(source=cells.source, box=cells.cell_box())
            for cells in strip.thumbnail_boxes()
            if cells.source is not None and cells.size.width > 0 and cells.size.height > 0
        ]

    def _update_faces(self, decision: PhotoDecision) -> None:
        """Show face close-ups for the current photo when the panel is open."""
        panel = self.query_one(FacePanel)
        if not panel.is_visible:
            return
        source = self._paths.resolve(decision)
        report = self.faces.request(FaceRequest(source=source, on_done=partial(self._on_face_report, source)))
        if report is None:
            panel.show_loading(decision)
            return
        panel.show_report(report, decision)

    def _on_face_report(self, source: Path, report: FaceReport) -> None:
        """Show a face report if its photo is still the current one."""
        decision = self._current_decision()
        if decision is None or self._paths.resolve(decision) != source:
            return
        self.query_one(FacePanel).show_report(report, decision)

    def _trigger_retrain(self, entry: OverrideEntry) -> None:
        """Bump the retrain counter; maybe_retrain reads the full override log to fit."""
        trainer_ctx = TasteTrainerInput(overrides=[entry], profile_path=TASTE_PROFILE_PATH)
        try:
            maybe_retrain(trainer_ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("taste retrain failed unexpectedly (%s): %s", type(exc).__name__, exc)

    def _write_log_batch(self, entry: UndoEntry) -> None:
        """Log worker: append override entries, bumping retrain when the action counts."""
        for logged in entry.logged:
            log_override(logged)
        if entry.retrain_bumps:
            for logged in entry.logged:
                self._trigger_retrain(logged)

    def _revert_log_batch(self, entry: UndoEntry) -> None:
        """Log worker: remove undone entries and take back their retrain bumps."""
        try:
            remove_overrides(entry.logged)
        except OSError as exc:
            logger.warning("override log undo failed: %s", exc)
        if entry.retrain_bumps:
            unwind_retrain_counter(TASTE_PROFILE_PATH, entry.retrain_bumps)

    def flush_log_jobs(self) -> None:
        """Block until queued override-log writes finish (tests and save)."""
        self._log_jobs.submit(lambda: None).result()

    def _apply_batch(self, batch: ChangeBatch) -> None:
        """Apply label changes as one undoable step and log them off the UI thread."""
        entry = self._ledger.apply(batch)
        if entry is not None and entry.logged:
            self._log_jobs.submit(self._write_log_batch, entry)

    def _decide(self, label: DecisionLabel) -> None:
        """Label the current photo, then advance to the next one in the queue."""
        if not self._queue_indices:
            return
        self.latency.start()
        change = LabelChange(index=self._queue_indices[self._photo_index], label=label)
        self._apply_batch(ChangeBatch(changes=[change], origin=OVERRIDE_ORIGIN_SINGLE, bumps_retrain=True))
        if self._photo_index < len(self._queue_indices) - 1:
            self._photo_index += 1
        self._display_current()

    def _switch_queue(self, queue_index: int) -> None:
        """Switch to a different review queue."""
        self._queue_index = queue_index
        self._photo_index = 0
        self._rebuild_queue()
        self._display_current()

    def action_keep(self) -> None:
        """Mark current photo as keeper and advance."""
        self._decide("keeper")

    def action_reject(self) -> None:
        """Mark current photo as rejected and advance."""
        self._decide("rejected")

    def action_mark_duplicate(self) -> None:
        """Mark current photo as duplicate and advance."""
        self._decide("duplicate")

    def action_curate(self) -> None:
        """Promote current photo to the curated select queue and advance."""
        self._decide("select")

    def _go_to(self, position: int) -> None:
        """Move the cursor to a queue position if it is valid and different."""
        if not 0 <= position < len(self._queue_indices) or position == self._photo_index:
            return
        self.latency.start()
        self._photo_index = position
        self._display_current()

    def action_next_photo(self) -> None:
        """Navigate to the next photo in the queue."""
        self._go_to(self._photo_index + 1)

    def action_prev_photo(self) -> None:
        """Navigate to the previous photo in the queue."""
        self._go_to(self._photo_index - 1)

    def action_first_photo(self) -> None:
        """Navigate to the first photo in the queue."""
        self._go_to(0)

    def action_last_photo(self) -> None:
        """Navigate to the last photo in the queue."""
        self._go_to(len(self._queue_indices) - 1)

    def action_undo(self) -> None:
        """Undo the last action (single, bulk, stack or cluster) and return to its photo."""
        entry = self._ledger.undo()
        if entry is None:
            self.notify("Nothing to undo", timeout=2)
            return
        if entry.logged:
            self._log_jobs.submit(self._revert_log_batch, entry)
        self._move_cursor_to_undone(entry)
        self._display_current()

    def _move_cursor_to_undone(self, entry: UndoEntry) -> None:
        """Put the cursor on the first photo the undone action touched, if in this queue."""
        paths = [str(p.decision.photo.path) for p in entry.priors] + entry.reviewed_only
        if not paths:
            return
        for position, index in enumerate(self._queue_indices):
            if str(self._session.decisions[index].photo.path) == paths[0]:
                self._photo_index = position
                return

    def action_toggle_scores(self) -> None:
        """Toggle the score detail panel."""
        self.query_one(ScorePanel).toggle_visible()

    def action_toggle_filmstrip(self) -> None:
        """Show or hide the filmstrip."""
        strip = self.query_one(Filmstrip)
        strip.toggle_class("hidden")
        if not strip.has_class("hidden"):
            self.call_after_refresh(self._update_filmstrip)

    def action_toggle_faces(self) -> None:
        """Show or hide the face close-up panel."""
        self.query_one(FacePanel).toggle()
        decision = self._current_decision()
        if decision is not None:
            self._update_faces(decision)

    def action_help(self) -> None:
        """Show the key reference."""
        self.push_screen(HelpScreen())

    def action_compare(self) -> None:
        """Open compare mode for the current photo's burst stack."""
        decision = self._current_decision()
        group_id = _burst_group_id(decision) if decision is not None else None
        if decision is None or group_id is None:
            self.notify("No burst stack for this photo", timeout=2)
            return
        source = CompareSource(
            group_id=group_id,
            load_frames=partial(self._stack_frames, group_id),
            pick=partial(self._pick_in_stack, group_id),
            undo=self.action_undo,
            start_path=self._paths.resolve(decision),
        )
        self.push_screen(CompareView(source), self._on_compare_closed)

    def _stack_frames(self, group_id: int) -> list[StackFrame]:
        """Return the stack's frames with current labels, in capture order."""
        frames: list[StackFrame] = []
        for index, decision in enumerate(self._session.decisions):
            if _burst_group_id(decision) != group_id:
                continue
            burst = decision.stage1.burst if decision.stage1 else None
            frames.append(StackFrame(
                index=index,
                source=self._paths.resolve(decision),
                filename=decision.photo.filename,
                label=decision.decision,
                sharpness=frame_sharpness(decision),
                is_ai_pick=bool(burst and burst.is_burst_winner),
            ))
        return frames

    def _pick_in_stack(self, group_id: int, picked: StackFrame) -> None:
        """Keep the picked frame and reject the rest of its stack, as one undo step."""
        winner_label: DecisionLabel = "select" if picked.label == "select" else "keeper"
        changes = [
            LabelChange(index=frame.index, label=winner_label if frame.index == picked.index else "rejected")
            for frame in self._stack_frames(group_id)
        ]
        self._apply_batch(ChangeBatch(changes=changes, origin=OVERRIDE_ORIGIN_BURST))

    def _on_compare_closed(self, _result: object) -> None:
        """Refresh the main view after compare mode."""
        self._display_current()

    def _pin_resolved_destinations(self) -> None:
        """Record recovered on-disk locations so moves start from where files really are."""
        for decision in self._session.decisions:
            cached = self._paths.cached(decision)
            if cached is not None and cached != decision.photo.path and cached.exists():
                decision.destination = cached

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
        self._pin_resolved_destinations()
        move_error = self._attempt_execute_moves()
        if move_error is not None:
            self._report_save_failure(f"{SAVE_FAILED_PREFIX}{move_error}")
            return
        report_error = self._attempt_write_report()
        if report_error is not None:
            self._report_save_failure(
                f"Photos moved; report write failed: {report_error}"
            )
            return
        self._clear_state_file()
        self._complete_save()

    def _attempt_execute_moves(self) -> str | None:
        """Run execute_moves; return an error message on failure, else None."""
        try:
            execute_moves(self._session.decisions, self._config)
        except OSError as exc:
            self.log.warning("review save failed during move: %s", exc)
            return str(exc)
        return None

    def _attempt_write_report(self) -> str | None:
        """Build the summary and write the report; return an error message on failure."""
        try:
            self._session.summary = _build_summary(self._session.decisions)
            write_report(self._session, overwrite=True)
        except Exception as exc:  # noqa: BLE001 - moves already happened, must not crash
            self.log.warning("review report write failed: %s", exc)
            return str(exc)
        return None

    def _clear_state_file(self) -> None:
        """Remove the autosave state file after a successful commit."""
        state_path = _state_path(self._session)
        if state_path.exists():
            state_path.unlink()

    def _report_save_failure(self, message: str) -> None:
        """Show a save-failure banner and clear the in-progress flag."""
        self._save_in_progress = False
        self._status_message = message
        self._update_info_bar(self._current_decision())

    def _complete_save(self) -> None:
        """Show the save-complete banner and schedule app exit."""
        self._status_message = SAVE_COMPLETE_MESSAGE
        self._update_info_bar(self._current_decision())
        self.log.info("review save complete")
        self.set_timer(SAVE_COMPLETE_DELAY_SECONDS, self.exit)

    def action_quit_no_save(self) -> None:
        """Ask before quitting without saving."""
        self.push_screen(ConfirmQuitScreen(len(self._overrides)), self._on_quit_answer)

    def _on_quit_answer(self, should_quit: bool | None) -> None:
        """Quit only on an explicit yes."""
        if should_quit:
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
        """Keep all photos in the current queue (one undo step)."""
        self._bulk_apply("keeper")

    def action_bulk_reject(self) -> None:
        """Reject all photos in the current queue (one undo step)."""
        self._bulk_apply("rejected")

    def action_reject_cluster(self) -> None:
        """Reject every member of the current photo's burst stack (one undo step)."""
        decision = self._current_decision()
        if decision is None:
            return
        group_id = _burst_group_id(decision)
        if group_id is None:
            self._decide("rejected")
            return
        changes = [LabelChange(index=frame.index, label="rejected") for frame in self._stack_frames(group_id)]
        self._apply_batch(ChangeBatch(changes=changes, origin=OVERRIDE_ORIGIN_BULK, bumps_retrain=True))
        self._display_current()

    def _bulk_apply(self, label: DecisionLabel) -> None:
        """Apply a label to all photos in the current queue."""
        changes = [LabelChange(index=idx, label=label) for idx in self._queue_indices]
        self._apply_batch(ChangeBatch(changes=changes, origin=OVERRIDE_ORIGIN_BULK))
        self._display_current()

    def action_auto_accept(self) -> None:
        """Auto-accept VLM recommendations with confidence above threshold."""
        changes = [
            LabelChange(index=i, label=self._vlm_label(d))
            for i, d in enumerate(self._session.decisions)
            if d.decision == "uncertain" and d.stage3 is not None
            and d.stage3.confidence > TUI_AUTOSAVE_BATCH_CONFIDENCE
        ]
        self._apply_batch(ChangeBatch(changes=changes, origin=OVERRIDE_ORIGIN_AUTO))
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
