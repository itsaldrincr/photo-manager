"""Stage 4 curation orchestrator — clusters keepers and picks the best."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from cull.config import (
    CLUSTER_THRESHOLD,
    CullConfig,
)
from cull.dashboard import _S4UpdateInput
from cull.models import (
    CurationResult,
    CuratorSelection,
    Stage1Result,
)
from cull.stage2.portrait import PortraitResult
from cull.stage4.cluster import ClusterInput, cluster_by_similarity
from cull.stage4.diversity import MmrContext, MmrInput, select as diversity_select
from cull.stage4.narrative_flow import NarrativeFlowInput, check as narrative_check
from cull.stage4.peak_action import PeakActionInput, pick_winner as pick_action_winner
from cull.stage4.peak_portrait import PeakPortraitInput, pick_winner as pick_portrait_winner
from cull.stage4.tournament import (
    TournamentContext,
    TournamentInput,
    run as run_tournament,
)

logger = logging.getLogger(__name__)

DIVERSITY_OVERSAMPLE: int = 2
MIN_CANDIDATES_FOR_TOURNAMENT: int = 2
PORTRAIT_FACE_VOTE_RATIO: float = 0.5
MOTION_VOTE_RATIO: float = 0.5


class CuratorInput(BaseModel):
    """Input bundle for the Stage 4 curation orchestrator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    keepers: list[Path]
    encodings: dict[str, Any]
    composite_scores: dict[str, float]
    config: CullConfig
    dashboard: Any = None
    s1_results: dict[str, Stage1Result] = {}
    portraits: dict[str, PortraitResult] = {}
    search_embeddings: Any = None  # np.ndarray | None — CLIP rows for diversity
    search_path_to_row: dict[str, int] | None = None
    vlm_session: Any = None  # VlmSession — Any avoids import cycle


class _ClusterClassifyInput(BaseModel):
    """Bundle for cluster peak-type classification."""

    cluster: list[Path]
    s1_results: dict[str, Stage1Result]
    portraits: dict[str, PortraitResult]


class _WinnerPickInput(BaseModel):
    """Input bundle for _pick_cluster_winner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cluster: list[Path]
    scores: dict[str, float]
    config: CullConfig
    s1_results: dict[str, Stage1Result] = {}
    portraits: dict[str, PortraitResult] = {}
    reporter: _S4ProgressReporter | None = None


class _SelectionBuildInput(BaseModel):
    """Input bundle for _build_selections."""

    winners: list[Path]
    cluster_lookup: dict[str, tuple[int, int]]
    scores: dict[str, float]
    reasons: dict[str, str | None]


class _SelectWinnersInput(BaseModel):
    """Input bundle for _select_winners."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    clusters: list[list[Path]]
    scores: dict[str, float]
    config: CullConfig
    s1_results: dict[str, Stage1Result] = {}
    portraits: dict[str, PortraitResult] = {}
    reporter: _S4ProgressReporter | None = None


class _S4ProgressReporter(BaseModel):
    """Progress reporter for Stage 4 curator — sends tallies to dashboard."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dashboard: Any = None
    clusters_found: int = 0
    selected: int = 0
    vlm_tiebreaks: int = 0

    def mark_clusters(self, count: int) -> None:
        """Record the cluster count and update dashboard."""
        if self.dashboard is None:
            return
        self.clusters_found = count
        self.dashboard.update_stage4(self._build_update())

    def mark_winner(self, cluster_id: int) -> None:
        """Record a winner selection and update dashboard."""
        if self.dashboard is None:
            return
        self.selected += 1
        update = self._build_update()
        update.current_cluster_id = cluster_id
        self.dashboard.update_stage4(update)

    def mark_tournament_match(self) -> None:
        """Record a tournament pairwise match and update dashboard."""
        if self.dashboard is None:
            return
        self.vlm_tiebreaks += 1
        self.dashboard.update_stage4(self._build_update())

    def _build_update(self) -> _S4UpdateInput:
        """Return a fresh update payload from current tallies."""
        return _S4UpdateInput(
            clusters_found=self.clusters_found,
            selected=self.selected,
            vlm_tiebreaks=self.vlm_tiebreaks,
        )


class _ClusterRunInput(BaseModel):
    """Input bundle for _run_clustering."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    curator_input: CuratorInput
    target: int
    threshold: float
    reporter: _S4ProgressReporter | None = None


class _DiversityRunInput(BaseModel):
    """Input bundle for diversity selection over winners."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    winners: list[Path]
    curator_input: CuratorInput
    target: int


class _TrimInput(BaseModel):
    """Input bundle for trimming candidates by composite score."""

    candidates: list[Path]
    scores: dict[str, float]
    limit: int


class _TournamentRunInput(BaseModel):
    """Input bundle for tournament run over diversified winners."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidates: list[Path]
    curator_input: CuratorInput
    reporter: _S4ProgressReporter | None = None


class _NarrativeRunInput(BaseModel):
    """Input bundle for narrative flow regularisation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    selections: list[CuratorSelection]
    curator_input: CuratorInput


class _FinalBuildInput(BaseModel):
    """Input bundle for assembling CurationResult from all stage outputs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    clusters: list[list[Path]]
    winners: list[Path]
    reasons: dict[str, str | None]
    curator_input: CuratorInput
    target: int
    threshold: float
    elapsed: float
    tournament_matches: int
    narrative_score: float


def _classify_cluster(classify_in: _ClusterClassifyInput) -> str:
    """Return peak type (portrait/action/composite) for a cluster."""
    portrait_votes = sum(
        1 for p in classify_in.cluster
        if classify_in.portraits.get(str(p), PortraitResult()).has_face
    )
    motion_votes = sum(
        1 for p in classify_in.cluster
        if _is_motion_photo(p, classify_in.s1_results)
    )
    total = len(classify_in.cluster)
    if total == 0:
        return "composite"
    if portrait_votes / total >= PORTRAIT_FACE_VOTE_RATIO:
        return "portrait"
    if motion_votes / total >= MOTION_VOTE_RATIO:
        return "action"
    return "composite"


def _is_motion_photo(path: Path, s1_results: dict[str, Stage1Result]) -> bool:
    """Return True if Stage 1 marked this photo as motion-blurred."""
    s1 = s1_results.get(str(path))
    if s1 is None:
        return False
    return s1.blur.is_motion_blur


def _pick_composite_winner(
    cluster: list[Path], scores: dict[str, float]
) -> Path:
    """Return the cluster member with the highest composite score."""
    return max(cluster, key=lambda p: scores.get(str(p), 0.0))


def _pick_portrait_winner(winner_in: _WinnerPickInput) -> Path:
    """Route to peak_portrait.pick_winner for a portrait cluster."""
    portrait_in = PeakPortraitInput(
        burst_members=winner_in.cluster, config=winner_in.config
    )
    winner, _peak = pick_portrait_winner(portrait_in)
    return winner


def _pick_action_winner(winner_in: _WinnerPickInput) -> Path:
    """Route to peak_action.pick_winner for an action cluster."""
    action_in = PeakActionInput(
        burst_members=winner_in.cluster, config=winner_in.config
    )
    winner, _peak = pick_action_winner(action_in)
    return winner


def _dispatch_by_peak_type(
    winner_in: _WinnerPickInput, peak_type: str
) -> Path:
    """Route winner selection to the correct peak picker by type."""
    if peak_type == "portrait":
        return _pick_portrait_winner(winner_in)
    if peak_type == "action":
        return _pick_action_winner(winner_in)
    return _pick_composite_winner(winner_in.cluster, winner_in.scores)


def _pick_cluster_winner(
    winner_in: _WinnerPickInput,
) -> tuple[Path, str | None]:
    """Classify cluster and route to the correct peak picker."""
    cluster = winner_in.cluster
    if len(cluster) == 1:
        return cluster[0], None
    classify_in = _ClusterClassifyInput(
        cluster=cluster, s1_results=winner_in.s1_results, portraits=winner_in.portraits
    )
    peak_type = _classify_cluster(classify_in)
    try:
        winner = _dispatch_by_peak_type(winner_in, peak_type)
    except Exception as exc:
        logger.warning("Peak picker failed (%s); composite fallback: %s", peak_type, exc)
        winner = _pick_composite_winner(cluster, winner_in.scores)
        peak_type = "composite_fallback"
    return winner, peak_type


def _build_selections(build_in: _SelectionBuildInput) -> list[CuratorSelection]:
    """Map winner paths to CuratorSelection records using cluster metadata."""
    selections: list[CuratorSelection] = []
    for path in build_in.winners:
        key = str(path)
        cluster_id, cluster_size = build_in.cluster_lookup.get(key, (0, 1))
        selections.append(
            CuratorSelection(
                path=path,
                cluster_id=cluster_id,
                cluster_size=cluster_size,
                composite=build_in.scores.get(key, 0.0),
                is_vlm_winner=False,
                reason=build_in.reasons.get(key),
            )
        )
    return selections


def _build_cluster_lookup(
    clusters: list[list[Path]],
) -> dict[str, tuple[int, int]]:
    """Build a path-to-(cluster_id, cluster_size) index from cluster list."""
    lookup: dict[str, tuple[int, int]] = {}
    for cluster_id, cluster in enumerate(clusters):
        for path in cluster:
            lookup[str(path)] = (cluster_id, len(cluster))
    return lookup


class _ProcessClusterInput(BaseModel):
    """Input bundle for _process_cluster."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cluster: list[Path]
    cluster_id: int
    scores: dict[str, float]
    config: CullConfig
    s1_results: dict[str, Stage1Result] = {}
    portraits: dict[str, PortraitResult] = {}
    reporter: _S4ProgressReporter | None = None


class _ClusterWinnerResult(BaseModel):
    """Result bundle from _process_cluster."""

    winner: Path
    reason: str | None = None


def _process_cluster(process_in: _ProcessClusterInput) -> _ClusterWinnerResult:
    """Process a single cluster: pick winner, record selection, return result."""
    pick_in = _WinnerPickInput(
        cluster=process_in.cluster,
        scores=process_in.scores,
        config=process_in.config,
        s1_results=process_in.s1_results,
        portraits=process_in.portraits,
        reporter=process_in.reporter,
    )
    winner, reason = _pick_cluster_winner(pick_in)
    if process_in.reporter is not None:
        process_in.reporter.mark_winner(process_in.cluster_id)
    return _ClusterWinnerResult(winner=winner, reason=reason)


def _select_winners(
    select_in: _SelectWinnersInput,
) -> tuple[list[Path], dict[str, str | None]]:
    """Pick one winner per cluster; return winners and reasons."""
    reasons: dict[str, str | None] = {}
    winners: list[Path] = []
    for cluster_id, cluster in enumerate(select_in.clusters):
        process_in = _ProcessClusterInput(
            cluster=cluster, cluster_id=cluster_id, scores=select_in.scores,
            config=select_in.config, s1_results=select_in.s1_results,
            portraits=select_in.portraits, reporter=select_in.reporter,
        )
        result = _process_cluster(process_in)
        winners.append(result.winner)
        reasons[str(result.winner)] = result.reason
    return winners, reasons


def _build_mmr_context(curator_input: CuratorInput, target: int) -> MmrContext | None:
    """Return an MmrContext from search-cache embeddings, or None on miss."""
    embeddings = curator_input.search_embeddings
    path_to_row = curator_input.search_path_to_row
    if embeddings is None or path_to_row is None:
        return None
    return MmrContext(
        embeddings=np.asarray(embeddings),
        path_to_row=path_to_row,
        target_count=target,
    )


def _trim_by_composite(trim_in: _TrimInput) -> list[Path]:
    """Return top-limit candidates by composite score (fallback when no MMR)."""
    return sorted(
        trim_in.candidates, key=lambda p: trim_in.scores.get(str(p), 0.0), reverse=True
    )[:trim_in.limit]


def _run_diversity(div_in: _DiversityRunInput) -> list[Path]:
    """Return the diversity-selected subset of winners via MMR."""
    oversample = max(div_in.target, DIVERSITY_OVERSAMPLE * div_in.target)
    trim_in = _TrimInput(
        candidates=div_in.winners,
        scores=div_in.curator_input.composite_scores,
        limit=oversample,
    )
    pool = _trim_by_composite(trim_in)
    context = _build_mmr_context(div_in.curator_input, div_in.target)
    if context is None:
        logger.info("Diversity skipped (no search cache); using top-%d by composite", div_in.target)
        return pool[: div_in.target]
    mmr_in = MmrInput(candidates=pool, scores=div_in.curator_input.composite_scores)
    return diversity_select(mmr_in, context)


def _tournament_match_budget(candidate_count: int) -> int:
    """Return the number of matches a full single-elim bracket will play."""
    if candidate_count < MIN_CANDIDATES_FOR_TOURNAMENT:
        return 0
    return candidate_count - 1


def _run_tournament(tour_in: _TournamentRunInput) -> tuple[list[Path], int]:
    """Run tournament over candidates; return ordered bracket and match count."""
    if len(tour_in.candidates) < MIN_CANDIDATES_FOR_TOURNAMENT:
        return tour_in.candidates, 0
    ctx = TournamentContext(
        s1_results=tour_in.curator_input.s1_results,
        composite_scores=tour_in.curator_input.composite_scores,
    )
    inp = TournamentInput(
        candidates=tour_in.candidates,
        config=tour_in.curator_input.config,
        session=tour_in.curator_input.vlm_session,
    )
    try:
        ordered = run_tournament(inp, ctx)
    except Exception as exc:
        logger.warning("Tournament failed; keeping diversity order: %s", exc)
        return tour_in.candidates, 0
    matches = _tournament_match_budget(len(tour_in.candidates))
    _report_tournament_matches(tour_in.reporter, matches)
    return ordered, matches


def _report_tournament_matches(
    reporter: _S4ProgressReporter | None, matches: int
) -> None:
    """Emit one dashboard update per virtual tournament match."""
    if reporter is None:
        return
    for _ in range(matches):
        reporter.mark_tournament_match()


def _run_narrative(
    narr_in: _NarrativeRunInput,
) -> tuple[list[CuratorSelection], float]:
    """Run narrative-flow regularisation; fall back to input on failure."""
    candidates = {str(p): p for p in narr_in.curator_input.keepers}
    flow_in = NarrativeFlowInput(
        selections=narr_in.selections, candidates=candidates
    )
    try:
        return narrative_check(flow_in)
    except Exception as exc:
        logger.warning("Narrative flow check failed: %s", exc)
        return narr_in.selections, 0.0


def _empty_result(threshold: float, target: int) -> CurationResult:
    """Return a CurationResult indicating no photos were selected."""
    return CurationResult(
        is_enabled=True,
        target_count=target,
        actual_count=0,
        cluster_count=0,
        vlm_tiebreakers=0,
        threshold_used=threshold,
        elapsed_seconds=0.0,
        selected=[],
    )


def _all_keeper_selections(curator_input: CuratorInput) -> list[CuratorSelection]:
    """Build CuratorSelection list for all keepers (no clustering)."""
    return _build_selections(
        _SelectionBuildInput(
            winners=curator_input.keepers,
            cluster_lookup={},
            scores=curator_input.composite_scores,
            reasons={},
        )
    )


def _make_all_keeper_result(
    curator_input: CuratorInput, threshold: float
) -> CurationResult:
    """Return all keepers as selections when keepers count is within target."""
    t0 = time.perf_counter()
    keepers = curator_input.keepers
    raw_target = curator_input.config.curate_target
    effective_target = raw_target if raw_target is not None else len(keepers)
    selections = _all_keeper_selections(curator_input)
    elapsed = time.perf_counter() - t0
    return CurationResult(
        is_enabled=True,
        target_count=effective_target,
        actual_count=len(selections),
        cluster_count=0,
        vlm_tiebreakers=0,
        threshold_used=threshold,
        elapsed_seconds=elapsed,
        selected=selections,
    )


def _build_clusters(run_in: _ClusterRunInput) -> list[list[Path]]:
    """Run similarity clustering and report cluster count to dashboard."""
    ci = run_in.curator_input
    cluster_in = ClusterInput(
        encodings=ci.encodings, paths=ci.keepers, threshold=run_in.threshold
    )
    clusters = cluster_by_similarity(cluster_in)
    if run_in.reporter is not None:
        run_in.reporter.mark_clusters(len(clusters))
    return clusters


def _pick_all_winners(
    clusters: list[list[Path]], run_in: _ClusterRunInput
) -> tuple[list[Path], dict[str, str | None]]:
    """Pick a winner for every cluster and return paths and reasons."""
    ci = run_in.curator_input
    select_in = _SelectWinnersInput(
        clusters=clusters, scores=ci.composite_scores, config=ci.config,
        s1_results=ci.s1_results, portraits=ci.portraits, reporter=run_in.reporter,
    )
    return _select_winners(select_in)


def _start_dashboard_peak(dashboard: Any, cluster_count: int) -> None:
    """Announce start of Stage 4 peak sub-phase on the dashboard."""
    if dashboard is None:
        return
    dashboard.start_stage4_peak(cluster_count)


def _complete_dashboard_peak(dashboard: Any, t0: float) -> None:
    """Mark the Stage 4 peak sub-phase complete with elapsed time."""
    if dashboard is None:
        return
    dashboard.complete_stage4_peak(time.perf_counter() - t0)


def _start_dashboard_diversity(dashboard: Any, pool_size: int) -> None:
    """Announce start of Stage 4 diversity sub-phase on the dashboard."""
    if dashboard is None:
        return
    dashboard.start_stage4_diversity(pool_size)


def _complete_dashboard_diversity(dashboard: Any, t0: float) -> None:
    """Mark the Stage 4 diversity sub-phase complete with elapsed time."""
    if dashboard is None:
        return
    dashboard.complete_stage4_diversity(time.perf_counter() - t0)


def _start_dashboard_tournament(dashboard: Any, candidate_count: int) -> None:
    """Announce start of Stage 4 tournament sub-phase on the dashboard."""
    if dashboard is None:
        return
    dashboard.start_stage4_tournament(candidate_count)


def _complete_dashboard_tournament(dashboard: Any, t0: float) -> None:
    """Mark the Stage 4 tournament sub-phase complete with elapsed time."""
    if dashboard is None:
        return
    dashboard.complete_stage4_tournament(time.perf_counter() - t0)


def _start_dashboard_narrative(dashboard: Any, selection_count: int) -> None:
    """Announce start of Stage 4 narrative sub-phase on the dashboard."""
    if dashboard is None:
        return
    dashboard.start_stage4_narrative(selection_count)


def _complete_dashboard_narrative(dashboard: Any, t0: float) -> None:
    """Mark the Stage 4 narrative sub-phase complete with elapsed time."""
    if dashboard is None:
        return
    dashboard.complete_stage4_narrative(time.perf_counter() - t0)


def _run_peak_phase(run_in: _ClusterRunInput) -> tuple[list[list[Path]], list[Path], dict[str, str | None]]:
    """Cluster keepers and pick per-cluster winners; bookend dashboard phase."""
    ci = run_in.curator_input
    t_peak = time.perf_counter()
    _start_dashboard_peak(ci.dashboard, len(ci.keepers))
    clusters = _build_clusters(run_in)
    winners, reasons = _pick_all_winners(clusters, run_in)
    _complete_dashboard_peak(ci.dashboard, t_peak)
    return clusters, winners, reasons


def _run_diversity_phase(winners: list[Path], run_in: _ClusterRunInput) -> list[Path]:
    """Run MMR diversity over winners; bookend dashboard phase."""
    ci = run_in.curator_input
    t0 = time.perf_counter()
    _start_dashboard_diversity(ci.dashboard, len(winners))
    div_in = _DiversityRunInput(
        winners=winners, curator_input=ci, target=run_in.target
    )
    diversified = _run_diversity(div_in)
    _complete_dashboard_diversity(ci.dashboard, t0)
    return diversified


def _run_tournament_phase(
    candidates: list[Path], run_in: _ClusterRunInput
) -> tuple[list[Path], int]:
    """Run tournament over candidates; bookend dashboard phase."""
    ci = run_in.curator_input
    t0 = time.perf_counter()
    _start_dashboard_tournament(ci.dashboard, len(candidates))
    tour_in = _TournamentRunInput(
        candidates=candidates, curator_input=ci, reporter=run_in.reporter
    )
    ordered, matches = _run_tournament(tour_in)
    _complete_dashboard_tournament(ci.dashboard, t0)
    return ordered, matches


def _run_narrative_phase(
    selections: list[CuratorSelection], run_in: _ClusterRunInput
) -> tuple[list[CuratorSelection], float]:
    """Run narrative flow check over selections; bookend dashboard phase."""
    ci = run_in.curator_input
    t0 = time.perf_counter()
    _start_dashboard_narrative(ci.dashboard, len(selections))
    narr_in = _NarrativeRunInput(selections=selections, curator_input=ci)
    new_selections, score = _run_narrative(narr_in)
    _complete_dashboard_narrative(ci.dashboard, t0)
    return new_selections, score


def _assemble_initial_selections(
    build_in: _FinalBuildInput, ordered: list[Path]
) -> list[CuratorSelection]:
    """Build CuratorSelection list for the tournament-ordered candidates."""
    cluster_lookup = _build_cluster_lookup(build_in.clusters)
    sel_in = _SelectionBuildInput(
        winners=ordered,
        cluster_lookup=cluster_lookup,
        scores=build_in.curator_input.composite_scores,
        reasons=build_in.reasons,
    )
    return _build_selections(sel_in)


def _final_curation_result(
    build_in: _FinalBuildInput, selections: list[CuratorSelection]
) -> CurationResult:
    """Assemble a CurationResult from the final per-phase outputs."""
    return CurationResult(
        is_enabled=True,
        target_count=build_in.target,
        actual_count=len(selections),
        cluster_count=len(build_in.clusters),
        vlm_tiebreakers=build_in.tournament_matches,
        threshold_used=build_in.threshold,
        elapsed_seconds=build_in.elapsed,
        selected=selections,
        narrative_flow_score=build_in.narrative_score,
    )


def _execute_phases(run_in: _ClusterRunInput, t0: float) -> CurationResult:
    """Execute peak → diversity → tournament → narrative and assemble result."""
    clusters, winners, reasons = _run_peak_phase(run_in)
    diversified = _run_diversity_phase(winners, run_in)
    ordered, matches = _run_tournament_phase(diversified, run_in)
    build_in = _FinalBuildInput(
        clusters=clusters, winners=ordered, reasons=reasons,
        curator_input=run_in.curator_input, target=run_in.target,
        threshold=run_in.threshold, elapsed=0.0,
        tournament_matches=matches, narrative_score=0.0,
    )
    selections = _assemble_initial_selections(build_in, ordered)
    final_selections, score = _run_narrative_phase(selections, run_in)
    build_in.elapsed = time.perf_counter() - t0
    build_in.narrative_score = score
    return _final_curation_result(build_in, final_selections)


def curate(curator_input: CuratorInput) -> CurationResult:
    """Cluster keepers, pick winners, and return a CurationResult."""
    config = curator_input.config
    if config.curate_target is None:
        raise ValueError("curate_target must be set before calling curate()")
    keepers = curator_input.keepers
    target = config.curate_target
    threshold = CLUSTER_THRESHOLD[config.preset]
    if len(keepers) == 0:
        return _empty_result(threshold, target)
    if len(keepers) <= target:
        return _make_all_keeper_result(curator_input, threshold)
    t0 = time.perf_counter()
    reporter = _S4ProgressReporter(dashboard=curator_input.dashboard)
    run_in = _ClusterRunInput(
        curator_input=curator_input, target=target,
        threshold=threshold, reporter=reporter,
    )
    return _execute_phases(run_in, t0)
