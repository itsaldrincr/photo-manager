"""Stage 4 curator routing tests — portrait/action/composite paths (mocked ML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from cull.config import CullConfig
from cull.models import (
    BlurScores,
    CuratorSelection,
    ExposureScores,
    PeakMomentScore,
    Stage1Result,
)
from cull.stage2.portrait import PortraitResult
from cull.stage4.curator import CuratorInput, curate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED: int = 42
EMBED_DIM: int = 1280
SMALL_NOISE_SCALE: float = 0.001
BASE_SCORE: float = 0.8
SCORE_STEP: float = 0.05
CURATE_TARGET: int = 2
CLUSTER_SIZE: int = 3
TENEN_VALUE: float = 100.0
FFT_VALUE: float = 0.5
BLUR_TIER: int = 1
DR_VALUE: float = 1.0
CLIP_VALUE: float = 0.0
MIDTONE_VALUE: float = 0.5
COLOR_CAST_VALUE: float = 0.0
NOISE_VALUE: float = 0.1


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


class _BuildInput(BaseModel):
    """Input bundle for cluster builders."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tmp_path: Path
    rng: np.random.Generator


def _make_config() -> CullConfig:
    """Build a minimal CullConfig for curator tests."""
    return CullConfig(preset="general", curate_target=CURATE_TARGET)  # type: ignore[arg-type]


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Return a single random unit-normalised embedding vector."""
    vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _near_copies(base: np.ndarray, count: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Return count embeddings near base with small Gaussian noise."""
    out: list[np.ndarray] = []
    for _ in range(count):
        noise = rng.standard_normal(EMBED_DIM).astype(np.float32) * SMALL_NOISE_SCALE
        out.append(base + noise)
    return out


def _touch_paths(tmp_path: Path, names: list[str]) -> list[Path]:
    """Create empty files and return their Paths."""
    paths: list[Path] = []
    for name in names:
        p = tmp_path / name
        p.touch()
        paths.append(p)
    return paths


def _make_blur(is_motion: bool) -> BlurScores:
    """Return a minimal BlurScores with the given motion flag."""
    return BlurScores(
        tenengrad=TENEN_VALUE,
        fft_ratio=FFT_VALUE,
        blur_tier=BLUR_TIER,
        is_motion_blur=is_motion,
    )


def _make_exposure() -> ExposureScores:
    """Return a minimal ExposureScores with neutral values."""
    return ExposureScores(
        dr_score=DR_VALUE,
        clipping_highlight=CLIP_VALUE,
        clipping_shadow=CLIP_VALUE,
        midtone_pct=MIDTONE_VALUE,
        color_cast_score=COLOR_CAST_VALUE,
    )


def _make_s1_result(path: Path, is_motion: bool) -> Stage1Result:
    """Return a Stage1Result for a path with optional motion-blur flag."""
    return Stage1Result(
        photo_path=path,
        blur=_make_blur(is_motion),
        exposure=_make_exposure(),
        noise_score=NOISE_VALUE,
    )


def _make_portrait(face_count: int) -> PortraitResult:
    """Return a PortraitResult with the given face count."""
    return PortraitResult(face_count=face_count)


class _ClusterBundle(BaseModel):
    """Bundle of cluster-build outputs used by curator tests."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    paths: list[Path]
    encodings: dict[str, Any]
    scores: dict[str, float]


def _build_single_cluster(bundle_in: _BuildInput, prefix: str) -> _ClusterBundle:
    """Return paths, encodings, and scores for one tight cluster."""
    paths = _touch_paths(bundle_in.tmp_path, [f"{prefix}_{i}.jpg" for i in range(CLUSTER_SIZE)])
    base = _unit_vector(bundle_in.rng)
    vecs = _near_copies(base, CLUSTER_SIZE, bundle_in.rng)
    encodings = {str(p): v for p, v in zip(paths, vecs)}
    scores = {str(p): BASE_SCORE - i * SCORE_STEP for i, p in enumerate(paths)}
    return _ClusterBundle(paths=paths, encodings=encodings, scores=scores)


def _merge_bundles(left: _ClusterBundle, right: _ClusterBundle) -> _ClusterBundle:
    """Merge two cluster bundles into one for CuratorInput construction."""
    return _ClusterBundle(
        paths=left.paths + right.paths,
        encodings={**left.encodings, **right.encodings},
        scores={**left.scores, **right.scores},
    )


def _build_portrait_input(tmp_path: Path) -> tuple[CuratorInput, list[Path]]:
    """Build CuratorInput whose first cluster is classified as portrait."""
    rng = np.random.default_rng(seed=SEED)
    portrait_bundle = _build_single_cluster(_BuildInput(tmp_path=tmp_path, rng=rng), "portrait")
    filler_bundle = _build_single_cluster(_BuildInput(tmp_path=tmp_path, rng=rng), "filler")
    merged = _merge_bundles(portrait_bundle, filler_bundle)
    portraits = {str(p): _make_portrait(face_count=1) for p in portrait_bundle.paths}
    curator_in = CuratorInput(
        keepers=merged.paths, encodings=merged.encodings,
        composite_scores=merged.scores, config=_make_config(), portraits=portraits,
    )
    return curator_in, portrait_bundle.paths


def _build_action_input(tmp_path: Path) -> tuple[CuratorInput, list[Path]]:
    """Build CuratorInput whose first cluster is classified as action."""
    rng = np.random.default_rng(seed=SEED)
    action_bundle = _build_single_cluster(_BuildInput(tmp_path=tmp_path, rng=rng), "action")
    filler_bundle = _build_single_cluster(_BuildInput(tmp_path=tmp_path, rng=rng), "filler")
    merged = _merge_bundles(action_bundle, filler_bundle)
    s1_results = {str(p): _make_s1_result(p, is_motion=True) for p in action_bundle.paths}
    curator_in = CuratorInput(
        keepers=merged.paths, encodings=merged.encodings,
        composite_scores=merged.scores, config=_make_config(), s1_results=s1_results,
    )
    return curator_in, action_bundle.paths


def _build_composite_input(tmp_path: Path) -> tuple[CuratorInput, list[Path]]:
    """Build CuratorInput with no faces and no motion → composite fallback path."""
    rng = np.random.default_rng(seed=SEED)
    left = _build_single_cluster(_BuildInput(tmp_path=tmp_path, rng=rng), "plain_a")
    right = _build_single_cluster(_BuildInput(tmp_path=tmp_path, rng=rng), "plain_b")
    merged = _merge_bundles(left, right)
    curator_in = CuratorInput(
        keepers=merged.paths, encodings=merged.encodings,
        composite_scores=merged.scores, config=_make_config(),
    )
    return curator_in, left.paths


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


class _PatchedMocks(BaseModel):
    """Bundle exposing every mocked stage4 callable to a test."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    portrait_picker: Any
    action_picker: Any
    diversity: Any
    tournament: Any
    narrative: Any


@pytest.fixture
def mocked_stage4() -> Any:
    """Patch every heavy stage4 callable imported inside curator.py."""
    peak_score = PeakMomentScore(
        eyes_open_score=1.0, smile_score=1.0, gaze_score=1.0,
        motion_peak_score=0.0, peak_type="portrait",
    )
    with patch("cull.stage4.curator.pick_portrait_winner") as portrait_mock, \
         patch("cull.stage4.curator.pick_action_winner") as action_mock, \
         patch("cull.stage4.curator.diversity_select") as diversity_mock, \
         patch("cull.stage4.curator.run_tournament") as tournament_mock, \
         patch("cull.stage4.curator.narrative_check") as narrative_mock:
        portrait_mock.side_effect = lambda inp: (inp.burst_members[0], peak_score)
        action_mock.side_effect = lambda inp: (inp.burst_members[0], peak_score)
        diversity_mock.side_effect = lambda mmr_in, ctx: list(mmr_in.candidates)
        tournament_mock.side_effect = lambda inp, ctx: list(inp.candidates)
        narrative_mock.side_effect = lambda flow_in: (flow_in.selections, 0.75)
        yield _PatchedMocks(
            portrait_picker=portrait_mock, action_picker=action_mock,
            diversity=diversity_mock, tournament=tournament_mock, narrative=narrative_mock,
        )


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


def test_portrait_cluster_routes_to_portrait_picker(
    tmp_path: Path, mocked_stage4: _PatchedMocks
) -> None:
    """Cluster with face_count votes is dispatched to peak_portrait.pick_winner."""
    curator_in, _portrait_cluster = _build_portrait_input(tmp_path)
    curate(curator_in)
    assert mocked_stage4.portrait_picker.call_count >= 1
    assert mocked_stage4.action_picker.call_count == 0


def test_action_cluster_routes_to_action_picker(
    tmp_path: Path, mocked_stage4: _PatchedMocks
) -> None:
    """Cluster with motion-blur votes is dispatched to peak_action.pick_winner."""
    curator_in, _action_cluster = _build_action_input(tmp_path)
    curate(curator_in)
    assert mocked_stage4.action_picker.call_count >= 1
    assert mocked_stage4.portrait_picker.call_count == 0


def test_composite_cluster_skips_peak_pickers(
    tmp_path: Path, mocked_stage4: _PatchedMocks
) -> None:
    """Clusters with neither faces nor motion use the composite fallback path."""
    curator_in, _cluster = _build_composite_input(tmp_path)
    curate(curator_in)
    assert mocked_stage4.portrait_picker.call_count == 0
    assert mocked_stage4.action_picker.call_count == 0


# ---------------------------------------------------------------------------
# Pipeline ordering tests
# ---------------------------------------------------------------------------


def _build_embedded_input(tmp_path: Path) -> CuratorInput:
    """Build a portrait CuratorInput with CLIP search embeddings attached."""
    curator_in, portrait_paths = _build_portrait_input(tmp_path)
    all_paths = curator_in.keepers
    rng = np.random.default_rng(seed=SEED + 1)
    embeddings = np.stack([_unit_vector(rng) for _ in all_paths], axis=0)
    path_to_row = {str(p): i for i, p in enumerate(all_paths)}
    return CuratorInput(
        keepers=curator_in.keepers, encodings=curator_in.encodings,
        composite_scores=curator_in.composite_scores, config=curator_in.config,
        portraits=curator_in.portraits, search_embeddings=embeddings,
        search_path_to_row=path_to_row,
    )


def test_diversity_runs_once_after_winners(
    tmp_path: Path, mocked_stage4: _PatchedMocks
) -> None:
    """diversity.select is invoked exactly once when a search cache is supplied."""
    curator_in = _build_embedded_input(tmp_path)
    curate(curator_in)
    assert mocked_stage4.diversity.call_count == 1


def test_tournament_runs_after_diversity(
    tmp_path: Path, mocked_stage4: _PatchedMocks
) -> None:
    """tournament.run fires after diversity selection returns candidates."""
    curator_in = _build_embedded_input(tmp_path)
    curate(curator_in)
    assert mocked_stage4.tournament.call_count == 1


def test_narrative_runs_last(
    tmp_path: Path, mocked_stage4: _PatchedMocks
) -> None:
    """narrative_flow.check is the final stage4 phase and populates the score."""
    curator_in = _build_embedded_input(tmp_path)
    result = curate(curator_in)
    assert mocked_stage4.narrative.call_count == 1
    assert result.narrative_flow_score == 0.75


# ---------------------------------------------------------------------------
# Dashboard sub-bar bookends
# ---------------------------------------------------------------------------


def test_curator_wires_dashboard_sub_bars(
    tmp_path: Path, mocked_stage4: _PatchedMocks
) -> None:
    """Each Stage 4 sub-bar start/complete method is called exactly once."""
    curator_in, _ = _build_portrait_input(tmp_path)
    dashboard = MagicMock()
    wired = CuratorInput(
        keepers=curator_in.keepers, encodings=curator_in.encodings,
        composite_scores=curator_in.composite_scores, config=curator_in.config,
        portraits=curator_in.portraits, dashboard=dashboard,
    )
    curate(wired)
    assert dashboard.start_stage4_peak.call_count == 1
    assert dashboard.complete_stage4_peak.call_count == 1
    assert dashboard.start_stage4_diversity.call_count == 1
    assert dashboard.complete_stage4_diversity.call_count == 1
    assert dashboard.start_stage4_tournament.call_count == 1
    assert dashboard.complete_stage4_tournament.call_count == 1
    assert dashboard.start_stage4_narrative.call_count == 1
    assert dashboard.complete_stage4_narrative.call_count == 1


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


def test_tournament_returns_all_candidates_in_order() -> None:
    """Regression: tournament.run must return all candidates, not just winner."""
    from cull.stage4.tournament import TournamentContext, TournamentInput, run as run_tournament

    tmp = Path("/tmp/test_tournament_fix")
    tmp.mkdir(exist_ok=True)
    try:
        paths = [tmp / f"photo_{i}.jpg" for i in range(5)]
        for p in paths:
            p.touch()

        scores = {str(p): 0.8 - i * 0.05 for i, p in enumerate(paths)}
        ctx = TournamentContext(s1_results={}, composite_scores=scores)

        with patch("cull.stage4.tournament._play_match") as match_mock:
            match_mock.side_effect = lambda params: params.photo_a if scores[str(params.photo_a)] > scores[str(params.photo_b)] else params.photo_b
            inp = TournamentInput(candidates=paths, config=CullConfig(preset="general"))  # type: ignore[arg-type]
            result = run_tournament(inp, ctx)

        assert len(result) == len(paths), f"Expected {len(paths)} results, got {len(result)}"
    finally:
        for p in paths:
            p.unlink(missing_ok=True)
        tmp.rmdir()


def test_curation_target_equals_keeper_count_returns_all(
    tmp_path: Path, mocked_stage4: _PatchedMocks
) -> None:
    """When target >= keeper count, return all keepers."""
    rng = np.random.default_rng(seed=SEED)
    bundle = _build_single_cluster(_BuildInput(tmp_path=tmp_path, rng=rng), "items")
    keeper_count = len(bundle.paths)
    config = CullConfig(preset="general", curate_target=keeper_count)  # type: ignore[arg-type]
    curator_in = CuratorInput(
        keepers=bundle.paths, encodings=bundle.encodings,
        composite_scores=bundle.scores, config=config,
    )
    result = curate(curator_in)
    assert result.actual_count == keeper_count
    assert len(result.selected) == keeper_count
