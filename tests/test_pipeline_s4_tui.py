"""Integration tests for Stage 4 curation dashboard wiring via pipeline._run_s4."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pydantic import BaseModel, ConfigDict

from cull.config import CullConfig
from cull.models import PhotoDecision, PhotoMeta, Stage2Result
from cull.pipeline import (
    _S4RunInput,
    _StageRunCtx,
    _StagesResult,
    _Stage1Output,
    _Stage2Output,
    _run_s4,
)
from cull.stage2.fusion import FusionResult


@pytest.fixture(autouse=True)
def _stub_stage4_ml() -> Any:
    """Stub ML-heavy stage4 dependencies so tests never load real models."""
    with patch("cull.stage4.curator.run_tournament") as mock_tournament, \
         patch("cull.stage4.curator.narrative_check") as mock_narrative, \
         patch("cull.stage4.curator.diversity_select") as mock_diversity:
        mock_tournament.side_effect = lambda inp, ctx: list(inp.candidates)
        mock_narrative.side_effect = lambda flow_in: (flow_in.selections, 0.0)
        mock_diversity.side_effect = lambda mmr_in, ctx: list(mmr_in.candidates)
        yield

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM: int = 1280
TARGET_COUNT: int = 2
TARGET_LARGE: int = 5
SEED: int = 42
CLUSTER_NOISE_SCALE: float = 0.001
BASE_SCORE: float = 0.8
SCORE_STEP: float = 0.1
DEFAULT_SCORE: float = 0.5


class _TestInputParams(BaseModel):
    """Bundle for test input construction (3+ params)."""

    curate_target: int | None
    keeper_count: int
    cluster_count: int


class _ClusterEmbeddingsInput(BaseModel):
    """Bundle for cluster embeddings generation."""

    cluster_id: int
    size: int


class _StageOutputsInput(BaseModel):
    """Bundle for stage outputs construction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    keeper_paths: list[Path]
    scores: dict[str, float]
    encodings: dict[str, Any]


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_base_vector(rng: np.random.Generator) -> np.ndarray:
    """Return a single random unit-normalised embedding vector."""
    vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm


def _make_cluster_embeddings(
    emb_in: _ClusterEmbeddingsInput,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate cluster_size embeddings for cluster_id."""
    base = _make_base_vector(rng)
    result = []
    for _ in range(emb_in.size):
        noise = rng.standard_normal(EMBED_DIM).astype(np.float32) * CLUSTER_NOISE_SCALE
        result.append(base + noise)
    return result


def _make_keeper_paths(tmp_path: Path, count: int) -> list[Path]:
    """Create count empty files and return their Paths."""
    paths: list[Path] = []
    for i in range(count):
        p = tmp_path / f"keeper_{i:03d}.jpg"
        p.touch()
        paths.append(p)
    return paths


def _make_encodings_and_scores(
    keeper_paths: list[Path],
    cluster_count: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Generate embeddings and composite scores across clusters."""
    rng = np.random.default_rng(seed=SEED)
    cluster_size = max(1, len(keeper_paths) // cluster_count)
    encodings: dict[str, Any] = {}
    scores: dict[str, float] = {}
    offset = 0
    for cid in range(cluster_count):
        end = min(offset + cluster_size, len(keeper_paths))
        cluster_paths = keeper_paths[offset:end]
        emb_in = _ClusterEmbeddingsInput(cluster_id=cid, size=len(cluster_paths))
        embeddings = _make_cluster_embeddings(emb_in, rng)
        for path, emb in zip(cluster_paths, embeddings):
            encodings[str(path)] = emb
            scores[str(path)] = BASE_SCORE - (cid * SCORE_STEP)
        offset = end
    return encodings, scores


def _make_stage_outputs(stg_in: _StageOutputsInput) -> _StagesResult:
    """Build S1 and S2 output structures with encodings and fusion results."""
    s1_out = _Stage1Output(encodings=stg_in.encodings)
    s2_out = _Stage2Output()
    for path in stg_in.keeper_paths:
        score = stg_in.scores.get(str(path), DEFAULT_SCORE)
        stage2_result = Stage2Result(
            photo_path=path,
            topiq=score,
            laion_aesthetic=score,
            clipiqa=score,
            composite=score,
        )
        s2_out.results[str(path)] = FusionResult(stage2=stage2_result, routing="KEEPER")
    return _StagesResult(s1_out=s1_out, s2_out=s2_out)


def _build_s4_input(
    tmp_path: Path,
    params: _TestInputParams,
) -> tuple[_S4RunInput, list[Path]]:
    """Assemble _S4RunInput with synthetic embeddings spanning clusters."""
    keeper_paths = _make_keeper_paths(tmp_path, params.keeper_count)
    encodings, scores = _make_encodings_and_scores(keeper_paths, params.cluster_count)
    stg_in = _StageOutputsInput(keeper_paths=keeper_paths, scores=scores, encodings=encodings)
    stages = _make_stage_outputs(stg_in)
    decisions = [PhotoDecision(photo=PhotoMeta(path=p, filename=p.name), decision="keeper") for p in keeper_paths]
    config = CullConfig(preset="general", curate_target=params.curate_target)  # type: ignore[arg-type]
    dashboard = MagicMock()
    ctx = _StageRunCtx(config=config, paths=keeper_paths, dashboard=dashboard)
    return _S4RunInput(stages=stages, decisions=decisions, ctx=ctx), keeper_paths


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_run_s4_calls_dashboard_bookends_when_curate_set(tmp_path: Path) -> None:
    """Start and complete calls happen when curate_target is set and keepers exist."""
    params = _TestInputParams(curate_target=TARGET_COUNT, keeper_count=4, cluster_count=2)
    s4_in, _ = _build_s4_input(tmp_path, params)
    result = _run_s4(s4_in)
    assert result is not None
    assert s4_in.ctx.dashboard.start_stage4.call_count == 1
    assert s4_in.ctx.dashboard.start_stage4.call_args[1]["target"] == TARGET_COUNT
    assert s4_in.ctx.dashboard.complete_stage4.call_count == 1
    call_args = s4_in.ctx.dashboard.complete_stage4.call_args[0]
    assert isinstance(call_args[0], float)
    assert call_args[0] >= 0.0


def test_run_s4_updates_dashboard_during_clustering(tmp_path: Path) -> None:
    """Update calls occur during clustering with clusters_found > 0."""
    params = _TestInputParams(curate_target=TARGET_COUNT, keeper_count=5, cluster_count=2)
    s4_in, _ = _build_s4_input(tmp_path, params)
    result = _run_s4(s4_in)
    assert result is not None
    assert s4_in.ctx.dashboard.update_stage4.call_count >= 1
    # Check at least one update has clusters_found > 0
    found_cluster_update = False
    for call in s4_in.ctx.dashboard.update_stage4.call_args_list:
        update_input = call[0][0]
        if update_input.clusters_found > 0:
            found_cluster_update = True
            break
    assert found_cluster_update, "Expected at least one update with clusters_found > 0"


def test_run_s4_updates_dashboard_per_cluster_winner(tmp_path: Path) -> None:
    """Per-winner updates carry incrementing selected values."""
    params = _TestInputParams(curate_target=TARGET_LARGE, keeper_count=6, cluster_count=3)
    expected_cluster_count = params.cluster_count
    s4_in, _ = _build_s4_input(tmp_path, params)
    result = _run_s4(s4_in)
    assert result is not None
    # Collect all selected values from updates
    selected_values = []
    for call in s4_in.ctx.dashboard.update_stage4.call_args_list:
        update_input = call[0][0]
        selected_values.append(update_input.selected)
    # Should have at least 3 updates with selected >= 1
    selected_at_1_or_more = sum(1 for v in selected_values if v >= 1)
    assert selected_at_1_or_more >= expected_cluster_count, f"Expected at least {expected_cluster_count} updates with selected >= 1, got {selected_at_1_or_more}"
    # Check max selected is >= cluster count (3)
    assert max(selected_values) >= expected_cluster_count, f"Expected max selected >= {expected_cluster_count}, got {max(selected_values)}"


def test_run_s4_skips_dashboard_when_curate_target_none(tmp_path: Path) -> None:
    """No dashboard calls when curate_target is None."""
    params = _TestInputParams(curate_target=None, keeper_count=4, cluster_count=2)
    s4_in, _ = _build_s4_input(tmp_path, params)
    result = _run_s4(s4_in)
    assert result is None
    assert s4_in.ctx.dashboard.start_stage4.call_count == 0
    assert s4_in.ctx.dashboard.update_stage4.call_count == 0
    assert s4_in.ctx.dashboard.complete_stage4.call_count == 0


def test_run_s4_skips_dashboard_when_no_keepers(tmp_path: Path) -> None:
    """No dashboard calls when keepers list is empty."""
    params = _TestInputParams(curate_target=TARGET_LARGE, keeper_count=0, cluster_count=1)
    s4_in, _ = _build_s4_input(tmp_path, params)
    result = _run_s4(s4_in)
    assert result is None
    assert s4_in.ctx.dashboard.start_stage4.call_count == 0
    assert s4_in.ctx.dashboard.update_stage4.call_count == 0
    assert s4_in.ctx.dashboard.complete_stage4.call_count == 0
