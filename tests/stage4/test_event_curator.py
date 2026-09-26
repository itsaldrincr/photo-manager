"""Stage 4 event curation: event_score ranking, MMR at lambda 0.75, no tournament."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from cull.config import EVENT_MMR_LAMBDA, CullConfig
from cull.stage4.curator import CuratorInput, curate

TARGET: int = 3
PHOTO_COUNT: int = 6
EMBED_DIM: int = 8


def _curator_input(tmp_path: Path, has_embeddings: bool) -> CuratorInput:
    """Return six candidates whose event score falls with their index."""
    paths = [tmp_path / f"p{i}.jpg" for i in range(PHOTO_COUNT)]
    event_scores = {str(p): 5.0 - i for i, p in enumerate(paths)}
    # Composite deliberately ranks the photos in reverse, so a composite-driven
    # curator would pick the wrong three.
    composites = {str(p): 0.1 * i for i, p in enumerate(paths)}
    embeddings = np.eye(PHOTO_COUNT, EMBED_DIM, dtype=np.float32) if has_embeddings else None
    return CuratorInput(
        keepers=paths, encodings={}, composite_scores=composites,
        config=CullConfig(preset="event", curate_target=TARGET),
        search_embeddings=embeddings,
        search_path_to_row={str(p): i for i, p in enumerate(paths)} if has_embeddings else None,
        event_scores=event_scores,
    )


def test_event_curator_never_runs_tournament_or_narrative(tmp_path: Path) -> None:
    """Event curation skips the VLM tournament, the narrative swap and clustering."""
    with patch("cull.stage4.curator.run_tournament") as tournament, \
         patch("cull.stage4.curator.narrative_check") as narrative, \
         patch("cull.stage4.curator.cluster_by_similarity") as clustering:
        result = curate(_curator_input(tmp_path, has_embeddings=True))
    tournament.assert_not_called()
    narrative.assert_not_called()
    clustering.assert_not_called()
    assert result.vlm_tiebreakers == 0
    assert result.narrative_flow_score is None


def test_event_curator_picks_top_event_scores_with_mmr(tmp_path: Path) -> None:
    """MMR runs at lambda 0.75 over candidates ranked by event_score."""
    seen = {}

    def _spy(mmr_in, context):  # noqa: ANN001, ANN202
        seen["lambda"] = context.lambda_quality
        seen["order"] = [p.name for p in mmr_in.candidates]
        return list(mmr_in.candidates)[: context.target_count]

    with patch("cull.stage4.event_curator.diversity_select", side_effect=_spy):
        result = curate(_curator_input(tmp_path, has_embeddings=True))
    assert seen["lambda"] == EVENT_MMR_LAMBDA
    assert seen["order"] == [f"p{i}.jpg" for i in range(PHOTO_COUNT)]
    assert [s.path.name for s in result.selected] == ["p0.jpg", "p1.jpg", "p2.jpg"]


def test_event_curator_real_mmr_prefers_higher_ratings(tmp_path: Path) -> None:
    """With orthogonal embeddings, real MMR keeps the event_score order."""
    result = curate(_curator_input(tmp_path, has_embeddings=True))
    assert [s.path.name for s in result.selected] == ["p0.jpg", "p1.jpg", "p2.jpg"]
    assert result.selected[0].reason == "event score 5.000"


def test_event_curator_without_search_cache_takes_top_n(tmp_path: Path) -> None:
    """Without CLIP rows, the top N by event_score are picked."""
    result = curate(_curator_input(tmp_path, has_embeddings=False))
    assert [s.path.name for s in result.selected] == ["p0.jpg", "p1.jpg", "p2.jpg"]
