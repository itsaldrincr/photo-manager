"""ARCHIVED candidate scorer (Task A, gate FAILED) — face-aware burst winner.

This is the face-aware burst winner scorer evaluated in the signal-quality wave
and REJECTED at the gate (benchmarks/runs/burst_selection_report.md): blending a
MediaPipe eyes-open signal with blur made winner_matches_owner_rate WORSE
(0.56 -> 0.48 pooled), because the owner's implicit burst keepers on the test
corpus do not favor open eyes — in one group the owner kept a genuine near-blink
(EAR 0.099) that this scorer correctly rejected. It is kept here, out of the
production tree, only so the attempted scorer is reproducible from the report.
It is NOT imported by any production code.

Ranks by within-group max-normalized blur blended with a per-member eyes-open
signal (neutral for faceless members). Max-normalization (not min-max) keeps a
less-sharp-but-open frame at a proportional blur value so the eyes term can
overturn a marginally-sharper blink.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

from cull.stage1.burst import BurstScoringInput

logger = logging.getLogger(__name__)

# Weights sum to 1.0; eyes-open reference and the faceless-member neutral score.
BURST_RANK_BLUR_WEIGHT: float = 0.5
BURST_RANK_EAR_WEIGHT: float = 0.5
BURST_RANK_EAR_OPEN_REF: float = 0.28
BURST_RANK_NO_FACE_EYES_SCORE: float = 0.5


def _blur_norm(rank_in: BurstScoringInput) -> dict[str, float]:
    """Normalize group blur scores by the group max into [0, 1]; all-zero maps to 1.0."""
    values = {str(p): rank_in.blur_scores.get(str(p), 0.0) for p in rank_in.group}
    hi = max(values.values(), default=0.0)
    if hi < 1e-9:
        return {key: 1.0 for key in values}
    return {key: value / hi for key, value in values.items()}


def _eyes_open_score(image: Any) -> float:
    """Return an eyes-open score in [0, 1] for the primary face, neutral if none."""
    from cull.stage2.portrait import compute_ear, detect_faces  # noqa: PLC0415

    faces = detect_faces(image)
    if not faces:
        return BURST_RANK_NO_FACE_EYES_SCORE
    return min(1.0, compute_ear(faces[0]) / BURST_RANK_EAR_OPEN_REF)


def _member_eyes_score(path: Path) -> float:
    """Load one burst member and return its eyes-open score (neutral on read failure)."""
    image = cv2.imread(str(path))
    if image is None:
        logger.warning("Cannot read %s for burst face ranking; using neutral score", path)
        return BURST_RANK_NO_FACE_EYES_SCORE
    return _eyes_open_score(image)


def _rank_scores(rank_in: BurstScoringInput) -> dict[str, float]:
    """Return each member's blended score: weighted normalized blur + eyes-open."""
    blur = _blur_norm(rank_in)
    return {
        str(p): BURST_RANK_BLUR_WEIGHT * blur[str(p)]
        + BURST_RANK_EAR_WEIGHT * _member_eyes_score(p)
        for p in rank_in.group
    }


def select_face_aware_winner(rank_in: BurstScoringInput) -> tuple[Path, list[Path]]:
    """Return (winner, losers) ranked by the blended blur + eyes-open score."""
    scores = _rank_scores(rank_in)
    ranked = sorted(rank_in.group, key=lambda p: scores[str(p)], reverse=True)
    return ranked[0], ranked[1:]
