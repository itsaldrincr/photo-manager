"""Burst detection: temporal clustering, visual confirmation, and winner selection."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import exifread
import imagehash
from PIL import Image
from pydantic import BaseModel, Field

from cull.config import (
    BLUR_DHASH_HAMMING_MAX,
    CullConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local models
# ---------------------------------------------------------------------------


class BurstScoringInput(BaseModel):
    """Input bundle for selecting the best photo in a burst group."""

    group: list[Path]
    blur_scores: dict[str, float]


class _BurstInput(BaseModel):
    """Input bundle for detect_bursts function."""

    image_paths: list[Path]
    config: CullConfig
    blur_scores: dict[str, float] | None = None


class BurstResult(BaseModel):
    """Output of burst detection across all images."""

    groups: list[list[Path]] = Field(default_factory=list)
    winners: list[Path] = Field(default_factory=list)
    losers: list[Path] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TimestampedPhoto = tuple[Path, datetime | None]


def _read_exif_datetime(path: Path) -> datetime | None:
    """Return EXIF DateTimeOriginal for a single file, or None."""
    try:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, stop_tag="EXIF DateTimeOriginal", details=False)
        raw = tags.get("EXIF DateTimeOriginal")
        if raw is None:
            return None
        return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
    except (OSError, ValueError, TypeError):
        logger.debug("EXIF read failed for %s", path)
        return None


def _mtime_as_datetime(path: Path) -> datetime:
    """Return the file modification time as a datetime."""
    return datetime.fromtimestamp(path.stat().st_mtime)


def _dhash_distance(path_a: Path, path_b: Path) -> int:
    """Return the Hamming distance between dHash values of two images."""
    hash_a = imagehash.dhash(Image.open(path_a))
    hash_b = imagehash.dhash(Image.open(path_b))
    return hash_a - hash_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_timestamps(image_paths: list[Path]) -> list[TimestampedPhoto]:
    """Return (path, datetime) pairs using EXIF with mtime fallback."""
    result: list[TimestampedPhoto] = []
    for path in image_paths:
        dt = _read_exif_datetime(path)
        if dt is None:
            dt = _mtime_as_datetime(path)
            logger.debug("Using mtime for %s", path)
        result.append((path, dt))
    return result


def cluster_by_time(timestamped: list[TimestampedPhoto], gap_seconds: float) -> list[list[Path]]:
    """Group photos into bursts where consecutive timestamps differ by ≤ gap_seconds."""
    if not timestamped:
        return []
    sorted_photos = sorted(timestamped, key=lambda t: t[1] or datetime.min)
    groups: list[list[Path]] = [[sorted_photos[0][0]]]
    prev_dt = sorted_photos[0][1]
    for path, dt in sorted_photos[1:]:
        gap = abs((dt - prev_dt).total_seconds()) if dt and prev_dt else gap_seconds + 1
        if gap <= gap_seconds:
            groups[-1].append(path)
        else:
            groups.append([path])
        if dt is not None:
            prev_dt = dt
    return [g for g in groups if len(g) > 1]


def confirm_burst_visually(group: list[Path]) -> list[list[Path]]:
    """Split a temporal group into visually similar sub-groups using dHash."""
    if not group:
        return []
    confirmed: list[list[Path]] = [[group[0]]]
    for path in group[1:]:
        placed = False
        for sub_group in confirmed:
            dist = _dhash_distance(sub_group[0], path)
            if dist <= BLUR_DHASH_HAMMING_MAX:
                sub_group.append(path)
                placed = True
                break
        if not placed:
            confirmed.append([path])
    return [g for g in confirmed if len(g) > 1]


def select_burst_winner(scoring_input: BurstScoringInput) -> tuple[Path, list[Path]]:
    """Return (winner, losers) where winner has the highest blur score."""
    group = scoring_input.group
    blur_scores = scoring_input.blur_scores
    ranked = sorted(group, key=lambda p: blur_scores.get(str(p), 0.0), reverse=True)
    winner = ranked[0]
    losers = ranked[1:]
    return winner, losers


def detect_bursts(burst_in: _BurstInput) -> BurstResult:
    """Detect burst groups, confirm visually, and select winners."""
    blur_scores = burst_in.blur_scores if burst_in.blur_scores is not None else {}
    timestamped = read_timestamps(burst_in.image_paths)
    temporal_groups = cluster_by_time(timestamped, burst_in.config.burst_gap)
    all_groups: list[list[Path]] = []
    for group in temporal_groups:
        visual_groups = confirm_burst_visually(group)
        all_groups.extend(visual_groups)
    winners: list[Path] = []
    losers: list[Path] = []
    for group in all_groups:
        scoring_input = BurstScoringInput(group=group, blur_scores=blur_scores)
        winner, group_losers = select_burst_winner(scoring_input)
        winners.append(winner)
        losers.extend(group_losers)
    return BurstResult(groups=all_groups, winners=winners, losers=losers)
