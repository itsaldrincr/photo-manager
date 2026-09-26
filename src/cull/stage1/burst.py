"""Burst detection: temporal clustering, visual confirmation, and winner selection."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import exifread
import imagehash
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from cull.config import (
    BLUR_DHASH_HAMMING_MAX,
    BURST_GAP_NO_SUBSEC_SECONDS,
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

# dHash compares a 9x8 thumbnail, so a JPEG DCT-scaled decode near this size
# gives the same hash as a full decode at a fraction of the cost.
DHASH_DECODE_PX: int = 256


class CaptureTime(BaseModel):
    """Capture time of one photo and whether it has sub-second resolution."""

    path: Path
    time: datetime | None
    has_subsec: bool = True


def _subsec_fraction(raw: object) -> timedelta | None:
    """Return EXIF SubSecTimeOriginal digits ("45" = 0.45 s) as a timedelta, or None."""
    digits = str(raw).strip() if raw is not None else ""
    if not digits.isdigit():
        return None
    return timedelta(seconds=float(f"0.{digits}"))


def _read_exif_capture(path: Path) -> CaptureTime | None:
    """Return EXIF DateTimeOriginal plus SubSecTimeOriginal for one file, or None."""
    try:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, stop_tag="EXIF SubSecTimeOriginal", details=False)
        raw = tags.get("EXIF DateTimeOriginal")
        if raw is None:
            return None
        whole = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
    except (OSError, ValueError, TypeError):
        logger.debug("EXIF read failed for %s", path)
        return None
    fraction = _subsec_fraction(tags.get("EXIF SubSecTimeOriginal"))
    if fraction is None:
        return CaptureTime(path=path, time=whole, has_subsec=False)
    return CaptureTime(path=path, time=whole + fraction)


def _mtime_as_datetime(path: Path) -> datetime | None:
    """Return the file modification time as a datetime, or None if unreadable."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        logger.warning("Cannot stat %s for burst timestamp; dropping from burst pass", path)
        return None


def _dhash(path: Path) -> imagehash.ImageHash:
    """Return the dHash of path from a reduced-size decode."""
    with Image.open(path) as img:
        img.draft("L", (DHASH_DECODE_PX, DHASH_DECODE_PX))
        return imagehash.dhash(img)


def _dhash_distance(path_a: Path, path_b: Path) -> int | None:
    """Return the Hamming distance between dHash values, or None if either is unreadable."""
    try:
        return _dhash(path_a) - _dhash(path_b)
    except (OSError, UnidentifiedImageError) as exc:
        logger.warning("Cannot hash %s or %s for burst pass: %s", path_a, path_b, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_capture_times(image_paths: list[Path]) -> list[CaptureTime]:
    """Return capture times from EXIF (with sub-seconds when present), else mtime."""
    result: list[CaptureTime] = []
    for path in image_paths:
        capture = _read_exif_capture(path)
        if capture is None:
            logger.debug("Using mtime for %s", path)
            capture = CaptureTime(path=path, time=_mtime_as_datetime(path))
        result.append(capture)
    return result


def read_timestamps(image_paths: list[Path]) -> list[TimestampedPhoto]:
    """Return (path, datetime) pairs using EXIF with mtime fallback."""
    return [(c.path, c.time) for c in read_capture_times(image_paths)]


def _within_gap(pair: tuple[CaptureTime, CaptureTime], gap_seconds: float) -> bool:
    """Return True if two frames are close enough in time to share a burst.

    Whole-second timestamps (e.g. Fuji JPEGs) put consecutive burst frames
    a full second apart, which a sub-second gap would always split, so the
    gap widens when either frame lacks sub-seconds.
    """
    prev, cur = pair
    if prev.time is None or cur.time is None:
        return False
    limit = gap_seconds
    if not (prev.has_subsec and cur.has_subsec):
        limit = max(gap_seconds, BURST_GAP_NO_SUBSEC_SECONDS)
    return abs((cur.time - prev.time).total_seconds()) <= limit


def cluster_captures(captures: list[CaptureTime], gap_seconds: float) -> list[list[Path]]:
    """Group photos into bursts where consecutive capture times are within the gap."""
    if not captures:
        return []
    ordered = sorted(captures, key=lambda c: c.time or datetime.min)
    groups: list[list[Path]] = [[ordered[0].path]]
    prev = ordered[0]
    for cur in ordered[1:]:
        if _within_gap((prev, cur), gap_seconds):
            groups[-1].append(cur.path)
        else:
            groups.append([cur.path])
        if cur.time is not None:
            prev = cur
    return [g for g in groups if len(g) > 1]


def cluster_by_time(timestamped: list[TimestampedPhoto], gap_seconds: float) -> list[list[Path]]:
    """Group (path, datetime) pairs into bursts where consecutive timestamps differ by ≤ gap_seconds."""
    return cluster_captures([CaptureTime(path=p, time=dt) for p, dt in timestamped], gap_seconds)


def confirm_burst_visually(group: list[Path]) -> list[list[Path]]:
    """Split a temporal group into visually similar sub-groups using dHash.

    A photo that cannot be hashed (deleted/unreadable between scan and this
    pass) never matches any sub-group representative — _dhash_distance
    returns None for it instead of raising — so it either lands in its own
    singleton group (dropped by the final length filter) or is simply
    skipped when compared against a broken representative.
    """
    if not group:
        return []
    confirmed: list[list[Path]] = [[group[0]]]
    for path in group[1:]:
        placed = False
        for sub_group in confirmed:
            dist = _dhash_distance(sub_group[0], path)
            if dist is not None and dist <= BLUR_DHASH_HAMMING_MAX:
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
    captures = read_capture_times(burst_in.image_paths)
    temporal_groups = cluster_captures(captures, burst_in.config.burst_gap)
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
