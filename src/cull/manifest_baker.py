"""Smart manifest baker — walks a corpus dir and writes manifest.json.

Categories are assigned by combining two signals:
1. Filename prefix for synth_* photos (author-injected, prefix is authoritative).
2. Content-aware burst detection via EXIF timestamps + dHash visual confirmation
   (same algorithm cull uses at runtime in src/cull/stage1/burst.py).

Everything else is classified as 'real'. Drop a real burst sequence into the
corpus and the baker will catch it without requiring filename gymnastics.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from cull.config import BURST_GAP_DEFAULT_SECONDS
from cull.stage1.burst import (
    cluster_by_time,
    confirm_burst_visually,
    read_timestamps,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME: str = "manifest.json"
HASH_CHUNK_BYTES: int = 65536
PHOTO_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg"})
JSON_INDENT: int = 2

CATEGORY_REAL: str = "real"
CATEGORY_BURST: str = "burst"
CATEGORY_SYNTH_PREFIXES: dict[str, str] = {
    "synth_blur_": "synth_blur",
    "synth_dupe_": "synth_dupe",
    "synth_motion_": "synth_motion",
    "synth_noisy_": "synth_noisy",
    "synth_overexp_": "synth_overexp",
    "synth_underexp_": "synth_underexp",
}


class ManifestEntry(BaseModel):
    """One file in the corpus manifest."""

    name: str
    sha256: str
    bytes: int
    category: str


class ManifestBakeRequest(BaseModel):
    """Input for a manifest bake run."""

    corpus_dir: Path


class ManifestBakeResult(BaseModel):
    """Output of a manifest bake run with category histogram."""

    corpus_dir: Path
    manifest_path: Path
    entry_count: int
    category_counts: dict[str, int]


class ManifestBakeError(Exception):
    """Raised when the corpus dir is missing or contains no photos."""


def _scan_photos(corpus_dir: Path) -> list[Path]:
    """Return sorted list of JPEG files directly under corpus_dir."""
    if not corpus_dir.is_dir():
        raise ManifestBakeError(f"corpus dir not found: {corpus_dir}")
    photos = [
        p for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS
    ]
    if not photos:
        raise ManifestBakeError(f"no JPEG files found in {corpus_dir}")
    return sorted(photos)


def _compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file streamed in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _classify_synth(filename: str) -> str | None:
    """Return synth_* category if the filename matches a known prefix, else None."""
    for prefix, category in CATEGORY_SYNTH_PREFIXES.items():
        if filename.startswith(prefix):
            return category
    return None


def _detect_burst_set(photo_paths: list[Path]) -> set[Path]:
    """Return the set of non-synth photo paths that belong to a confirmed burst group.

    Synth photos are excluded because they inherit their source's EXIF timestamp
    and would trigger false burst clusters against their source photo.
    """
    non_synth = [p for p in photo_paths if _classify_synth(p.name) is None]
    timestamped = read_timestamps(non_synth)
    temporal_groups = cluster_by_time(timestamped, BURST_GAP_DEFAULT_SECONDS)
    burst_set: set[Path] = set()
    for group in temporal_groups:
        for confirmed in confirm_burst_visually(group):
            burst_set.update(confirmed)
    return burst_set


def _classify_one(path: Path, burst_set: set[Path]) -> str:
    """Return the category for one photo via synth-prefix → burst → real."""
    synth = _classify_synth(path.name)
    if synth is not None:
        return synth
    if path in burst_set:
        return CATEGORY_BURST
    return CATEGORY_REAL


def _build_entry(path: Path, burst_set: set[Path]) -> ManifestEntry:
    """Construct one ManifestEntry from disk: hash, byte size, category."""
    return ManifestEntry(
        name=path.name,
        sha256=_compute_sha256(path),
        bytes=path.stat().st_size,
        category=_classify_one(path, burst_set),
    )


def _build_entries(photos: list[Path]) -> list[ManifestEntry]:
    """Compute burst set once and build a ManifestEntry for every photo."""
    burst_set = _detect_burst_set(photos)
    return [_build_entry(p, burst_set) for p in photos]


def _category_histogram(entries: list[ManifestEntry]) -> dict[str, int]:
    """Return a {category: count} dict aggregated across all entries."""
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.category] = counts.get(entry.category, 0) + 1
    return counts


def _write_manifest(corpus_dir: Path, entries: list[ManifestEntry]) -> Path:
    """Serialise the entry list to manifest.json under corpus_dir; return path."""
    manifest_path = corpus_dir / MANIFEST_FILENAME
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(entries),
        "files": [entry.model_dump() for entry in entries],
    }
    manifest_path.write_text(json.dumps(payload, indent=JSON_INDENT))
    logger.info("Manifest written: %s (%d entries)", manifest_path, len(entries))
    return manifest_path


def bake_manifest(request: ManifestBakeRequest) -> ManifestBakeResult:
    """Walk corpus_dir, classify every photo, and write a fresh manifest.json."""
    photos = _scan_photos(request.corpus_dir)
    logger.info("Baking manifest for %d photos in %s", len(photos), request.corpus_dir)
    entries = _build_entries(photos)
    manifest_path = _write_manifest(request.corpus_dir, entries)
    return ManifestBakeResult(
        corpus_dir=request.corpus_dir,
        manifest_path=manifest_path,
        entry_count=len(entries),
        category_counts=_category_histogram(entries),
    )
