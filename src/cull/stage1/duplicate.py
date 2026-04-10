"""CNN-based duplicate detection via imagededup."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cull.config import BLUR_CNN_SIMILARITY_EXACT

logger = logging.getLogger(__name__)

_CNN_INSTANCE: object | None = None
_IMPORT_FAILED: bool = False


class DuplicateGroup(BaseModel):
    """A group of duplicate images with similarity scores."""

    paths: list[Path]
    similarities: list[float] = Field(default_factory=list)


class DuplicateResult(BaseModel):
    """Output of duplicate detection across an image directory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    duplicate_groups: list[DuplicateGroup] = Field(default_factory=list)
    encodings: dict[str, Any] = Field(default_factory=dict)


def _build_duplicate_groups(duplicates_map: dict, image_dir: Path) -> list[DuplicateGroup]:
    """Convert the imagededup duplicates map into DuplicateGroup objects."""
    seen: set[str] = set()
    groups: list[DuplicateGroup] = []
    for source, dupes in duplicates_map.items():
        if not dupes or source in seen:
            continue
        members = [source] + dupes
        seen.update(members)
        paths = [image_dir / name for name in members]
        groups.append(DuplicateGroup(paths=paths))
    return groups


def _build_encodings_map(raw_encodings: dict, image_dir: Path) -> dict[str, Any]:
    """Build encodings map keyed by full str(path)."""
    return {str(image_dir / name): vec for name, vec in raw_encodings.items()}


def _load_cnn() -> object | None:
    """Lazy-load imagededup CNN, returning None if unavailable."""
    global _CNN_INSTANCE, _IMPORT_FAILED  # noqa: PLW0603
    if _IMPORT_FAILED:
        return None
    if _CNN_INSTANCE is not None:
        return _CNN_INSTANCE
    try:
        from imagededup.methods import CNN  # noqa: PLC0415

        _CNN_INSTANCE = CNN()
        return _CNN_INSTANCE
    except ImportError:
        _IMPORT_FAILED = True
        logger.warning("imagededup not available — skipping CNN duplicate detection")
        return None


def _unload_cnn() -> None:
    """Drop the cached CNN encoder to free memory after Stage 1 completes."""
    global _CNN_INSTANCE  # noqa: PLW0603
    _CNN_INSTANCE = None
    gc.collect()


def _run_cnn_encoding(cnn: object, image_dir: Path) -> tuple[dict, dict]:
    """Run CNN encode and find_duplicates; return (raw_encodings, duplicates_map)."""
    from cull.io_silence import _silence_stdio  # noqa: PLC0415

    threshold = BLUR_CNN_SIMILARITY_EXACT
    with _silence_stdio():
        raw_encodings = cnn.encode_images(image_dir=str(image_dir), recursive=True)  # type: ignore[attr-defined]
        duplicates_map = cnn.find_duplicates(  # type: ignore[attr-defined]
            encoding_map=raw_encodings,
            min_similarity_threshold=threshold,
            scores=False,
        )
    return raw_encodings, duplicates_map


def find_duplicates(image_dir: Path) -> DuplicateResult:
    """Detect duplicate images in image_dir using CNN embeddings."""
    from cull.io_silence import _silence_stdio  # noqa: PLC0415

    with _silence_stdio():
        cnn = _load_cnn()
    if cnn is None:
        return DuplicateResult()
    logger.info(
        "Running CNN duplicate detection (threshold=%.2f)",
        BLUR_CNN_SIMILARITY_EXACT,
    )
    try:
        raw_encodings, duplicates_map = _run_cnn_encoding(cnn, image_dir)
    except (RuntimeError, ValueError, OSError, TypeError):
        logger.exception("CNN duplicate detection failed for %s", image_dir)
        return DuplicateResult()
    groups = _build_duplicate_groups(duplicates_map, image_dir)
    encodings = _build_encodings_map(raw_encodings, image_dir)
    logger.info("Found %d duplicate groups", len(groups))
    return DuplicateResult(duplicate_groups=groups, encodings=encodings)
