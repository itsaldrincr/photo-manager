"""Regression test — source scan must not re-ingest prior pipeline output dirs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from cull.pipeline import _scan_with_dashboard
from cull.router import CURATED_DIR, REVIEW_DIR

PHOTO_BYTES: bytes = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


def _write_photo(path: Path) -> None:
    """Create a minimal fake JPEG file at path, making parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PHOTO_BYTES)


def _make_already_culled_tree(tmp_path: Path) -> list[Path]:
    """Build fresh source photos plus a pre-existing _review/_curated subtree."""
    fresh = [tmp_path / "new_photo_01.jpg", tmp_path / "new_photo_02.jpg"]
    for path in fresh:
        _write_photo(path)
    _write_photo(tmp_path / REVIEW_DIR / "_rejected" / "old_rejected.jpg")
    _write_photo(tmp_path / REVIEW_DIR / "_duplicates" / "old_duplicate.jpg")
    _write_photo(tmp_path / CURATED_DIR / "_selects" / "old_select.jpg")
    return fresh


def test_scan_excludes_prior_review_and_curated_dirs(tmp_path: Path) -> None:
    """Re-running a scan on an already-culled folder must skip _review/_curated."""
    fresh = _make_already_culled_tree(tmp_path)
    found = _scan_with_dashboard(tmp_path, MagicMock())
    assert found == sorted(fresh)


def test_scan_ignores_nested_review_dirs_at_any_depth(tmp_path: Path) -> None:
    """A _review dir nested under a subfolder must also be excluded."""
    fresh = tmp_path / "session_a" / "keeper.jpg"
    _write_photo(fresh)
    _write_photo(tmp_path / "session_a" / REVIEW_DIR / "_uncertain" / "old.jpg")
    found = _scan_with_dashboard(tmp_path, MagicMock())
    assert found == [fresh]
