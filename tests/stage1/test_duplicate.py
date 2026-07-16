"""Regression test — CNN duplicate detection must not re-ingest pipeline output dirs."""

from __future__ import annotations

from cull.router import CURATED_DIR, REVIEW_DIR
from cull.stage1.duplicate import _exclude_pipeline_output, _is_pipeline_output_name


def test_is_pipeline_output_name_flags_review_subtree() -> None:
    """A relative name nested under _review must be flagged as pipeline output."""
    assert _is_pipeline_output_name(f"{REVIEW_DIR}/_rejected/old.jpg") is True


def test_is_pipeline_output_name_flags_curated_subtree() -> None:
    """A relative name nested under _curated must be flagged as pipeline output."""
    assert _is_pipeline_output_name(f"{CURATED_DIR}/_selects/old.jpg") is True


def test_is_pipeline_output_name_allows_fresh_photos() -> None:
    """A relative name outside _review/_curated must not be flagged."""
    assert _is_pipeline_output_name("session_a/new_photo.jpg") is False
    assert _is_pipeline_output_name("new_photo.jpg") is False


def test_exclude_pipeline_output_drops_review_and_curated_entries() -> None:
    """Filtering the raw encoding map must keep only fresh, un-routed photos."""
    raw_encodings = {
        "new_photo_01.jpg": [0.1, 0.2],
        f"{REVIEW_DIR}/_duplicates/old_duplicate.jpg": [0.3, 0.4],
        f"{CURATED_DIR}/_selects/old_select.jpg": [0.5, 0.6],
        "session_a/new_photo_02.jpg": [0.7, 0.8],
    }
    filtered = _exclude_pipeline_output(raw_encodings)
    assert filtered == {
        "new_photo_01.jpg": [0.1, 0.2],
        "session_a/new_photo_02.jpg": [0.7, 0.8],
    }
