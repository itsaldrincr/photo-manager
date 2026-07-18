"""Unit tests for Kitty upload-identity dedup helpers in photo_view.

These exercise the pure identity-cache/hash logic directly with a fake
``_emit_kitty_delete`` capturing calls, with no Textual app or real
terminal involved.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from cull.tui import photo_view as pv

FAKE_HASH_A: str = "hash-a"
FAKE_HASH_B: str = "hash-b"


@pytest.fixture(autouse=True)
def _reset_identity_cache() -> Iterator[None]:
    """Clear the module-level upload-identity cache before and after each test."""
    pv._upload_identity_cache.clear()
    yield
    pv._upload_identity_cache.clear()


def test_reserve_image_id_reuses_existing_hash() -> None:
    """A repeated content hash returns the same image id without re-upload."""
    first = pv._reserve_image_id(FAKE_HASH_A)
    second = pv._reserve_image_id(FAKE_HASH_A)
    assert first.is_new_upload is True
    assert second.is_new_upload is False
    assert first.image_id == second.image_id


def test_reserve_image_id_distinct_hash_gets_new_id() -> None:
    """Different content hashes reserve distinct image ids."""
    first = pv._reserve_image_id(FAKE_HASH_A)
    second = pv._reserve_image_id(FAKE_HASH_B)
    assert first.image_id != second.image_id
    assert second.is_new_upload is True


def test_reserve_image_id_eviction_emits_delete() -> None:
    """Evicting the oldest identity-cache entry deletes its terminal image."""
    deletes: list[int] = []
    original = pv._emit_kitty_delete
    pv._emit_kitty_delete = lambda image_id: deletes.append(image_id)  # type: ignore[assignment]
    try:
        first = pv._reserve_image_id("hash-0")
        for i in range(1, pv.CACHE_MAX_ENTRIES):
            pv._reserve_image_id(f"hash-{i}")
        assert deletes == []
        pv._reserve_image_id("hash-overflow")
        assert deletes == [first.image_id]
    finally:
        pv._emit_kitty_delete = original


def test_build_upload_sequence_if_new_skips_reused_identity() -> None:
    """No APC upload sequence is built when the identity was already known."""
    identity_new = pv._ImageIdentity(image_id=1, is_new_upload=True)
    identity_reused = pv._ImageIdentity(image_id=1, is_new_upload=False)
    assert pv._build_upload_sequence_if_new(b"png-bytes", identity_new) != ""
    assert pv._build_upload_sequence_if_new(b"png-bytes", identity_reused) == ""


def test_invalidate_image_id_removes_stale_mapping() -> None:
    """Invalidating an image id lets its hash be treated as new again."""
    identity = pv._reserve_image_id(FAKE_HASH_A)
    pv._invalidate_image_id(identity.image_id)
    again = pv._reserve_image_id(FAKE_HASH_A)
    assert again.is_new_upload is True


def test_hash_png_is_deterministic_and_content_sensitive() -> None:
    """Identical bytes hash identically; different bytes hash differently."""
    assert pv._hash_png(b"abc") == pv._hash_png(b"abc")
    assert pv._hash_png(b"abc") != pv._hash_png(b"abd")
