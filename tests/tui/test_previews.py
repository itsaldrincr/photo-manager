"""Preview cache: EXIF orientation, aspect-correct sizing, disk reuse, pruning, ordering."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from cull.tui import previews
from cull.tui.previews import PixelBox, PreviewJob, PreviewScheduler, make_spec

from tests.tui._helpers import JpegSpec, write_jpeg

# EXIF orientation 6: stored landscape, displayed rotated 90 degrees clockwise.
ORIENTATION_ROTATE_90_CW: int = 6


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    """A landscape-stored frame tagged orientation 6 previews as portrait."""
    source = write_jpeg(JpegSpec(path=tmp_path / "rotated.jpg", size=(300, 200), orientation=ORIENTATION_ROTATE_90_CW))
    preview = previews.render_preview(make_spec(source, PixelBox(width=1000, height=1000)))
    assert (preview.width, preview.height) == (200, 300)
    with Image.open(preview.path) as image:
        assert image.size == (200, 300)


def test_preview_fits_box_preserving_aspect(tmp_path: Path) -> None:
    """A 3:2 frame in a 640x640 box renders at 640x427."""
    source = write_jpeg(JpegSpec(path=tmp_path / "wide.jpg", size=(3000, 2000)))
    preview = previews.render_preview(make_spec(source, PixelBox(width=640, height=640)))
    assert (preview.width, preview.height) == (640, 427)


def test_second_render_reuses_disk_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-opening a session reads the cached PNG instead of decoding the JPEG again."""
    source = write_jpeg(JpegSpec(path=tmp_path / "a.jpg"))
    spec = make_spec(source, PixelBox(width=256, height=256))
    first = previews.render_preview(spec)

    def _no_decode(_spec: object) -> None:
        raise AssertionError("decoded twice")

    monkeypatch.setattr(previews, "decode_fitted", _no_decode)
    assert previews.render_preview(spec) == first


def test_cache_key_changes_when_file_changes(tmp_path: Path) -> None:
    """A new mtime means a new cache entry (the photo was edited)."""
    source = write_jpeg(JpegSpec(path=tmp_path / "a.jpg"))
    spec = make_spec(source, PixelBox(width=256, height=256))
    before = previews.cache_path(spec, source.stat())
    os.utime(source, (1, 1))
    assert previews.cache_path(spec, source.stat()) != before


def test_prune_removes_least_recently_used(preview_cache: Path) -> None:
    """Pruning deletes the oldest files first until under budget."""
    for index, name in enumerate(("old", "mid", "new")):
        entry = preview_cache / f"{name}.png"
        entry.write_bytes(b"x" * 100)
        os.utime(entry, (index + 1, index + 1))
    assert previews.prune_cache(150) == 2
    assert [p.name for p in preview_cache.glob("*.png")] == ["new.png"]


def test_scheduler_takes_nearest_first_and_reserves_urgent_worker(tmp_path: Path) -> None:
    """The urgent-only worker skips background jobs; others take the queue head."""
    scheduler = PreviewScheduler(lambda spec, preview: None)
    urgent = PreviewJob(spec=make_spec(tmp_path / "u.jpg", PixelBox(width=64, height=64)), is_urgent=True)
    background = PreviewJob(spec=make_spec(tmp_path / "b.jpg", PixelBox(width=64, height=64)))
    scheduler.prioritise([background, urgent])
    assert scheduler._take(True) == urgent
    assert scheduler._take(True) is None
    assert scheduler._take(False) == background


def test_box_is_bucketed_so_small_resizes_share_previews(tmp_path: Path) -> None:
    """A one-cell resize keeps the same spec (and cache entry)."""
    assert make_spec(tmp_path / "a.jpg", PixelBox(width=1001, height=700)) == make_spec(
        tmp_path / "a.jpg", PixelBox(width=1020, height=680),
    )
