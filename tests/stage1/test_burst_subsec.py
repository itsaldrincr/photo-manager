"""Burst grouping honours EXIF sub-second time and widens the gap without it."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cull.config import CullConfig
from cull.stage1 import burst
from cull.stage1.burst import _BurstInput, _dhash_distance, detect_bursts

EXIF_IFD: int = 0x8769
DATETIME_ORIGINAL: int = 0x9003
SUBSEC_TIME_ORIGINAL: int = 0x9291
LARGE_PX: tuple[int, int] = (4000, 3000)
REDUCED_DECODE_MAX_PX: int = 1000


def _jpeg(path: Path, when: str) -> Path:
    """Write a JPEG stamped "YYYY:MM:DD HH:MM:SS[.subsec]" in EXIF.

    The part after the dot goes to SubSecTimeOriginal; without it the tag is absent.
    """
    whole, _, subsec = when.partition(".")
    ifd = {DATETIME_ORIGINAL: whole}
    if subsec:
        ifd[SUBSEC_TIME_ORIGINAL] = subsec
    exif = Image.Exif()
    exif[EXIF_IFD] = ifd
    Image.new("RGB", (32, 32), (90, 90, 90)).save(path, exif=exif.tobytes())
    return path


def _groups(paths: list[Path]) -> list[list[Path]]:
    """Run detect_bursts at the default 0.5 s gap with every pair visually alike."""
    with patch.object(burst, "_dhash_distance", return_value=0):
        return detect_bursts(_BurstInput(image_paths=paths, config=CullConfig())).groups


def test_no_subsec_frames_one_second_apart_group(tmp_path: Path) -> None:
    """Frames stamped one whole second apart with no sub-second tag form one burst."""
    a = _jpeg(tmp_path / "a.jpg", "2024:06:01 12:00:00")
    b = _jpeg(tmp_path / "b.jpg", "2024:06:01 12:00:01")
    assert len(_groups([a, b])) == 1


def test_subsec_frames_across_a_second_boundary_group(tmp_path: Path) -> None:
    """12:00:00.90 and 12:00:01.10 are 0.2 s apart and must group."""
    a = _jpeg(tmp_path / "a.jpg", "2024:06:01 12:00:00.90")
    b = _jpeg(tmp_path / "b.jpg", "2024:06:01 12:00:01.10")
    assert len(_groups([a, b])) == 1


def test_subsec_frames_far_apart_do_not_group(tmp_path: Path) -> None:
    """12:00:00.10 and 12:00:01.90 are 1.8 s apart and must not group at 0.5 s."""
    a = _jpeg(tmp_path / "a.jpg", "2024:06:01 12:00:00.10")
    b = _jpeg(tmp_path / "b.jpg", "2024:06:01 12:00:01.90")
    assert _groups([a, b]) == []


def test_dhash_uses_reduced_decode(tmp_path: Path) -> None:
    """dHash only needs 9x8 pixels, so a large JPEG must not be decoded at full size."""
    path = tmp_path / "large.jpg"
    Image.new("RGB", LARGE_PX, (10, 200, 30)).save(path)
    sizes: list[tuple[int, int]] = []

    def _record(img: Image.Image) -> int:
        sizes.append(img.size)
        return 0

    with patch.object(burst.imagehash, "dhash", side_effect=_record):
        _dhash_distance(path, path)
    assert sizes and max(max(s) for s in sizes) <= REDUCED_DECODE_MAX_PX
