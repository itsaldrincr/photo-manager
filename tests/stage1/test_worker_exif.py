"""Tests for cull.stage1.worker EXIF extraction — capture_time and ExifSummary."""

from __future__ import annotations

from pathlib import Path

import pytest

from cull.models import ExifSummary
from cull.stage1.worker import (
    _build_exif_summary,
    _parse_capture_time,
    _parse_focal_length,
    _parse_int_tag,
    _read_exif_summary,
    EXIF_TAG_APERTURE,
    EXIF_TAG_DATETIME_ORIGINAL,
    EXIF_TAG_FOCAL_LENGTH,
    EXIF_TAG_ISO,
    EXIF_TAG_SHUTTER,
)

EASTER_VIGIL_FIXTURE: Path = (
    Path(__file__).resolve().parents[2] / "fixtures" / "easter_vigil" / "DSCF0006.JPG"
)


# ---------------------------------------------------------------------------
# Tag-parsing unit tests (hand-built tags dicts, no real file needed)
# ---------------------------------------------------------------------------


def test_parse_capture_time_valid_tag() -> None:
    """A well-formed DateTimeOriginal tag parses into the expected datetime."""
    tags = {EXIF_TAG_DATETIME_ORIGINAL: "2024:11:06 06:54:38"}
    parsed = _parse_capture_time(tags)
    assert parsed is not None
    assert parsed.isoformat() == "2024-11-06T06:54:38"


def test_parse_capture_time_missing_tag_returns_none() -> None:
    """A tags dict without DateTimeOriginal yields None rather than raising."""
    assert _parse_capture_time({}) is None


def test_parse_capture_time_malformed_tag_returns_none() -> None:
    """A malformed DateTimeOriginal string yields None rather than raising."""
    tags = {EXIF_TAG_DATETIME_ORIGINAL: "not-a-date"}
    assert _parse_capture_time(tags) is None


def test_parse_int_tag_valid() -> None:
    """A numeric ISO tag parses to an int."""
    assert _parse_int_tag({EXIF_TAG_ISO: "3200"}, EXIF_TAG_ISO) == 3200


def test_parse_int_tag_missing_returns_none() -> None:
    """A missing ISO tag returns None."""
    assert _parse_int_tag({}, EXIF_TAG_ISO) is None


def test_parse_focal_length_simple_ratio() -> None:
    """A simplified 'N' focal length string parses as a plain float."""
    assert _parse_focal_length({EXIF_TAG_FOCAL_LENGTH: "35"}) == pytest.approx(35.0)


def test_parse_focal_length_fraction_string() -> None:
    """A 'num/den' focal length string parses via Fraction."""
    assert _parse_focal_length({EXIF_TAG_FOCAL_LENGTH: "50/1"}) == pytest.approx(50.0)


def test_parse_focal_length_zero_denominator_returns_none() -> None:
    """A malformed '0/0' ratio (seen on real cameras for missing aperture) yields None."""
    assert _parse_focal_length({EXIF_TAG_FOCAL_LENGTH: "0/0"}) is None


def test_build_exif_summary_all_fields_present() -> None:
    """A tags dict with all four EXIF anomaly keys builds a fully-populated ExifSummary."""
    tags = {
        EXIF_TAG_ISO: "1000",
        EXIF_TAG_SHUTTER: "1/280",
        EXIF_TAG_APERTURE: "9/2",
        EXIF_TAG_FOCAL_LENGTH: "16",
    }
    summary = _build_exif_summary(tags)
    assert summary == ExifSummary(iso=1000, shutter="1/280", aperture="9/2", focal_length_mm=16.0)


def test_build_exif_summary_all_fields_missing_returns_none() -> None:
    """A tags dict with none of the anomaly keys returns None, not an all-None summary."""
    assert _build_exif_summary({}) is None


# ---------------------------------------------------------------------------
# End-to-end: real EXIF read from a fixture photo (skips if fixture absent)
# ---------------------------------------------------------------------------


def _require_fixture() -> None:
    if not EASTER_VIGIL_FIXTURE.exists():
        pytest.skip(f"fixture corpus not present locally: {EASTER_VIGIL_FIXTURE}")


def test_read_exif_summary_real_photo() -> None:
    """_read_exif_summary extracts real capture_time and camera settings from a JPEG."""
    _require_fixture()
    capture_time, exif = _read_exif_summary(EASTER_VIGIL_FIXTURE)
    assert capture_time is not None
    assert exif is not None
    assert exif.iso is not None
    assert exif.shutter is not None
    assert exif.focal_length_mm is not None
