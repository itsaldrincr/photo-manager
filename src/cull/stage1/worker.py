"""Stage 1 assessment worker: picklable single-image pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import exifread
import numpy as np
from pydantic import BaseModel, ConfigDict

from cull.config import CullConfig
from cull.models import ExifSummary
from cull.stage1.blur import (
    BlurResult,
    _ArrayAssessInput,
    _resize_to_long_edge,
    assess_blur_from_array,
)
from cull.stage1.exposure import ExposureResult, assess_exposure_from_array
from cull.stage1.geometry import GeometryResult, assess_geometry_from_array
from cull.stage1.noise import NoiseResult, assess_noise_from_array

logger = logging.getLogger(__name__)

EXIF_TAG_DATETIME_ORIGINAL: str = "EXIF DateTimeOriginal"
EXIF_TAG_ISO: str = "EXIF ISOSpeedRatings"
EXIF_TAG_SHUTTER: str = "EXIF ExposureTime"
EXIF_TAG_APERTURE: str = "EXIF FNumber"
EXIF_TAG_FOCAL_LENGTH: str = "EXIF FocalLength"
EXIF_DATETIME_FORMAT: str = "%Y:%m:%d %H:%M:%S"


@dataclass(frozen=True)
class Stage1WorkerResult:
    """Complete Stage 1 assessment result for one image."""

    image_path: Path
    blur: BlurResult
    exposure: ExposureResult
    noise: NoiseResult
    geometry: GeometryResult
    capture_time: datetime | None = None
    exif: ExifSummary | None = None


class _DecodedImage(BaseModel):
    """Two decodes of a photo, shared across all four Stage 1 assessors.

    blur/exposure/noise operate on one shared resized-BGR array (three
    redundant cv2.imread + resize calls collapse into one). geometry runs
    on full-resolution grayscale, matching its original behaviour, which
    never resized. cv2's IMREAD_GRAYSCALE decode path and cv2.cvtColor
    applied to an already-decoded BGR image round differently by up to
    1 level per pixel, which is enough to nudge RANSAC's near-zero tilt
    selection — so geometry keeps its own IMREAD_GRAYSCALE read rather
    than deriving grayscale from the color decode. This still collapses
    the four cv2.imread calls per photo down to two.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resized_bgr: np.ndarray
    full_gray: np.ndarray


def _parse_capture_time(tags: dict[str, Any]) -> datetime | None:
    """Parse EXIF DateTimeOriginal into a datetime, or None when absent/malformed."""
    raw = tags.get(EXIF_TAG_DATETIME_ORIGINAL)
    if raw is None:
        return None
    try:
        return datetime.strptime(str(raw), EXIF_DATETIME_FORMAT)
    except ValueError:
        return None


def _parse_int_tag(tags: dict[str, Any], key: str) -> int | None:
    """Parse an EXIF tag as an int, or None when absent/malformed."""
    raw = tags.get(key)
    if raw is None:
        return None
    try:
        return int(str(raw))
    except ValueError:
        return None


def _parse_focal_length(tags: dict[str, Any]) -> float | None:
    """Parse EXIF FocalLength as millimetres, or None when absent/malformed."""
    raw = tags.get(EXIF_TAG_FOCAL_LENGTH)
    if raw is None:
        return None
    try:
        return float(Fraction(str(raw)))
    except (ValueError, ZeroDivisionError):
        return None


def _build_exif_summary(tags: dict[str, Any]) -> ExifSummary | None:
    """Build an ExifSummary from parsed tags; None when every field is missing."""
    summary = ExifSummary(
        iso=_parse_int_tag(tags, EXIF_TAG_ISO),
        shutter=str(tags[EXIF_TAG_SHUTTER]) if EXIF_TAG_SHUTTER in tags else None,
        aperture=str(tags[EXIF_TAG_APERTURE]) if EXIF_TAG_APERTURE in tags else None,
        focal_length_mm=_parse_focal_length(tags),
    )
    if summary.model_dump(exclude_none=True):
        return summary
    return None


def _read_exif_summary(image_path: Path) -> tuple[datetime | None, ExifSummary | None]:
    """Read capture time and camera settings from EXIF in a single file pass."""
    try:
        with open(image_path, "rb") as fh:
            tags = exifread.process_file(fh, details=False, extract_thumbnail=False)
    except OSError:
        logger.debug("EXIF read failed for %s", image_path)
        return None, None
    return _parse_capture_time(tags), _build_exif_summary(tags)


def _decode_once(image_path: Path) -> _DecodedImage:
    """Read the image from disk once per required variant (color + grayscale)."""
    raw_bgr = cv2.imread(str(image_path))
    full_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw_bgr is None or full_gray is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return _DecodedImage(
        resized_bgr=_resize_to_long_edge(raw_bgr),
        full_gray=full_gray,
    )


def assess_one(image_path: Path, config: CullConfig) -> Stage1WorkerResult:
    """Run Stage 1 assessment on a single image, decoding it exactly once."""
    decoded = _decode_once(image_path)
    blur_in = _ArrayAssessInput(image=decoded.resized_bgr, path=image_path)
    blur_result = assess_blur_from_array(blur_in, config)
    exposure_result = assess_exposure_from_array(decoded.resized_bgr)
    noise_result = assess_noise_from_array(decoded.resized_bgr)
    geometry_result = assess_geometry_from_array(decoded.full_gray, image_path)
    capture_time, exif = _read_exif_summary(image_path)
    return Stage1WorkerResult(
        image_path=image_path,
        blur=blur_result,
        exposure=exposure_result,
        noise=noise_result,
        geometry=geometry_result,
        capture_time=capture_time,
        exif=exif,
    )
