"""Subprocess worker: candidate occlusion-v2 signals for a photo corpus.

The production texture-variance ratio (cull.stage2.portrait.detect_occlusion)
is a no-op on real occluders (occlusion_recal_report.md). This worker extracts
three cheaper-than-VLM candidate signals per face, all computable from the same
full-res BGR image + MediaPipe landmarks the production detector consumes, so a
winner can later be dropped into detect_occlusion unchanged:

  1. blendshape activation collapse — std/mean of FaceLandmarker blendshape
     scores (occluded regions produce flat activations),
  2. skin-outside-hull fraction — skin-colored pixels inside the face box but
     outside the landmark convex hull (a hand reaching over the face),
  3. boundary edge density — strong edges crossing the face-hull boundary band
     (an occluder edge cuts across; an open face's boundary is smooth).

Also records the existing texture ratio for reference. One MediaPipe model
resident, serial. Signals are threshold-independent (raw values), so operating
points are swept afterwards without re-running inference.

Usage:
    python3 benchmarks/occlusion_v2_signals.py <manifest.json> <output.json>
    (manifest: {"image_paths": [...]}, same shape as weaklabel manifests)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from weaklabel_models import WeaklabelManifest

logger = logging.getLogger(__name__)

_NEUTRAL_BLENDSHAPE = "_neutral"
_SKIN_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
_SKIN_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)
_CANNY_LOW = 50
_CANNY_HIGH = 150
_HULL_BAND_PX = 4
_TOTAL_LANDMARK_COUNT = 468


class SignalReading(BaseModel):
    """One photo's candidate occlusion signals (None when no face detected)."""

    name: str
    path: str
    has_face: bool
    blendshape_std: float | None = None
    blendshape_mean: float | None = None
    skin_outside_hull_frac: float | None = None
    boundary_edge_density: float | None = None
    texture_ratio: float | None = None


class SignalRunOutput(BaseModel):
    """Full output of one occlusion_v2_signals.py invocation."""

    readings: list[SignalReading] = []


class _Face(BaseModel):
    """A detected face: full BGR image + its normalized landmark list."""

    model_config = {"arbitrary_types_allowed": True}

    image: Any
    landmarks: list


def _build_face_landmarker() -> Any:
    """Construct a blendshapes-enabled FaceLandmarker from the shared model cache."""
    import mediapipe as mp  # noqa: PLC0415
    from cull.config import (  # noqa: PLC0415
        FACE_LANDMARKER_FILENAME,
        PORTRAIT_FACE_DETECTION_CONFIDENCE_MIN,
        ModelCacheConfig,
    )

    cache = ModelCacheConfig.from_env()
    model_path = cache.mediapipe_dir / FACE_LANDMARKER_FILENAME
    base_opts = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=PORTRAIT_FACE_DETECTION_CONFIDENCE_MIN,
        output_face_blendshapes=True,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def _landmark_pixels(face: _Face) -> np.ndarray:
    """Return an (N, 2) int array of landmark pixel coordinates."""
    h, w = face.image.shape[:2]
    pts = [
        (int(lm.x * w), int(lm.y * h))
        for lm in face.landmarks[:_TOTAL_LANDMARK_COUNT]
    ]
    return np.array(pts, dtype=np.int32)


def _hull_mask(face: _Face) -> np.ndarray:
    """Return a filled convex-hull mask (uint8, 0/255) over the face landmarks."""
    h, w = face.image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(_landmark_pixels(face))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _face_bbox(pts: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return the clamped pixel bbox (x1, y1, x2, y2) enclosing landmark points."""
    h, w = shape
    x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
    x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _skin_mask(image: np.ndarray) -> np.ndarray:
    """Return a YCrCb skin-probability mask (uint8, 0/255)."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, _SKIN_YCRCB_LOWER, _SKIN_YCRCB_UPPER)


def _skin_outside_hull_frac(face: _Face) -> float:
    """Return the fraction of face-box pixels that are skin but outside the hull."""
    pts = _landmark_pixels(face)
    x1, y1, x2, y2 = _face_bbox(pts, face.image.shape[:2])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    outside = cv2.bitwise_and(_skin_mask(face.image), cv2.bitwise_not(_hull_mask(face)))
    box_outside = outside[y1:y2, x1:x2]
    return float(np.count_nonzero(box_outside)) / float(box_outside.size)


def _boundary_edge_density(face: _Face) -> float:
    """Return the density of Canny edges within a band around the face-hull boundary."""
    gray = cv2.cvtColor(face.image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH)
    hull = _hull_mask(face)
    kernel = np.ones((_HULL_BAND_PX * 2 + 1, _HULL_BAND_PX * 2 + 1), np.uint8)
    band = cv2.subtract(cv2.dilate(hull, kernel), cv2.erode(hull, kernel))
    band_area = float(np.count_nonzero(band))
    if band_area < 1.0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(edges, band))) / band_area


def _blendshape_stats(scores: list[float]) -> tuple[float, float]:
    """Return (std, mean) of non-neutral blendshape scores; (0, 0) if empty."""
    if not scores:
        return 0.0, 0.0
    arr = np.array(scores, dtype=np.float64)
    return float(arr.std()), float(arr.mean())


def _blendshape_scores(detection: Any) -> list[float]:
    """Return non-neutral blendshape scores from a FaceLandmarker result."""
    if not detection.face_blendshapes:
        return []
    return [
        cat.score
        for cat in detection.face_blendshapes[0]
        if cat.category_name != _NEUTRAL_BLENDSHAPE
    ]


def _signals_for_face(face: _Face, scores: list[float]) -> dict[str, float]:
    """Compute every candidate signal for one detected face."""
    from cull.stage2.portrait import detect_occlusion  # noqa: PLC0415

    std, mean = _blendshape_stats(scores)
    return {
        "blendshape_std": std,
        "blendshape_mean": mean,
        "skin_outside_hull_frac": _skin_outside_hull_frac(face),
        "boundary_edge_density": _boundary_edge_density(face),
        "texture_ratio": detect_occlusion(face.image, face.landmarks),
    }


def _signal_one(image_path: Path, landmarker: Any) -> SignalReading:
    """Detect the primary face and compute every candidate occlusion signal."""
    base = {"name": image_path.name, "path": str(image_path)}
    image = cv2.imread(str(image_path))
    if image is None:
        return SignalReading(**base, has_face=False)
    import mediapipe as mp  # noqa: PLC0415

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not detection.face_landmarks:
        return SignalReading(**base, has_face=False)
    face = _Face(image=image, landmarks=list(detection.face_landmarks[0]))
    signals = _signals_for_face(face, _blendshape_scores(detection))
    return SignalReading(**base, has_face=True, **signals)


def _run(manifest_path: Path) -> SignalRunOutput:
    """Compute candidate signals for every manifest image serially."""
    from cull.env_bootstrap import bootstrap_default  # noqa: PLC0415

    bootstrap_default()
    manifest = WeaklabelManifest.model_validate_json(manifest_path.read_text())
    landmarker = _build_face_landmarker()
    readings: list[SignalReading] = []
    for index, image_path in enumerate(manifest.image_paths):
        readings.append(_signal_one(image_path, landmarker))
        logger.info("[%d/%d] %s", index + 1, len(manifest.image_paths), image_path.name)
    landmarker.close()
    return SignalRunOutput(readings=readings)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if len(sys.argv) != 3:
        raise SystemExit("usage: occlusion_v2_signals.py <manifest.json> <output.json>")
    output = _run(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(output.model_dump_json(indent=2))
    faces = sum(1 for r in output.readings if r.has_face)
    logger.info("done: %d readings, %d with faces", len(output.readings), faces)


if __name__ == "__main__":
    main()
