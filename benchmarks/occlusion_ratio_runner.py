"""Subprocess worker: production occlusion ratios for a photo corpus.

Runs the REAL production path — cull.stage2.portrait.detect_faces followed by
detect_occlusion on the full-resolution decode — one MediaPipe model resident,
serial. Output ratios are threshold-independent, so any candidate
PORTRAIT_FACE_OCCLUSION_MIN can be swept over them afterwards without
re-running inference.

Usage:
    python3 benchmarks/occlusion_ratio_runner.py <manifest.json> <output.json>
    (manifest: {"image_paths": [...]}, same shape as weaklabel manifests)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
from pydantic import BaseModel

from weaklabel_models import WeaklabelManifest

logger = logging.getLogger(__name__)


class RatioReading(BaseModel):
    """One photo's production occlusion ratio (None when no face detected)."""

    name: str
    path: str
    has_face: bool
    occlusion_ratio: float | None = None


class RatioRunOutput(BaseModel):
    """Full output of one occlusion_ratio_runner.py invocation."""

    readings: list[RatioReading] = []


def _ratio_one(image_path: Path) -> RatioReading:
    """Detect the primary face and compute its production occlusion ratio."""
    from cull.stage2.portrait import detect_faces, detect_occlusion  # noqa: PLC0415

    base = {"name": image_path.name, "path": str(image_path)}
    image = cv2.imread(str(image_path))
    if image is None:
        return RatioReading(**base, has_face=False)
    landmark_sets = detect_faces(image)
    if not landmark_sets:
        return RatioReading(**base, has_face=False)
    ratio = detect_occlusion(image, landmark_sets[0])
    return RatioReading(**base, has_face=True, occlusion_ratio=ratio)


def _run(manifest_path: Path) -> RatioRunOutput:
    """Compute ratios for every manifest image serially."""
    from cull.env_bootstrap import bootstrap_default  # noqa: PLC0415

    bootstrap_default()
    manifest = WeaklabelManifest.model_validate_json(manifest_path.read_text())
    readings: list[RatioReading] = []
    for index, image_path in enumerate(manifest.image_paths):
        readings.append(_ratio_one(image_path))
        logger.info("[%d/%d] %s", index + 1, len(manifest.image_paths), image_path.name)
    return RatioRunOutput(readings=readings)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if len(sys.argv) != 3:
        raise SystemExit("usage: occlusion_ratio_runner.py <manifest.json> <output.json>")
    output = _run(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(output.model_dump_json(indent=2))
    faces = sum(1 for r in output.readings if r.has_face)
    logger.info("done: %d readings, %d with faces", len(output.readings), faces)


if __name__ == "__main__":
    main()
