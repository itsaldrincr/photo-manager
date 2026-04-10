"""Stage 1 assessment worker: picklable single-image pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from cull.config import CullConfig
from cull.stage1.blur import BlurResult, assess_blur
from cull.stage1.exposure import ExposureResult, assess_exposure
from cull.stage1.geometry import GeometryResult, assess_geometry
from cull.stage1.noise import NoiseResult, assess_noise

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage1WorkerResult:
    """Complete Stage 1 assessment result for one image."""

    image_path: Path
    blur: BlurResult
    exposure: ExposureResult
    noise: NoiseResult
    geometry: GeometryResult


def assess_one(image_path: Path, config: CullConfig) -> Stage1WorkerResult:
    """Run Stage 1 assessment on a single image."""
    blur_result = assess_blur(image_path, config)
    exposure_result = assess_exposure(image_path)
    noise_result = assess_noise(image_path)
    geometry_result = assess_geometry(image_path)
    return Stage1WorkerResult(
        image_path=image_path,
        blur=blur_result,
        exposure=exposure_result,
        noise=noise_result,
        geometry=geometry_result,
    )
