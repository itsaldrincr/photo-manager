"""Deterministic mock scorer functions for pytest — NO real ML models loaded.

All mock scores fall in MOCK_SCORE_BAND ([0.900, 0.999]), which is intentionally
above the typical real-model distribution (~0.2–0.7) so any mock output is
visually unmistakable from real-model output.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from cull.models import BlurScores, GeometryScore
from cull.stage1.blur import BlurResult
from cull.stage1.exposure import (
    ClippingResult,
    ColorCastResult,
    ExposureResult,
)
from cull.stage1.geometry import GeometryResult
from cull.stage1.noise import NoiseResult

MOCK_SCORE_BAND: tuple[float, float] = (0.900, 0.999)
MOCK_SCORE_RANGE: int = 99000
MOCK_SCORE_BASE: float = 0.900
MOCK_SCORE_DIVISOR: float = 1000000.0
PIL_BYTES_SLICE: int = 1024
FINGERPRINT_MODULUS: int = 2**31
FINGERPRINT_SCALE: float = 1e6


def _tensor_fingerprint(tensor: torch.Tensor) -> int:
    """Return deterministic int derived from tensor content."""
    return int((tensor.sum().item() * FINGERPRINT_SCALE) % FINGERPRINT_MODULUS)


def _pil_fingerprint(pil_image: object) -> int:
    """Return deterministic int derived from first PIL_BYTES_SLICE bytes of image."""
    raw_bytes = pil_image.tobytes()[:PIL_BYTES_SLICE]  # type: ignore[union-attr]
    digest = hashlib.sha256(raw_bytes).digest()
    return int.from_bytes(digest[:4], "big") % FINGERPRINT_MODULUS


def _salt_fingerprint(salt: str) -> int:
    """Return deterministic int derived from salt string (process-stable)."""
    digest = hashlib.sha256(salt.encode()).digest()
    return int.from_bytes(digest[:4], "big") % FINGERPRINT_MODULUS


def _int_to_mock_score(fingerprint: int, salt: str) -> float:
    """Map (fingerprint, salt) to a float in MOCK_SCORE_BAND."""
    return MOCK_SCORE_BASE + ((fingerprint ^ _salt_fingerprint(salt)) % MOCK_SCORE_RANGE) / MOCK_SCORE_DIVISOR


def mock_score_topiq_batch(tensor_batch: torch.Tensor, device: str) -> list[float]:
    """Return deterministic mock TOPIQ scores for each image in the batch."""
    count = tensor_batch.shape[0]
    return [
        _int_to_mock_score(_tensor_fingerprint(tensor_batch[i]), "topiq")
        for i in range(count)
    ]


def mock_score_clipiqa_batch(tensor_batch: torch.Tensor, device: str) -> list[float]:
    """Return deterministic mock CLIP-IQA+ scores for each image in the batch."""
    count = tensor_batch.shape[0]
    return [
        _int_to_mock_score(_tensor_fingerprint(tensor_batch[i]), "clipiqa")
        for i in range(count)
    ]


def mock_score_aesthetic_batch(
    pil_images: list | None, embeddings: object = None
) -> list[float]:
    """Return deterministic mock aesthetic scores for each PIL image or embedding."""
    if embeddings is not None and isinstance(embeddings, torch.Tensor):
        count = embeddings.shape[0]
        return [
            _int_to_mock_score(_tensor_fingerprint(embeddings[i]), "aesthetic")
            for i in range(count)
        ]
    if pil_images is None:
        return []
    return [
        _int_to_mock_score(_pil_fingerprint(img), "aesthetic")
        for img in pil_images
    ]


def mock_score_topiq(image_tensor: torch.Tensor, device: str) -> float:
    """Return deterministic mock TOPIQ score for a single image tensor."""
    return mock_score_topiq_batch(image_tensor, device)[0]


def mock_score_clipiqa(image_tensor: torch.Tensor, device: str) -> float:
    """Return deterministic mock CLIP-IQA+ score for a single image tensor."""
    return mock_score_clipiqa_batch(image_tensor, device)[0]


def mock_score_aesthetic(image_tensor: torch.Tensor, device: str) -> float:
    """Return deterministic mock aesthetic score for a single image tensor."""
    return _int_to_mock_score(_tensor_fingerprint(image_tensor), "aesthetic")


# ---------------------------------------------------------------------------
# Stage 1 mock constants
# ---------------------------------------------------------------------------

STAGE1_BLUR_TIER_MODULUS: int = 3
STAGE1_BOOL_MODULUS: int = 2


def _path_fingerprint(image_path: Path) -> int:
    """Return deterministic int derived from the path string only."""
    digest = hashlib.sha256(str(image_path).encode()).digest()
    return int.from_bytes(digest[:4], "big") % FINGERPRINT_MODULUS


def _mock_float(fingerprint: int, salt: str) -> float:
    """Map (fingerprint, salt) to a float in MOCK_SCORE_BAND."""
    return _int_to_mock_score(fingerprint, salt)


def _mock_bool(fingerprint: int, salt: str) -> bool:
    """Derive a deterministic bool from fingerprint and salt."""
    return bool((fingerprint ^ _salt_fingerprint(salt)) % STAGE1_BOOL_MODULUS)


def _mock_blur_scores(fingerprint: int) -> BlurScores:
    """Build a deterministic BlurScores from fingerprint."""
    return BlurScores(
        tenengrad=_mock_float(fingerprint, "tenengrad"),
        fft_ratio=_mock_float(fingerprint, "fft_ratio"),
        blur_tier=(fingerprint % STAGE1_BLUR_TIER_MODULUS) + 1,
        subject_sharpness=_mock_float(fingerprint, "subject_sharpness"),
        background_sharpness=_mock_float(fingerprint, "background_sharpness"),
        is_bokeh=_mock_bool(fingerprint, "is_bokeh"),
        is_motion_blur=_mock_bool(fingerprint, "is_motion_blur"),
    )


def mock_assess_blur(image_path: Path, config: object) -> BlurResult:
    """Return deterministic BlurResult derived from path string only."""
    fp = _path_fingerprint(image_path)
    return BlurResult(
        path=image_path,
        scores=_mock_blur_scores(fp),
        is_blurry=_mock_bool(fp, "is_blurry"),
    )


def _mock_clipping(fingerprint: int) -> ClippingResult:
    """Build a deterministic ClippingResult from fingerprint."""
    return ClippingResult(
        highlight_r=_mock_float(fingerprint, "highlight_r"),
        highlight_g=_mock_float(fingerprint, "highlight_g"),
        highlight_b=_mock_float(fingerprint, "highlight_b"),
        shadow_r=_mock_float(fingerprint, "shadow_r"),
        shadow_g=_mock_float(fingerprint, "shadow_g"),
        shadow_b=_mock_float(fingerprint, "shadow_b"),
        highlight_pct=_mock_float(fingerprint, "highlight_pct"),
        shadow_pct=_mock_float(fingerprint, "shadow_pct"),
    )


def _mock_color_cast(fingerprint: int) -> ColorCastResult:
    """Build a deterministic ColorCastResult from fingerprint."""
    return ColorCastResult(
        mean_a=_mock_float(fingerprint, "mean_a"),
        mean_b=_mock_float(fingerprint, "mean_b"),
        cast_score=_mock_float(fingerprint, "cast_score"),
    )


def mock_assess_exposure(image_path: Path) -> ExposureResult:
    """Return deterministic ExposureResult derived from path string only."""
    fp = _path_fingerprint(image_path)
    return ExposureResult(
        clipping=_mock_clipping(fp),
        dynamic_range=_mock_float(fp, "dynamic_range"),
        midtone_pct=_mock_float(fp, "midtone_pct"),
        color_cast=_mock_color_cast(fp),
        has_highlight_clip=_mock_bool(fp, "has_highlight_clip"),
        has_shadow_clip=_mock_bool(fp, "has_shadow_clip"),
        has_color_cast=_mock_bool(fp, "has_color_cast"),
        has_low_dr=_mock_bool(fp, "has_low_dr"),
    )


def mock_assess_noise(image_path: Path) -> NoiseResult:
    """Return deterministic NoiseResult derived from path string only."""
    fp = _path_fingerprint(image_path)
    return NoiseResult(
        noise_score=_mock_float(fp, "noise_score"),
        is_noisy=_mock_bool(fp, "is_noisy"),
    )


def _mock_geometry_scores(fingerprint: int) -> GeometryScore:
    """Build a deterministic GeometryScore from fingerprint."""
    return GeometryScore(
        tilt_degrees=_mock_float(fingerprint, "tilt_degrees"),
        keystone_degrees=_mock_float(fingerprint, "keystone_degrees"),
        confidence=_mock_float(fingerprint, "geom_confidence"),
        has_horizon=_mock_bool(fingerprint, "has_horizon"),
        has_verticals=_mock_bool(fingerprint, "has_verticals"),
    )


def mock_assess_geometry(image_path: Path) -> GeometryResult:
    """Return deterministic GeometryResult derived from path string only."""
    fp = _path_fingerprint(image_path)
    return GeometryResult(path=image_path, scores=_mock_geometry_scores(fp))
