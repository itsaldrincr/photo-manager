"""Builders shared by the TUI tests: tiny JPEGs, sessions, and an async runner."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from io import BytesIO
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from cull._pipeline.orchestrator import SessionResult, SessionSummary
from cull.models import (
    BlurScores,
    BurstInfo,
    DecisionLabel,
    ExposureScores,
    PhotoDecision,
    PhotoMeta,
    Stage1Result,
    Stage2Result,
    TasteScore,
)

EXIF_ORIENTATION_TAG: int = 0x0112
SETTLE_TIMEOUT_SECONDS: float = 5.0


class JpegSpec(BaseModel):
    """A test JPEG to write."""

    path: Path
    size: tuple[int, int] = (300, 200)
    colour: tuple[int, int, int] = (10, 20, 30)
    orientation: int | None = None


def write_jpeg(spec: JpegSpec) -> Path:
    """Write a solid-colour JPEG, optionally tagged with an EXIF orientation."""
    image = Image.new("RGB", spec.size, color=spec.colour)
    exif = Image.Exif()
    if spec.orientation is not None:
        exif[EXIF_ORIENTATION_TAG] = spec.orientation
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    spec.path.write_bytes(buffer.getvalue())
    return spec.path


def _stage1(path: Path, burst: BurstInfo | None) -> Stage1Result:
    """Build a minimal passing Stage 1 result."""
    return Stage1Result(
        photo_path=path,
        blur=BlurScores(tenengrad=1.0, fft_ratio=1.0, blur_tier=1),
        exposure=ExposureScores(
            dr_score=1.0, clipping_highlight=0.0, clipping_shadow=0.0, midtone_pct=0.5, color_cast_score=0.0,
        ),
        noise_score=0.0,
        burst=burst,
    )


class PhotoSpec(BaseModel):
    """One photo in a test session."""

    name: str
    label: DecisionLabel = "uncertain"
    taste: float | None = None
    burst: BurstInfo | None = None
    composite: float = 0.9


def make_decision(folder: Path, photo: PhotoSpec) -> PhotoDecision:
    """Write the photo's JPEG and build its decision."""
    path = write_jpeg(JpegSpec(path=folder / photo.name))
    taste = None
    if photo.taste is not None:
        taste = TasteScore(probability=photo.taste, label_count_at_score=10, weight_applied=0.2, model_version="t")
    stage2 = Stage2Result(
        photo_path=path, topiq=0.5, laion_aesthetic=0.5, clipiqa=0.5, composite=photo.composite, taste=taste,
    )
    return PhotoDecision(
        photo=PhotoMeta(path=path, filename=photo.name),
        decision=photo.label,
        stage1=_stage1(path, photo.burst),
        stage2=stage2,
        stage_reached=2,
    )


def make_session(folder: Path, photos: list[PhotoSpec]) -> SessionResult:
    """Build a session whose photos exist on disk under ``folder``."""
    decisions = [make_decision(folder, photo) for photo in photos]
    return SessionResult(
        source_path=str(folder), total_photos=len(decisions), summary=SessionSummary(), decisions=decisions,
    )


def numbered(count: int, label: DecisionLabel = "uncertain") -> list[PhotoSpec]:
    """Return ``count`` photos named p00.jpg, p01.jpg, ..."""
    return [PhotoSpec(name=f"p{i:02d}.jpg", label=label) for i in range(count)]


def run(body: Callable[[], Awaitable[None]]) -> None:
    """Run an async test body synchronously (no pytest-asyncio dependency)."""
    asyncio.run(body())


async def wait_until(condition: Callable[[], bool]) -> None:
    """Poll the event loop until ``condition`` holds; fail after a timeout."""
    deadline = time.monotonic() + SETTLE_TIMEOUT_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met before timeout")
        await asyncio.sleep(0.01)
