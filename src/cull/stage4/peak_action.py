"""DIS optical-flow burst peak detection for action photo selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from cull.config import CullConfig, MOTION_PEAK_WINDOW
from cull.models import PeakMomentScore

if TYPE_CHECKING:
    import cv2 as _cv2_type

logger = logging.getLogger(__name__)


def _get_cv2() -> "_cv2_type":
    """Lazy-load cv2 to avoid import cost during test collection."""
    import cv2  # noqa: PLC0415
    return cv2


class PeakActionInput(BaseModel):
    """Input bundle for burst peak action detection."""

    burst_members: list[Path]
    config: CullConfig


def _compute_flow_pair(prev_gray: np.ndarray, next_gray: np.ndarray) -> float:
    """Compute mean optical-flow magnitude between two grayscale frames."""
    cv2 = _get_cv2()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow: np.ndarray = dis.calc(prev_gray, next_gray, None)
    magnitude: np.ndarray = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(magnitude.mean())


def _load_gray(path: Path) -> np.ndarray:
    """Load an image file as a grayscale uint8 array."""
    cv2 = _get_cv2()
    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise ValueError(f"Cannot read image: {path}")
    return frame


def _inflection_index(magnitudes: list[float], window: int) -> int:
    """Return index of the frame with maximum motion within the peak window."""
    half = window // 2
    mid = len(magnitudes) // 2
    lo = max(0, mid - half)
    hi = min(len(magnitudes), mid + half + 1)
    windowed = magnitudes[lo:hi]
    return lo + int(np.argmax(windowed))


def _make_action_score(motion_peak_score: float) -> PeakMomentScore:
    """Construct a PeakMomentScore for action peak detection."""
    return PeakMomentScore(
        eyes_open_score=0.0,
        smile_score=0.0,
        gaze_score=0.0,
        motion_peak_score=motion_peak_score,
        peak_type="action",
    )


def pick_winner(inp: PeakActionInput) -> tuple[Path, PeakMomentScore]:
    """Select the motion-inflection peak frame from a burst sequence."""
    members = inp.burst_members
    if len(members) < 2:
        return members[0], _make_action_score(0.0)
    grays = [_load_gray(p) for p in members]
    magnitudes = [
        _compute_flow_pair(grays[i], grays[i + 1])
        for i in range(len(grays) - 1)
    ]
    logger.debug("Flow magnitudes: %s", magnitudes)
    peak_idx = _inflection_index(magnitudes, MOTION_PEAK_WINDOW)
    peak_score = magnitudes[peak_idx]
    return members[peak_idx], _make_action_score(peak_score)
