"""Group photos: eyes-closed must consider every prominent face, not only the first."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cull.config import CullConfig
from cull.stage2 import portrait
from cull.stage2.portrait import _LEFT_EYE_INDICES, _RIGHT_EYE_INDICES, assess_portrait_from_array

LANDMARK_COUNT: int = 478
IMAGE_PX: int = 400
# Eye outlines in face-box units; EAR is scale-invariant (open ~0.40, closed ~0.02).
OPEN_EYE: list[tuple[float, float]] = [(0.2, 0.4), (0.3, 0.44), (0.4, 0.44), (0.5, 0.4), (0.4, 0.36), (0.3, 0.36)]
CLOSED_EYE: list[tuple[float, float]] = [(0.2, 0.4), (0.3, 0.401), (0.4, 0.401), (0.5, 0.4), (0.4, 0.399), (0.3, 0.399)]
LARGE_FACE: tuple[float, float, float] = (0.05, 0.05, 0.40)
PEER_FACE: tuple[float, float, float] = (0.55, 0.05, 0.30)
TINY_FACE: tuple[float, float, float] = (0.80, 0.80, 0.10)


def _face(box: tuple[float, float, float], eye: list[tuple[float, float]]) -> list[SimpleNamespace]:
    """Return normalised landmarks filling a square face box (x0, y0, side) with the given eyes."""
    x0, y0, side = box
    corners = [(x0, y0), (x0 + side, y0 + side)]
    landmarks = [
        SimpleNamespace(x=corners[i % 2][0], y=corners[i % 2][1], visibility=None)
        for i in range(LANDMARK_COUNT)
    ]
    for indices in (_LEFT_EYE_INDICES, _RIGHT_EYE_INDICES):
        for (u, v), idx in zip(eye, indices):
            landmarks[idx] = SimpleNamespace(x=x0 + u * side, y=y0 + v * side, visibility=None)
    return landmarks


@pytest.fixture()
def no_emotion(monkeypatch) -> None:
    """Stub the EmotiEffLib pass so no model loads."""
    monkeypatch.setattr(
        portrait, "_detect_emotion_reading",
        lambda crop: portrait._EmotionReading(label="", valence=None, arousal=None),
    )


def _assess(monkeypatch, faces: list[list[SimpleNamespace]]) -> portrait.PortraitResult:
    """Run portrait assessment with detect_faces mocked to return faces."""
    monkeypatch.setattr(portrait, "detect_faces", lambda image: faces)
    image = np.zeros((IMAGE_PX, IMAGE_PX, 3), dtype=np.uint8)
    return assess_portrait_from_array(image, CullConfig(is_portrait=True))


def test_prominent_second_face_with_closed_eyes_flags_photo(monkeypatch, no_emotion) -> None:
    """A peer-sized face with closed eyes makes the group photo eyes-closed."""
    result = _assess(monkeypatch, [_face(LARGE_FACE, OPEN_EYE), _face(PEER_FACE, CLOSED_EYE)])
    assert result.face_count == 2
    assert result.eyes_closed is True


def test_tiny_background_face_with_closed_eyes_is_ignored(monkeypatch, no_emotion) -> None:
    """A face under 25% of the largest face's area does not count as prominent."""
    result = _assess(monkeypatch, [_face(LARGE_FACE, OPEN_EYE), _face(TINY_FACE, CLOSED_EYE)])
    assert result.eyes_closed is False


def test_all_prominent_faces_open_is_not_flagged(monkeypatch, no_emotion) -> None:
    """Every prominent face open means no eyes-closed flag."""
    result = _assess(monkeypatch, [_face(LARGE_FACE, OPEN_EYE), _face(PEER_FACE, OPEN_EYE)])
    assert result.eyes_closed is False
