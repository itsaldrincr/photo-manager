"""Face bbox from a full-resolution portrait pass must land on the face in pil_1280."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from cull.config import CullConfig
from cull.stage2 import portrait
from cull.stage2.portrait import assess_portrait_from_array
from cull.stage2.subject_blur import SubjectBlurInput, compute_subject_blur

FULL_RES_PX: int = 2560
PIL_1280_PX: int = 1280
FACE_FRAC_X0: float = 0.625
FACE_FRAC_X1: float = 0.9375
LANDMARK_COUNT: int = 478
SOFT_VALUE: int = 40
SHARP_VALUE: int = 220


def _full_res_bgr() -> np.ndarray:
    """Return a flat full-res frame with sharp stripes only inside the face region."""
    img = np.full((FULL_RES_PX, FULL_RES_PX, 3), SOFT_VALUE, dtype=np.uint8)
    lo, hi = int(FACE_FRAC_X0 * FULL_RES_PX), int(FACE_FRAC_X1 * FULL_RES_PX)
    for col in range(lo, hi, 8):
        img[lo:hi, col:col + 4] = SHARP_VALUE
    return img


def _face_landmarks() -> list[SimpleNamespace]:
    """Return landmarks spread over the face region in normalised coordinates."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(FACE_FRAC_X0, FACE_FRAC_X1, size=(LANDMARK_COUNT, 2))
    return [SimpleNamespace(x=float(x), y=float(y), visibility=None) for x, y in coords]


def test_full_res_face_bbox_scores_the_sharp_face_on_pil_1280(monkeypatch) -> None:
    """The face crop on the 1280px image must cover the sharp face, not an empty strip."""
    monkeypatch.setattr(portrait, "detect_faces", lambda image: [_face_landmarks()])
    monkeypatch.setattr(
        portrait, "_detect_emotion_reading",
        lambda crop: portrait._EmotionReading(label="", valence=None, arousal=None),
    )
    full_res = _full_res_bgr()
    result = assess_portrait_from_array(full_res, CullConfig(is_portrait=True))
    pil_1280 = Image.fromarray(full_res).resize((PIL_1280_PX, PIL_1280_PX))
    score = compute_subject_blur(SubjectBlurInput(pil_1280=pil_1280, portrait=result))
    assert score.subject_region_source == "face"
    assert score.tenengrad > 1000.0
