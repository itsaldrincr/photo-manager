"""Subprocess worker for expression_eval.py — loads exactly one heavy model.

Runs in its own process so py-feat's torch/timm backend and EmotiEffLib's
onnxruntime backend are never resident at the same time, and so RSS deltas
measured around model load are not polluted by the other stage.

Usage:
    python3 benchmarks/expression_eval_runner.py pyfeat <manifest.json> <output.json>
    python3 benchmarks/expression_eval_runner.py emotiefflib <manifest.json> <output.json>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import psutil
from pydantic import BaseModel

from expression_eval_models import (
    EmotiEfflibReading,
    EmotiEfflibStageOutput,
    PyFeatReading,
    PyFeatStageOutput,
    au43_eyes_closed,
    va_quadrant,
)
from portrait_eval_models import StageTiming

HAAR_SCALE_FACTOR: float = 1.1
HAAR_MIN_NEIGHBORS: int = 5
HAAR_MIN_FACE_PX: int = 60


class RunnerArgs(BaseModel):
    """Parsed CLI arguments."""

    stage: str
    input_path: Path
    output_path: Path


def _parse_args(argv: list[str]) -> RunnerArgs:
    """Validate and bundle the three positional arguments."""
    if len(argv) != 4 or argv[1] not in {"pyfeat", "emotiefflib"}:
        raise SystemExit(
            "usage: expression_eval_runner.py {pyfeat|emotiefflib} <manifest.json> <output.json>"
        )
    return RunnerArgs(stage=argv[1], input_path=Path(argv[2]), output_path=Path(argv[3]))


def _rss_mb() -> float:
    """Return this process's current resident set size in MB."""
    return round(psutil.Process().memory_info().rss / 1e6, 1)


def _load_manifest(path: Path) -> list[Path]:
    """Read a JSON list of photo paths."""
    return [Path(p) for p in json.loads(path.read_text())]


# ---------------------------------------------------------------------------
# py-feat Detectorv2 stage
#
# py-feat 2.0.3 imports torchcodec (for video decoding) at module load time,
# and torchcodec's dylibs require a system ffmpeg install that is absent on
# this host. We never decode video here, so the video decoder is stubbed out
# before importing feat — a workaround for a real integration blocker, noted
# in the report rather than silently worked around.
# ---------------------------------------------------------------------------


def _stub_torchcodec_video_decoder() -> None:
    """Stub torchcodec.decoders.VideoDecoder so `import feat` succeeds without ffmpeg."""
    import types  # noqa: PLC0415

    fake_decoders = types.ModuleType("torchcodec.decoders")
    fake_decoders.VideoDecoder = type("VideoDecoder", (), {})
    fake_torchcodec = types.ModuleType("torchcodec")
    fake_torchcodec.decoders = fake_decoders
    sys.modules["torchcodec"] = fake_torchcodec
    sys.modules["torchcodec.decoders"] = fake_decoders


def _build_pyfeat_detector() -> Any:
    """Construct a py-feat Detectorv2 on the best available device."""
    _stub_torchcodec_video_decoder()
    from feat.detector_v2 import Detectorv2  # noqa: PLC0415

    return Detectorv2(device="auto")


def _primary_face_row(fex: Any) -> Any:
    """Return the highest FaceScore row from a py-feat Fex frame, or None."""
    import pandas as pd  # noqa: PLC0415

    if fex.empty or pd.isna(fex["FaceScore"]).all():
        return None
    return fex.loc[fex["FaceScore"].idxmax()]


_PYFEAT_EMOTION_COLUMNS: list[str] = [
    "Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger",
]


class _PyFeatRowResult(BaseModel):
    """A detected face's raw row paired with its photo name and detect() latency."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    row: Any
    latency_seconds: float


def _pyfeat_reading_from_row(result: _PyFeatRowResult) -> PyFeatReading:
    """Build a PyFeatReading from the primary detected face's row."""
    row = result.row
    aus = {au: float(row[au]) for au in row.index if au.startswith("AU")}
    emotions = row[_PYFEAT_EMOTION_COLUMNS].astype(float)
    valence, arousal = float(row["valence"]), float(row["arousal"])
    return PyFeatReading(
        name=result.name, has_face=True, latency_seconds=result.latency_seconds,
        face_score=float(row["FaceScore"]), action_units=aus,
        dominant_emotion=str(emotions.idxmax()), valence=valence, arousal=arousal,
        au43_eyes_closed=au43_eyes_closed(aus.get("AU43", 0.0)),
        va_quadrant=va_quadrant(valence, arousal),
    )


def _detect_one_pyfeat(photo_path: Path, detector: Any) -> PyFeatReading:
    """Run one photo through Detectorv2.detect(); time the detect() call only."""
    started = time.monotonic()
    fex = detector.detect([str(photo_path)], data_type="image", progress_bar=False)
    latency = time.monotonic() - started
    row = _primary_face_row(fex)
    if row is None:
        return PyFeatReading(name=photo_path.name, has_face=False, latency_seconds=latency)
    return _pyfeat_reading_from_row(
        _PyFeatRowResult(name=photo_path.name, row=row, latency_seconds=latency)
    )


def _run_pyfeat_stage(args: RunnerArgs) -> PyFeatStageOutput:
    """Load Detectorv2 once, then process every candidate photo."""
    photo_paths = _load_manifest(args.input_path)
    rss_before = _rss_mb()
    started = time.monotonic()
    detector = _build_pyfeat_detector()
    timing = StageTiming(
        load_seconds=round(time.monotonic() - started, 3),
        rss_before_load_mb=rss_before, rss_after_load_mb=_rss_mb(),
    )
    readings = [_detect_one_pyfeat(p, detector) for p in photo_paths]
    return PyFeatStageOutput(timing=timing, readings=readings)


# ---------------------------------------------------------------------------
# EmotiEffLib stage
#
# EmotiEffLib ships no face detector — it expects a pre-cropped face. We use
# OpenCV's bundled Haar cascade (the same detector family DeepFace's default
# backend uses) purely for cropping; that crop cost is excluded from the
# measured "model call" latency below, since EmotiEffLib itself is a
# recognition-only model (unlike py-feat/DeepFace, which bundle detection).
# ---------------------------------------------------------------------------


def _build_haar_cascade() -> Any:
    """Load OpenCV's bundled frontal-face Haar cascade."""
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


def _crop_primary_face(image: Any, cascade: Any) -> Any:
    """Return the largest detected face crop (RGB), or None if none found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=HAAR_SCALE_FACTOR, minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=(HAAR_MIN_FACE_PX, HAAR_MIN_FACE_PX),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)


def _build_emotiefflib_recognizer() -> Any:
    """Construct the EmotiEffLib ONNX recognizer for enet_b0_8_va_mtl."""
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer  # noqa: PLC0415

    return EmotiEffLibRecognizer(engine="onnx", model_name="enet_b0_8_va_mtl")


class _EmotiEfflibPrediction(BaseModel):
    """A photo name paired with one predict_emotions() call's raw output."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    labels: list[str]
    scores: Any
    latency_seconds: float


def _emotiefflib_reading_from_scores(prediction: _EmotiEfflibPrediction) -> EmotiEfflibReading:
    """Build an EmotiEfflibReading from one predict_emotions() call's output."""
    valence = float(prediction.scores[0, -2])
    arousal = float(prediction.scores[0, -1])
    return EmotiEfflibReading(
        name=prediction.name, has_face=True, dominant_emotion=prediction.labels[0],
        valence=valence, arousal=arousal, va_quadrant=va_quadrant(valence, arousal),
        latency_seconds=prediction.latency_seconds,
    )


class _EmotiEfflibModels(BaseModel):
    """The Haar cascade (cropper) and recognizer used for one EmotiEffLib pass."""

    model_config = {"arbitrary_types_allowed": True}

    cascade: Any
    recognizer: Any


def _detect_one_emotiefflib(photo_path: Path, models: _EmotiEfflibModels) -> EmotiEfflibReading:
    """Crop the primary face (untimed), then time only the recognizer's forward call."""
    image = cv2.imread(str(photo_path))
    crop = _crop_primary_face(image, models.cascade) if image is not None else None
    if crop is None:
        return EmotiEfflibReading(name=photo_path.name, has_face=False)
    started = time.monotonic()
    labels, scores = models.recognizer.predict_emotions(crop, logits=True)
    latency = time.monotonic() - started
    return _emotiefflib_reading_from_scores(
        _EmotiEfflibPrediction(name=photo_path.name, labels=labels, scores=scores, latency_seconds=latency)
    )


def _run_emotiefflib_stage(args: RunnerArgs) -> EmotiEfflibStageOutput:
    """Load the recognizer once, then process every candidate photo."""
    photo_paths = _load_manifest(args.input_path)
    cascade = _build_haar_cascade()
    rss_before = _rss_mb()
    started = time.monotonic()
    recognizer = _build_emotiefflib_recognizer()
    timing = StageTiming(
        load_seconds=round(time.monotonic() - started, 3),
        rss_before_load_mb=rss_before, rss_after_load_mb=_rss_mb(),
    )
    models = _EmotiEfflibModels(cascade=cascade, recognizer=recognizer)
    readings = [_detect_one_emotiefflib(p, models) for p in photo_paths]
    return EmotiEfflibStageOutput(timing=timing, readings=readings)


def main() -> None:
    """Entry point: run the requested stage and write JSON to the output path."""
    args = _parse_args(sys.argv)
    result = (
        _run_pyfeat_stage(args) if args.stage == "pyfeat" else _run_emotiefflib_stage(args)
    )
    args.output_path.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
