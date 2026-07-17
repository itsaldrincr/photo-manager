"""Subprocess worker for portrait_eval.py — loads exactly one heavy model.

Runs in its own process so DeepFace's TensorFlow backend and MediaPipe's
XNNPACK backend are never resident at the same time, and so RSS deltas
measured around model load are not polluted by the other stage.

Usage:
    python3 benchmarks/portrait_eval_runner.py mediapipe <manifest.json> <output.json>
    python3 benchmarks/portrait_eval_runner.py deepface <mediapipe_output.json> <output.json>
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

from portrait_eval_models import (
    BlendshapeScores,
    DeepfaceReading,
    DeepfaceStageOutput,
    MediapipeReading,
    MediapipeStageOutput,
    StageTiming,
    blendshape_eyes_closed,
    bucket_from_blendshapes,
    bucket_from_deepface_label,
)

MIN_FACE_DETECTION_CONFIDENCE: float = 0.5
SINGLE_FACE_MAX: int = 1

_BLENDSHAPE_FIELD_NAMES: dict[str, str] = {
    "eyeBlinkLeft": "eye_blink_left",
    "eyeBlinkRight": "eye_blink_right",
    "mouthSmileLeft": "mouth_smile_left",
    "mouthSmileRight": "mouth_smile_right",
    "browDownLeft": "brow_down_left",
    "browDownRight": "brow_down_right",
    "mouthFrownLeft": "mouth_frown_left",
    "mouthFrownRight": "mouth_frown_right",
    "jawOpen": "jaw_open",
    "browInnerUp": "brow_inner_up",
}


class RunnerArgs(BaseModel):
    """Parsed CLI arguments."""

    stage: str
    input_path: Path
    output_path: Path


class _DetectionOutcome(BaseModel):
    """A landmarker detection result paired with its source photo and latency."""

    model_config = {"arbitrary_types_allowed": True}

    photo_path: Path
    result: Any
    latency_seconds: float


def _parse_args(argv: list[str]) -> RunnerArgs:
    """Validate and bundle the three positional arguments."""
    if len(argv) != 4 or argv[1] not in {"mediapipe", "deepface"}:
        raise SystemExit(
            "usage: portrait_eval_runner.py {mediapipe|deepface} <input.json> <output.json>"
        )
    return RunnerArgs(stage=argv[1], input_path=Path(argv[2]), output_path=Path(argv[3]))


def _rss_mb() -> float:
    """Return this process's current resident set size in MB."""
    return round(psutil.Process().memory_info().rss / 1e6, 1)


def _blendshape_scores_from_categories(categories: list[Any]) -> BlendshapeScores:
    """Convert MediaPipe blendshape Category objects into a BlendshapeScores model."""
    scores = {name: c.score for c in categories if (name := _BLENDSHAPE_FIELD_NAMES.get(c.category_name))}
    return BlendshapeScores(**scores)


def _bootstrap_offline_env() -> None:
    """Pin model cache paths via cull's own bootstrap (reused, not modified)."""
    from cull.env_bootstrap import bootstrap_default  # noqa: PLC0415

    bootstrap_default()


def _build_face_landmarker() -> Any:
    """Construct a blendshapes-enabled FaceLandmarker from the shared model cache."""
    import mediapipe as mp  # noqa: PLC0415
    from cull.config import FACE_LANDMARKER_FILENAME, ModelCacheConfig  # noqa: PLC0415

    cache = ModelCacheConfig.from_env()
    model_path = cache.mediapipe_dir / FACE_LANDMARKER_FILENAME
    base_opts = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=SINGLE_FACE_MAX,
        min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
        output_face_blendshapes=True,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def _detect_one_mediapipe(photo_path: Path, landmarker: Any) -> MediapipeReading:
    """Run one photo through the blendshapes-enabled landmarker; time detect() only."""
    import mediapipe as mp  # noqa: PLC0415
    from cull.stage2.portrait import compute_ear, is_eyes_closed  # noqa: PLC0415

    image = cv2.imread(str(photo_path))
    if image is None:
        return MediapipeReading(name=photo_path.name, path=str(photo_path), has_face=False)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    started = time.monotonic()
    result = landmarker.detect(mp_image)
    latency = time.monotonic() - started
    if not result.face_landmarks:
        return MediapipeReading(
            name=photo_path.name, path=str(photo_path), has_face=False, latency_seconds=latency
        )
    outcome = _DetectionOutcome(photo_path=photo_path, result=result, latency_seconds=latency)
    return _mediapipe_reading_from_result(outcome)


def _mediapipe_reading_from_result(outcome: _DetectionOutcome) -> MediapipeReading:
    """Build a MediapipeReading from a detected FaceLandmarker result."""
    from cull.stage2.portrait import compute_ear, is_eyes_closed  # noqa: PLC0415

    landmarks = outcome.result.face_landmarks[0]
    ear_closed = is_eyes_closed(compute_ear(landmarks))
    scores = _blendshape_scores_from_categories(outcome.result.face_blendshapes[0])
    return MediapipeReading(
        name=outcome.photo_path.name,
        path=str(outcome.photo_path),
        has_face=True,
        latency_seconds=outcome.latency_seconds,
        ear_eyes_closed=ear_closed,
        blendshape_eyes_closed=blendshape_eyes_closed(scores),
        expression_bucket=bucket_from_blendshapes(scores),
    )


def _run_mediapipe_stage(args: RunnerArgs) -> MediapipeStageOutput:
    """Load the blendshapes landmarker once, then process every candidate photo."""
    photo_paths = [Path(p) for p in json.loads(args.input_path.read_text())]
    _bootstrap_offline_env()
    rss_before = _rss_mb()
    started = time.monotonic()
    landmarker = _build_face_landmarker()
    timing = StageTiming(
        load_seconds=round(time.monotonic() - started, 3),
        rss_before_load_mb=rss_before,
        rss_after_load_mb=_rss_mb(),
    )
    readings = [_detect_one_mediapipe(p, landmarker) for p in photo_paths]
    landmarker.close()
    return MediapipeStageOutput(timing=timing, readings=readings)


def _dominant_emotion_label(result: object) -> str:
    """Extract the dominant_emotion field from a DeepFace.analyze() result."""
    emotions = result[0] if isinstance(result, list) else result
    return str(emotions.get("dominant_emotion", ""))  # type: ignore[union-attr]


def _analyze_one_deepface(photo_path: Path) -> DeepfaceReading:
    """Run one photo through DeepFace.analyze(); time the analyze() call only."""
    from deepface import DeepFace  # noqa: PLC0415

    image = cv2.imread(str(photo_path))
    started = time.monotonic()
    result = DeepFace.analyze(image, actions=["emotion"], enforce_detection=False)
    latency = time.monotonic() - started
    label = _dominant_emotion_label(result)
    return DeepfaceReading(
        name=photo_path.name,
        latency_seconds=latency,
        dominant_emotion=label,
        expression_bucket=bucket_from_deepface_label(label),
    )


def _run_deepface_stage(args: RunnerArgs) -> DeepfaceStageOutput:
    """Load DeepFace's emotion model once (via a warm-up call), then process faces."""
    mediapipe_output = MediapipeStageOutput.model_validate_json(args.input_path.read_text())
    face_photos = [Path(r.path) for r in mediapipe_output.readings if r.has_face]
    if not face_photos:
        raise SystemExit("no face-bearing photos in mediapipe stage output")
    _bootstrap_offline_env()
    rss_before = _rss_mb()
    started = time.monotonic()
    warmup = _analyze_one_deepface(face_photos[0])
    timing = StageTiming(
        load_seconds=round(time.monotonic() - started, 3),
        rss_before_load_mb=rss_before,
        rss_after_load_mb=_rss_mb(),
    )
    readings = [warmup] + [_analyze_one_deepface(p) for p in face_photos[1:]]
    return DeepfaceStageOutput(timing=timing, readings=readings)


def main() -> None:
    """Entry point: run the requested stage and write JSON to the output path."""
    args = _parse_args(sys.argv)
    result = _run_mediapipe_stage(args) if args.stage == "mediapipe" else _run_deepface_stage(args)
    args.output_path.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
