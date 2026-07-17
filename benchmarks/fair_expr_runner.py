"""Subprocess worker for the fair DeepFace-vs-EmotiEffLib test.

Loads exactly one heavy model per process (DeepFace/TensorFlow, EmotiEffLib/
onnxruntime, or MediaPipe), then exits — mirrors expression_eval_runner.py so
the ONE-heavy-model-resident rule holds across all three test layers.

Both emotion models receive IDENTICAL pre-cropped face images with their own
internal face detection disabled/absent — this is the fairness contract the
task requires, not a convenience shortcut.

Usage:
    python3 benchmarks/fair_expr_runner.py deepface <manifest.json> <output.json>
    python3 benchmarks/fair_expr_runner.py emotiefflib <manifest.json> <output.json>
    python3 benchmarks/fair_expr_runner.py mediapipe_bbox <manifest.json> <output.json>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
from pydantic import BaseModel

from fair_expr_models import BboxReading, FACE_CROP_MARGIN_FRACTION, ModelReading, RunnerManifest, RunnerStageOutput

STAGES: set[str] = {"deepface", "emotiefflib", "mediapipe_bbox"}
SINGLE_FACE_MAX: int = 1


class RunnerArgs(BaseModel):
    """Parsed CLI arguments."""

    stage: str
    input_path: Path
    output_path: Path


def _parse_args(argv: list[str]) -> RunnerArgs:
    """Validate and bundle the three positional arguments."""
    if len(argv) != 4 or argv[1] not in STAGES:
        raise SystemExit(f"usage: fair_expr_runner.py {{{'|'.join(sorted(STAGES))}}} <manifest.json> <output.json>")
    return RunnerArgs(stage=argv[1], input_path=Path(argv[2]), output_path=Path(argv[3]))


def _load_manifest(path: Path) -> RunnerManifest:
    """Read and validate the runner manifest."""
    return RunnerManifest.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# DeepFace stage — enforce_detection=False: the crop IS the face, no re-detect.
# ---------------------------------------------------------------------------


def _dominant_emotion_label(result: object) -> str:
    """Extract the dominant_emotion field from a DeepFace.analyze() result."""
    emotions = result[0] if isinstance(result, list) else result
    return str(emotions.get("dominant_emotion", ""))  # type: ignore[union-attr]


def _analyze_one_deepface(image_path: Path) -> ModelReading:
    """Run one pre-cropped face image through DeepFace.analyze(); time analyze() only."""
    from deepface import DeepFace  # noqa: PLC0415

    image = cv2.imread(str(image_path))
    if image is None:
        return ModelReading(name=image_path.name, has_face=False)
    started = time.monotonic()
    result = DeepFace.analyze(image, actions=["emotion"], enforce_detection=False)
    latency = time.monotonic() - started
    return ModelReading(
        name=image_path.name, has_face=True, latency_seconds=latency,
        raw_label=_dominant_emotion_label(result).strip().lower(),
    )


def _run_deepface_stage(manifest: RunnerManifest) -> RunnerStageOutput:
    """Load DeepFace's emotion model via a warm-up call, then score every image."""
    from cull.env_bootstrap import bootstrap_default  # noqa: PLC0415

    bootstrap_default()
    paths = manifest.image_paths
    warmup = _analyze_one_deepface(paths[0])
    readings = [warmup] + [_analyze_one_deepface(p) for p in paths[1:]]
    return RunnerStageOutput(stage="deepface", readings=readings)


# ---------------------------------------------------------------------------
# EmotiEffLib stage — recognizer.predict_emotions() on the raw crop, no cascade.
# ---------------------------------------------------------------------------


def _build_emotiefflib_recognizer() -> Any:
    """Construct the EmotiEffLib ONNX recognizer for enet_b0_8_va_mtl."""
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer  # noqa: PLC0415

    return EmotiEffLibRecognizer(engine="onnx", model_name="enet_b0_8_va_mtl")


def _analyze_one_emotiefflib(image_path: Path, recognizer: Any) -> ModelReading:
    """Run one pre-cropped face image through predict_emotions(); time it only."""
    image = cv2.imread(str(image_path))
    if image is None:
        return ModelReading(name=image_path.name, has_face=False)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    started = time.monotonic()
    labels, scores = recognizer.predict_emotions(rgb, logits=True)
    latency = time.monotonic() - started
    return ModelReading(
        name=image_path.name, has_face=True, latency_seconds=latency, raw_label=labels[0],
        valence=float(scores[0, -2]), arousal=float(scores[0, -1]),
    )


def _run_emotiefflib_stage(manifest: RunnerManifest) -> RunnerStageOutput:
    """Load the recognizer once, then score every pre-cropped face image."""
    recognizer = _build_emotiefflib_recognizer()
    readings = [_analyze_one_emotiefflib(p, recognizer) for p in manifest.image_paths]
    return RunnerStageOutput(stage="emotiefflib", readings=readings)


# ---------------------------------------------------------------------------
# MediaPipe bbox/crop stage — shared face-gate + identical-crop producer for
# Layers B and C. Builds its own landmarker rather than importing
# portrait_eval_runner.py's private helpers, to keep that file untouched.
# ---------------------------------------------------------------------------


def _build_face_landmarker() -> Any:
    """Construct a FaceLandmarker from the shared offline model cache."""
    import mediapipe as mp  # noqa: PLC0415
    from cull.config import FACE_LANDMARKER_FILENAME, PORTRAIT_FACE_DETECTION_CONFIDENCE_MIN  # noqa: PLC0415
    from cull.env_bootstrap import bootstrap_default  # noqa: PLC0415

    cache = bootstrap_default()
    base_opts = mp.tasks.BaseOptions(model_asset_path=str(cache.mediapipe_dir / FACE_LANDMARKER_FILENAME))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_opts, running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=SINGLE_FACE_MAX, min_face_detection_confidence=PORTRAIT_FACE_DETECTION_CONFIDENCE_MIN,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


class _LandmarkBbox(BaseModel):
    """A face bbox in pixel coordinates, clamped to the source image bounds."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


def _bbox_from_landmarks(landmarks: list[Any], image_shape: tuple[int, int]) -> _LandmarkBbox:
    """Compute a margin-padded pixel bbox from normalized face landmarks."""
    height, width = image_shape
    xs, ys = [pt.x for pt in landmarks], [pt.y for pt in landmarks]
    x_span, y_span = (max(xs) - min(xs)) * width, (max(ys) - min(ys)) * height
    x_pad, y_pad = x_span * FACE_CROP_MARGIN_FRACTION, y_span * FACE_CROP_MARGIN_FRACTION
    return _LandmarkBbox(
        x_min=max(0, int(min(xs) * width - x_pad)), y_min=max(0, int(min(ys) * height - y_pad)),
        x_max=min(width, int(max(xs) * width + x_pad)), y_max=min(height, int(max(ys) * height + y_pad)),
    )


class _CropTarget(BaseModel):
    """A source photo path paired with the directory to write its face crop into."""

    photo_path: Path
    crop_dir: Path


def _detect_and_crop_one(target: _CropTarget, landmarker: Any) -> BboxReading:
    """Detect the primary face, crop with margin, and save it to crop_dir."""
    import mediapipe as mp  # noqa: PLC0415

    image = cv2.imread(str(target.photo_path))
    if image is None:
        return BboxReading(name=target.photo_path.name, path=str(target.photo_path), has_face=False)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return BboxReading(name=target.photo_path.name, path=str(target.photo_path), has_face=False)
    bbox = _bbox_from_landmarks(result.face_landmarks[0], image.shape[:2])
    crop = image[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]
    target.crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = target.crop_dir / target.photo_path.name
    cv2.imwrite(str(crop_path), crop)
    return BboxReading(
        name=target.photo_path.name, path=str(target.photo_path), has_face=True,
        crop_path=str(crop_path), face_height_px=bbox.y_max - bbox.y_min,
    )


def _run_mediapipe_bbox_stage(manifest: RunnerManifest) -> RunnerStageOutput:
    """Load the landmarker once, then face-gate and crop every candidate photo."""
    if manifest.crop_dir is None:
        raise SystemExit("mediapipe_bbox stage requires manifest.crop_dir")
    landmarker = _build_face_landmarker()
    bboxes = [
        _detect_and_crop_one(_CropTarget(photo_path=p, crop_dir=manifest.crop_dir), landmarker)
        for p in manifest.image_paths
    ]
    landmarker.close()
    return RunnerStageOutput(stage="mediapipe_bbox", bboxes=bboxes)


_STAGE_RUNNERS: dict[str, Any] = {
    "deepface": _run_deepface_stage,
    "emotiefflib": _run_emotiefflib_stage,
    "mediapipe_bbox": _run_mediapipe_bbox_stage,
}


def main() -> None:
    """Entry point: run the requested stage and write JSON to the output path."""
    args = _parse_args(sys.argv)
    manifest = _load_manifest(args.input_path)
    result = _STAGE_RUNNERS[args.stage](manifest)
    args.output_path.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
