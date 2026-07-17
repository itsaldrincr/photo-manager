"""Tests for cull.stage2.portrait: face_landmarker resolver + EmotiEffLib fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cull.config import FACE_LANDMARKER_FILENAME, ModelCacheConfig
from cull.model_cache import ConfigError
from cull.stage2 import portrait
from cull.stage2.portrait import _resolve_face_landmarker_path, detect_expression


def _build_cache(tmp_path: Path) -> ModelCacheConfig:
    """Return a ModelCacheConfig rooted at an empty tmp_path."""
    return ModelCacheConfig(
        root=tmp_path,
        hf_home=tmp_path / "hf",
        torch_home=tmp_path / "torch",
        emotieff_dir=tmp_path / "emotieff",
        mediapipe_dir=tmp_path / "mediapipe",
    )


def test_resolve_face_landmarker_path_raises_on_missing(tmp_path: Path) -> None:
    """_resolve_face_landmarker_path must raise ConfigError mentioning 'cull setup'."""
    cache = _build_cache(tmp_path)
    with pytest.raises(ConfigError) as excinfo:
        _resolve_face_landmarker_path(cache)
    assert "cull setup" in str(excinfo.value)


def test_resolve_face_landmarker_path_returns_existing(tmp_path: Path) -> None:
    """_resolve_face_landmarker_path must return the path when the .task file exists."""
    cache = _build_cache(tmp_path)
    cache.mediapipe_dir.mkdir(parents=True, exist_ok=True)
    expected = cache.mediapipe_dir / FACE_LANDMARKER_FILENAME
    expected.write_bytes(b"fake face landmarker task payload")
    result = _resolve_face_landmarker_path(cache)
    assert result == expected


class _FakeRecognizerFailing:
    """Stand-in whose predict_emotions always raises."""

    @staticmethod
    def predict_emotions(image: np.ndarray, logits: bool = True) -> tuple:
        """Simulate an EmotiEffLib inference failure."""
        raise RuntimeError("simulated emotiefflib failure")


class _FakeRecognizerHappy:
    """Stand-in that always reports 'Happiness' with fixed valence/arousal."""

    @staticmethod
    def predict_emotions(image: np.ndarray, logits: bool = True) -> tuple:
        """Return a single 'Happiness' reading with valence=0.30, arousal=0.10."""
        scores = np.array([[0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.30, 0.10]])
        return ["Happiness"], scores


class _FakeRecognizerContempt:
    """Stand-in that always reports 'Contempt' (the 8-class label with no 7-class match)."""

    @staticmethod
    def predict_emotions(image: np.ndarray, logits: bool = True) -> tuple:
        """Return a single 'Contempt' reading."""
        scores = np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.10, 0.05]])
        return ["Contempt"], scores


def test_detect_expression_from_array_handles_emotiefflib_failure(monkeypatch) -> None:
    """detect_expression_from_array must swallow EmotiEffLib errors and return empty string."""
    monkeypatch.setattr(
        "cull.emotieff_loader.get_emotieff_recognizer",
        lambda: _FakeRecognizerFailing(),
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert portrait.detect_expression_from_array(image) == ""


def test_detect_expression_from_array_maps_happiness(monkeypatch) -> None:
    """detect_expression_from_array must map EmotiEffLib 'Happiness' to 'happy'."""
    monkeypatch.setattr(
        "cull.emotieff_loader.get_emotieff_recognizer",
        lambda: _FakeRecognizerHappy(),
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert portrait.detect_expression_from_array(image) == "happy"


def test_detect_emotion_reading_captures_valence_arousal(monkeypatch) -> None:
    """_detect_emotion_reading must surface valence/arousal alongside the mapped label."""
    monkeypatch.setattr(
        "cull.emotieff_loader.get_emotieff_recognizer",
        lambda: _FakeRecognizerHappy(),
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    reading = portrait._detect_emotion_reading(image)
    assert reading.label == "happy"
    assert reading.valence == pytest.approx(0.30)
    assert reading.arousal == pytest.approx(0.10)


def test_detect_expression_from_array_folds_contempt_into_disgust(monkeypatch) -> None:
    """detect_expression_from_array must fold the 8-class 'Contempt' label into 'disgust'."""
    monkeypatch.setattr(
        "cull.emotieff_loader.get_emotieff_recognizer",
        lambda: _FakeRecognizerContempt(),
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert portrait.detect_expression_from_array(image) == "disgust"


def test_detect_expression_reads_image_and_delegates(tmp_path: Path, monkeypatch) -> None:
    """detect_expression must read image bytes then delegate to detect_expression_from_array."""
    monkeypatch.setattr(
        "cull.emotieff_loader.get_emotieff_recognizer",
        lambda: _FakeRecognizerHappy(),
    )
    image_path = tmp_path / "face.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
    assert detect_expression(image_path) == "happy"


def test_detect_expression_missing_file_returns_empty() -> None:
    """detect_expression must return an empty string when the image cannot be read."""
    assert detect_expression(Path("/tmp/does-not-exist.jpg")) == ""
