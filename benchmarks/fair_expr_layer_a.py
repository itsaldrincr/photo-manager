"""Layer A: labeled benchmark anchor (RAF-DB, HuggingFace mirror).

Dataset: deanngkl/raf-db-7emotions — a reupload of the Real-world Affective
Faces Database (Li & Deng, 2017) as 100x100 aligned face crops. The image
`path` field preserves RAF-DB's own train_/test_ file-name prefixes, so we
recover the ORIGINAL test partition rather than inventing our own split.

PROVENANCE CAVEAT (verified by inspecting the raw parquet, documented in the
report, not hidden): six of the seven classes (anger, disgust, fear,
happiness, sadness, surprise) are pure RAF-DB test-partition images. The
"neutral" class in THIS MIRROR is 100% sourced from AffectNet instead of
RAF-DB (every neutral row's path starts with "affectnet.zip", zero
train_/test_ RAF-DB neutral rows exist in the reupload) — an upstream
mislabeling of the mirror, not a RAF-DB property. We use it anyway (AffectNet
is itself a credible, separately-labeled academic dataset) because the task
requires a neutral class for the sad/fear/neutral/angry sub-analysis, and
flag it rather than silently treating it as RAF-DB.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from pydantic import BaseModel

from fair_expr_metrics import LabelPairs, accuracy, build_confusion_matrix, macro_f1, per_class_accuracy
from fair_expr_models import (
    CANONICAL_CLASSES, LAYER_A_DATASET_ID, LAYER_A_SAMPLES_PER_CLASS, LAYER_A_SEED,
    RAFDB_LABEL_MAP, SUB_ANALYSIS_CLASSES, canonicalize,
)

RAFDB_LABEL_ID_TO_NAME: dict[int, str] = {
    0: "anger", 1: "disgust", 2: "fear", 3: "happiness", 4: "neutral", 5: "sadness", 6: "surprise",
}
RAFDB_TEST_PREFIX: str = "test_"
NEUTRAL_SOURCE_PREFIX: str = "affectnet.zip"
SHARD_COUNT: int = 5

LICENSE_NOTE: str = (
    "RAF-DB's original license requires a research-use EULA with the authors; this HF "
    "reupload carries no explicit license tag. Used here for internal, non-redistributed "
    "benchmarking only — a known gray area, flagged not resolved (mirrors this project's "
    "existing pattern for non-permissive dependencies)."
)
NEUTRAL_PROVENANCE_NOTE: str = (
    "The 'neutral' class in this mirror is sourced entirely from AffectNet, not RAF-DB "
    "(verified: every neutral row's image path starts with 'affectnet.zip', zero "
    "train_/test_ RAF-DB rows). All other six classes are genuine RAF-DB test-partition "
    "images (path prefix 'test_')."
)


def _download_shards() -> list[Path]:
    """Download all five RAF-DB parquet shards from the HF hub."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return [
        Path(hf_hub_download(LAYER_A_DATASET_ID, f"data/train-0000{i}-of-00005.parquet", repo_type="dataset"))
        for i in range(SHARD_COUNT)
    ]


class _RawRow(BaseModel):
    """One dataset row's decoded fields, before any sampling decision."""

    model_config = {"arbitrary_types_allowed": True}

    image_path_field: str
    image_bytes: bytes
    label_id: int


def _read_rows(shard_paths: list[Path]) -> list[_RawRow]:
    """Read image bytes, path, and label from every parquet shard."""
    rows: list[_RawRow] = []
    for shard_path in shard_paths:
        table = pq.read_table(shard_path, columns=["image", "label"])
        images, labels = table.column("image").to_pylist(), table.column("label").to_pylist()
        rows += [
            _RawRow(image_path_field=img["path"], image_bytes=img["bytes"], label_id=label)
            for img, label in zip(images, labels)
        ]
    return rows


def _is_eligible(row: _RawRow, raw_class: str) -> bool:
    """Return True if a row belongs to its class's intended source partition."""
    if raw_class == "neutral":
        return row.image_path_field.startswith(NEUTRAL_SOURCE_PREFIX)
    return row.image_path_field.startswith(RAFDB_TEST_PREFIX)


def _group_eligible_rows(rows: list[_RawRow]) -> dict[str, list[_RawRow]]:
    """Group rows by RAF-DB raw class name, keeping only each class's eligible source."""
    grouped: dict[str, list[_RawRow]] = {name: [] for name in RAFDB_LABEL_ID_TO_NAME.values()}
    for row in rows:
        raw_class = RAFDB_LABEL_ID_TO_NAME[row.label_id]
        if _is_eligible(row, raw_class):
            grouped[raw_class].append(row)
    return grouped


class LayerASample(BaseModel):
    """The written Layer A image manifest: photo paths and their true canonical labels."""

    image_paths: list[Path]
    true_labels: dict[str, str]


class ClassSampleRequest(BaseModel):
    """One RAF-DB raw class's eligible rows, paired with the write destination."""

    model_config = {"arbitrary_types_allowed": True}

    raw_class: str
    rows: list[_RawRow]
    output_dir: Path


def _write_class_sample(request: ClassSampleRequest) -> LayerASample:
    """Deterministically sample and write one class's images to request.output_dir."""
    import random  # noqa: PLC0415

    rng = random.Random(LAYER_A_SEED)
    chosen = rng.sample(request.rows, min(LAYER_A_SAMPLES_PER_CLASS, len(request.rows)))
    canonical = canonicalize(request.raw_class, RAFDB_LABEL_MAP)
    paths, labels = [], {}
    for i, row in enumerate(chosen):
        out_path = request.output_dir / f"{request.raw_class}_{i:03d}.jpg"
        Image.open(io.BytesIO(row.image_bytes)).convert("RGB").save(out_path, "JPEG")
        paths.append(out_path)
        labels[out_path.name] = canonical
    return LayerASample(image_paths=paths, true_labels=labels)


def build_layer_a_sample(output_dir: Path) -> LayerASample:
    """Download RAF-DB, sample LAYER_A_SAMPLES_PER_CLASS per class, write crops to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = _group_eligible_rows(_read_rows(_download_shards()))
    samples = [
        _write_class_sample(ClassSampleRequest(raw_class=raw_class, rows=rows, output_dir=output_dir))
        for raw_class, rows in grouped.items()
    ]
    all_paths = [p for s in samples for p in s.image_paths]
    all_labels = {name: label for s in samples for name, label in s.true_labels.items()}
    return LayerASample(image_paths=all_paths, true_labels=all_labels)


# ---------------------------------------------------------------------------
# Metrics: consume runner readings against LayerASample.true_labels
# ---------------------------------------------------------------------------


class LayerAResult(BaseModel):
    """One model's full Layer A scoring: overall + per-class + sub-analysis metrics."""

    model: str
    n: int
    accuracy: float
    macro_f1: float
    per_class_accuracy: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    sub_analysis_macro_f1: float
    sub_analysis_confusion_matrix: dict[str, dict[str, int]]


class ScoredReadings(BaseModel):
    """A model's raw readings paired with the ground truth to score them against."""

    model: str
    true_labels: dict[str, str]
    raw_labels: dict[str, str]
    label_map: dict[str, str]


def score_layer_a(scored: ScoredReadings) -> LayerAResult:
    """Canonicalize raw predictions and compute all Layer A metrics for one model."""
    names = [name for name in scored.true_labels if name in scored.raw_labels]
    true = [scored.true_labels[n] for n in names]
    pred = [canonicalize(scored.raw_labels[n], scored.label_map) for n in names]
    pairs = LabelPairs(true_labels=true, pred_labels=pred)
    sub_pairs = LabelPairs(
        true_labels=[t for t in true if t in SUB_ANALYSIS_CLASSES],
        pred_labels=[p for t, p in zip(true, pred) if t in SUB_ANALYSIS_CLASSES],
    )
    return LayerAResult(
        model=scored.model, n=len(names), accuracy=accuracy(pairs), macro_f1=macro_f1(pairs, CANONICAL_CLASSES),
        per_class_accuracy=per_class_accuracy(pairs, CANONICAL_CLASSES), confusion_matrix=build_confusion_matrix(pairs),
        sub_analysis_macro_f1=macro_f1(sub_pairs, SUB_ANALYSIS_CLASSES),
        sub_analysis_confusion_matrix=build_confusion_matrix(sub_pairs),
    )
