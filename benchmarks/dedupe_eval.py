"""Evaluate DINOv2-small embeddings against the production imagededup CNN +
dHash pipeline for near-duplicate/burst detection.

Measurement only — no production code is modified. Builds a labeled corpus
of controlled near-duplicates under a scratch directory (outside the repo),
scores it with three methods, and writes a decision report.

Usage:
    python3 benchmarks/dedupe_eval.py
"""

from __future__ import annotations

import gc
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import imagehash
import numpy as np
import torch
from PIL import Image, ImageEnhance
from pydantic import BaseModel, ConfigDict, Field

from cull.config import BLUR_CNN_SIMILARITY_EXACT, BLUR_DHASH_HAMMING_MAX
from cull.stage1.burst import cluster_by_time, read_timestamps
from cull.stage1.duplicate import DuplicateResult, _load_cnn, _unload_cnn, find_duplicates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_DIR: Path = Path("/Users/alrelador/Desktop/cull-test")
SCRATCH_ROOT: Path = Path(
    "/private/tmp/claude-501/-Users-alrelador/85225a69-a5ad-4e09-b879-f167553ba959/scratchpad"
)
EVAL_DIR: Path = SCRATCH_ROOT / "dedupe_eval" / "images"
REPORT_PATH: Path = Path(__file__).resolve().parent / "runs" / "dedupe_eval_report.md"

BASE_PHOTO_COUNT: int = 20
JPEG_REENCODE_QUALITY: int = 70
JPEG_HIGH_QUALITY: int = 95
RESIZE_SCALE: float = 0.70
CROP_FRACTION: float = 0.10
BRIGHTNESS_FACTOR: float = 1.10
ROTATION_DEGREES: float = 1.5
BURST_GAP_SECONDS: float = 0.5

CNN_SIMILARITY_THRESHOLD: float = BLUR_CNN_SIMILARITY_EXACT
DHASH_HAMMING_MAX: int = BLUR_DHASH_HAMMING_MAX
DHASH_BITS: int = 64
DHASH_SIMILARITY_THRESHOLD: float = 1.0 - (DHASH_HAMMING_MAX / DHASH_BITS)

DINOV2_MODEL_ID: str = "facebook/dinov2-small"
DINOV2_BATCH_SIZE: int = 8
DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"
THRESHOLD_SWEEP: list[float] = [round(0.80 + 0.01 * i, 2) for i in range(20)]

BYTES_PER_FLOAT32: int = 4
BYTES_PER_MB: int = 1_000_000
DISAGREEMENT_EXAMPLE_LIMIT: int = 20
LATENCY_BUDGET_MS: float = 250.0
RECALL_GAIN_THRESHOLD: float = 0.15


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PhotoRecord(BaseModel):
    """One photo in the eval corpus: its group and variant type."""

    name: str
    group_id: str
    variant_type: str


class LatencyResult(BaseModel):
    """Per-photo latency samples for one method."""

    mean_ms: float
    p95_ms: float
    samples_ms: list[float] = Field(default_factory=list)


class OverallMetrics(BaseModel):
    """Aggregate pairwise precision/recall/F1 at one threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


class VariantMetrics(BaseModel):
    """Recall for one variant type at a method's chosen threshold."""

    variant_type: str
    recall: float
    matched: int
    total: int


class MethodReport(BaseModel):
    """Full evaluation results for one method."""

    method_name: str
    overall: OverallMetrics
    per_variant: list[VariantMetrics]
    timing: LatencyResult
    model_memory_mb: float
    disagreement_examples: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """Swap/augment/no-change decision with supporting rationale."""

    decision: str
    rationale: list[str]


class ConfusionCounts(BaseModel):
    """Raw pairwise confusion counts at one threshold."""

    tp: int
    fp: int
    fn: int
    tn: int


class ConfusionQuery(BaseModel):
    """Inputs for computing pairwise confusion counts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    similarity: np.ndarray
    group_ids: list[str]
    threshold: float


class VariantRecallQuery(BaseModel):
    """Inputs for computing per-variant-type recall."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    similarity: np.ndarray
    records: list[PhotoRecord]
    threshold: float


class SweepQuery(BaseModel):
    """Inputs for a best-F1 threshold sweep."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    similarity: np.ndarray
    group_ids: list[str]


class DecisionInput(BaseModel):
    """The three method reports under judgment."""

    cnn: MethodReport
    dhash: MethodReport
    dinov2: MethodReport


class ReportInput(BaseModel):
    """Everything needed to render the markdown report and console summary."""

    records: list[PhotoRecord]
    natural_burst_count: int
    cnn: MethodReport
    dhash: MethodReport
    dinov2: MethodReport
    recommendation: Recommendation


class VariantWriteInput(BaseModel):
    """Inputs for writing one transformed variant photo to disk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Any
    group_id: str
    variant_name: str
    eval_dir: Path


class EmbedJob(BaseModel):
    """Inputs for batch-embedding a corpus with DINOv2."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    records: list[PhotoRecord]
    eval_dir: Path
    processor: Any
    model: Any


class DisagreementQuery(BaseModel):
    """Inputs for finding pairs where a weak method misses and a strong one hits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    records: list[PhotoRecord]
    weak_similarity: np.ndarray
    weak_threshold: float
    strong_similarity: np.ndarray
    strong_threshold: float


# ---------------------------------------------------------------------------
# Test-set construction
# ---------------------------------------------------------------------------


def select_base_photos(source_dir: Path) -> list[Path]:
    """Return up to BASE_PHOTO_COUNT source JPEGs, sorted for determinism."""
    return sorted(source_dir.glob("*.JPG"))[:BASE_PHOTO_COUNT]


def _variant_reencode(image: Image.Image) -> Image.Image:
    """Return the image unchanged; JPEG re-encode quality is applied at save time."""
    return image


def _variant_resize(image: Image.Image) -> Image.Image:
    """Resize the image to RESIZE_SCALE of its original dimensions."""
    width, height = image.size
    new_size = (int(width * RESIZE_SCALE), int(height * RESIZE_SCALE))
    return image.resize(new_size, Image.LANCZOS)


def _variant_crop(image: Image.Image) -> Image.Image:
    """Crop CROP_FRACTION off the right edge of the image."""
    width, height = image.size
    kept_width = int(width * (1 - CROP_FRACTION))
    return image.crop((0, 0, kept_width, height))


def _variant_brightness(image: Image.Image) -> Image.Image:
    """Increase brightness by BRIGHTNESS_FACTOR."""
    return ImageEnhance.Brightness(image).enhance(BRIGHTNESS_FACTOR)


def _variant_rotate(image: Image.Image) -> Image.Image:
    """Rotate the image by ROTATION_DEGREES, keeping canvas size fixed."""
    return image.rotate(ROTATION_DEGREES, resample=Image.BICUBIC, fillcolor=(0, 0, 0))


VARIANT_TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "reencode": _variant_reencode,
    "resize": _variant_resize,
    "crop": _variant_crop,
    "brightness": _variant_brightness,
    "rotation": _variant_rotate,
}


def _write_variant(
    transform: Callable[[Image.Image], Image.Image], spec: VariantWriteInput
) -> PhotoRecord:
    """Apply one transform and save the resulting variant photo."""
    out_path = spec.eval_dir / f"{spec.group_id}__{spec.variant_name}.jpg"
    variant_image = transform(spec.image)
    quality = JPEG_REENCODE_QUALITY if spec.variant_name == "reencode" else JPEG_HIGH_QUALITY
    variant_image.save(out_path, "JPEG", quality=quality)
    return PhotoRecord(name=out_path.name, group_id=spec.group_id, variant_type=spec.variant_name)


def build_group_files(base_path: Path, eval_dir: Path) -> list[PhotoRecord]:
    """Write one base photo plus its controlled variants into eval_dir."""
    group_id = base_path.stem
    base_out = eval_dir / f"{group_id}__base.jpg"
    shutil.copy2(base_path, base_out)
    records = [PhotoRecord(name=base_out.name, group_id=group_id, variant_type="base")]
    image = Image.open(base_path).convert("RGB")
    for variant_name, transform in VARIANT_TRANSFORMS.items():
        spec = VariantWriteInput(
            image=image, group_id=group_id, variant_name=variant_name, eval_dir=eval_dir
        )
        records.append(_write_variant(transform, spec))
    return records


def build_test_set(source_dir: Path, eval_dir: Path) -> list[PhotoRecord]:
    """Build the labeled near-duplicate corpus under eval_dir."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    records: list[PhotoRecord] = []
    for base_path in select_base_photos(source_dir):
        records.extend(build_group_files(base_path, eval_dir))
    return records


def find_natural_bursts(photo_paths: list[Path]) -> list[list[Path]]:
    """Return genuine EXIF burst groups (<BURST_GAP_SECONDS apart) in the source corpus."""
    timestamped = read_timestamps(photo_paths)
    return cluster_by_time(timestamped, BURST_GAP_SECONDS)


# ---------------------------------------------------------------------------
# Similarity matrices
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine similarity matrix for a set of vectors."""
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return norm @ norm.T


def build_cnn_similarity(
    records: list[PhotoRecord], eval_dir: Path
) -> tuple[np.ndarray, DuplicateResult]:
    """Compute pairwise CNN cosine similarity via the production duplicate.py path."""
    result = find_duplicates(eval_dir)
    vectors = np.stack([np.asarray(result.encodings[str(eval_dir / r.name)]) for r in records])
    return cosine_similarity_matrix(vectors), result


def build_dhash_similarity(records: list[PhotoRecord], eval_dir: Path) -> np.ndarray:
    """Compute pairwise dHash similarity (1 - normalized Hamming distance)."""
    hashes = [imagehash.dhash(Image.open(eval_dir / r.name)) for r in records]
    n = len(hashes)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = 1.0 - (hashes[i] - hashes[j]) / DHASH_BITS
    return sim


def _load_dinov2() -> tuple[Any, Any]:
    """Load the DINOv2-small processor and model onto DEVICE."""
    from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID, use_fast=True)
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID)
    model.to(DEVICE).eval()
    return processor, model


def _unload_dinov2(processor: Any, model: Any) -> None:
    """Drop DINOv2 references and free memory."""
    del processor, model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()


def _embed_all(job: EmbedJob) -> np.ndarray:
    """Batch-embed the corpus with DINOv2, returning pooled CLS embeddings."""
    vectors = []
    for start in range(0, len(job.records), DINOV2_BATCH_SIZE):
        batch = job.records[start : start + DINOV2_BATCH_SIZE]
        images = [Image.open(job.eval_dir / r.name).convert("RGB") for r in batch]
        inputs = job.processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = job.model(**inputs)
        vectors.append(out.pooler_output.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def build_dinov2_similarity(records: list[PhotoRecord], eval_dir: Path) -> np.ndarray:
    """Compute pairwise DINOv2-small cosine similarity for the corpus."""
    processor, model = _load_dinov2()
    embeddings = _embed_all(EmbedJob(records=records, eval_dir=eval_dir, processor=processor, model=model))
    _unload_dinov2(processor, model)
    return cosine_similarity_matrix(embeddings)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def pairwise_confusion(query: ConfusionQuery) -> ConfusionCounts:
    """Compute TP/FP/FN/TN across all photo pairs at query.threshold."""
    n = len(query.group_ids)
    iu, ju = np.triu_indices(n, k=1)
    same = np.array([query.group_ids[i] == query.group_ids[j] for i, j in zip(iu, ju)])
    predicted = query.similarity[iu, ju] >= query.threshold
    return ConfusionCounts(
        tp=int(np.sum(same & predicted)),
        fp=int(np.sum(predicted & ~same)),
        fn=int(np.sum(same & ~predicted)),
        tn=int(np.sum(~predicted & ~same)),
    )


def confusion_to_metrics(counts: ConfusionCounts, threshold: float) -> OverallMetrics:
    """Reduce confusion counts into precision/recall/F1 at threshold."""
    precision = counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if (counts.tp + counts.fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return OverallMetrics(
        threshold=threshold, precision=precision, recall=recall, f1=f1,
        true_positive=counts.tp, false_positive=counts.fp,
        false_negative=counts.fn, true_negative=counts.tn,
    )


def per_variant_recall(query: VariantRecallQuery) -> list[VariantMetrics]:
    """Return recall of each base-vs-variant pair, grouped by variant type."""
    base_index = {r.group_id: i for i, r in enumerate(query.records) if r.variant_type == "base"}
    metrics: list[VariantMetrics] = []
    for variant_type in VARIANT_TRANSFORMS:
        matched = total = 0
        for i, record in enumerate(query.records):
            if record.variant_type != variant_type:
                continue
            base_i = base_index[record.group_id]
            total += 1
            matched += int(query.similarity[i, base_i] >= query.threshold)
        recall = matched / total if total else 0.0
        metrics.append(VariantMetrics(variant_type=variant_type, recall=recall, matched=matched, total=total))
    return metrics


def sweep_best_threshold(query: SweepQuery) -> OverallMetrics:
    """Return the OverallMetrics for the threshold with the highest F1."""
    best: OverallMetrics | None = None
    for threshold in THRESHOLD_SWEEP:
        counts = pairwise_confusion(
            ConfusionQuery(similarity=query.similarity, group_ids=query.group_ids, threshold=threshold)
        )
        metrics = confusion_to_metrics(counts, threshold)
        if best is None or metrics.f1 > best.f1:
            best = metrics
    assert best is not None
    return best


def find_disagreements(query: DisagreementQuery) -> list[str]:
    """Return base-vs-variant pairs where the weak method misses but the strong one hits."""
    base_index = {r.group_id: i for i, r in enumerate(query.records) if r.variant_type == "base"}
    examples: list[str] = []
    for i, record in enumerate(query.records):
        if record.variant_type == "base":
            continue
        j = base_index[record.group_id]
        weak_score = query.weak_similarity[i, j]
        strong_score = query.strong_similarity[i, j]
        if weak_score < query.weak_threshold <= strong_score:
            examples.append(
                f"{record.group_id}/{record.variant_type}: weak_sim={weak_score:.3f} "
                f"(miss) vs strong_sim={strong_score:.3f} (hit)"
            )
    return examples


def model_param_memory_mb(model: Any) -> float:
    """Return the theoretical fp32 parameter memory footprint of a torch model in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * BYTES_PER_FLOAT32 / BYTES_PER_MB


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


def _latency_from_samples(samples: list[float]) -> LatencyResult:
    """Reduce raw millisecond samples into mean/p95 LatencyResult."""
    array = np.array(samples)
    return LatencyResult(mean_ms=float(array.mean()), p95_ms=float(np.percentile(array, 95)), samples_ms=samples)


def measure_cnn_latency(records: list[PhotoRecord], eval_dir: Path) -> LatencyResult:
    """Time per-photo CNN encode_image calls using the production CNN instance."""
    cnn = _load_cnn()
    if cnn is None:
        raise RuntimeError("imagededup CNN unavailable")
    samples = []
    for record in records:
        start = time.perf_counter()
        cnn.encode_image(str(eval_dir / record.name))  # type: ignore[attr-defined]
        samples.append((time.perf_counter() - start) * 1000)
    return _latency_from_samples(samples)


def measure_dhash_latency(records: list[PhotoRecord], eval_dir: Path) -> LatencyResult:
    """Time per-photo dHash computation."""
    samples = []
    for record in records:
        start = time.perf_counter()
        imagehash.dhash(Image.open(eval_dir / record.name))
        samples.append((time.perf_counter() - start) * 1000)
    return _latency_from_samples(samples)


def measure_dinov2_latency(records: list[PhotoRecord], eval_dir: Path) -> LatencyResult:
    """Time per-photo single-image DINOv2 forward passes."""
    processor, model = _load_dinov2()
    samples = []
    for record in records:
        image = Image.open(eval_dir / record.name).convert("RGB")
        start = time.perf_counter()
        inputs = processor(images=[image], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inputs)
        samples.append((time.perf_counter() - start) * 1000)
    _unload_dinov2(processor, model)
    return _latency_from_samples(samples)


def cnn_model_memory_mb() -> float:
    """Return the fp32 parameter memory footprint of the production CNN in MB."""
    cnn = _load_cnn()
    if cnn is None:
        raise RuntimeError("imagededup CNN unavailable")
    memory = model_param_memory_mb(cnn.model)  # type: ignore[attr-defined]
    _unload_cnn()
    return memory


def dinov2_model_memory_mb() -> float:
    """Return the fp32 parameter memory footprint of DINOv2-small in MB."""
    processor, model = _load_dinov2()
    memory = model_param_memory_mb(model)
    _unload_dinov2(processor, model)
    return memory


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------


def evaluate_cnn(records: list[PhotoRecord]) -> MethodReport:
    """Evaluate the production imagededup CNN duplicate detector."""
    similarity, result = build_cnn_similarity(records, EVAL_DIR)
    logger.info("production find_duplicates() reported %d groups", len(result.duplicate_groups))
    group_ids = [r.group_id for r in records]
    query = ConfusionQuery(similarity=similarity, group_ids=group_ids, threshold=CNN_SIMILARITY_THRESHOLD)
    overall = confusion_to_metrics(pairwise_confusion(query), CNN_SIMILARITY_THRESHOLD)
    variant_query = VariantRecallQuery(similarity=similarity, records=records, threshold=CNN_SIMILARITY_THRESHOLD)
    per_variant = per_variant_recall(variant_query)
    timing = measure_cnn_latency(records, EVAL_DIR)
    memory = cnn_model_memory_mb()
    return MethodReport(
        method_name="imagededup CNN (production)", overall=overall, per_variant=per_variant,
        timing=timing, model_memory_mb=memory,
    )


def evaluate_dhash(records: list[PhotoRecord]) -> MethodReport:
    """Evaluate the production dHash burst-confirmation gate as a pairwise classifier."""
    similarity = build_dhash_similarity(records, EVAL_DIR)
    group_ids = [r.group_id for r in records]
    query = ConfusionQuery(similarity=similarity, group_ids=group_ids, threshold=DHASH_SIMILARITY_THRESHOLD)
    overall = confusion_to_metrics(pairwise_confusion(query), DHASH_SIMILARITY_THRESHOLD)
    variant_query = VariantRecallQuery(similarity=similarity, records=records, threshold=DHASH_SIMILARITY_THRESHOLD)
    per_variant = per_variant_recall(variant_query)
    timing = measure_dhash_latency(records, EVAL_DIR)
    return MethodReport(
        method_name="dHash gate (production)", overall=overall, per_variant=per_variant,
        timing=timing, model_memory_mb=0.0,
    )


def evaluate_dinov2(records: list[PhotoRecord], dhash_similarity: np.ndarray) -> MethodReport:
    """Evaluate facebook/dinov2-small with a best-F1 threshold swept on this corpus."""
    similarity = build_dinov2_similarity(records, EVAL_DIR)
    group_ids = [r.group_id for r in records]
    overall = sweep_best_threshold(SweepQuery(similarity=similarity, group_ids=group_ids))
    variant_query = VariantRecallQuery(similarity=similarity, records=records, threshold=overall.threshold)
    per_variant = per_variant_recall(variant_query)
    timing = measure_dinov2_latency(records, EVAL_DIR)
    memory = dinov2_model_memory_mb()
    disagreements = find_disagreements(DisagreementQuery(
        records=records, weak_similarity=dhash_similarity, weak_threshold=DHASH_SIMILARITY_THRESHOLD,
        strong_similarity=similarity, strong_threshold=overall.threshold,
    ))
    return MethodReport(
        method_name="facebook/dinov2-small", overall=overall, per_variant=per_variant,
        timing=timing, model_memory_mb=memory, disagreement_examples=disagreements,
    )


# ---------------------------------------------------------------------------
# Decision rubric
# ---------------------------------------------------------------------------


def _variant_recall(report: MethodReport, variant_name: str) -> float:
    """Return one method's recall for a single variant type."""
    return next(v.recall for v in report.per_variant if v.variant_type == variant_name)


def _hardest_variant_gap(inputs: DecisionInput) -> float:
    """Return DINOv2's largest recall gain over both baselines on crop/rotation."""
    gaps = []
    for variant_name in ("crop", "rotation"):
        cnn_r = _variant_recall(inputs.cnn, variant_name)
        dhash_r = _variant_recall(inputs.dhash, variant_name)
        dinov2_r = _variant_recall(inputs.dinov2, variant_name)
        gaps.append(dinov2_r - max(cnn_r, dhash_r))
    return max(gaps)


def decide_recommendation(inputs: DecisionInput) -> Recommendation:
    """Apply the ADD-DINOV2-TIER vs NO-CHANGE decision rubric."""
    gap = _hardest_variant_gap(inputs)
    latency_ok = inputs.dinov2.timing.mean_ms <= LATENCY_BUDGET_MS
    material_gain = gap >= RECALL_GAIN_THRESHOLD
    decision = "ADD-DINOV2-TIER" if material_gain and latency_ok else "NO-CHANGE"
    rationale = [
        f"largest DINOv2 recall gain over CNN/dhash on crop+rotation: {gap:+.3f} "
        f"(materiality threshold: {RECALL_GAIN_THRESHOLD:+.3f})",
        f"DINOv2 mean latency {inputs.dinov2.timing.mean_ms:.2f}ms/photo vs budget "
        f"{LATENCY_BUDGET_MS:.0f}ms/photo ({'OK' if latency_ok else 'OVER BUDGET'})",
        f"overall recall — CNN {inputs.cnn.overall.recall:.3f}, dhash "
        f"{inputs.dhash.overall.recall:.3f}, DINOv2 {inputs.dinov2.overall.recall:.3f}",
    ]
    return Recommendation(decision=decision, rationale=rationale)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _render_header(outcome: ReportInput) -> str:
    """Render the report title and corpus/method description."""
    group_count = sum(1 for r in outcome.records if r.variant_type == "base")
    return (
        f"# DINOv2-small vs imagededup CNN + dHash — Dedupe Eval\n\n"
        f"Corpus: {len(outcome.records)} photos across {group_count} groups "
        f"({group_count} base + {len(VARIANT_TRANSFORMS)} controlled variants each). "
        f"Natural EXIF bursts (<{BURST_GAP_SECONDS}s apart) found in source corpus: "
        f"{outcome.natural_burst_count}.\n\n"
        f"- Method A1: production `imagededup` CNN (`cull/stage1/duplicate.py`, "
        f"threshold={CNN_SIMILARITY_THRESHOLD}).\n"
        f"- Method A2: production dHash gate (`cull/stage1/burst.py`, "
        f"hamming<={DHASH_HAMMING_MAX} -> similarity>={DHASH_SIMILARITY_THRESHOLD:.3f}).\n"
        f"- Method B: `facebook/dinov2-small` cosine similarity, best-F1 threshold "
        f"swept on this corpus."
    )


def _render_overall_table(outcome: ReportInput) -> str:
    """Render the overall precision/recall/F1 table."""
    lines = [
        "## Overall pairwise precision / recall / F1", "",
        "| method | threshold | precision | recall | f1 | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for report in (outcome.cnn, outcome.dhash, outcome.dinov2):
        o = report.overall
        lines.append(
            f"| {report.method_name} | {o.threshold:.3f} | {o.precision:.3f} | "
            f"{o.recall:.3f} | {o.f1:.3f} | {o.true_positive} | {o.false_positive} | "
            f"{o.false_negative} | {o.true_negative} |"
        )
    return "\n".join(lines)


def _variant_recall_cell(report: MethodReport, variant_name: str) -> str:
    """Render one recall cell as 'recall (matched/total)'."""
    match = next(v for v in report.per_variant if v.variant_type == variant_name)
    return f"{match.recall:.3f} ({match.matched}/{match.total})"


def _render_variant_table(outcome: ReportInput) -> str:
    """Render the per-variant-type recall table."""
    reports = [outcome.cnn, outcome.dhash, outcome.dinov2]
    header = "| variant | " + " | ".join(r.method_name for r in reports) + " |"
    lines = ["## Recall by variant type", "", header, "|---|" + "---|" * len(reports)]
    for variant_name in VARIANT_TRANSFORMS:
        cells = [_variant_recall_cell(r, variant_name) for r in reports]
        lines.append(f"| {variant_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_latency_table(outcome: ReportInput) -> str:
    """Render the per-method latency and model memory table."""
    lines = [
        "## Latency and model memory", "",
        "| method | mean ms/photo | p95 ms/photo | model memory (MB, fp32) |",
        "|---|---|---|---|",
    ]
    for report in (outcome.cnn, outcome.dhash, outcome.dinov2):
        lines.append(
            f"| {report.method_name} | {report.timing.mean_ms:.2f} | "
            f"{report.timing.p95_ms:.2f} | {report.model_memory_mb:.2f} |"
        )
    return "\n".join(lines)


def _render_disagreements(outcome: ReportInput) -> str:
    """Render base-vs-variant pairs where dHash missed but DINOv2 caught it."""
    lines = ["## Where hash fails but DINOv2 succeeds", ""]
    examples = outcome.dinov2.disagreement_examples
    if not examples:
        lines.append("No cases found where the dHash gate missed and DINOv2 caught it.")
        return "\n".join(lines)
    lines.extend(f"- {example}" for example in examples[:DISAGREEMENT_EXAMPLE_LIMIT])
    return "\n".join(lines)


def _render_recommendation(outcome: ReportInput) -> str:
    """Render the final decision and its rationale."""
    rec = outcome.recommendation
    lines = [f"## Decision: {rec.decision}", ""]
    lines.extend(f"- {reason}" for reason in rec.rationale)
    return "\n".join(lines)


def write_report(outcome: ReportInput) -> None:
    """Render the full markdown report to REPORT_PATH."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        _render_header(outcome), _render_overall_table(outcome), _render_variant_table(outcome),
        _render_latency_table(outcome), _render_disagreements(outcome), _render_recommendation(outcome),
    ]
    REPORT_PATH.write_text("\n\n".join(sections) + "\n")


def print_summary(outcome: ReportInput) -> None:
    """Print a compact console summary of the eval results."""
    for report in (outcome.cnn, outcome.dhash, outcome.dinov2):
        o = report.overall
        logger.info(
            "%s: F1=%.3f P=%.3f R=%.3f @thr=%.3f | %.2fms/photo | %.1fMB",
            report.method_name, o.f1, o.precision, o.recall, o.threshold,
            report.timing.mean_ms, report.model_memory_mb,
        )
    logger.info("Decision: %s", outcome.recommendation.decision)
    for reason in outcome.recommendation.rationale:
        logger.info("  - %s", reason)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full DINOv2-vs-hash dedupe evaluation and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = build_test_set(SOURCE_DIR, EVAL_DIR)
    natural_bursts = find_natural_bursts(select_base_photos(SOURCE_DIR))
    logger.info("Built %d photos; %d natural EXIF bursts found", len(records), len(natural_bursts))

    cnn_report = evaluate_cnn(records)
    gc.collect()
    dhash_report = evaluate_dhash(records)
    gc.collect()
    dhash_similarity = build_dhash_similarity(records, EVAL_DIR)
    dinov2_report = evaluate_dinov2(records, dhash_similarity)
    gc.collect()

    recommendation = decide_recommendation(
        DecisionInput(cnn=cnn_report, dhash=dhash_report, dinov2=dinov2_report)
    )
    outcome = ReportInput(
        records=records, natural_burst_count=len(natural_bursts),
        cnn=cnn_report, dhash=dhash_report, dinov2=dinov2_report, recommendation=recommendation,
    )
    write_report(outcome)
    print_summary(outcome)
    logger.info("Report written to %s", REPORT_PATH)


if __name__ == "__main__":
    main()
