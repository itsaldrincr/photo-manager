"""Pure metric functions for the three-layer fair test: no I/O, no heavy imports."""

from __future__ import annotations

import statistics
from collections import Counter

from pydantic import BaseModel

from fair_expr_models import CanonicalLabel

# ---------------------------------------------------------------------------
# Layer A: classification metrics
# ---------------------------------------------------------------------------


class LabelPairs(BaseModel):
    """Matched true/predicted canonical labels for a set of images."""

    true_labels: list[str]
    pred_labels: list[str]


ConfusionMatrix = dict[str, dict[str, int]]


def build_confusion_matrix(pairs: LabelPairs) -> ConfusionMatrix:
    """Build a true-label -> predicted-label -> count confusion matrix."""
    matrix: ConfusionMatrix = {}
    for true_label, pred_label in zip(pairs.true_labels, pairs.pred_labels):
        row = matrix.setdefault(true_label, {})
        row[pred_label] = row.get(pred_label, 0) + 1
    return matrix


def per_class_accuracy(pairs: LabelPairs, classes: list[str]) -> dict[str, float]:
    """Return recall (per-class accuracy) for each class: TP / class support."""
    matrix = build_confusion_matrix(pairs)
    result: dict[str, float] = {}
    for label in classes:
        row = matrix.get(label, {})
        support = sum(row.values())
        result[label] = round(row.get(label, 0) / support, 4) if support else 0.0
    return result


def _precision_recall(pairs: LabelPairs, label: str) -> tuple[float, float]:
    """Return (precision, recall) for one class against the full pair set."""
    true_positive = sum(
        1 for t, p in zip(pairs.true_labels, pairs.pred_labels) if t == label and p == label
    )
    predicted_positive = sum(1 for p in pairs.pred_labels if p == label)
    actual_positive = sum(1 for t in pairs.true_labels if t == label)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    return precision, recall


def macro_f1(pairs: LabelPairs, classes: list[str]) -> float:
    """Return the unweighted mean per-class F1 score across the given classes."""
    scores: list[float] = []
    for label in classes:
        precision, recall = _precision_recall(pairs, label)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def accuracy(pairs: LabelPairs) -> float:
    """Return overall accuracy across all pairs."""
    if not pairs.true_labels:
        return 0.0
    correct = sum(1 for t, p in zip(pairs.true_labels, pairs.pred_labels) if t == p)
    return round(correct / len(pairs.true_labels), 4)


def subset_pairs(pairs: LabelPairs, classes: list[str]) -> LabelPairs:
    """Restrict a pair set to rows whose true label is in the given class subset."""
    kept = [(t, p) for t, p in zip(pairs.true_labels, pairs.pred_labels) if t in classes]
    return LabelPairs(true_labels=[t for t, _ in kept], pred_labels=[p for _, p in kept])


# ---------------------------------------------------------------------------
# Layer B: stability metrics
# ---------------------------------------------------------------------------


class FlipRateInput(BaseModel):
    """Baseline labels paired with one perturbation's labels, by matching index."""

    baseline_labels: list[str]
    perturbed_labels: list[str]


def flip_rate(flip_input: FlipRateInput) -> float:
    """Return the fraction of predictions that changed vs the unperturbed baseline."""
    pairs = list(zip(flip_input.baseline_labels, flip_input.perturbed_labels))
    if not pairs:
        return 0.0
    flips = sum(1 for base, pert in pairs if base != pert)
    return round(flips / len(pairs), 4)


class DriftInput(BaseModel):
    """Baseline values paired with one perturbation's values, by matching index."""

    baseline_values: list[float]
    perturbed_values: list[float]


def mean_abs_drift(drift_input: DriftInput) -> float:
    """Return the mean absolute delta between baseline and perturbed values."""
    pairs = list(zip(drift_input.baseline_values, drift_input.perturbed_values))
    if not pairs:
        return 0.0
    deltas = [abs(base - pert) for base, pert in pairs]
    return round(sum(deltas) / len(deltas), 4)


# ---------------------------------------------------------------------------
# Layer C: distribution / separation metrics
# ---------------------------------------------------------------------------


def label_histogram(labels: list[str]) -> dict[str, int]:
    """Count occurrences of each canonical label."""
    return dict(Counter(labels))


NEGATIVE_LABELS: set[CanonicalLabel] = {"sad", "angry", "fear", "disgust"}


def happy_rate(labels: list[str]) -> float:
    """Return the fraction of labels equal to 'happy'."""
    return round(labels.count("happy") / len(labels), 4) if labels else 0.0


def negative_rate(labels: list[str]) -> float:
    """Return the fraction of labels in the negative-emotion set."""
    if not labels:
        return 0.0
    negative = sum(1 for label in labels if label in NEGATIVE_LABELS)
    return round(negative / len(labels), 4)


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, population stdev) for a list of values, 0.0 for <2 samples."""
    if not values:
        return 0.0, 0.0
    mean = round(sum(values) / len(values), 4)
    stdev = round(statistics.pstdev(values), 4) if len(values) >= 2 else 0.0
    return mean, stdev


class CohensDInput(BaseModel):
    """Two independent samples to compare via Cohen's d."""

    sample_a: list[float]
    sample_b: list[float]


def cohens_d(d_input: CohensDInput) -> float:
    """Return Cohen's d (standardized mean difference) between two samples."""
    a, b = d_input.sample_a, d_input.sample_b
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    var_a, var_b = statistics.pvariance(a), statistics.pvariance(b)
    pooled_std = ((var_a * len(a) + var_b * len(b)) / (len(a) + len(b))) ** 0.5
    return round((mean_a - mean_b) / pooled_std, 4) if pooled_std else 0.0
