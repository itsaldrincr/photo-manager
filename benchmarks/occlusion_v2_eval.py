"""Analysis driver: evaluate occlusion-v2 candidate signals against weak labels.

Pure arithmetic — no model inference. Consumes candidate signals from
occlusion_v2_signals.py and Gemma double-pass occlusion weak labels, joins on
photo name, and for each candidate signal (and the best pair as a z-score
ensemble) computes ROC-AUC plus the best-F1 operating point subject to the
clean-false-flag ceiling. Applies the real-corpus half of the Task B gate:

    a candidate clears the real-corpus bar only if F1 >= REAL_F1_MIN at an
    operating point whose clean-face false-flag rate <= MAX_CLEAN_FALSE_FLAG.

The synthetic half (F1 within 0.05 of the texture ratio's 0.776) is only worth
computing for a candidate that clears the real bar, and requires regenerating
the synthetic occluder set (git dde308a); this driver reports whether the real
bar was cleared and hands off that decision.

Weak labels are MODEL-GENERATED (labeler: gemma-4-12b); trust is double-pass
agreement only, never self-confidence (benchmarks/LOG.md 2026-07-18).

Usage:
    python3 benchmarks/occlusion_v2_eval.py <config.json> <report.md>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from occlusion_v2_signals import SignalRunOutput
from weaklabel_models import WeaklabelRunOutput, compute_agreement

REAL_F1_MIN: float = 0.55
MAX_CLEAN_FALSE_FLAG: float = 0.10
SYNTHETIC_F1_TARGET: float = 0.776
SYNTHETIC_F1_TOLERANCE: float = 0.05
_CANDIDATES: list[str] = [
    "blendshape_std", "blendshape_mean",
    "skin_outside_hull_frac", "boundary_edge_density", "texture_ratio",
]


class EvalConfig(BaseModel):
    """Paths to every input artifact of the occlusion-v2 evaluation."""

    signals_path: Path
    labels_v1_path: Path
    labels_v2_path: Path


class LabeledSignals(BaseModel):
    """One photo joined across weak label and all candidate signal values."""

    name: str
    is_occluded: bool
    signals: dict[str, float]


class OperatingPoint(BaseModel):
    """Best-F1 operating point for one candidate under the clean-FP ceiling."""

    candidate: str
    auc: float
    occluded_when_high: bool
    threshold: float
    precision: float
    recall: float
    f1: float
    clean_false_flag_rate: float


def _load_agreement(config: EvalConfig) -> list:
    """Return double-pass-agreed occlusion labels."""
    run_v1 = WeaklabelRunOutput.model_validate_json(config.labels_v1_path.read_text())
    run_v2 = WeaklabelRunOutput.model_validate_json(config.labels_v2_path.read_text())
    return compute_agreement((run_v1, run_v2)).agreed


def _load_rows(config: EvalConfig) -> list[LabeledSignals]:
    """Join agreed weak labels with candidate signals on photo name."""
    agreed = _load_agreement(config)
    signals = SignalRunOutput.model_validate_json(config.signals_path.read_text())
    by_name = {r.name: r for r in signals.readings if r.has_face}
    rows: list[LabeledSignals] = []
    for label in agreed:
        reading = by_name.get(label.name)
        if reading is None:
            continue
        values = {c: getattr(reading, c) for c in _CANDIDATES}
        if any(v is None for v in values.values()):
            continue
        rows.append(LabeledSignals(
            name=label.name, is_occluded=label.label == "True", signals=values,
        ))
    return rows


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return ROC-AUC of scores vs binary labels via the rank statistic."""
    pos, neg = labels.sum(), (~labels.astype(bool)).sum()
    if pos == 0 or neg == 0:
        return 0.5
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels.astype(bool)].sum() - pos * (pos + 1) / 2) / (pos * neg))


class _Column(BaseModel):
    """An occluded-positive score column plus the aligned label vector."""

    model_config = {"arbitrary_types_allowed": True}

    positives: object
    labels: object


def _f1_at(threshold: float, column: _Column) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for `predicted = positives >= threshold`."""
    positives, labels = column.positives, column.labels
    pred = positives >= threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _clean_false_flag(threshold: float, column: _Column) -> float:
    """Return the share of clean (non-occluded) faces flagged at this threshold."""
    clean = ~column.labels
    if clean.sum() == 0:
        return 0.0
    return float(((column.positives >= threshold) & clean).sum()) / float(clean.sum())


def _best_operating_point(candidate: str, column: _Column) -> OperatingPoint:
    """Sweep thresholds; pick max-F1 point with clean-FP <= ceiling."""
    positives = np.asarray(column.positives, dtype=np.float64)
    labels = np.asarray(column.labels, dtype=bool)
    norm = _Column(positives=positives, labels=labels)
    auc = _auc(labels.astype(int), positives)
    best = OperatingPoint(
        candidate=candidate, auc=auc, occluded_when_high=True,
        threshold=float("inf"), precision=0.0, recall=0.0, f1=0.0, clean_false_flag_rate=0.0,
    )
    for threshold in np.unique(positives):
        if _clean_false_flag(threshold, norm) > MAX_CLEAN_FALSE_FLAG:
            continue
        precision, recall, f1 = _f1_at(threshold, norm)
        if f1 > best.f1:
            best = OperatingPoint(
                candidate=candidate, auc=auc, occluded_when_high=True, threshold=float(threshold),
                precision=precision, recall=recall, f1=f1,
                clean_false_flag_rate=_clean_false_flag(threshold, norm),
            )
    return best


def _occluded_positive_column(rows: list[LabeledSignals], candidate: str) -> _Column:
    """Return a score column oriented so higher = more occluded (AUC-guided)."""
    labels = np.array([r.is_occluded for r in rows], dtype=bool)
    raw = np.array([r.signals[candidate] for r in rows], dtype=np.float64)
    if _auc(labels.astype(int), raw) < 0.5:
        raw = -raw
    return _Column(positives=raw, labels=labels)


def _zscore(values: np.ndarray) -> np.ndarray:
    """Return z-scored values; zeros if degenerate."""
    std = values.std()
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def _evaluate_candidates(rows: list[LabeledSignals]) -> list[OperatingPoint]:
    """Return the operating point for every single candidate, best F1 first."""
    points = [
        _best_operating_point(c, _occluded_positive_column(rows, c))
        for c in _CANDIDATES
    ]
    return sorted(points, key=lambda p: p.f1, reverse=True)


def _best_pair_ensemble(rows: list[LabeledSignals], singles: list[OperatingPoint]) -> OperatingPoint:
    """Z-score-average the two highest-AUC candidates and evaluate the ensemble."""
    top_two = sorted(singles, key=lambda p: p.auc, reverse=True)[:2]
    labels = np.array([r.is_occluded for r in rows], dtype=bool)
    columns = [_occluded_positive_column(rows, p.candidate) for p in top_two]
    combined = np.mean([_zscore(np.asarray(c.positives, dtype=np.float64)) for c in columns], axis=0)
    name = "+".join(p.candidate for p in top_two)
    return _best_operating_point(f"ensemble({name})", _Column(positives=combined, labels=labels))


def _row_line(point: OperatingPoint) -> str:
    """Render one markdown table row for an operating point."""
    return (
        f"| {point.candidate} | {point.auc:.3f} | {point.threshold:.4g} | "
        f"{point.precision:.3f} | {point.recall:.3f} | {point.f1:.3f} | "
        f"{point.clean_false_flag_rate:.3f} |"
    )


class _ReportInput(BaseModel):
    """Everything the report renderer needs."""

    rows: list[LabeledSignals]
    singles: list[OperatingPoint]
    ensemble: OperatingPoint


def _verdict_line(best: OperatingPoint) -> str:
    """Return the real-corpus gate verdict line for the best candidate."""
    cleared = best.f1 >= REAL_F1_MIN and best.clean_false_flag_rate <= MAX_CLEAN_FALSE_FLAG
    if cleared:
        return (
            f"## Verdict: REAL BAR CLEARED by {best.candidate} "
            f"(F1 {best.f1:.3f} >= {REAL_F1_MIN}, clean-FP {best.clean_false_flag_rate:.3f} "
            f"<= {MAX_CLEAN_FALSE_FLAG}) — proceed to synthetic gate "
            f"(F1 within {SYNTHETIC_F1_TOLERANCE} of {SYNTHETIC_F1_TARGET})."
        )
    return (
        f"## Verdict: FAIL — best candidate {best.candidate} F1 {best.f1:.3f} "
        f"< {REAL_F1_MIN} at clean-FP <= {MAX_CLEAN_FALSE_FLAG}. No production change; "
        f"texture-ratio detect_occlusion stays."
    )


def _render_report(report_in: _ReportInput) -> str:
    """Render the full occlusion-v2 evaluation report as markdown."""
    occ = sum(1 for r in report_in.rows if r.is_occluded)
    all_points = report_in.singles + [report_in.ensemble]
    best = max(all_points, key=lambda p: p.f1)
    header = [
        "# Occlusion v2 candidate signals — Gemma weak labels",
        "",
        "PROVENANCE: labels are MODEL-GENERATED weak labels (labeler: gemma-4-12b,",
        "prompts occlusion_v1 + occlusion_v2, double-pass agreement). NOT human",
        "ground truth. Signals are cheap, computed from the same full-res image +",
        "MediaPipe landmarks the production detector consumes.",
        "",
        f"- Joined agreed labels with a detected face: {len(report_in.rows)} "
        f"({occ} occluded, {len(report_in.rows) - occ} clean)",
        f"- Operating points chosen to maximize F1 s.t. clean-FP <= {MAX_CLEAN_FALSE_FLAG}",
        "",
        "## Per-candidate operating points (higher score = more occluded)",
        "",
        "| candidate | ROC-AUC | threshold | precision | recall | F1 | clean-FP |",
        "|---|---|---|---|---|---|---|",
    ]
    body = [_row_line(p) for p in report_in.singles] + [_row_line(report_in.ensemble)]
    return "\n".join(header + body + ["", _verdict_line(best), ""])


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: occlusion_v2_eval.py <config.json> <report.md>")
    config = EvalConfig.model_validate_json(Path(sys.argv[1]).read_text())
    rows = _load_rows(config)
    singles = _evaluate_candidates(rows)
    ensemble = _best_pair_ensemble(rows, singles)
    report_in = _ReportInput(rows=rows, singles=singles, ensemble=ensemble)
    Path(sys.argv[2]).write_text(_render_report(report_in))
    best = max(singles + [ensemble], key=lambda p: p.f1)
    print(json.dumps({  # noqa: T201 — CLI result surface
        "best_candidate": best.candidate, "best_f1": best.f1,
        "clean_false_flag_rate": best.clean_false_flag_rate,
        "real_bar_cleared": best.f1 >= REAL_F1_MIN and best.clean_false_flag_rate <= MAX_CLEAN_FALSE_FLAG,
    }, indent=2))


if __name__ == "__main__":
    main()
