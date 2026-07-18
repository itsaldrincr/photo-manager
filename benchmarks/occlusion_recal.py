"""Analysis driver: recalibrate PORTRAIT_FACE_OCCLUSION_MIN from weak labels.

Pure arithmetic — no model inference. Consumes artifacts produced by
gemma_weaklabel_runner.py (two paraphrased passes) and
occlusion_ratio_runner.py (production ratios for the real corpus AND the
synthetic occluder eval set), sweeps candidate thresholds, and applies the
promotion gate:

    adopt a new threshold only if it improves F1 on the weak-labeled REAL
    corpus by >= 0.05 WITHOUT dropping synthetic F1 by more than 0.05.

Weak labels are MODEL-GENERATED (labeler: gemma-4-12b) — never presented as
human ground truth; trust comes from double-pass agreement only.

Usage:
    python3 benchmarks/occlusion_recal.py <config.json> <report.md>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

from occlusion_ratio_runner import RatioRunOutput
from weaklabel_models import AgreementResult, WeaklabelRunOutput, compute_agreement

SWEEP_START: float = 0.02
SWEEP_STOP: float = 0.80
SWEEP_STEP: float = 0.02
CURRENT_THRESHOLD: float = 0.32
REAL_F1_MIN_GAIN: float = 0.05
SYNTHETIC_F1_MAX_DROP: float = 0.05


class RecalConfig(BaseModel):
    """Paths to every input artifact of the recalibration analysis."""

    corpus_path: Path
    labels_v1_path: Path
    labels_v2_path: Path
    real_ratios_path: Path
    synthetic_ratios_path: Path
    synthetic_manifest_path: Path
    verdict_out_path: Path


class LabeledRatio(BaseModel):
    """One photo joined across weak label and production ratio."""

    name: str
    context: str
    is_occluded: bool
    ratio: float


class SweepPoint(BaseModel):
    """P/R/F1 of `occluded = ratio < threshold` at one threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float


def _prf1(threshold: float, rows: list[tuple[bool, float]]) -> SweepPoint:
    """Compute P/R/F1 at one threshold over (label, ratio) rows."""
    tp = sum(1 for occluded, ratio in rows if occluded and ratio < threshold)
    fp = sum(1 for occluded, ratio in rows if not occluded and ratio < threshold)
    fn = sum(1 for occluded, ratio in rows if occluded and ratio >= threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return SweepPoint(threshold=threshold, precision=precision, recall=recall, f1=f1)


def _sweep(rows: list[tuple[bool, float]]) -> list[SweepPoint]:
    """P/R/F1 at every candidate threshold in the sweep grid."""
    steps = int(round((SWEEP_STOP - SWEEP_START) / SWEEP_STEP)) + 1
    return [_prf1(round(SWEEP_START + i * SWEEP_STEP, 2), rows) for i in range(steps)]


def _load_agreement(config: RecalConfig) -> AgreementResult:
    """Compute double-pass agreement from the two labeling runs."""
    run_v1 = WeaklabelRunOutput.model_validate_json(config.labels_v1_path.read_text())
    run_v2 = WeaklabelRunOutput.model_validate_json(config.labels_v2_path.read_text())
    return compute_agreement((run_v1, run_v2))


def _load_real_rows(config: RecalConfig) -> list[LabeledRatio]:
    """Join agreed weak labels with production ratios on photo name."""
    agreement = _load_agreement(config)
    contexts = {r["name"]: r["context"] for r in json.loads(config.corpus_path.read_text())}
    ratios = RatioRunOutput.model_validate_json(config.real_ratios_path.read_text())
    ratio_by_name = {r.name: r.occlusion_ratio for r in ratios.readings if r.has_face}
    return [
        LabeledRatio(
            name=label.name, context=contexts.get(label.name, "?"),
            is_occluded=label.label == "True", ratio=ratio_by_name[label.name],
        )
        for label in agreement.agreed if ratio_by_name.get(label.name) is not None
    ]


def _load_synthetic_rows(config: RecalConfig) -> list[tuple[bool, float]]:
    """Join synthetic eval labels with production ratios on file name."""
    labels = {r["file"]: bool(r["label"]) for r in json.loads(config.synthetic_manifest_path.read_text())}
    ratios = RatioRunOutput.model_validate_json(config.synthetic_ratios_path.read_text())
    return [
        (labels[r.name], r.occlusion_ratio)
        for r in ratios.readings
        if r.has_face and r.occlusion_ratio is not None and r.name in labels
    ]


class GateVerdict(BaseModel):
    """Promotion-gate outcome for the best candidate threshold."""

    adopted: bool
    current_threshold: float = CURRENT_THRESHOLD
    best_threshold: float
    real_f1_current: float
    real_f1_best: float
    synthetic_f1_current: float
    synthetic_f1_at_best: float


def _judge(sweeps: tuple[list[SweepPoint], list[SweepPoint]]) -> GateVerdict:
    """Apply the promotion gate to the best real-corpus threshold."""
    real_sweep, synthetic_sweep = sweeps
    synthetic_by_threshold = {p.threshold: p for p in synthetic_sweep}
    real_current = next(p for p in real_sweep if p.threshold == CURRENT_THRESHOLD)
    synthetic_current = synthetic_by_threshold[CURRENT_THRESHOLD]
    best = max(real_sweep, key=lambda p: p.f1)
    synthetic_at_best = synthetic_by_threshold[best.threshold]
    adopted = (
        best.f1 >= real_current.f1 + REAL_F1_MIN_GAIN
        and synthetic_at_best.f1 >= synthetic_current.f1 - SYNTHETIC_F1_MAX_DROP
    )
    return GateVerdict(
        adopted=adopted, best_threshold=best.threshold,
        real_f1_current=real_current.f1, real_f1_best=best.f1,
        synthetic_f1_current=synthetic_current.f1, synthetic_f1_at_best=synthetic_at_best.f1,
    )


def _sweep_table(points: list[SweepPoint]) -> str:
    """Render a markdown P/R/F1 table for the interesting sweep range."""
    lines = ["| threshold | precision | recall | F1 |", "|---|---|---|---|"]
    lines += [
        f"| {p.threshold:.2f} | {p.precision:.3f} | {p.recall:.3f} | {p.f1:.3f} |"
        for p in points if 0.05 <= p.threshold <= 0.60
    ]
    return "\n".join(lines)


class _ReportInput(BaseModel):
    """Everything the report renderer needs."""

    config: RecalConfig
    real_rows: list[LabeledRatio]
    agreement: AgreementResult
    real_sweep: list[SweepPoint]
    synthetic_sweep: list[SweepPoint]
    verdict: GateVerdict


def _context_counts(rows: list[LabeledRatio]) -> str:
    """Summarize corpus composition per context as `ctx n (occluded k)`."""
    contexts = sorted({r.context for r in rows})
    parts = [
        f"{c}: {sum(1 for r in rows if r.context == c)} "
        f"({sum(1 for r in rows if r.context == c and r.is_occluded)} occluded)"
        for c in contexts
    ]
    return "; ".join(parts)


_PROVENANCE_LINES: list[str] = [
    "# Occlusion threshold recalibration — Gemma weak labels vs synthetic eval",
    "",
    "PROVENANCE: labels are MODEL-GENERATED weak labels (labeler: gemma-4-12b,",
    "prompts occlusion_v1 + occlusion_v2, temperature 0.0, double-pass",
    "agreement filter). They are NOT human ground truth; every number below",
    "inherits that caveat. Gemma self-reported confidence is uninformative",
    "(benchmarks/LOG.md 2026-07-18) and was NOT used for gating.",
    "",
]


def _corpus_section(report_in: _ReportInput) -> list[str]:
    """Render the weak-label corpus summary lines."""
    agreement = report_in.agreement
    kept, disagreed = len(agreement.agreed), len(agreement.disagreed_names)
    return [
        "## Weak-label corpus",
        "",
        f"- Face-gated real photos labeled: {kept + disagreed + len(agreement.parse_error_names)}",
        f"- Double-pass agreement kept: {kept} | disagreed (dropped): {disagreed} "
        f"| unparsed (dropped): {len(agreement.parse_error_names)}",
        f"- Disagreement rate (comparable photos): {agreement.disagreement_rate:.3f}",
        f"- Joined with a production ratio (face detected): {len(report_in.real_rows)}",
        f"- Composition — {_context_counts(report_in.real_rows)}",
        "",
    ]


def _gate_section(verdict: GateVerdict) -> list[str]:
    """Render the gate summary and final verdict lines."""
    outcome = (
        f"ADOPT {verdict.best_threshold:.2f}" if verdict.adopted
        else f"KEEP {verdict.current_threshold:.2f}"
    )
    return [
        "## Gate",
        "",
        f"- Current threshold {verdict.current_threshold:.2f}: real F1 "
        f"{verdict.real_f1_current:.3f}, synthetic F1 {verdict.synthetic_f1_current:.3f}",
        f"- Best real-corpus threshold {verdict.best_threshold:.2f}: real F1 "
        f"{verdict.real_f1_best:.3f}, synthetic F1 {verdict.synthetic_f1_at_best:.3f}",
        f"- Rule: adopt only if real F1 gain >= {REAL_F1_MIN_GAIN:.2f} and synthetic "
        f"F1 drop <= {SYNTHETIC_F1_MAX_DROP:.2f}",
        "",
        f"## Verdict: {outcome}",
        "",
    ]


def _render_report(report_in: _ReportInput) -> str:
    """Render the full recalibration report as markdown."""
    return "\n".join(
        _PROVENANCE_LINES
        + _corpus_section(report_in)
        + ["## Real-corpus sweep (weak labels)", "", _sweep_table(report_in.real_sweep), ""]
        + ["## Synthetic-occluder sweep (labels from generated eval set)", "",
           _sweep_table(report_in.synthetic_sweep), ""]
        + _gate_section(report_in.verdict)
    )


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: occlusion_recal.py <config.json> <report.md>")
    config = RecalConfig.model_validate_json(Path(sys.argv[1]).read_text())
    real_rows = _load_real_rows(config)
    real_sweep = _sweep([(r.is_occluded, r.ratio) for r in real_rows])
    synthetic_sweep = _sweep(_load_synthetic_rows(config))
    verdict = _judge((real_sweep, synthetic_sweep))
    report_in = _ReportInput(
        config=config, real_rows=real_rows, agreement=_load_agreement(config),
        real_sweep=real_sweep, synthetic_sweep=synthetic_sweep, verdict=verdict,
    )
    Path(sys.argv[2]).write_text(_render_report(report_in))
    config.verdict_out_path.write_text(verdict.model_dump_json(indent=2))
    print(verdict.model_dump_json(indent=2))  # noqa: T201 — CLI result surface


if __name__ == "__main__":
    main()
