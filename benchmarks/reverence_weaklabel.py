"""Analysis driver: EmotiEffLib valence/arousal vs Gemma reverence weak labels.

Pure arithmetic — no model inference. Consumes two paraphrased Gemma
expression passes (gemma_weaklabel_runner.py) plus EmotiEffLib readings for
the same face crops (fair_expr_runner.py emotiefflib stage), and reports
whether reverent/prayerful faces separate from distressed faces in
valence/arousal space. Prototype only: proposes, never wires, scoring rules.

Weak labels are MODEL-GENERATED (labeler: gemma-4-12b) — not human ground
truth; trust comes from double-pass agreement only.

Usage:
    python3 benchmarks/reverence_weaklabel.py <config.json> <report.md>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from pydantic import BaseModel

from fair_expr_models import RunnerStageOutput
from weaklabel_models import EXPRESSION_VOCAB, WeaklabelRunOutput, compute_agreement

KEY_CONTRAST: tuple[str, str] = ("reverent/prayerful", "distressed")
SECONDARY_CONTRAST: tuple[str, str] = ("reverent/prayerful", "joyful")
MIN_GROUP_FOR_EFFECT: int = 5


class ReverenceConfig(BaseModel):
    """Paths to every input artifact of the reverence prototype analysis."""

    corpus_path: Path
    labels_v1_path: Path
    labels_v2_path: Path
    emotieff_path: Path


class JoinedFace(BaseModel):
    """One face with its agreed Gemma label and EmotiEffLib reading."""

    name: str
    context: str
    gemma_label: str
    emotieff_label: str
    valence: float
    arousal: float


def _load_joined(config: ReverenceConfig) -> list[JoinedFace]:
    """Join agreed Gemma labels with EmotiEffLib readings on crop name."""
    runs = (
        WeaklabelRunOutput.model_validate_json(config.labels_v1_path.read_text()),
        WeaklabelRunOutput.model_validate_json(config.labels_v2_path.read_text()),
    )
    agreement = compute_agreement(runs)
    contexts = {r["name"]: r["context"] for r in json.loads(config.corpus_path.read_text())}
    emotieff = {
        r.name: r
        for r in RunnerStageOutput.model_validate_json(config.emotieff_path.read_text()).readings
        if r.has_face and r.valence is not None and r.arousal is not None
    }
    return [
        JoinedFace(
            name=label.name, context=contexts.get(label.name, "?"), gemma_label=label.label,
            emotieff_label=emotieff[label.name].raw_label or "?",
            valence=emotieff[label.name].valence, arousal=emotieff[label.name].arousal,
        )
        for label in agreement.agreed if label.name in emotieff
    ]


class GroupStats(BaseModel):
    """Mean/std of valence and arousal for one Gemma label group."""

    label: str
    count: int
    valence_mean: float
    valence_std: float
    arousal_mean: float
    arousal_std: float


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, population std) of a non-empty value list."""
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(variance)


def _group_stats(faces: list[JoinedFace]) -> list[GroupStats]:
    """Per-Gemma-label valence/arousal statistics, vocabulary order."""
    stats = []
    for label in EXPRESSION_VOCAB:
        group = [f for f in faces if f.gemma_label == label]
        if not group:
            continue
        v_mean, v_std = _mean_std([f.valence for f in group])
        a_mean, a_std = _mean_std([f.arousal for f in group])
        stats.append(GroupStats(
            label=label, count=len(group), valence_mean=v_mean, valence_std=v_std,
            arousal_mean=a_mean, arousal_std=a_std,
        ))
    return stats


def _cohens_d(groups: tuple[list[float], list[float]]) -> float:
    """Cohen's d between two value lists (pooled std)."""
    first, second = groups
    mean_a, std_a = _mean_std(first)
    mean_b, std_b = _mean_std(second)
    pooled = math.sqrt((std_a ** 2 + std_b ** 2) / 2)
    return (mean_a - mean_b) / pooled if pooled > 1e-9 else 0.0


def _auc(groups: tuple[list[float], list[float]]) -> float:
    """Rank-based ROC-AUC of separating group A (positive) from group B."""
    positives, negatives = groups
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in positives for n in negatives
    )
    return wins / (len(positives) * len(negatives))


class ContrastResult(BaseModel):
    """Effect sizes for one label-pair contrast in valence/arousal space."""

    label_a: str
    label_b: str
    n_a: int
    n_b: int
    valence_cohens_d: float | None = None
    arousal_cohens_d: float | None = None
    valence_auc: float | None = None
    arousal_auc: float | None = None


class _ContrastInput(BaseModel):
    """Faces plus the label pair to contrast."""

    faces: list[JoinedFace]
    pair: tuple[str, str]


def _contrast(contrast_in: _ContrastInput) -> ContrastResult:
    """Compute effect sizes for one label pair when both groups are big enough."""
    label_a, label_b = contrast_in.pair
    group_a = [f for f in contrast_in.faces if f.gemma_label == label_a]
    group_b = [f for f in contrast_in.faces if f.gemma_label == label_b]
    result = ContrastResult(label_a=label_a, label_b=label_b, n_a=len(group_a), n_b=len(group_b))
    if len(group_a) < MIN_GROUP_FOR_EFFECT or len(group_b) < MIN_GROUP_FOR_EFFECT:
        return result
    valence_pair = ([f.valence for f in group_a], [f.valence for f in group_b])
    arousal_pair = ([f.arousal for f in group_a], [f.arousal for f in group_b])
    result.valence_cohens_d = _cohens_d(valence_pair)
    result.arousal_cohens_d = _cohens_d(arousal_pair)
    result.valence_auc = _auc(valence_pair)
    result.arousal_auc = _auc(arousal_pair)
    return result


def _crosstab_lines(faces: list[JoinedFace]) -> list[str]:
    """Markdown cross-tab of Gemma label x EmotiEffLib label."""
    emotieff_labels = sorted({f.emotieff_label for f in faces})
    header = "| gemma \\ emotieff | " + " | ".join(emotieff_labels) + " |"
    rows = [header, "|" + "---|" * (len(emotieff_labels) + 1)]
    for gemma_label in EXPRESSION_VOCAB:
        group = [f for f in faces if f.gemma_label == gemma_label]
        if not group:
            continue
        counts = [sum(1 for f in group if f.emotieff_label == e) for e in emotieff_labels]
        rows.append(f"| {gemma_label} | " + " | ".join(str(c) for c in counts) + " |")
    return rows


def _stats_lines(stats: list[GroupStats]) -> list[str]:
    """Markdown table of per-label valence/arousal statistics."""
    rows = [
        "| gemma label | n | valence mean±sd | arousal mean±sd |",
        "|---|---|---|---|",
    ]
    rows += [
        f"| {s.label} | {s.count} | {s.valence_mean:+.3f}±{s.valence_std:.3f} "
        f"| {s.arousal_mean:+.3f}±{s.arousal_std:.3f} |"
        for s in stats
    ]
    return rows


def _contrast_lines(contrast: ContrastResult) -> list[str]:
    """Markdown lines for one label-pair contrast."""
    lines = [
        f"## Contrast: {contrast.label_a} vs {contrast.label_b}",
        "",
        f"- Group sizes: {contrast.label_a} {contrast.n_a}, {contrast.label_b} {contrast.n_b}",
    ]
    if contrast.valence_cohens_d is None:
        lines.append(
            f"- Groups too small for effect sizes (need >= {MIN_GROUP_FOR_EFFECT} each) "
            "— separation question UNANSWERABLE on this corpus."
        )
        return lines + [""]
    lines += [
        f"- Valence: Cohen's d {contrast.valence_cohens_d:+.2f}, AUC {contrast.valence_auc:.3f}",
        f"- Arousal: Cohen's d {contrast.arousal_cohens_d:+.2f}, AUC {contrast.arousal_auc:.3f}",
        "",
    ]
    return lines


def _render_report(faces: list[JoinedFace]) -> str:
    """Render the full prototype report as markdown."""
    header = [
        "# Reverence weak-label prototype — EmotiEffLib V/A vs Gemma labels",
        "",
        "PROVENANCE: expression labels are MODEL-GENERATED weak labels (labeler:",
        "gemma-4-12b, prompts expression_v1 + expression_v2, double-pass agreement",
        "filter). NOT human ground truth. This is a prototype analysis only — no",
        "production scoring change is implied or implemented.",
        "",
        f"- Joined faces (agreed label + EmotiEffLib reading): {len(faces)}",
        f"- By context: vigil {sum(1 for f in faces if f.context == 'vigil')}, "
        f"weddings {sum(1 for f in faces if f.context == 'weddings')}",
        "",
    ]
    body = ["## Cross-tab", ""] + _crosstab_lines(faces) + [""]
    body += ["## Valence/arousal by Gemma label", ""] + _stats_lines(_group_stats(faces)) + [""]
    body += _contrast_lines(_contrast(_ContrastInput(faces=faces, pair=KEY_CONTRAST)))
    body += _contrast_lines(_contrast(_ContrastInput(faces=faces, pair=SECONDARY_CONTRAST)))
    return "\n".join(header + body)


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: reverence_weaklabel.py <config.json> <report.md>")
    config = ReverenceConfig.model_validate_json(Path(sys.argv[1]).read_text())
    faces = _load_joined(config)
    Path(sys.argv[2]).write_text(_render_report(faces))
    print(f"report written: {len(faces)} joined faces")  # noqa: T201 — CLI result surface


if __name__ == "__main__":
    main()
