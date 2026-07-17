"""Report assembly: verdict rubric + markdown rendering for the fair test.

Pure functions over already-computed layer results — no I/O, no heavy imports.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel

from fair_expr_layer_a import LayerAResult, LICENSE_NOTE, NEUTRAL_PROVENANCE_NOTE
from fair_expr_layer_b import RobustnessResult
from fair_expr_layer_c import CorpusDistribution, SeparationEffect
from fair_expr_models import LAYER_A_DATASET_ID, MACRO_F1_TOLERANCE, SEPARATION_TOLERANCE_FRACTION

# ---------------------------------------------------------------------------
# Verdict rubric
# ---------------------------------------------------------------------------


class LayerCSurvivors(BaseModel):
    """How many photos per corpus survived mediapipe face-gating."""

    by_corpus: dict[str, tuple[int, int]]  # corpus -> (pre_gate, post_gate)


class FairTestVerdict(BaseModel):
    """The rubric's three per-layer checks and the resulting REPLACE/KEEP/MIXED call."""

    layer_a_pass: bool
    layer_b_pass: bool
    layer_c_pass: bool
    verdict: str
    reasoning: list[str]


class RubricInput(BaseModel):
    """Everything the rubric needs: both models' Layer A/B/C headline numbers."""

    deepface_macro_f1: float
    emotiefflib_macro_f1: float
    deepface_mean_flip_rate: float
    emotiefflib_mean_flip_rate: float
    deepface_happy_rate_diff: float
    emotiefflib_happy_rate_diff: float


def _mean_flip_rate(result: RobustnessResult) -> float:
    """Average flip rate across all perturbation types for one model."""
    rates = list(result.flip_rate_by_perturbation.values())
    return round(sum(rates) / len(rates), 4) if rates else 0.0


class LayerResults(BaseModel):
    """All three layers' raw per-model results, bundled for rubric extraction."""

    model_config = {"arbitrary_types_allowed": True}

    layer_a: list[LayerAResult]
    layer_b: list[RobustnessResult]
    layer_c_effects: list[SeparationEffect]


def rubric_input_from_results(results: LayerResults) -> RubricInput:
    """Extract the four headline comparisons the rubric needs from the three layers."""
    a_by_model = {r.model: r for r in results.layer_a}
    b_by_model = {r.model: r for r in results.layer_b}
    c_by_model = {e.model: e for e in results.layer_c_effects}
    return RubricInput(
        deepface_macro_f1=a_by_model["deepface"].macro_f1, emotiefflib_macro_f1=a_by_model["emotiefflib"].macro_f1,
        deepface_mean_flip_rate=_mean_flip_rate(b_by_model["deepface"]),
        emotiefflib_mean_flip_rate=_mean_flip_rate(b_by_model["emotiefflib"]),
        deepface_happy_rate_diff=c_by_model["deepface"].happy_rate_diff,
        emotiefflib_happy_rate_diff=c_by_model["emotiefflib"].happy_rate_diff,
    )


def apply_verdict_rubric(rubric_input: RubricInput) -> FairTestVerdict:
    """Apply the task's REPLACE/KEEP/MIXED rubric to the three layers' headline numbers."""
    layer_a_pass = rubric_input.emotiefflib_macro_f1 >= rubric_input.deepface_macro_f1 - MACRO_F1_TOLERANCE
    layer_b_pass = rubric_input.emotiefflib_mean_flip_rate <= rubric_input.deepface_mean_flip_rate
    separation_floor = rubric_input.deepface_happy_rate_diff * (1 - SEPARATION_TOLERANCE_FRACTION)
    layer_c_pass = rubric_input.emotiefflib_happy_rate_diff >= separation_floor
    verdict = _resolve_verdict(LayerPassFlags(layer_a=layer_a_pass, layer_b=layer_b_pass, layer_c=layer_c_pass))
    reasoning = [
        f"Layer A: EmotiEffLib macro-F1 {rubric_input.emotiefflib_macro_f1:.3f} vs DeepFace "
        f"{rubric_input.deepface_macro_f1:.3f} (needs >= DeepFace-{MACRO_F1_TOLERANCE}) -> {'PASS' if layer_a_pass else 'FAIL'}",
        f"Layer B: EmotiEffLib mean flip-rate {rubric_input.emotiefflib_mean_flip_rate:.3f} vs DeepFace "
        f"{rubric_input.deepface_mean_flip_rate:.3f} (needs <=) -> {'PASS' if layer_b_pass else 'FAIL'}",
        f"Layer C: EmotiEffLib happy-rate separation {rubric_input.emotiefflib_happy_rate_diff:.3f} vs DeepFace "
        f"{rubric_input.deepface_happy_rate_diff:.3f} (needs >= {separation_floor:.3f}, i.e. within "
        f"{SEPARATION_TOLERANCE_FRACTION:.0%}) -> {'PASS' if layer_c_pass else 'FAIL'}",
    ]
    return FairTestVerdict(
        layer_a_pass=layer_a_pass, layer_b_pass=layer_b_pass, layer_c_pass=layer_c_pass,
        verdict=verdict, reasoning=reasoning,
    )


class LayerPassFlags(BaseModel):
    """The three layers' rubric pass/fail flags."""

    layer_a: bool
    layer_b: bool
    layer_c: bool


def _resolve_verdict(flags: LayerPassFlags) -> str:
    """Map the three layer pass/fail flags to REPLACE / KEEP / MIXED."""
    if flags.layer_a and flags.layer_b and flags.layer_c:
        return "REPLACE DeepFace with EmotiEffLib"
    blockers = [
        name for name, passed in
        (("Layer A (accuracy)", flags.layer_a), ("Layer B (stability)", flags.layer_b), ("Layer C (separation)", flags.layer_c))
        if not passed
    ]
    return f"MIXED — blocked by: {', '.join(blockers)}"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConfusionMatrixSpec(BaseModel):
    """A confusion matrix and the class order to render it in, with a section title."""

    title: str
    matrix: dict[str, dict[str, int]]
    classes: list[str]


def _confusion_matrix_lines(spec: ConfusionMatrixSpec) -> list[str]:
    """Render a confusion matrix as a markdown table (rows=true, cols=predicted)."""
    header = "| true \\ pred | " + " | ".join(spec.classes) + " |"
    sep = "|---" * (len(spec.classes) + 1) + "|"
    rows = [
        f"| {true_label} | " + " | ".join(str(spec.matrix.get(true_label, {}).get(c, 0)) for c in spec.classes) + " |"
        for true_label in spec.classes
    ]
    return [f"#### {spec.title}", "", header, sep, *rows, ""]


class LayerASectionInput(BaseModel):
    """Layer A results plus the canonical and sub-analysis class orderings to render."""

    model_config = {"arbitrary_types_allowed": True}

    results: list[LayerAResult]
    classes: list[str]
    sub_classes: list[str]


def _layer_a_section(section_input: LayerASectionInput) -> list[str]:
    """Render the full Layer A section: dataset provenance, metrics, confusion matrices."""
    results, classes, sub_classes = section_input.results, section_input.classes, section_input.sub_classes
    lines = [
        "## Layer A — labeled benchmark anchor (RAF-DB)", "",
        f"Dataset: `{LAYER_A_DATASET_ID}` (RAF-DB, Li & Deng 2017, HF mirror). "
        "Test images use the mirror's preserved train_/test_ RAF-DB filename prefix — "
        "this recovers RAF-DB's OFFICIAL test partition, not a self-defined split.", "",
        f"- License: {LICENSE_NOTE}",
        f"- Provenance caveat: {NEUTRAL_PROVENANCE_NOTE}", "",
        "### EmotiEffLib 8-class -> dataset 7-class taxonomy mapping", "",
        "anger->angry, disgust->disgust, fear->fear, happiness->happy, neutral->neutral, "
        "sadness->sad, surprise->surprise, **contempt has no RAF-DB/DeepFace equivalent** "
        "(scored as a miss whenever predicted).", "",
        "| model | n | accuracy | macro-F1 | sub-analysis macro-F1 (sad/fear/neutral/angry) |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {r.model} | {r.n} | {r.accuracy:.3f} | {r.macro_f1:.3f} | {r.sub_analysis_macro_f1:.3f} |" for r in results]
    lines.append("")
    lines.append("### Per-class accuracy")
    lines += ["", "| model | " + " | ".join(classes) + " |", "|---" * (len(classes) + 1) + "|"]
    lines += [
        f"| {r.model} | " + " | ".join(f"{r.per_class_accuracy.get(c, 0.0):.3f}" for c in classes) + " |"
        for r in results
    ]
    lines.append("")
    for r in results:
        lines += _confusion_matrix_lines(
            ConfusionMatrixSpec(title=f"{r.model} full confusion matrix", matrix=r.confusion_matrix, classes=classes)
        )
        lines += _confusion_matrix_lines(ConfusionMatrixSpec(
            title=f"{r.model} sad/fear/neutral/angry confusion matrix",
            matrix=r.sub_analysis_confusion_matrix, classes=sub_classes,
        ))
    return lines


def _layer_b_section(results: list[RobustnessResult]) -> list[str]:
    """Render the full Layer B section: flip-rate and VA-drift tables."""
    perturbations = sorted(next(r.flip_rate_by_perturbation.keys() for r in results if r.flip_rate_by_perturbation))
    lines = [
        "## Layer B — robustness, label-free", "",
        "This measures STABILITY (agreement with the model's own unperturbed prediction), "
        "not correctness. A stable model keeps its prediction under a perturbation.", "",
        "| model | n faces | " + " | ".join(perturbations) + " | mean flip-rate |",
        "|---" * (len(perturbations) + 3) + "|",
    ]
    for r in results:
        mean_rate = sum(r.flip_rate_by_perturbation.values()) / len(r.flip_rate_by_perturbation)
        row = " | ".join(f"{r.flip_rate_by_perturbation[p]:.3f}" for p in perturbations)
        lines.append(f"| {r.model} | {r.n_faces} | {row} | {mean_rate:.3f} |")
    lines.append("")
    ef = next((r for r in results if r.valence_drift_by_perturbation), None)
    if ef:
        lines += ["### EmotiEffLib valence/arousal drift (mean |delta|)", "",
                   "| perturbation | valence drift | arousal drift |", "|---|---|---|"]
        for p in perturbations:
            lines.append(f"| {p} | {ef.valence_drift_by_perturbation[p]:.3f} | {ef.arousal_drift_by_perturbation[p]:.3f} |")
        lines.append("")
    return lines


def _corpus_distribution_lines(distributions: list[CorpusDistribution]) -> list[str]:
    """Render per-corpus, per-model label histograms and happy/negative rates."""
    lines = [
        "| corpus | model | n | happy-rate | negative-rate | mean valence | std valence | mean arousal | std arousal |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in distributions:
        mv = f"{d.mean_valence:.3f}" if d.mean_valence is not None else "—"
        sv = f"{d.std_valence:.3f}" if d.std_valence is not None else "—"
        ma = f"{d.mean_arousal:.3f}" if d.mean_arousal is not None else "—"
        sa = f"{d.std_arousal:.3f}" if d.std_arousal is not None else "—"
        lines.append(f"| {d.corpus} | {d.model} | {d.n} | {d.happy_rate:.3f} | {d.negative_rate:.3f} | {mv} | {sv} | {ma} | {sa} |")
    return lines


class LayerCSectionInput(BaseModel):
    """Layer C distributions, separation effects, and gate survivor counts to render."""

    distributions: list[CorpusDistribution]
    effects: list[SeparationEffect]
    survivors: LayerCSurvivors


def _layer_c_section(section_input: LayerCSectionInput) -> list[str]:
    """Render the full Layer C section: survivor counts, distributions, separation effects."""
    distributions, effects, survivors = section_input.distributions, section_input.effects, section_input.survivors
    lines = ["## Layer C — context-distribution separation", "", "### Face-gate survivors", "",
             "| corpus | photos sampled | survived mediapipe gate |", "|---|---|---|"]
    lines += [f"| {corpus} | {pre} | {post} |" for corpus, (pre, post) in survivors.by_corpus.items()]
    lines += ["", "### Per-corpus distributions", ""] + _corpus_distribution_lines(distributions) + [""]
    lines += ["### Weddings-vs-Vigil separation effect size", "",
              "| model | happy-rate diff (weddings - vigil) | Cohen's d valence |", "|---|---|---|"]
    for e in effects:
        d = f"{e.cohens_d_valence:.3f}" if e.cohens_d_valence is not None else "—"
        lines.append(f"| {e.model} | {e.happy_rate_diff:.3f} | {d} |")
    return lines + [""]


class ReportSections(BaseModel):
    """All computed sections needed to render the final markdown report."""

    model_config = {"arbitrary_types_allowed": True}

    layer_a: list[LayerAResult]
    layer_b: list[RobustnessResult]
    layer_c_distributions: list[CorpusDistribution]
    layer_c_effects: list[SeparationEffect]
    layer_c_survivors: LayerCSurvivors
    verdict: FairTestVerdict
    classes: list[str]
    sub_classes: list[str]


def render_report_markdown(sections: ReportSections) -> str:
    """Assemble the full decision-grade markdown report."""
    lines = [
        "# Fair Three-Layer Test: EmotiEffLib (enet_b0_8_va_mtl, ONNX) vs DeepFace Emotion",
        "", f"Generated {_utc_now()}", "",
        f"## Verdict: {sections.verdict.verdict}", "",
    ] + [f"- {r}" for r in sections.verdict.reasoning] + [""]
    lines += _layer_a_section(LayerASectionInput(results=sections.layer_a, classes=sections.classes, sub_classes=sections.sub_classes))
    lines += _layer_b_section(sections.layer_b)
    lines += _layer_c_section(LayerCSectionInput(
        distributions=sections.layer_c_distributions, effects=sections.layer_c_effects, survivors=sections.layer_c_survivors,
    ))
    return "\n".join(lines) + "\n"
