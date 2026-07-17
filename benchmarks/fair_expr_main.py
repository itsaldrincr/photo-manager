"""Orchestrator: the three-layer fair test of EmotiEffLib vs DeepFace emotion.

Runs Layer A (labeled accuracy), Layer B (perturbation robustness), and
Layer C (context-distribution separation) serially, one heavy model resident
at a time (subprocess-per-stage), and writes a decision-grade report.

Usage:
    python3 benchmarks/fair_expr_main.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fair_expr_layer_a import build_layer_a_sample, score_layer_a, ScoredReadings
from fair_expr_layer_b import LayerBRequest, PerturbationSource, run_layer_b
from fair_expr_layer_c import (
    DistributionRequest, GateRequest, LAYER_C_CORPORA, NasCorpusSpec, SeparationRequest,
    _corpus_distribution, face_gate_corpus, gather_corpus, separation_effect,
)
from fair_expr_models import CANONICAL_CLASSES, DEEPFACE_LABEL_MAP, EMOTIEFFLIB_LABEL_MAP, RunnerManifest, SUB_ANALYSIS_CLASSES
from fair_expr_report import LayerCSurvivors, LayerResults, ReportSections, apply_verdict_rubric, render_report_markdown, rubric_input_from_results
from fair_expr_subprocess import StageInvocation, assert_no_stale_runners, run_stage

SCRATCH_ROOT: Path = Path(
    "/private/tmp/claude-501/-Users-alrelador/85225a69-a5ad-4e09-b879-f167553ba959/scratchpad/fair_expr"
)
RUNS_DIR: Path = Path(__file__).resolve().parent / "runs"
REPORT_PATH: Path = RUNS_DIR / "expression_fair_test_report.md"

_LABEL_MAPS: dict[str, dict[str, str]] = {"deepface": DEEPFACE_LABEL_MAP, "emotiefflib": EMOTIEFFLIB_LABEL_MAP}


def _run_both_models(image_paths: list[Path], out_prefix: Path) -> dict[str, list]:
    """Run the deepface and emotiefflib stages over one manifest, serially."""
    manifest = RunnerManifest(image_paths=image_paths)
    results: dict[str, list] = {}
    for stage in ("deepface", "emotiefflib"):
        output = run_stage(StageInvocation(
            stage=stage, manifest=manifest, output_path=Path(f"{out_prefix}_{stage}.json"),
        ))
        results[stage] = output.readings
    return results


# ---------------------------------------------------------------------------
# Layer A
# ---------------------------------------------------------------------------


def _run_layer_a() -> tuple[list, list[Path]]:
    """Prep the RAF-DB sample, score both models, return results + the 60-per-class paths."""
    sample = build_layer_a_sample(SCRATCH_ROOT / "layer_a" / "images")
    (RUNS_DIR / "fair_expr_layer_a_labels.json").write_text(json.dumps(sample.true_labels, indent=2))
    readings = _run_both_models(sample.image_paths, RUNS_DIR / "fair_expr_layer_a")
    results = [
        score_layer_a(ScoredReadings(
            model=stage, true_labels=sample.true_labels,
            raw_labels={r.name: r.raw_label or "" for r in readings[stage]}, label_map=_LABEL_MAPS[stage],
        ))
        for stage in ("deepface", "emotiefflib")
    ]
    for stage in ("deepface", "emotiefflib"):
        raw = [r.model_dump() for r in readings[stage]]
        (RUNS_DIR / f"fair_expr_layer_a_raw_{stage}.json").write_text(json.dumps(raw, indent=2))
    return results, sample.image_paths


# ---------------------------------------------------------------------------
# Layer B
# ---------------------------------------------------------------------------


_VIGIL_SPEC: NasCorpusSpec = next(spec for spec in LAYER_C_CORPORA if spec.name == "vigil")


def _vigil_crop_sources() -> list[PerturbationSource]:
    """Face-gate the 30 Vigil photos and return them as Layer B perturbation sources."""
    photos = gather_corpus(_VIGIL_SPEC, SCRATCH_ROOT / "layer_b" / "_unused")
    gated = face_gate_corpus(GateRequest(
        name="layer_b_vigil", photos=photos, crop_dir=SCRATCH_ROOT / "layer_b" / "vigil_crops",
    ))
    return [PerturbationSource(name=p.name, image_path=p) for p in gated.crop_paths]


def _run_layer_b(layer_a_paths: list[Path]) -> list:
    """Build the 50-face perturbation set (30 Vigil + 20 Layer A crops) and run robustness."""
    layer_a_sources = [PerturbationSource(name=p.name, image_path=p) for p in sorted(layer_a_paths)[:20]]
    sources = _vigil_crop_sources() + layer_a_sources
    results = run_layer_b(LayerBRequest(sources=sources, work_dir=SCRATCH_ROOT / "layer_b"))
    (RUNS_DIR / "fair_expr_layer_b_raw.json").write_text(
        json.dumps([r.model_dump() for r in results], indent=2)
    )
    return results


# ---------------------------------------------------------------------------
# Layer C
# ---------------------------------------------------------------------------


def _gate_one_corpus(spec: NasCorpusSpec) -> tuple:
    """Pull (or reuse) one corpus's photos and face-gate them."""
    raw_dir = SCRATCH_ROOT / "layer_c" / "raw" / spec.name
    crop_dir = SCRATCH_ROOT / "layer_c" / "crops" / spec.name
    photos = gather_corpus(spec, raw_dir)
    gated = face_gate_corpus(GateRequest(name=spec.name, photos=photos, crop_dir=crop_dir))
    return gated, (len(photos), len(gated.crop_paths))


def _run_layer_c() -> tuple[list, list, LayerCSurvivors]:
    """Pull/gate all three corpora, score both models, compute distributions + separation."""
    survivors: dict[str, tuple[int, int]] = {}
    distributions = []
    valence_by_corpus: dict[str, list[float]] = {}
    for spec in LAYER_C_CORPORA:
        gated, counts = _gate_one_corpus(spec)
        survivors[spec.name] = counts
        readings = _run_both_models(gated.crop_paths, RUNS_DIR / f"fair_expr_layer_c_{spec.name}")
        raw_dump = {stage: [r.model_dump() for r in readings[stage]] for stage in readings}
        (RUNS_DIR / f"fair_expr_layer_c_{spec.name}_raw.json").write_text(json.dumps(raw_dump, indent=2))
        for stage in ("deepface", "emotiefflib"):
            distributions.append(_corpus_distribution(DistributionRequest(
                corpus=spec.name, model=stage, readings=readings[stage], label_map=_LABEL_MAPS[stage],
            )))
        valence_by_corpus[spec.name] = [
            r.valence for r in readings["emotiefflib"] if r.has_face and r.valence is not None
        ]
    effects = [
        separation_effect(SeparationRequest(
            distributions=[d for d in distributions if d.model == stage], valence_by_corpus=valence_by_corpus,
        ))
        for stage in ("deepface", "emotiefflib")
    ]
    return distributions, effects, LayerCSurvivors(by_corpus=survivors)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all three layers serially and write the decision-grade report."""
    assert_no_stale_runners()
    RUNS_DIR.mkdir(exist_ok=True)
    layer_a_results, layer_a_paths = _run_layer_a()
    layer_b_results = _run_layer_b(layer_a_paths)
    layer_c_distributions, layer_c_effects, layer_c_survivors = _run_layer_c()
    rubric_input = rubric_input_from_results(
        LayerResults(layer_a=layer_a_results, layer_b=layer_b_results, layer_c_effects=layer_c_effects)
    )
    verdict = apply_verdict_rubric(rubric_input)
    report = render_report_markdown(ReportSections(
        layer_a=layer_a_results, layer_b=layer_b_results, layer_c_distributions=layer_c_distributions,
        layer_c_effects=layer_c_effects, layer_c_survivors=layer_c_survivors, verdict=verdict,
        classes=CANONICAL_CLASSES, sub_classes=SUB_ANALYSIS_CLASSES,
    ))
    REPORT_PATH.write_text(report)
    print(f"wrote {REPORT_PATH}")
    print(f"VERDICT: {verdict.verdict}")
    for line in verdict.reasoning:
        print(f"  {line}")


if __name__ == "__main__":
    main()
