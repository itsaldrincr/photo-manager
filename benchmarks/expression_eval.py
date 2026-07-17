"""Orchestrator: benchmark py-feat Detectorv2 and EmotiEffLib against the
already-measured DeepFace/MediaPipe baseline, and gate the keep/replace
decision for DeepFace's emotion head with measured evidence.

Runs each new candidate in its own subprocess (one heavy model resident at a
time), reuses the existing portrait_eval_* JSON for DeepFace/MediaPipe
numbers, and writes a durable decision report.

Usage:
    python3 benchmarks/expression_eval.py
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from expression_eval_models import (
    EmotiEfflibReading,
    EmotiEfflibStageOutput,
    PyFeatReading,
    PyFeatStageOutput,
    REPLACE_AU43_AGREEMENT_MIN,
    REPLACE_RSS_SAVINGS_MIN_MB,
    REPLACE_SPEEDUP_MIN,
)
from portrait_eval_models import (
    DeepfaceReading,
    DeepfaceStageOutput,
    MediapipeReading,
    MediapipeStageOutput,
    StageTiming,
    list_image_paths,
)

BENCH_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = BENCH_DIR / "runs"
REPORT_PATH: Path = RUNS_DIR / "expression_eval_report.md"
RUNNER_PATH: Path = BENCH_DIR / "expression_eval_runner.py"
PORTRAIT_MANIFEST_PATH: Path = RUNS_DIR / "portrait_eval_manifest.json"
PORTRAIT_MEDIAPIPE_PATH: Path = RUNS_DIR / "portrait_eval_mediapipe.json"
PORTRAIT_DEEPFACE_PATH: Path = RUNS_DIR / "portrait_eval_deepface.json"

RUNNER_TIMEOUT_SECONDS: int = 1800

DEEPFACE_LICENSE_NOTE: str = "MIT (deepface itself); backend model weights vary by provider."
PYFEAT_LICENSE_NOTE: str = (
    "MIT (py-feat code); face_multitask_v2 CHECKPOINT is research/non-commercial only — "
    "flagged, not resolved, by this eval."
)
EMOTIEFFLIB_LICENSE_NOTE: str = "Apache-2.0 (library and enet_b0_8_va_mtl weights) — fully permissive."


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""
    return round(sum(values) / len(values), 4) if values else 0.0


def _stdev(values: list[float]) -> float:
    """Return the population standard deviation, or 0.0 for <2 values."""
    return round(statistics.pstdev(values), 4) if len(values) >= 2 else 0.0


def _gather_photos() -> list[Path]:
    """Return the exact 30-photo corpus the DeepFace/MediaPipe baseline used."""
    override = os.environ.get("EXPRESSION_EVAL_DIR")
    if override:
        return list_image_paths(Path(override))
    if PORTRAIT_MANIFEST_PATH.exists():
        return [Path(p) for p in json.loads(PORTRAIT_MANIFEST_PATH.read_text())]
    raise SystemExit(
        "no photo corpus: set EXPRESSION_EVAL_DIR or run portrait_eval.py first "
        f"to produce {PORTRAIT_MANIFEST_PATH}"
    )


class _StageInvocation(BaseModel):
    """Inputs for one subprocess stage run."""

    stage: str
    input_path: Path
    output_path: Path


def _run_stage(invocation: _StageInvocation) -> None:
    """Invoke expression_eval_runner.py for one stage in a fresh subprocess."""
    argv = [
        sys.executable, str(RUNNER_PATH), invocation.stage,
        str(invocation.input_path), str(invocation.output_path),
    ]
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=RUNNER_TIMEOUT_SECONDS,
        check=False, cwd=BENCH_DIR,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{invocation.stage} stage failed: {completed.stderr.strip()[-3000:]}")


class _Baseline(BaseModel):
    """The already-measured MediaPipe and DeepFace stage outputs, reused as-is."""

    mediapipe: MediapipeStageOutput
    deepface: DeepfaceStageOutput


def _load_baseline() -> _Baseline:
    """Read the existing DeepFace/MediaPipe measurements from disk (not re-run)."""
    if not (PORTRAIT_MEDIAPIPE_PATH.exists() and PORTRAIT_DEEPFACE_PATH.exists()):
        raise SystemExit(
            "missing baseline: run benchmarks/portrait_eval.py first to produce "
            f"{PORTRAIT_MEDIAPIPE_PATH.name} and {PORTRAIT_DEEPFACE_PATH.name}"
        )
    return _Baseline(
        mediapipe=MediapipeStageOutput.model_validate_json(PORTRAIT_MEDIAPIPE_PATH.read_text()),
        deepface=DeepfaceStageOutput.model_validate_json(PORTRAIT_DEEPFACE_PATH.read_text()),
    )


# ---------------------------------------------------------------------------
# A. Cost table
# ---------------------------------------------------------------------------


class CostRow(BaseModel):
    """One model's measured latency and memory footprint."""

    model: str
    mean_latency_seconds: float
    load_seconds: float
    rss_delta_mb: float


class _CostRowSource(BaseModel):
    """A named model's timing plus the per-photo latencies to average."""

    model: str
    timing: StageTiming
    latencies: list[float]


def _cost_row(source: _CostRowSource) -> CostRow:
    """Reduce one model's timing and latency samples into a CostRow."""
    return CostRow(
        model=source.model, mean_latency_seconds=_mean(source.latencies),
        load_seconds=source.timing.load_seconds, rss_delta_mb=source.timing.rss_delta_mb,
    )


class _AllStageOutputs(BaseModel):
    """Every measured stage's output, bundled for cost-table construction."""

    baseline: _Baseline
    pyfeat: PyFeatStageOutput
    emotiefflib: EmotiEfflibStageOutput


def _build_cost_table(outputs: _AllStageOutputs) -> list[CostRow]:
    """Build the four-way latency/RSS cost table."""
    mp, df = outputs.baseline.mediapipe, outputs.baseline.deepface
    return [
        _cost_row(_CostRowSource(
            model="MediaPipe blendshapes", timing=mp.timing,
            latencies=[r.latency_seconds for r in mp.readings if r.has_face],
        )),
        _cost_row(_CostRowSource(
            model="DeepFace emotion", timing=df.timing,
            latencies=[r.latency_seconds for r in df.readings],
        )),
        _cost_row(_CostRowSource(
            model="Py-Feat Detectorv2", timing=outputs.pyfeat.timing,
            latencies=[r.latency_seconds for r in outputs.pyfeat.readings if r.has_face],
        )),
        _cost_row(_CostRowSource(
            model="EmotiEffLib enet_b0_8_va_mtl", timing=outputs.emotiefflib.timing,
            latencies=[r.latency_seconds for r in outputs.emotiefflib.readings if r.has_face],
        )),
    ]


# ---------------------------------------------------------------------------
# B. VA coherence check
# ---------------------------------------------------------------------------


class VaCoherenceStats(BaseModel):
    """Dispersion summary for one candidate's valence-arousal outputs."""

    candidate: str
    n: int
    mean_valence: float
    std_valence: float
    mean_arousal: float
    std_arousal: float
    dominant_quadrant: str
    dominant_quadrant_fraction: float


class _VaSample(BaseModel):
    """One photo's valence-arousal-quadrant triple."""

    valence: float
    arousal: float
    quadrant: str


class _VaCoherenceInput(BaseModel):
    """A candidate name paired with its per-photo VA samples."""

    candidate: str
    samples: list[_VaSample]


def _va_coherence(coherence_input: _VaCoherenceInput) -> VaCoherenceStats:
    """Compute mean/std/dominant-quadrant dispersion for one candidate's VA outputs."""
    samples = coherence_input.samples
    quadrant_counts = Counter(s.quadrant for s in samples)
    dominant_quadrant, dominant_count = quadrant_counts.most_common(1)[0]
    return VaCoherenceStats(
        candidate=coherence_input.candidate, n=len(samples),
        mean_valence=_mean([s.valence for s in samples]),
        std_valence=_stdev([s.valence for s in samples]),
        mean_arousal=_mean([s.arousal for s in samples]),
        std_arousal=_stdev([s.arousal for s in samples]),
        dominant_quadrant=dominant_quadrant,
        dominant_quadrant_fraction=round(dominant_count / len(samples), 4),
    )


class DeepfaceSpreadStats(BaseModel):
    """How many distinct categorical labels DeepFace scattered the corpus across."""

    n: int
    distinct_labels: int
    dominant_label: str
    dominant_label_fraction: float


def _deepface_spread(labels: list[str]) -> DeepfaceSpreadStats:
    """Summarize DeepFace's categorical label scatter for the VA coherence contrast."""
    counts = Counter(labels)
    dominant_label, dominant_count = counts.most_common(1)[0]
    return DeepfaceSpreadStats(
        n=len(labels), distinct_labels=len(counts), dominant_label=dominant_label,
        dominant_label_fraction=round(dominant_count / len(labels), 4),
    )


def _pyfeat_va_samples(readings: list[PyFeatReading]) -> list[_VaSample]:
    """Extract VA samples from face-bearing py-feat readings."""
    return [
        _VaSample(valence=r.valence, arousal=r.arousal, quadrant=r.va_quadrant)
        for r in readings if r.has_face and r.valence is not None
    ]


def _emotiefflib_va_samples(readings: list[EmotiEfflibReading]) -> list[_VaSample]:
    """Extract VA samples from face-bearing EmotiEffLib readings."""
    return [
        _VaSample(valence=r.valence, arousal=r.arousal, quadrant=r.va_quadrant)
        for r in readings if r.has_face and r.valence is not None
    ]


# ---------------------------------------------------------------------------
# C. AU sanity cross-check (AU43 eyes-closed vs EAR vs blendshape)
# ---------------------------------------------------------------------------


class Au43CrossCheck(BaseModel):
    """3-way agreement between EAR-, blendshape-, and AU43-based eyes-closed."""

    n_common: int
    all_three_agree_rate: float
    ear_vs_au43_rate: float
    blendshape_vs_au43_rate: float
    ear_vs_blendshape_rate: float


class _EyesClosedTriple(BaseModel):
    """One photo's three independent eyes-closed signals."""

    name: str
    ear: bool
    blendshape: bool
    au43: bool


def _eyes_closed_triples(mp_readings: list[MediapipeReading], pf_readings: list[PyFeatReading]) -> list[_EyesClosedTriple]:
    """Pair EAR/blendshape (MediaPipe) with AU43 (py-feat) by photo name."""
    pf_by_name = {r.name: r for r in pf_readings if r.au43_eyes_closed is not None}
    return [
        _EyesClosedTriple(
            name=mp.name, ear=mp.ear_eyes_closed, blendshape=mp.blendshape_eyes_closed,
            au43=pf_by_name[mp.name].au43_eyes_closed,
        )
        for mp in mp_readings
        if mp.name in pf_by_name and mp.ear_eyes_closed is not None and mp.blendshape_eyes_closed is not None
    ]


def _au43_cross_check(triples: list[_EyesClosedTriple]) -> Au43CrossCheck:
    """Compute pairwise and 3-way eyes-closed agreement rates."""
    if not triples:
        return Au43CrossCheck(
            n_common=0, all_three_agree_rate=0.0, ear_vs_au43_rate=0.0,
            blendshape_vs_au43_rate=0.0, ear_vs_blendshape_rate=0.0,
        )
    return Au43CrossCheck(
        n_common=len(triples),
        all_three_agree_rate=_mean([float(t.ear == t.blendshape == t.au43) for t in triples]),
        ear_vs_au43_rate=_mean([float(t.ear == t.au43) for t in triples]),
        blendshape_vs_au43_rate=_mean([float(t.blendshape == t.au43) for t in triples]),
        ear_vs_blendshape_rate=_mean([float(t.ear == t.blendshape) for t in triples]),
    )


# ---------------------------------------------------------------------------
# Decision rubric
# ---------------------------------------------------------------------------


class CandidateVerdict(BaseModel):
    """One candidate's rubric inputs and resulting keep/replace verdict."""

    name: str
    speedup_ratio: float
    rss_savings_mb: float
    au43_agreement_rate: float | None
    verdict: str
    reasoning: list[str]


class _RubricInput(BaseModel):
    """A candidate's measured speedup, RSS savings, and (if any) AU agreement."""

    name: str
    speedup_ratio: float
    rss_savings_mb: float
    au43_agreement_rate: float | None


def _apply_rubric(rubric_input: _RubricInput) -> CandidateVerdict:
    """Apply the REPLACE/KEEP rubric to one candidate's measured metrics."""
    reasons = [
        f"speedup {rubric_input.speedup_ratio:.2f}x (threshold {REPLACE_SPEEDUP_MIN:.1f}x)",
        f"RSS savings {rubric_input.rss_savings_mb:.1f} MB (threshold {REPLACE_RSS_SAVINGS_MIN_MB:.0f} MB)",
    ]
    if rubric_input.au43_agreement_rate is None:
        reasons.append("no AU43/eyes-closed signal to objectively validate")
        verdict = "KEEP"
    else:
        reasons.append(
            f"AU43-vs-EAR agreement {rubric_input.au43_agreement_rate:.1%} "
            f"(threshold {REPLACE_AU43_AGREEMENT_MIN:.0%})"
        )
        speed_ok = rubric_input.speedup_ratio >= REPLACE_SPEEDUP_MIN
        rss_ok = rubric_input.rss_savings_mb >= REPLACE_RSS_SAVINGS_MIN_MB
        au_ok = rubric_input.au43_agreement_rate >= REPLACE_AU43_AGREEMENT_MIN
        verdict = "REPLACE" if (speed_ok and rss_ok and au_ok) else "KEEP"
    return CandidateVerdict(
        name=rubric_input.name, speedup_ratio=rubric_input.speedup_ratio,
        rss_savings_mb=rubric_input.rss_savings_mb,
        au43_agreement_rate=rubric_input.au43_agreement_rate,
        verdict=verdict, reasoning=reasons,
    )


def _speedup_ratio(deepface_latency: float, candidate_latency: float) -> float:
    """Return how many times slower DeepFace is than a candidate."""
    return round(deepface_latency / candidate_latency, 2) if candidate_latency else 0.0


def _rss_savings_mb(deepface_rss_mb: float, candidate_rss_mb: float) -> float:
    """Return the RSS a candidate saves relative to DeepFace."""
    return round(deepface_rss_mb - candidate_rss_mb, 1)


def _pyfeat_rubric_input(outputs: _AllStageOutputs, au43_cross_check: Au43CrossCheck) -> _RubricInput:
    """Bundle Py-Feat's measured metrics for the REPLACE/KEEP rubric."""
    df_latency = _mean([r.latency_seconds for r in outputs.baseline.deepface.readings])
    pf_latency = _mean([r.latency_seconds for r in outputs.pyfeat.readings if r.has_face])
    return _RubricInput(
        name="Py-Feat Detectorv2", speedup_ratio=_speedup_ratio(df_latency, pf_latency),
        rss_savings_mb=_rss_savings_mb(
            outputs.baseline.deepface.timing.rss_delta_mb, outputs.pyfeat.timing.rss_delta_mb
        ),
        au43_agreement_rate=au43_cross_check.ear_vs_au43_rate if au43_cross_check.n_common else None,
    )


def _emotiefflib_rubric_input(outputs: _AllStageOutputs) -> _RubricInput:
    """Bundle EmotiEffLib's measured metrics for the REPLACE/KEEP rubric."""
    df_latency = _mean([r.latency_seconds for r in outputs.baseline.deepface.readings])
    ef_latency = _mean([r.latency_seconds for r in outputs.emotiefflib.readings if r.has_face])
    return _RubricInput(
        name="EmotiEffLib enet_b0_8_va_mtl", speedup_ratio=_speedup_ratio(df_latency, ef_latency),
        rss_savings_mb=_rss_savings_mb(
            outputs.baseline.deepface.timing.rss_delta_mb, outputs.emotiefflib.timing.rss_delta_mb
        ),
        au43_agreement_rate=None,
    )


def _final_recommendation(verdicts: list[CandidateVerdict]) -> str:
    """Roll per-candidate verdicts into one top-level keep/replace call."""
    replacements = [v.name for v in verdicts if v.verdict == "REPLACE"]
    if replacements:
        return f"REPLACE DeepFace with {', '.join(replacements)}"
    return "KEEP DeepFace (no candidate is both cheaper and objectively AU-validated)"


UNRESOLVED_NOTES: list[str] = [
    "Accuracy-for-reverence cannot be measured: no public dataset labels solemnity/reverence, so "
    "owner expression labeling on this corpus is required before any candidate's categorical or "
    "dimensional output can be scored against ground truth rather than just internal coherence.",
    "The VA coherence check (section B) is dispersion evidence, not accuracy: tighter clustering "
    "is evidence a signal COULD express reverence coherently, not proof that it does.",
    "py-feat's Detectorv2 device='auto' carries a known upstream FIXME (mixed cpu/mps ops can "
    "produce NaNs on some Mac configurations); this run resolved to MPS and produced non-NaN "
    "output on all 30 photos, but that is not a guarantee on other hardware/driver combinations.",
]


# ---------------------------------------------------------------------------
# Per-photo merged table
# ---------------------------------------------------------------------------


class MergedRow(BaseModel):
    """One photo's readings across all four models, for the full per-photo table."""

    name: str
    deepface_label: str | None
    pyfeat_emotion: str | None
    pyfeat_valence: float | None
    pyfeat_arousal: float | None
    pyfeat_au43_eyes_closed: bool | None
    emotiefflib_emotion: str | None
    emotiefflib_valence: float | None
    emotiefflib_arousal: float | None
    ear_eyes_closed: bool | None
    blendshape_eyes_closed: bool | None


class _MergeSources(BaseModel):
    """One photo's readings from each stage, gathered before merging."""

    name: str
    mp: MediapipeReading
    df: DeepfaceReading | None
    pf: PyFeatReading | None
    ef: EmotiEfflibReading | None


def _merge_one_row(sources: _MergeSources) -> MergedRow:
    """Build one MergedRow from a matched set of per-stage readings."""
    pf, ef = sources.pf, sources.ef
    return MergedRow(
        name=sources.name, deepface_label=sources.df.dominant_emotion if sources.df else None,
        pyfeat_emotion=pf.dominant_emotion if pf else None,
        pyfeat_valence=pf.valence if pf else None, pyfeat_arousal=pf.arousal if pf else None,
        pyfeat_au43_eyes_closed=pf.au43_eyes_closed if pf else None,
        emotiefflib_emotion=ef.dominant_emotion if ef else None,
        emotiefflib_valence=ef.valence if ef else None,
        emotiefflib_arousal=ef.arousal if ef else None,
        ear_eyes_closed=sources.mp.ear_eyes_closed,
        blendshape_eyes_closed=sources.mp.blendshape_eyes_closed,
    )


def _build_merged_rows(outputs: _AllStageOutputs) -> list[MergedRow]:
    """Join all four stages' readings by photo name."""
    mp_by_name = {r.name: r for r in outputs.baseline.mediapipe.readings if r.has_face}
    df_by_name = {r.name: r for r in outputs.baseline.deepface.readings}
    pf_by_name = {r.name: r for r in outputs.pyfeat.readings}
    ef_by_name = {r.name: r for r in outputs.emotiefflib.readings}
    return [
        _merge_one_row(_MergeSources(
            name=name, mp=mp_by_name[name], df=df_by_name.get(name),
            pf=pf_by_name.get(name), ef=ef_by_name.get(name),
        ))
        for name in sorted(mp_by_name)
    ]


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------


class ExpressionEvalReport(BaseModel):
    """The full decision report: cost, VA coherence, AU cross-check, verdicts."""

    generated_at: str
    photo_count: int
    cost_table: list[CostRow]
    deepface_spread: DeepfaceSpreadStats
    pyfeat_va: VaCoherenceStats
    emotiefflib_va: VaCoherenceStats
    au43_cross_check: Au43CrossCheck
    pyfeat_verdict: CandidateVerdict
    emotiefflib_verdict: CandidateVerdict
    final_recommendation: str
    licensing_flags: list[str]
    unresolved: list[str]
    merged_rows: list[MergedRow]


def _build_report(outputs: _AllStageOutputs) -> ExpressionEvalReport:
    """Reduce every stage's output into the final ExpressionEvalReport."""
    triples = _eyes_closed_triples(outputs.baseline.mediapipe.readings, outputs.pyfeat.readings)
    au43_cross_check = _au43_cross_check(triples)
    pyfeat_verdict = _apply_rubric(_pyfeat_rubric_input(outputs, au43_cross_check))
    emotiefflib_verdict = _apply_rubric(_emotiefflib_rubric_input(outputs))
    return ExpressionEvalReport(
        generated_at=_utc_now(), photo_count=len(outputs.baseline.mediapipe.readings),
        cost_table=_build_cost_table(outputs),
        deepface_spread=_deepface_spread([r.dominant_emotion for r in outputs.baseline.deepface.readings]),
        pyfeat_va=_va_coherence(_VaCoherenceInput(
            candidate="Py-Feat", samples=_pyfeat_va_samples(outputs.pyfeat.readings)
        )),
        emotiefflib_va=_va_coherence(_VaCoherenceInput(
            candidate="EmotiEffLib", samples=_emotiefflib_va_samples(outputs.emotiefflib.readings)
        )),
        au43_cross_check=au43_cross_check, pyfeat_verdict=pyfeat_verdict,
        emotiefflib_verdict=emotiefflib_verdict,
        final_recommendation=_final_recommendation([pyfeat_verdict, emotiefflib_verdict]),
        licensing_flags=[DEEPFACE_LICENSE_NOTE, PYFEAT_LICENSE_NOTE, EMOTIEFFLIB_LICENSE_NOTE],
        unresolved=UNRESOLVED_NOTES, merged_rows=_build_merged_rows(outputs),
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _report_header_lines(report: ExpressionEvalReport) -> list[str]:
    """Return the title and headline summary lines."""
    return [
        "# Expression Head Eval: DeepFace vs Py-Feat Detectorv2 vs EmotiEffLib",
        "", f"Generated {report.generated_at}", "",
        f"- Photos evaluated: {report.photo_count}",
        f"- **Final recommendation: {report.final_recommendation}**", "",
    ]


def _report_cost_table_lines(report: ExpressionEvalReport) -> list[str]:
    """Return section A: the four-way cost table."""
    lines = [
        "## A. Cost table", "",
        "| model | mean latency/photo | load time | RSS delta |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r.model} | {r.mean_latency_seconds*1000:.1f} ms | {r.load_seconds}s | {r.rss_delta_mb} MB |"
        for r in report.cost_table
    ]
    return lines + [""]


def _report_va_lines(report: ExpressionEvalReport) -> list[str]:
    """Return section B: the VA coherence check against DeepFace's label scatter."""
    ds = report.deepface_spread
    lines = [
        "## B. VA coherence check", "",
        f"- DeepFace: {ds.n} faces scattered across {ds.distinct_labels} distinct labels; "
        f"dominant label `{ds.dominant_label}` covers only {ds.dominant_label_fraction:.1%}.",
        "",
        "| candidate | n | mean valence | std valence | mean arousal | std arousal | "
        "dominant quadrant | dominant quadrant share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for va in (report.pyfeat_va, report.emotiefflib_va):
        lines.append(
            f"| {va.candidate} | {va.n} | {va.mean_valence} | {va.std_valence} | "
            f"{va.mean_arousal} | {va.std_arousal} | {va.dominant_quadrant} | "
            f"{va.dominant_quadrant_fraction:.1%} |"
        )
    return lines + ["", *_va_verdict_lines(report)]


def _va_verdict_lines(report: ExpressionEvalReport) -> list[str]:
    """State plainly whether VA clustering beats DeepFace's categorical scatter."""
    more_coherent = report.pyfeat_va.dominant_quadrant_fraction > report.deepface_spread.dominant_label_fraction
    verdict = (
        "more coherent than DeepFace's categorical scatter"
        if more_coherent else "roughly as noisy as DeepFace's categorical scatter"
    )
    return [f"> Py-Feat's VA quadrant clustering is **{verdict}** on this corpus.", ""]


def _report_au43_lines(report: ExpressionEvalReport) -> list[str]:
    """Return section C: the AU43/EAR/blendshape 3-way cross-check."""
    c = report.au43_cross_check
    return [
        "## C. AU sanity cross-check (eyes-closed)", "",
        f"- n={c.n_common} face photos with all three signals",
        f"- **3-way agreement (EAR == blendshape == AU43): {c.all_three_agree_rate:.1%}**",
        f"- EAR vs AU43: {c.ear_vs_au43_rate:.1%}",
        f"- blendshape vs AU43: {c.blendshape_vs_au43_rate:.1%}",
        f"- EAR vs blendshape (existing baseline): {c.ear_vs_blendshape_rate:.1%}",
        "",
    ]


def _report_verdict_lines(report: ExpressionEvalReport) -> list[str]:
    """Return the rubric verdicts, licensing flags, and unresolved caveats."""
    lines = ["## Rubric verdicts", ""]
    for v in (report.pyfeat_verdict, report.emotiefflib_verdict):
        lines.append(f"### {v.name}: {v.verdict}")
        lines += [f"- {r}" for r in v.reasoning]
        lines.append("")
    lines += ["## Licensing flags", ""] + [f"- {f}" for f in report.licensing_flags] + [""]
    lines += ["## Unresolved (pending owner reverence labels)", ""]
    lines += [f"- {n}" for n in report.unresolved] + [""]
    return lines


def _report_per_photo_lines(report: ExpressionEvalReport) -> list[str]:
    """Return the full per-photo table for all 30 face-bearing photos."""
    lines = [
        "## Per-photo table", "",
        "| photo | deepface | pyfeat emotion | pyfeat V | pyfeat A | pyfeat AU43-closed | "
        "emotiefflib emotion | emotiefflib V | emotiefflib A | EAR-closed | blendshape-closed |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [_per_photo_row(r) for r in report.merged_rows]
    return lines


def _per_photo_row(r: MergedRow) -> str:
    """Format one MergedRow as a markdown table row."""
    return (
        f"| {r.name} | {r.deepface_label} | {r.pyfeat_emotion} | {r.pyfeat_valence} | "
        f"{r.pyfeat_arousal} | {r.pyfeat_au43_eyes_closed} | {r.emotiefflib_emotion} | "
        f"{r.emotiefflib_valence} | {r.emotiefflib_arousal} | {r.ear_eyes_closed} | "
        f"{r.blendshape_eyes_closed} |"
    )


def _write_report_markdown(report: ExpressionEvalReport) -> None:
    """Write the durable, human-readable decision report."""
    RUNS_DIR.mkdir(exist_ok=True)
    lines = (
        _report_header_lines(report) + _report_cost_table_lines(report)
        + _report_va_lines(report) + _report_au43_lines(report)
        + _report_verdict_lines(report) + _report_per_photo_lines(report)
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def _print_summary(report: ExpressionEvalReport) -> None:
    """Print the headline numbers to stdout."""
    print(json.dumps({
        "photo_count": report.photo_count,
        "cost_table": [r.model_dump() for r in report.cost_table],
        "pyfeat_va": report.pyfeat_va.model_dump(),
        "emotiefflib_va": report.emotiefflib_va.model_dump(),
        "deepface_spread": report.deepface_spread.model_dump(),
        "au43_cross_check": report.au43_cross_check.model_dump(),
        "pyfeat_verdict": report.pyfeat_verdict.model_dump(),
        "emotiefflib_verdict": report.emotiefflib_verdict.model_dump(),
        "final_recommendation": report.final_recommendation,
    }, indent=2))


def main() -> None:
    """Gather photos, run both new candidate stages, merge with baseline, report."""
    RUNS_DIR.mkdir(exist_ok=True)
    photos = _gather_photos()
    manifest_path = RUNS_DIR / "expression_eval_manifest.json"
    pf_out_path = RUNS_DIR / "expression_eval_pyfeat.json"
    ef_out_path = RUNS_DIR / "expression_eval_emotiefflib.json"
    manifest_path.write_text(json.dumps([str(p) for p in photos]))
    _run_stage(_StageInvocation(stage="pyfeat", input_path=manifest_path, output_path=pf_out_path))
    _run_stage(_StageInvocation(stage="emotiefflib", input_path=manifest_path, output_path=ef_out_path))
    outputs = _AllStageOutputs(
        baseline=_load_baseline(),
        pyfeat=PyFeatStageOutput.model_validate_json(pf_out_path.read_text()),
        emotiefflib=EmotiEfflibStageOutput.model_validate_json(ef_out_path.read_text()),
    )
    report = _build_report(outputs)
    _write_report_markdown(report)
    _print_summary(report)


if __name__ == "__main__":
    main()
