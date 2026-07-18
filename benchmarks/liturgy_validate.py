"""Analysis driver: validate the liturgy V/A window before wiring a preset.

Pure arithmetic — no model inference. The reverence prototype
(reverence_weaklabel_report.md) proposed treating a face as calm-reverent (and
NOT penalizing sad/surprise or closed eyes) when EmotiEffLib valence is in
[-0.4, +0.2] and arousal in [-0.3, +0.3]. That window is only safe to wire if
it EXCLUDES genuinely distressed faces while still covering reverent ones.

Gate (handover Task C):
    PASS iff >= 90% of agreed-distressed faces fall OUTSIDE the window AND
    >= 70% of the 32 agreed-reverent faces fall INSIDE it.

Distressed V/A come from a freshly mined, Gemma-double-pass-agreed distressed
set (this wave); reverent V/A come from the prior reverence artifacts. All
labels are MODEL-GENERATED weak labels (gemma-4-12b, double-pass agreement).

Usage:
    python3 benchmarks/liturgy_validate.py <config.json> <report.md>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

from fair_expr_models import RunnerStageOutput
from weaklabel_models import WeaklabelRunOutput, compute_agreement

VALENCE_RANGE: tuple[float, float] = (-0.4, 0.2)
AROUSAL_RANGE: tuple[float, float] = (-0.3, 0.3)
DISTRESS_OUTSIDE_MIN: float = 0.90
REVERENT_INSIDE_MIN: float = 0.70
MIN_DISTRESSED_FACES: int = 20


class ValidateConfig(BaseModel):
    """Paths to every input artifact of the liturgy-window validation."""

    distressed_va_path: Path
    distressed_names_path: Path
    reverence_labels_v1_path: Path
    reverence_labels_v2_path: Path
    reverence_va_path: Path


class _VAReading(BaseModel):
    """One face's valence/arousal."""

    name: str
    valence: float
    arousal: float


def _in_window(reading: _VAReading) -> bool:
    """Return True if the reading's V/A falls inside the proposed reverent window."""
    return (
        VALENCE_RANGE[0] <= reading.valence <= VALENCE_RANGE[1]
        and AROUSAL_RANGE[0] <= reading.arousal <= AROUSAL_RANGE[1]
    )


def _va_by_name(stage_out: RunnerStageOutput) -> dict[str, _VAReading]:
    """Index valence/arousal readings by crop name (faces with V/A only)."""
    result: dict[str, _VAReading] = {}
    for r in stage_out.readings:
        if r.has_face and r.valence is not None and r.arousal is not None:
            result[r.name] = _VAReading(name=r.name, valence=r.valence, arousal=r.arousal)
    return result


def _load_stage(path: Path) -> RunnerStageOutput:
    """Load an EmotiEffLib RunnerStageOutput from disk."""
    return RunnerStageOutput.model_validate_json(path.read_text())


def _reverent_names(config: ValidateConfig) -> set[str]:
    """Return the names of the agreed reverent/prayerful faces."""
    v1 = WeaklabelRunOutput.model_validate_json(config.reverence_labels_v1_path.read_text())
    v2 = WeaklabelRunOutput.model_validate_json(config.reverence_labels_v2_path.read_text())
    agreed = compute_agreement((v1, v2)).agreed
    return {a.name for a in agreed if a.label == "reverent/prayerful"}


def _distressed_readings(config: ValidateConfig) -> list[_VAReading]:
    """Return V/A readings for the agreed-distressed mined faces."""
    names = set(json.loads(config.distressed_names_path.read_text()))
    va = _va_by_name(_load_stage(config.distressed_va_path))
    return [va[n] for n in names if n in va]


def _reverent_readings(config: ValidateConfig) -> list[_VAReading]:
    """Return V/A readings for the 32 agreed-reverent faces."""
    names = _reverent_names(config)
    va = _va_by_name(_load_stage(config.reverence_va_path))
    return [va[n] for n in names if n in va]


class GateResult(BaseModel):
    """Outcome of the liturgy-window validation gate."""

    distressed_n: int
    distressed_outside_frac: float
    reverent_n: int
    reverent_inside_frac: float
    enough_distressed: bool
    passed: bool


def _fraction(predicate_true: int, total: int) -> float:
    """Return predicate_true / total, or 0.0 when total is zero."""
    return predicate_true / total if total else 0.0


def _judge(distressed: list[_VAReading], reverent: list[_VAReading]) -> GateResult:
    """Apply the window gate to the distressed and reverent V/A sets."""
    d_outside = sum(1 for r in distressed if not _in_window(r))
    r_inside = sum(1 for r in reverent if _in_window(r))
    outside_frac = _fraction(d_outside, len(distressed))
    inside_frac = _fraction(r_inside, len(reverent))
    enough = len(distressed) >= MIN_DISTRESSED_FACES
    passed = (
        enough
        and outside_frac >= DISTRESS_OUTSIDE_MIN
        and inside_frac >= REVERENT_INSIDE_MIN
    )
    return GateResult(
        distressed_n=len(distressed), distressed_outside_frac=outside_frac,
        reverent_n=len(reverent), reverent_inside_frac=inside_frac,
        enough_distressed=enough, passed=passed,
    )


def _verdict_line(gate: GateResult) -> str:
    """Return the human verdict line for the gate result."""
    if not gate.enough_distressed:
        return (
            f"## Verdict: STOP (base rate) — only {gate.distressed_n} agreed-distressed "
            f"faces mined (< {MIN_DISTRESSED_FACES}). Window unvalidated; no preset. "
            "Distress does not occur at a useful base rate in the mined corpora."
        )
    if gate.passed:
        return (
            f"## Verdict: PASS — {gate.distressed_outside_frac:.0%} distressed outside "
            f"(>= {DISTRESS_OUTSIDE_MIN:.0%}) and {gate.reverent_inside_frac:.0%} reverent "
            f"inside (>= {REVERENT_INSIDE_MIN:.0%}). Window is safe to wire."
        )
    return (
        f"## Verdict: FAIL — distressed outside {gate.distressed_outside_frac:.0%} "
        f"(need {DISTRESS_OUTSIDE_MIN:.0%}), reverent inside {gate.reverent_inside_frac:.0%} "
        f"(need {REVERENT_INSIDE_MIN:.0%}). Window not safe; no preset."
    )


def _render_report(gate: GateResult) -> str:
    """Render the validation report as markdown."""
    return "\n".join([
        "# Liturgy V/A window validation — Gemma weak labels",
        "",
        "PROVENANCE: MODEL-GENERATED weak labels (gemma-4-12b, expression_v1 +",
        "expression_v2 double-pass agreement). NOT human ground truth.",
        "",
        f"- Proposed window: valence in {list(VALENCE_RANGE)}, arousal in {list(AROUSAL_RANGE)}",
        f"- Agreed-distressed faces (mined this wave): {gate.distressed_n}",
        f"- Distressed OUTSIDE window: {gate.distressed_outside_frac:.1%} "
        f"(gate >= {DISTRESS_OUTSIDE_MIN:.0%})",
        f"- Agreed-reverent faces (prior artifacts): {gate.reverent_n}",
        f"- Reverent INSIDE window: {gate.reverent_inside_frac:.1%} "
        f"(gate >= {REVERENT_INSIDE_MIN:.0%})",
        "",
        _verdict_line(gate),
        "",
    ])


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: liturgy_validate.py <config.json> <report.md>")
    config = ValidateConfig.model_validate_json(Path(sys.argv[1]).read_text())
    gate = _judge(_distressed_readings(config), _reverent_readings(config))
    Path(sys.argv[2]).write_text(_render_report(gate))
    print(gate.model_dump_json(indent=2))  # noqa: T201 — CLI result surface


if __name__ == "__main__":
    main()
