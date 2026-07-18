"""Fit wedding genre weights (implicit Edited/ labels) against unbiased ground truth.

Reads ground_truth.json (filename -> keep/reject label) and scored_rows.json
(per-photo raw fusion-input metrics from the real Stage 1+2 pipeline), then:

  1. Sweeps (ROUTING_AMBIGUOUS_MIN, ROUTING_KEEPER_MIN) under CURRENT holiday
     weights, minimizing auto-routing error subject to VLM-share <= 35%.
  2. Fits a standardized logistic regression predicting keep/reject from all
     11 GENRE_WEIGHTS metrics, 5-fold CV, maps coefficients to non-negative
     weights for the core 4 (topiq/laion/clipiqa/exposure).
  3. Compares current vs fitted core-4 weights on 5-fold CV routing accuracy
     (each evaluated at its own best threshold pair).
  4. Applies the >=3-point gate and prints a machine-readable verdict.

Usage:
    python3 fit_and_report.py <ground_truth.json> <scored_rows.json> <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

CORE_METRICS: list[str] = ["topiq", "laion_aesthetic", "clipiqa", "exposure"]
PENALTY_METRICS: list[str] = [
    "tilt_penalty", "palette_outlier_score", "exposure_drift_score", "exif_anomaly_score",
]
BONUS_METRICS: list[str] = ["composition", "taste_probability", "scene_start_bonus"]
ALL_METRICS: list[str] = CORE_METRICS + BONUS_METRICS + PENALTY_METRICS

CURRENT_HOLIDAY_WEIGHTS: dict[str, float] = {
    "topiq": 0.25, "laion_aesthetic": 0.40, "clipiqa": 0.25, "exposure": 0.10,
    "composition": 0.15, "taste": 0.15, "tilt_penalty": 0.05,
    "palette_outlier": 0.05, "exposure_drift": 0.05, "exif_anomaly": 0.03,
    "scene_start_bonus": 0.04,
}
CURRENT_KEEPER_MIN: float = 0.94
CURRENT_AMBIGUOUS_MIN: float = 0.85
VLM_SHARE_MAX: float = 0.35
GATE_MIN_IMPROVEMENT_POINTS: float = 3.0
N_FOLDS: int = 5
RANDOM_SEED: int = 0


def _load_matched_rows(gt_path: Path, rows_path: Path) -> tuple[list[dict], dict[str, Any]]:
    """Join ground truth labels to scored rows by day+orig_filename key; report gaps."""
    gt = json.loads(gt_path.read_text())
    scored = json.loads(rows_path.read_text())
    rows_by_key = scored["rows"]

    matched: list[dict] = []
    missing: list[str] = []
    for gt_name, meta in gt.items():
        key = gt_name
        row = rows_by_key.get(key)
        if row is None:
            missing.append(gt_name)
            continue
        entry = dict(row)
        entry["gt_filename"] = gt_name
        entry["label"] = meta["label"]
        entry["day"] = meta["day"]
        matched.append(entry)

    gap_report = {
        "ground_truth_total": len(gt),
        "matched": len(matched),
        "missing_from_scored": missing,
        "stage1_dropped": scored.get("stage1_dropped", []),
        "stage1_dropped_ground_truth_keepers": [
            name for name in missing
            if gt[name]["label"] == "keep"
        ],
    }
    return matched, gap_report


def _feature_matrix(rows: list[dict], metrics: list[str]) -> np.ndarray:
    """Assemble a raw (unstandardized) feature matrix, imputing None as 0.0."""
    return np.array(
        [[float(r.get(m) or 0.0) for m in metrics] for r in rows], dtype=np.float64
    )


def _labels(rows: list[dict]) -> np.ndarray:
    """Return a 0/1 label array: 1 = keep, 0 = reject."""
    return np.array([1 if r["label"] == "keep" else 0 for r in rows], dtype=np.int64)


def _current_composite(rows: list[dict]) -> np.ndarray:
    """Return the real pipeline's current-weights composite for each row."""
    return np.array([r["composite_current_weights"] for r in rows], dtype=np.float64)


class _ThresholdPair(NamedTuple):
    """One (ambiguous_min, keeper_min) routing threshold candidate."""

    ambiguous_min: float
    keeper_min: float


class _RouteEvalInput(NamedTuple):
    """Bundle of ground-truth-labeled composites plus a threshold candidate."""

    composite: np.ndarray
    y: np.ndarray
    thresholds: _ThresholdPair


def _route(eval_in: _RouteEvalInput) -> np.ndarray:
    """Return routing labels: 0=REJECT, 1=AMBIGUOUS, 2=KEEPER."""
    labels = np.zeros_like(eval_in.composite, dtype=np.int64)
    labels[eval_in.composite >= eval_in.thresholds.ambiguous_min] = 1
    labels[eval_in.composite >= eval_in.thresholds.keeper_min] = 2
    return labels


def _routing_error_and_share(eval_in: _RouteEvalInput) -> tuple[float, float]:
    """Return (auto-routing error rate, VLM/ambiguous share) for one threshold pair."""
    routing = _route(eval_in)
    vlm_share = float(np.mean(routing == 1))
    auto_mask = routing != 1
    if not auto_mask.any():
        return 1.0, vlm_share
    auto_decision = (routing[auto_mask] == 2).astype(np.int64)
    error = float(np.mean(auto_decision != eval_in.y[auto_mask]))
    return error, vlm_share


def _sweep_thresholds(
    composite: np.ndarray, y: np.ndarray
) -> list[dict[str, float]]:
    """Grid-sweep (ambiguous_min, keeper_min) pairs; return every evaluated point."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    results: list[dict[str, float]] = []
    for ambiguous_min in grid:
        for keeper_min in grid:
            if keeper_min <= ambiguous_min:
                continue
            thresholds = _ThresholdPair(float(ambiguous_min), float(keeper_min))
            error, vlm_share = _routing_error_and_share(
                _RouteEvalInput(composite=composite, y=y, thresholds=thresholds)
            )
            results.append({
                "ambiguous_min": thresholds.ambiguous_min, "keeper_min": thresholds.keeper_min,
                "error": error, "vlm_share": vlm_share,
            })
    return results


def _best_under_constraint(
    sweep: list[dict[str, float]], vlm_share_max: float
) -> dict[str, float] | None:
    """Return the lowest-error point with vlm_share <= vlm_share_max."""
    feasible = [p for p in sweep if p["vlm_share"] <= vlm_share_max]
    if not feasible:
        return None
    return min(feasible, key=lambda p: (p["error"], p["vlm_share"]))


def _frontier(sweep: list[dict[str, float]]) -> list[dict[str, float]]:
    """Return the best-error point at each of several VLM-share caps."""
    frontier = []
    for cap in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 1.00):
        best = _best_under_constraint(sweep, cap)
        if best is not None:
            frontier.append({"vlm_share_cap": cap, **best})
    return frontier


# ---------------------------------------------------------------------------
# Composite reconstruction — mirrors cull.stage2.fusion math exactly, so we
# can recompute the composite under alternate core-4 weights while holding
# every other term (composition/taste/tilt/reducer/portrait) at its real,
# pipeline-computed value. Constants copied from cull.config (holiday preset).
# ---------------------------------------------------------------------------

SUBJECT_BLUR_NORM_DIVISOR: float = 1000.0
HOLIDAY_QUALITY_POLICY: dict[str, float] = {
    "subject_blur_blend": 0.30, "bokeh_bonus": 0.05,
    "portrait_sharpness_bonus": 0.08, "eyes_closed_penalty": 0.18,
    "face_occlusion_penalty": 0.12,
}
TASTE_RAMP_LABELS: int = 20


def _clamp_unit(value: float) -> float:
    """Clamp a scalar into the closed [0, 1] interval."""
    return min(1.0, max(0.0, value))


def _blended_topiq(row: dict) -> float:
    """Replicate fusion._topiq_term: blend in subject-region sharpness + bokeh bonus."""
    tenengrad = row.get("subject_blur_tenengrad")
    if tenengrad is None:
        return float(row["topiq"])
    blend = HOLIDAY_QUALITY_POLICY["subject_blur_blend"]
    subject_score = min(1.0, max(0.0, tenengrad / SUBJECT_BLUR_NORM_DIVISOR))
    blended = (1.0 - blend) * float(row["topiq"]) + blend * subject_score
    if row.get("is_bokeh"):
        blended += HOLIDAY_QUALITY_POLICY["bokeh_bonus"]
    return _clamp_unit(blended)


def _taste_term(row: dict, taste_weight: float) -> float:
    """Replicate fusion._taste_term: ramp the taste weight by label count."""
    taste = row.get("taste_probability")
    if taste is None:
        return 0.0
    label_count = row.get("taste_label_count") or 0
    ramp = min(label_count / TASTE_RAMP_LABELS, 1.0)
    return ramp * taste_weight * taste


def _portrait_adjustment(row: dict) -> float:
    """Replicate fusion._portrait_adjustment for the holiday preset policy."""
    if row.get("eye_sharpness_left") is None and row.get("eye_sharpness_right") is None:
        eye_values = []
    else:
        eye_values = [
            v for v in (row.get("eye_sharpness_left"), row.get("eye_sharpness_right"))
            if v is not None
        ]
    delta = 0.0
    if eye_values:
        normalized_eye = min(1.0, max(0.0, (sum(eye_values) / len(eye_values)) / SUBJECT_BLUR_NORM_DIVISOR))
        delta += HOLIDAY_QUALITY_POLICY["portrait_sharpness_bonus"] * normalized_eye
    if row.get("is_eyes_closed"):
        delta -= HOLIDAY_QUALITY_POLICY["eyes_closed_penalty"]
    if row.get("is_face_occluded"):
        delta -= HOLIDAY_QUALITY_POLICY["face_occlusion_penalty"]
    return delta


def _reducer_term(row: dict, weights: dict[str, float]) -> float:
    """Replicate fusion._reducer_term: palette/exposure/EXIF/scene deltas."""
    palette = (row.get("palette_outlier_score") or 0.0) * weights.get("palette_outlier", 0.0)
    drift = (row.get("exposure_drift_score") or 0.0) * weights.get("exposure_drift", 0.0)
    anomaly = (row.get("exif_anomaly_score") or 0.0) * weights.get("exif_anomaly", 0.0)
    bonus = (row.get("scene_start_bonus") or 0.0) * weights.get("scene_start_bonus", 0.0)
    return bonus - palette - drift - anomaly


def _recompute_composite(row: dict, weights: dict[str, float]) -> float:
    """Recompute the full fusion composite for one row under candidate `weights`."""
    topiq = _blended_topiq(row)
    composition = row.get("composition") or 0.0
    raw = (
        weights["topiq"] * topiq
        + weights["laion_aesthetic"] * float(row["laion_aesthetic"])
        + weights["clipiqa"] * float(row["clipiqa"])
        + weights["exposure"] * float(row["exposure"])
        + weights.get("composition", 0.0) * composition
        + _taste_term(row, weights.get("taste", 0.0))
        - weights.get("tilt_penalty", 0.0) * (row.get("tilt_penalty") or 0.0)
        + _reducer_term(row, weights)
    )
    return _clamp_unit(raw + _portrait_adjustment(row))


# ---------------------------------------------------------------------------
# Logistic regression fit — standardized metrics -> keep/reject
# ---------------------------------------------------------------------------


def _fit_logreg(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Standardize X, fit a balanced logistic regression, return metric->coef."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)
    clf.fit(X_std, y)
    return dict(zip(ALL_METRICS, clf.coef_[0].tolist()))


def _map_core_weights(coefs: dict[str, float]) -> dict[str, float]:
    """Floor negative core coefficients at 0, normalize the core 4 to sum to 1.0."""
    floored = {m: max(coefs[m], 0.0) for m in CORE_METRICS}
    total = sum(floored.values())
    if total <= 0.0:
        return dict(zip(CORE_METRICS, [CURRENT_HOLIDAY_WEIGHTS[m] for m in CORE_METRICS]))
    return {m: floored[m] / total for m in CORE_METRICS}


_EXPECTED_SIGN: dict[str, int] = {
    "topiq": 1, "laion_aesthetic": 1, "clipiqa": 1, "exposure": 1,
    "composition": 1, "taste_probability": 1, "scene_start_bonus": 1,
    "tilt_penalty": -1, "palette_outlier_score": -1,
    "exposure_drift_score": -1, "exif_anomaly_score": -1,
}


def _non_core_disagreements(coefs: dict[str, float]) -> list[dict[str, Any]]:
    """Flag non-core metrics whose fitted sign disagrees with fusion's prior."""
    flagged = []
    for metric in BONUS_METRICS + PENALTY_METRICS:
        coef = coefs[metric]
        expected = _EXPECTED_SIGN[metric]
        agrees = (coef * expected) > 0
        flagged.append({
            "metric": metric, "coefficient": coef,
            "expected_sign": "positive" if expected > 0 else "negative",
            "agrees_with_current_design": agrees,
        })
    return flagged


# ---------------------------------------------------------------------------
# Cross-validated routing-accuracy comparison
# ---------------------------------------------------------------------------


def _routing_accuracy_at_best_threshold(
    composite: np.ndarray, y: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Return (accuracy, best-threshold-pair) under the VLM-share<=35% constraint."""
    sweep = _sweep_thresholds(composite, y)
    best = _best_under_constraint(sweep, VLM_SHARE_MAX)
    if best is None:
        return 0.0, {}
    return 1.0 - best["error"], best


def _weights_full(core: dict[str, float]) -> dict[str, float]:
    """Merge fitted core-4 weights with current non-core weights (unchanged)."""
    merged = dict(CURRENT_HOLIDAY_WEIGHTS)
    merged.update(core)
    return merged


def _cv_compare(rows: list[dict]) -> dict[str, Any]:
    """5-fold CV: current weights vs per-fold-fitted core weights, at each's best threshold."""
    y = _labels(rows)
    X_all = _feature_matrix(rows, ALL_METRICS)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    current_accs, fitted_accs, fold_weights = [], [], []
    for train_idx, test_idx in skf.split(X_all, y):
        test_rows = [rows[i] for i in test_idx]
        y_test = y[test_idx]
        current_composite = np.array(
            [_recompute_composite(r, CURRENT_HOLIDAY_WEIGHTS) for r in test_rows]
        )
        current_acc, _ = _routing_accuracy_at_best_threshold(current_composite, y_test)
        current_accs.append(current_acc)

        train_coefs = _fit_logreg(X_all[train_idx], y[train_idx])
        fold_core = _map_core_weights(train_coefs)
        fold_weights.append(fold_core)
        fitted_composite = np.array(
            [_recompute_composite(r, _weights_full(fold_core)) for r in test_rows]
        )
        fitted_acc, _ = _routing_accuracy_at_best_threshold(fitted_composite, y_test)
        fitted_accs.append(fitted_acc)

    return {
        "current_cv_accuracy_mean": float(np.mean(current_accs)),
        "current_cv_accuracy_folds": current_accs,
        "fitted_cv_accuracy_mean": float(np.mean(fitted_accs)),
        "fitted_cv_accuracy_folds": fitted_accs,
        "fitted_core_weights_per_fold": fold_weights,
        "improvement_points": float(
            (np.mean(fitted_accs) - np.mean(current_accs)) * 100.0
        ),
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


class _ReportCtx(NamedTuple):
    """Everything the markdown report needs, computed once in main()."""

    gap_report: dict[str, Any]
    reconstruction_check: dict[str, float]
    sweep_best: dict[str, float]
    frontier: list[dict[str, float]]
    full_coefs: dict[str, float]
    full_core_weights: dict[str, float]
    non_core_flags: list[dict[str, Any]]
    cv: dict[str, Any]
    gate_pass: bool


def _reconstruction_check(rows: list[dict]) -> dict[str, float]:
    """Validate _recompute_composite against the pipeline's real composite."""
    recomputed = np.array([_recompute_composite(r, CURRENT_HOLIDAY_WEIGHTS) for r in rows])
    actual = _current_composite(rows)
    diff = np.abs(recomputed - actual)
    return {
        "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
        "n": len(rows),
    }


def _section_corpus(ctx: _ReportCtx, rows: list[dict]) -> str:
    """Render the corpus + data-quality section."""
    gap = ctx.gap_report
    keep_n = sum(1 for r in rows if r["label"] == "keep")
    reject_n = len(rows) - keep_n
    lines = [
        "## Corpus",
        "",
        f"- Implicit-label photos (Backup/ matched via DINOv2 to Edited/ selects): "
        f"{gap['ground_truth_total']}",
        f"- Matched to a Stage 2 composite: {gap['matched']} ({keep_n} keep / {reject_n} reject)",
        f"- Dropped before Stage 2 (Stage 1 blur/noise/burst/duplicate reject): "
        f"{len(gap['missing_from_scored'])}",
        f"  - Of those, ground-truth KEEPERS dropped by Stage 1 (uncorrectable by "
        f"Stage 2 threshold/weight tuning): {len(gap['stage1_dropped_ground_truth_keepers'])} "
        f"{gap['stage1_dropped_ground_truth_keepers']}",
        "",
        "Composite-reconstruction sanity check (recomputed vs. real pipeline composite, "
        "current wedding weights):",
        f"- max abs diff: {ctx.reconstruction_check['max_abs_diff']:.6g}, "
        f"mean abs diff: {ctx.reconstruction_check['mean_abs_diff']:.6g} "
        f"(n={ctx.reconstruction_check['n']})",
        "",
    ]
    return "\n".join(lines)


def _section_threshold(ctx: _ReportCtx) -> str:
    """Render the routing-threshold sweep + frontier section."""
    lines = [
        "## Routing threshold sweep (current holiday weights)",
        "",
        f"Current thresholds: AMBIGUOUS_MIN={CURRENT_AMBIGUOUS_MIN}, "
        f"KEEPER_MIN={CURRENT_KEEPER_MIN}",
        "",
        f"Best under VLM-share <= {VLM_SHARE_MAX:.0%}: "
        f"AMBIGUOUS_MIN={ctx.sweep_best.get('ambiguous_min')}, "
        f"KEEPER_MIN={ctx.sweep_best.get('keeper_min')}, "
        f"error={ctx.sweep_best.get('error', float('nan')):.4f}, "
        f"vlm_share={ctx.sweep_best.get('vlm_share', float('nan')):.4f}",
        "",
        "Frontier (best error at each VLM-share cap):",
        "",
        "| vlm_share_cap | ambiguous_min | keeper_min | error | vlm_share |",
        "|---|---|---|---|---|",
    ]
    for p in ctx.frontier:
        lines.append(
            f"| {p['vlm_share_cap']:.0%} | {p['ambiguous_min']:.2f} | {p['keeper_min']:.2f} "
            f"| {p['error']:.4f} | {p['vlm_share']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_coefficients(ctx: _ReportCtx) -> str:
    """Render the full-data logistic regression coefficient + weight-mapping tables."""
    lines = [
        "## Logistic regression fit (full data, standardized metrics -> keep/reject)",
        "",
        "### Core 4 (topiq / laion_aesthetic / clipiqa / exposure)",
        "",
        "| metric | current weight | LR coefficient | fitted weight (floored, normalized) |",
        "|---|---|---|---|",
    ]
    for m in CORE_METRICS:
        lines.append(
            f"| {m} | {CURRENT_HOLIDAY_WEIGHTS[m]:.3f} | {ctx.full_coefs[m]:+.4f} "
            f"| {ctx.full_core_weights[m]:.3f} |"
        )
    lines += [
        "",
        "### Non-core (composition / taste / penalties / bonus) — reported only, "
        "NOT applied per task scope",
        "",
        "| metric | current weight | LR coefficient | expected sign | agrees |",
        "|---|---|---|---|---|",
    ]
    non_core_weight_key = {
        "composition": "composition", "taste_probability": "taste",
        "scene_start_bonus": "scene_start_bonus", "tilt_penalty": "tilt_penalty",
        "palette_outlier_score": "palette_outlier",
        "exposure_drift_score": "exposure_drift", "exif_anomaly_score": "exif_anomaly",
    }
    for flag in ctx.non_core_flags:
        cur = CURRENT_HOLIDAY_WEIGHTS[non_core_weight_key[flag["metric"]]]
        lines.append(
            f"| {flag['metric']} | {cur:.3f} | {flag['coefficient']:+.4f} "
            f"| {flag['expected_sign']} | {'yes' if flag['agrees_with_current_design'] else '**NO**'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_cv(ctx: _ReportCtx) -> str:
    """Render the 5-fold CV routing-accuracy comparison + gate verdict."""
    cv = ctx.cv
    lines = [
        "## 5-fold cross-validated routing accuracy (each config at its own best "
        f"threshold, VLM-share <= {VLM_SHARE_MAX:.0%})",
        "",
        "| fold | current weights acc | fitted weights acc |",
        "|---|---|---|",
    ]
    for i in range(N_FOLDS):
        lines.append(
            f"| {i + 1} | {cv['current_cv_accuracy_folds'][i]:.4f} "
            f"| {cv['fitted_cv_accuracy_folds'][i]:.4f} |"
        )
    lines += [
        f"| **mean** | **{cv['current_cv_accuracy_mean']:.4f}** "
        f"| **{cv['fitted_cv_accuracy_mean']:.4f}** |",
        "",
        f"Improvement: {cv['improvement_points']:+.2f} points "
        f"(gate requires >= {GATE_MIN_IMPROVEMENT_POINTS:.1f})",
        "",
        f"## Gate verdict: {'PASS' if ctx.gate_pass else 'FAIL'}",
        "",
    ]
    return "\n".join(lines)


_LIMITATIONS_TEXT: str = """## Honest limitations

- **Implicit labels, not triage labels.** "Keep" means the owner exported an
  edited version of the photo; "reject" means they did not. Editing selection
  conflates artistic preference, client requests, and near-duplicate choice —
  it is a noisier signal than the explicit keep/reject triage used for the
  holiday calibration.
- **Edited->Backup matching is itself model-based.** Pairs come from DINOv2
  nearest-neighbor cosine matching (edited exports are renamed, so filename
  matching is impossible). Only high-confidence matches (similarity/margin
  gated) are used; unmatched Edited files are excluded and reported.
- **Two-shoot corpus, one photographer, rehearsal-heavy.** Both shoots are
  wedding-rehearsal/dance events by the same owner. Weights fitted here may
  not transfer to ceremony/reception coverage.
- **Single-owner taste + taste-model confound.** As in the holiday
  calibration: the live taste term is trained on the disagreement-biased
  override log and is captured as-is in both baseline and fitted composites.
- **Per-shoot scoring.** Each shoot was scored as its own stage1+2 run so
  reducer stats use the correct population.
- **Stage 1 drop-outs are out of scope for weight fitting** and are reported
  separately; a dropped ground-truth keeper cannot be recovered by weight
  tuning.
"""


def _run_fit(rows: list[dict]) -> _ReportCtx:
    """Run the threshold sweep, full-data LR fit, and CV comparison."""
    check = _reconstruction_check(rows)
    print(f"Reconstruction check: max_abs_diff={check['max_abs_diff']:.6g}")

    current_composite = _current_composite(rows)
    y = _labels(rows)
    sweep = _sweep_thresholds(current_composite, y)
    sweep_best = _best_under_constraint(sweep, VLM_SHARE_MAX) or {}
    frontier = _frontier(sweep)

    full_coefs = _fit_logreg(_feature_matrix(rows, ALL_METRICS), y)
    full_core_weights = _map_core_weights(full_coefs)
    non_core_flags = _non_core_disagreements(full_coefs)

    cv = _cv_compare(rows)
    gate_pass = cv["improvement_points"] >= GATE_MIN_IMPROVEMENT_POINTS
    return _ReportCtx(
        gap_report={}, reconstruction_check=check, sweep_best=sweep_best,
        frontier=frontier, full_coefs=full_coefs, full_core_weights=full_core_weights,
        non_core_flags=non_core_flags, cv=cv, gate_pass=gate_pass,
    )


def _write_outputs(write_in: tuple[_ReportCtx, list[dict], Path]) -> None:
    """Render the markdown report + JSON artifact and write both to out_dir."""
    ctx, rows, out_dir = write_in
    report = "\n".join([
        "# Calibration report — wedding genre weights (implicit Edited/ labels)",
        "",
        _section_corpus(ctx, rows),
        _section_threshold(ctx),
        _section_coefficients(ctx),
        _section_cv(ctx),
        _LIMITATIONS_TEXT,
    ])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_report.md").write_text(report)
    (out_dir / "calibration_fit_result.json").write_text(json.dumps({
        "gap_report": ctx.gap_report, "reconstruction_check": ctx.reconstruction_check,
        "sweep_best": ctx.sweep_best, "frontier": ctx.frontier, "full_coefs": ctx.full_coefs,
        "full_core_weights": ctx.full_core_weights, "non_core_flags": ctx.non_core_flags,
        "cv": ctx.cv, "gate_pass": ctx.gate_pass,
    }, indent=2))
    print(f"Wrote report to {out_dir / 'calibration_report.md'}")
    print(f"GATE: {'PASS' if ctx.gate_pass else 'FAIL'} ({ctx.cv['improvement_points']:+.2f} points)")


def main() -> None:
    """Run the full fit + gate + report pipeline."""
    gt_path, rows_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows, gap_report = _load_matched_rows(gt_path, rows_path)
    print(f"Matched {len(rows)} / {gap_report['ground_truth_total']} ground-truth photos")

    ctx = _run_fit(rows)._replace(gap_report=gap_report)
    _write_outputs((ctx, rows, out_dir))


if __name__ == "__main__":
    main()
