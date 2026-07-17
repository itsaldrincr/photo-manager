---
id: task_712
name: face_occlusion_detection
state: DONE
step: 5 of 5
depends: []
checkpoint: 9c7e14
created: 2026-07-18
---

## Program (immutable — set at planning)
1. Build labeled occlusion eval set in scratchpad (30 negatives from ev_faces + ~90 synthetic positives: opaque rect lower/upper/left, skin-tone ellipse, heavy gaussian blur).
2. Prototype candidate occlusion signals (landmark-patch texture flatness, blendshape degeneracy, face-detection confidence drop, EAR/z asymmetry), measure P/R/F1 on eval set, pick winner.
3. Implement winner as new `detect_occlusion` in `src/cull/stage2/portrait.py` (same contract: occluded bool + occlusion_ratio float, lower = more occluded). Remove dead visibility-based path. Constants to `config.py`.
4. Unit tests (synthetic arrays + None-visibility regression) in `tests/test_portrait.py`; keep full suite green (`pytest tests -q -k "not golden and not perf"`).
5. E2E: `assess_portrait` on 5 clean + 5 occluded eval photos; confirm preset penalty applies via `_compute_face_metrics` → `fusion._portrait_adjustment`.

## Registers (mutable — agent writes after each step)
- step 1 done @ nonce 3f9a21 → built scratchpad/occlusion_eval: 27 real negatives (3 of 30 ev_faces skipped, no face detected) + 81 synthetic positives (27 × [opaque-rect lower/upper/left, skin-tone ellipse mouth/eye, heavy-gaussian-blur patch]). generate_eval_set.py + extract_features.py (pass-2 MediaPipe with blendshapes) + manifest.json/features_manifest.json.
- step 2 done @ nonce 5b8d02 → prototyped: detected-only gate (AUC 0.70), per-region Laplacian-variance min (AUC 0.85, absolute — resolution-fragile), min/median self-normalized ratio (AUC 0.79, resolution-invariant, chosen), blendshape-std (AUC 0.52, discarded). eval_candidates.py holds final reproducible table.
- step 3 done @ nonce 7a1f39 → implemented detect_occlusion(image, landmarks) in portrait.py: samples small IOD-scaled patches at 6 key regions (eyes/brows/nose/mouth), ratio = min(region variance)/median(region variance). Removed _is_landmark_visible + visibility-based detect_occlusion. Added PORTRAIT_OCCLUSION_PATCH_IOD_FRACTION=0.03, PORTRAIT_OCCLUSION_PATCH_MIN_HALF_PX=3 to config.py; updated PORTRAIT_FACE_OCCLUSION_MIN 0.70→0.32 (new scale). Removed dead PORTRAIT_LANDMARK_VISIBILITY_THRESHOLD.
- step 4 done @ nonce 2e6c88 → rewrote tests/test_portrait.py occlusion tests for the new (image, landmarks) signature + synthetic textured/flat-block images. Full suite: `pytest tests -q -k "not golden and not perf"` → 494 passed (repo baseline had drifted above the stated 481 due to task_701 already being merged; no regressions, exit 0).
- step 5 done @ nonce 9c7e14 → e2e_demo.py + direct assess_portrait() sweep over all 108 eval images: detected=79/108, confusion P=0.84 R=0.72 F1=0.78 (matches calibration). Fusion trace on DSCF0218_rect_upper.jpg (wedding preset): face_occluded=True → _portrait_adjustment delta -0.04 vs counterfactual +0.08 with face_occluded forced False — isolated penalty contribution = 0.12, exactly PRESET_QUALITY_POLICY["wedding"]["face_occlusion_penalty"]. Confirms the previously-dead penalty now fires.

## Working Memory (scratch values the agent carries forward)
- Eval set root: /private/tmp/claude-501/-Users-alrelador/85225a69-a5ad-4e09-b879-f167553ba959/scratchpad/occlusion_eval/
- Contract: detect_occlusion consumed only by `_compute_face_metrics` (portrait.py:245-255); `_assemble_result` derives face_occluded = occlusion < PORTRAIT_FACE_OCCLUSION_MIN.
- Downstream: PortraitScores.is_face_occluded → fusion._portrait_adjustment subtracts policy["face_occlusion_penalty"] (config.py PRESET_QUALITY_POLICY, 0.00-0.12 by preset).
- No FaceDetector (BlazeFace) .task model cached — only face_landmarker.task. Avoid downloading new heavy model; candidate (c) face-detector-confidence must reuse FaceLandmarker's own detection, which does not expose a raw score in FaceLandmarkerResult — document as a limitation.
- Only one heavy model process (MediaPipe FaceLandmarker) at a time; batch eval passes.

## Acceptance Criteria
- [ ] Eval set built and documented (limitations noted: synthetic ≠ natural occlusion).
- [ ] Candidate signals measured with P/R/F1 table.
- [ ] detect_occlusion reimplemented, no visibility-based dead code remains.
- [ ] Constants added to config.py, no magic numbers in portrait.py.
- [ ] pytest tests -q -k "not golden and not perf" passes with baseline count (481) or documented delta.
- [ ] E2E demonstration on 10 eval photos (5 clean/5 occluded) with fusion penalty trace.

## Transition Rules
- IF current step DONE → increment step, update Registers, continue
- IF all steps DONE → set state: VERIFY, self-check acceptance criteria
- IF verify passes → set state: DONE, update MAP.md flag
- IF verify fails → set state to failed step number, note what failed
