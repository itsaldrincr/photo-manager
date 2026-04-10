"""Pipeline shim — re-exports the frozen cull_fast contract from cull._pipeline.

The second import block keeps nine monkeypatch attachment points live as
`cull.pipeline.*` attributes so existing test fixtures using
`monkeypatch.setattr("cull.pipeline.X", ...)` resolve without rewriting.
"""

from __future__ import annotations

from cull._pipeline import (
    SessionResult, SessionSummary, SessionTiming, _apply_exposure_to_scores,
    _assemble_session, _build_all_decisions, _build_curator_input,
    _build_decision, _build_photo_meta, _build_reducer_input, _build_run_ctx,
    _build_summary, _classify_s2_routing, _DecisionCtx, _decide_label,
    _execute_run, _execute_stages_inline, _finalize_run, _load_tensor_batch,
    _make_dashboard, _PipelineCtx, _PipelineRunInput, _reroute_after_patch,
    _resolve_vlm_session_scope, _RunState, _run_portrait_if_needed, _run_s1,
    _run_s2, _run_s2_reducer, _run_s3, _run_s3_if_configured, _run_s4,
    _run_with_session, _S2ReducerRunInput, _S2RunInput, _S3MaybeRunInput,
    _S3RunInput, _S4RunInput, _scan_with_dashboard, _SessionInput,
    _Stage1LoopInput, _Stage1Output, _Stage2LoopInput, _Stage2Output,
    _Stage2ScoreInput, _StageRunCtx, _StagesResult, _StageTimings,
    _unload_imagededup_cnn, _unload_stage2_models, run_pipeline,
)
from cull._pipeline.stage1_runner import _run_stage1_loop  # noqa: F401  contract
from cull._pipeline.stage2_runner import (  # noqa: F401  test contract
    _BatchCtx, _ChunkInput, _populate_saliency_cache, _process_batch,
    _run_shared_clip_forward, _run_stage2_loop,
)
from cull._pipeline.stage2_scoring import (  # noqa: F401  test+monkeypatch surface
    _apply_composition_to_scores, _apply_subject_blur_to_scores,
    _apply_taste_to_scores, _build_composition_inputs, _build_subject_blur_input,
    _CompositionBuildInput, _DualLoadInput, _DualPilBatch, _load_dual_pil_batch,
)
from cull._pipeline.stage3_runner import _run_stage3_loop  # noqa: F401  contract
from cull.saliency import compute_saliency  # noqa: F401  monkeypatch surface
from cull.stage2.aesthetic import _get_head, score_aesthetic_batch  # noqa: F401
from cull.stage2.iqa import score_clipiqa_batch, score_topiq_batch  # noqa: F401

__all__: list[str] = [
    "run_pipeline", "SessionResult", "SessionSummary", "SessionTiming",
    "_PipelineRunInput", "_RunState", "_StageRunCtx", "_StagesResult",
    "_StageTimings", "_PipelineCtx", "_SessionInput", "_run_s1",
    "_run_stage1_loop", "_run_s2", "_run_s2_reducer", "_run_s3",
    "_run_stage3_loop", "_run_s3_if_configured", "_run_s4",
    "_Stage1LoopInput", "_Stage1Output", "_S2RunInput", "_S2ReducerRunInput",
    "_S3RunInput", "_S3MaybeRunInput", "_S4RunInput", "_Stage2Output",
    "_Stage2LoopInput", "_Stage2ScoreInput", "_DecisionCtx", "_load_tensor_batch",
    "_apply_exposure_to_scores", "_classify_s2_routing", "_run_portrait_if_needed",
    "_finalize_run", "_make_dashboard", "_scan_with_dashboard",
    "_unload_stage2_models", "_unload_imagededup_cnn",
    "_execute_run", "_execute_stages_inline", "_build_curator_input",
    "_assemble_session", "_decide_label", "_build_decision", "_build_all_decisions",
    "_build_summary", "_build_photo_meta", "_build_reducer_input",
    "_reroute_after_patch", "_BatchCtx", "_ChunkInput", "_process_batch",
    "_populate_saliency_cache", "_run_shared_clip_forward", "_run_stage2_loop",
    "_build_composition_inputs", "_build_subject_blur_input",
    "_CompositionBuildInput", "_DualLoadInput", "_DualPilBatch",
    "_load_dual_pil_batch", "_resolve_vlm_session_scope", "_run_with_session",
    "_build_run_ctx",
]
