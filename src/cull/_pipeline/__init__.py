"""Private pipeline subpackage — re-exports the frozen cull_fast contract.

This package owns the post-700-series split of `cull.pipeline` into per-stage
modules. The public `cull.pipeline` shim re-imports from here, so the frozen
cull_fast import contract continues to resolve via either path:

    from cull._pipeline import run_pipeline, _S2RunInput, ...
    from cull.pipeline   import run_pipeline, _S2RunInput, ...

Every symbol listed in `__all__` is part of the frozen contract verified by
`tests/cull_fast/test_imports_stable.py`. Adding or removing entries here is a
contract change and must be coordinated with the manifest in
`plans/refactor_700_series_manifest.md`.
"""

from __future__ import annotations

from cull._pipeline.decision_assembly import (
    _build_all_decisions,
    _build_decision,
    _build_photo_meta,
    _build_summary,
    _decide_label,
    _DecisionCtx,
)
from cull._pipeline.orchestrator import (
    _assemble_session,
    _build_run_ctx,
    _execute_run,
    _execute_stages_inline,
    _finalize_run,
    _make_dashboard,
    _PipelineCtx,
    _PipelineRunInput,
    _resolve_vlm_session_scope,
    _RunState,
    _run_with_session,
    _scan_with_dashboard,
    _SessionInput,
    _StageRunCtx,
    _StagesResult,
    _StageTimings,
    _unload_imagededup_cnn,
    _unload_stage2_models,
    run_pipeline,
    SessionResult,
    SessionSummary,
    SessionTiming,
)
from cull._pipeline.stage1_runner import (
    _run_s1,
    _Stage1LoopInput,
    _Stage1Output,
)
from cull._pipeline.stage2_reducer import (
    _build_reducer_input,
    _reroute_after_patch,
    _run_s2_reducer,
    _S2ReducerRunInput,
)
from cull._pipeline.stage2_runner import (
    _classify_s2_routing,
    _run_portrait_if_needed,
    _run_s2,
    _S2RunInput,
    _Stage2LoopInput,
    _Stage2Output,
    _Stage2ScoreInput,
)
from cull._pipeline.stage2_scoring import (
    _apply_exposure_to_scores,
    _load_tensor_batch,
)
from cull._pipeline.stage3_runner import (
    _run_s3,
    _run_s3_if_configured,
    _S3MaybeRunInput,
    _S3RunInput,
)
from cull._pipeline.stage4_curator import (
    _build_curator_input,
    _run_s4,
    _S4RunInput,
)

__all__: list[str] = [
    "run_pipeline",
    "SessionResult",
    "SessionTiming",
    "SessionSummary",
    "_PipelineRunInput",
    "_RunState",
    "_StageRunCtx",
    "_StagesResult",
    "_StageTimings",
    "_PipelineCtx",
    "_SessionInput",
    "_run_s1",
    "_run_s3",
    "_run_s3_if_configured",
    "_run_s2",
    "_run_s2_reducer",
    "_run_s4",
    "_Stage1LoopInput",
    "_Stage1Output",
    "_S2RunInput",
    "_S2ReducerRunInput",
    "_S3RunInput",
    "_S3MaybeRunInput",
    "_S4RunInput",
    "_Stage2Output",
    "_Stage2LoopInput",
    "_Stage2ScoreInput",
    "_DecisionCtx",
    "_load_tensor_batch",
    "_apply_exposure_to_scores",
    "_classify_s2_routing",
    "_run_portrait_if_needed",
    "_finalize_run",
    "_make_dashboard",
    "_scan_with_dashboard",
    "_unload_stage2_models",
    "_unload_imagededup_cnn",
    "_execute_run",
    "_execute_stages_inline",
    "_resolve_vlm_session_scope",
    "_run_with_session",
    "_build_run_ctx",
    "_build_curator_input",
    "_assemble_session",
    "_decide_label",
    "_build_decision",
    "_build_all_decisions",
    "_build_summary",
    "_build_photo_meta",
    "_build_reducer_input",
    "_reroute_after_patch",
]
