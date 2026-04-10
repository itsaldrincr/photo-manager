"""Contract tests: assert that every cull.* symbol fast mode depends on is importable."""

from __future__ import annotations

import importlib
import inspect
import logging

import pytest

logger = logging.getLogger(__name__)

LOAD_TENSOR_BATCH_PARAMS: list[str] = ["paths"]
LOAD_TENSOR_BATCH_RETURN_KEYWORDS: tuple[str, ...] = (
    "tuple", "torch.Tensor", "list", "Image.Image",
)

# Contract change rationale: `_unload_vlm_models` was removed from this list
# (and from cull.pipeline / cull._pipeline __all__) as part of the VLM offline
# integration. The vlm_session context manager in cull._pipeline.orchestrator
# now handles model unloading, so the standalone helper is gone. Similarly,
# `select_model` was replaced by `cull.vlm_registry.resolve_alias`, which is
# asserted via REQUIRED_VLM_REGISTRY_SYMBOLS below. See
# plans/vlm_offline_integration_manifest.md (Breaking Changes 1 and 2) for the
# full rationale.
REQUIRED_PIPELINE_SYMBOLS: list[str] = [
    "_run_s1",
    "_run_s2",
    "_run_s3",
    "_run_s4",
    "_S2RunInput",
    "_Stage2Output",
    "_Stage2LoopInput",
    "_Stage2ScoreInput",
    "_StagesResult",
    "_StageRunCtx",
    "_StageTimings",
    "_PipelineRunInput",
    "_S3RunInput",
    "_S4RunInput",
    "_RunState",
    "_DecisionCtx",
    "_SessionInput",
    "SessionResult",
    "_unload_stage2_models",
    "_resolve_vlm_session_scope",
    "_run_stage1_loop",
    "_load_tensor_batch",
    "_apply_exposure_to_scores",
    "_run_stage3_loop",
    "_run_portrait_if_needed",
    "_classify_s2_routing",
    "_make_dashboard",
    "_scan_with_dashboard",
    "_finalize_run",
    "_assemble_session",
    "_build_all_decisions",
]

REQUIRED_FUSION_SYMBOLS: list[str] = [
    "IqaScores",
    "compute_composite",
    "FusionResult",
]

REQUIRED_IQA_SYMBOLS: list[str] = [
    "select_device",
    "unload_metrics",
]

REQUIRED_AESTHETIC_SYMBOLS: list[str] = [
    "_normalize_aesthetic",
]

REQUIRED_CONFIG_SYMBOLS: list[str] = [
    "GENRE_WEIGHTS",
    "CullConfig",
    "ROUTING_KEEPER_MIN",
    "ROUTING_AMBIGUOUS_MIN",
    "STAGE2_BATCH_SIZE",
    "STAGE_IQA",
    "STAGE_VLM",
]

REQUIRED_DASHBOARD_SYMBOLS: list[str] = [
    "_Stage2UpdateInput",
    "Dashboard",
]

REQUIRED_VLM_REGISTRY_SYMBOLS: list[str] = [
    "resolve_alias",
]

IQA_SCORES_REQUIRED_FIELDS: set[str] = {
    "photo_path",
    "topiq",
    "laion_aesthetic",
    "clipiqa",
    "exposure",
}


def _assert_symbol(module_name: str, symbol_name: str) -> None:
    """Assert that module_name.symbol_name exists; log the missing symbol on failure."""
    module = importlib.import_module(module_name)
    if not hasattr(module, symbol_name):
        logger.error("Missing symbol: %s.%s", module_name, symbol_name)
    assert hasattr(module, symbol_name), (
        f"Symbol '{symbol_name}' not found in module '{module_name}'"
    )


@pytest.mark.parametrize("symbol", REQUIRED_PIPELINE_SYMBOLS)
def test_pipeline_symbols_exist(symbol: str) -> None:
    """Assert each pipeline symbol required by fast mode exists."""
    _assert_symbol("cull.pipeline", symbol)


@pytest.mark.parametrize("symbol", REQUIRED_FUSION_SYMBOLS)
def test_fusion_symbols_exist(symbol: str) -> None:
    """Assert each fusion symbol required by fast mode exists."""
    _assert_symbol("cull.stage2.fusion", symbol)


@pytest.mark.parametrize("symbol", REQUIRED_IQA_SYMBOLS)
def test_iqa_symbols_exist(symbol: str) -> None:
    """Assert each IQA symbol required by fast mode exists."""
    _assert_symbol("cull.stage2.iqa", symbol)


@pytest.mark.parametrize("symbol", REQUIRED_AESTHETIC_SYMBOLS)
def test_aesthetic_symbols_exist(symbol: str) -> None:
    """Assert each aesthetic symbol required by fast mode exists."""
    _assert_symbol("cull.stage2.aesthetic", symbol)


@pytest.mark.parametrize("symbol", REQUIRED_CONFIG_SYMBOLS)
def test_config_symbols_exist(symbol: str) -> None:
    """Assert each config symbol required by fast mode exists."""
    _assert_symbol("cull.config", symbol)


@pytest.mark.parametrize("symbol", REQUIRED_DASHBOARD_SYMBOLS)
def test_dashboard_symbols_exist(symbol: str) -> None:
    """Assert each dashboard symbol required by fast mode exists."""
    _assert_symbol("cull.dashboard", symbol)


@pytest.mark.parametrize("symbol", REQUIRED_VLM_REGISTRY_SYMBOLS)
def test_vlm_registry_symbols_exist(symbol: str) -> None:
    """Assert each VLM registry symbol required by fast mode exists."""
    _assert_symbol("cull.vlm_registry", symbol)


def test_iqa_scores_model_fields_contain_required_keys() -> None:
    """Assert IqaScores.model_fields contains the exact required field names."""
    module = importlib.import_module("cull.stage2.fusion")
    iqa_cls = getattr(module, "IqaScores")
    actual_fields: set[str] = set(iqa_cls.model_fields.keys())
    missing = IQA_SCORES_REQUIRED_FIELDS - actual_fields
    if missing:
        logger.error("IqaScores is missing required fields: %s", missing)
    assert not missing, (
        f"IqaScores.model_fields missing required keys: {missing}"
    )


def test_load_tensor_batch_signature_frozen() -> None:
    """`_load_tensor_batch(paths)` signature is frozen for pipeline_fast compatibility."""
    from cull.pipeline import _load_tensor_batch  # noqa: PLC0415

    sig = inspect.signature(_load_tensor_batch)
    assert list(sig.parameters) == LOAD_TENSOR_BATCH_PARAMS, (
        f"_load_tensor_batch params drifted: {list(sig.parameters)}"
    )
    return_annotation_str = str(sig.return_annotation)
    for keyword in LOAD_TENSOR_BATCH_RETURN_KEYWORDS:
        assert keyword in return_annotation_str, (
            f"_load_tensor_batch return annotation missing '{keyword}': "
            f"{return_annotation_str}"
        )


def test_pipeline_fast_direct_imports_succeed() -> None:
    """pipeline_fast's direct import list from cull.pipeline must still resolve."""
    from cull.pipeline import (  # noqa: PLC0415, F401
        _apply_exposure_to_scores,
        _classify_s2_routing,
        _load_tensor_batch,
        _run_portrait_if_needed,
        _S2RunInput,
        _Stage2LoopInput,
        _Stage2Output,
        _Stage2ScoreInput,
        _unload_stage2_models,
    )
