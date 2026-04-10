"""Integration test for pipeline._run_s2_reducer ordering and in-place patching."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from cull.config import CullConfig
from cull.models import (
    BlurScores,
    ExposureScores,
    Stage1Result,
    Stage2Result,
)
from cull.pipeline import (
    _S2ReducerRunInput,
    _Stage1Output,
    _Stage2Output,
    _StageRunCtx,
    _execute_stages_inline,
    _run_s2_reducer,
)
from cull.stage2.fusion import FusionResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHOTO_COUNT: int = 5
OUTLIER_INDEX: int = 2
INLIER_TOPIQ: float = 0.6
INLIER_LAION: float = 0.6
INLIER_CLIPIQA: float = 0.6
INLIER_DR_SCORE: float = 0.6
OUTLIER_DR_SCORE: float = 0.95
INLIER_BLUR_TENENGRAD: float = 100.0
BASE_BLUR_FFT: float = 0.5
BASE_NOISE: float = 0.1
BASELINE_COMPOSITE: float = 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blur() -> BlurScores:
    """Build a baseline BlurScores instance for synthetic Stage1Results."""
    return BlurScores(
        tenengrad=INLIER_BLUR_TENENGRAD, fft_ratio=BASE_BLUR_FFT, blur_tier=1,
    )


def _make_exposure(dr: float) -> ExposureScores:
    """Build an ExposureScores with the given dynamic-range scalar."""
    return ExposureScores(
        dr_score=dr,
        clipping_highlight=0.0,
        clipping_shadow=0.0,
        midtone_pct=0.5,
        color_cast_score=0.0,
    )


def _make_stage1(path: Path, dr: float) -> Stage1Result:
    """Build a Stage1Result with no exif / capture_time (None defaults)."""
    return Stage1Result(
        photo_path=path,
        blur=_make_blur(),
        exposure=_make_exposure(dr),
        noise_score=BASE_NOISE,
    )


def _make_fusion(path: Path) -> FusionResult:
    """Build a FusionResult using the real Stage2Result model."""
    stage2 = Stage2Result(
        photo_path=path,
        topiq=INLIER_TOPIQ,
        laion_aesthetic=INLIER_LAION,
        clipiqa=INLIER_CLIPIQA,
        composite=BASELINE_COMPOSITE,
    )
    return FusionResult(stage2=stage2, routing="AMBIGUOUS")


def _make_corpus(tmp_path: Path) -> tuple[_Stage1Output, _Stage2Output]:
    """Build a 5-photo corpus where index OUTLIER_INDEX has a skewed exposure drift."""
    s1_out = _Stage1Output()
    s2_out = _Stage2Output()
    for i in range(PHOTO_COUNT):
        path = tmp_path / f"photo_{i:02d}.jpg"
        path.touch()
        is_outlier = i == OUTLIER_INDEX
        dr = OUTLIER_DR_SCORE if is_outlier else INLIER_DR_SCORE
        s1_out.results[str(path)] = _make_stage1(path, dr)
        s2_out.results[str(path)] = _make_fusion(path)
    return s1_out, s2_out


def _make_run_ctx(tmp_path: Path) -> _StageRunCtx:
    """Construct a _StageRunCtx with a MagicMock dashboard for sub-bar assertions."""
    return _StageRunCtx(
        config=CullConfig(preset="general"),
        paths=[tmp_path],
        dashboard=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_s2_reducer_patches_composites_in_place(tmp_path: Path) -> None:
    """The reducer must mutate every fusion's composite (different from baseline)."""
    s1_out, s2_out = _make_corpus(tmp_path)
    ctx = _make_run_ctx(tmp_path)
    baseline_composite = {
        key: fusion.stage2.composite for key, fusion in s2_out.results.items()
    }
    output = _run_s2_reducer(_S2ReducerRunInput(s2_out=s2_out, s1_out=s1_out, ctx=ctx))
    assert output.patched_count == PHOTO_COUNT
    outlier_path = str(tmp_path / f"photo_{OUTLIER_INDEX:02d}.jpg")
    assert s2_out.results[outlier_path].stage2.composite != baseline_composite[outlier_path]
    assert s2_out.results[outlier_path].stage2.shoot_stats is not None


def test_run_s2_reducer_emits_dashboard_sub_bar(tmp_path: Path) -> None:
    """Dashboard sub-bar bookends and per-photo updates must fire."""
    s1_out, s2_out = _make_corpus(tmp_path)
    ctx = _make_run_ctx(tmp_path)
    _run_s2_reducer(_S2ReducerRunInput(s2_out=s2_out, s1_out=s1_out, ctx=ctx))
    assert ctx.dashboard.start_stage2_reducer.call_count == 1
    assert ctx.dashboard.start_stage2_reducer.call_args[0][0] == PHOTO_COUNT
    assert ctx.dashboard.update_stage2_reducer.call_count == PHOTO_COUNT
    assert ctx.dashboard.complete_stage2_reducer.call_count == 1


def _patch_pipeline_stages(
    monkeypatch, call_log: list[str]
) -> tuple[_Stage1Output, _Stage2Output]:
    """Replace pipeline stage functions with logging stubs and return shared outputs."""
    s1_stub = _Stage1Output()
    s2_stub = _Stage2Output()
    fake_entry = SimpleNamespace(
        alias="mock", display_name="mock-model", directory=Path("/tmp/mock"),
    )

    monkeypatch.setattr(
        "cull._pipeline.orchestrator._run_s1", lambda _ctx: (call_log.append("s1") or s1_stub)
    )
    monkeypatch.setattr(
        "cull._pipeline.orchestrator._run_s2", lambda _r: (call_log.append("s2") or s2_stub)
    )
    monkeypatch.setattr(
        "cull._pipeline.orchestrator._run_s2_reducer", lambda _r: call_log.append("s2_reducer")
    )
    monkeypatch.setattr(
        "cull._pipeline.stage3_runner._run_s3", lambda _r: (call_log.append("s3") or {})
    )
    monkeypatch.setattr(
        "cull._pipeline.stage3_runner.resolve_alias", lambda _alias: fake_entry
    )
    monkeypatch.setattr("cull._pipeline.orchestrator._unload_stage2_models", lambda: None)
    return s1_stub, s2_stub


def test_execute_stages_inline_invokes_reducer_between_s2_and_s3(
    tmp_path: Path, monkeypatch
) -> None:
    """_run_s2_reducer must be called exactly once between _run_s2 and _run_s3."""
    call_log: list[str] = []
    _patch_pipeline_stages(monkeypatch, call_log)
    ctx = _make_run_ctx(tmp_path)
    _execute_stages_inline(ctx)
    assert call_log.count("s2_reducer") == 1, f"reducer call_log={call_log}"
    s2_idx = call_log.index("s2")
    reducer_idx = call_log.index("s2_reducer")
    s3_idx = call_log.index("s3")
    assert s2_idx < reducer_idx < s3_idx, f"order broken: {call_log}"
