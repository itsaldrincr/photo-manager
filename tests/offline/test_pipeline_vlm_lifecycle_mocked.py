"""End-to-end mocked pipeline VLM lifecycle tests.

Verifies that:
  - Stage 3 and Stage 4 share one VlmSession instance.
  - The session is unloaded exactly once at pipeline end.
  - When VLM is not needed, the session CM is never entered.

No real mlx_vlm or model weights are loaded. All external scoring
entry points are replaced via monkeypatch.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict

from cull._pipeline.stage1_runner import _Stage1Output
from cull._pipeline.stage2_runner import _Stage2Output
from cull.config import CullConfig
from cull._pipeline.orchestrator import SessionSummary
from cull.models import CurationResult, PhotoDecision, PhotoMeta
from cull.pipeline import _PipelineRunInput, run_pipeline
from cull.stage3.prompt import PromptContext
from cull.stage4.vlm_tiebreak import CuratorTiebreakCallInput, CuratorTiebreakInput
from cull.vlm_session import VlmGenerateInput

logger = logging.getLogger(__name__)

GRADIENT_SIZE: int = 32
CORPUS_COUNT: int = 3
COLOR_CHANNEL_RANGE: int = 256
COLOR_SHIFT_HALF: int = 128


# ---------------------------------------------------------------------------
# Recording fake VLM session
# ---------------------------------------------------------------------------


class RecordingFakeVlmSession(BaseModel):
    """Stub VlmSession that records calls and tracks unload."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    entry: Any = None
    model: Any = None
    processor: Any = None
    config: Any = None
    generate_call_count: int = 0
    is_unloaded: bool = False

    def generate(self, call_in: VlmGenerateInput) -> str:
        """Record the call and return stub JSON."""
        self.generate_call_count += 1
        return '{"sharpness":0.9,"exposure":0.9,"composition":0.9,"keeper":true,"confidence":0.9,"flags":[]}'

    def unload(self) -> None:
        """Mark session as unloaded."""
        self.is_unloaded = True


# ---------------------------------------------------------------------------
# Session CM factory and entry tracker
# ---------------------------------------------------------------------------


class _SessionCmState(BaseModel):
    """Mutable state shared between factory and CM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    entry_count: int = 0
    session: RecordingFakeVlmSession | None = None


def _make_fake_vlm_session_cm(
    state: _SessionCmState,
) -> contextlib.AbstractContextManager:
    """Return a CM that yields a RecordingFakeVlmSession and unloads on exit."""

    @contextlib.contextmanager
    def _cm(alias: str) -> Iterator[RecordingFakeVlmSession]:
        state.entry_count += 1
        session = RecordingFakeVlmSession()
        state.session = session
        try:
            yield session
        finally:
            session.unload()

    return _cm


# ---------------------------------------------------------------------------
# Stage 3 / Stage 4 recording stubs
# ---------------------------------------------------------------------------


class _ScoreRecorder(BaseModel):
    """Records the session instance passed to score_photo."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    session: Any = None

    def record(self, call_in: Any) -> Any:
        """Capture the session and return a minimal Stage3Result."""
        self.session = call_in.session
        return _make_stage3_result(call_in.request.image_path)


class _TiebreakRecorder(BaseModel):
    """Records the session instance passed to compare_photos."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    session: Any = None

    def record(self, call_in: Any) -> Any:
        """Capture the session and return a minimal tiebreak result."""
        self.session = call_in.session
        return _make_tiebreak_result(call_in.tiebreak_input)


def _make_stage3_result(image_path: Path) -> Any:
    """Build a minimal Stage3Result for the given path."""
    from cull.models import Stage3Result  # noqa: PLC0415

    return Stage3Result(photo_path=image_path, is_parse_error=False)


def _make_tiebreak_result(tiebreak_input: Any) -> Any:
    """Build a minimal CuratorTiebreakResult picking photo_a."""
    from cull.stage4.vlm_tiebreak import CuratorTiebreakResult  # noqa: PLC0415

    return CuratorTiebreakResult(
        winner=tiebreak_input.photo_a,
        reason="stub",
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Corpus and config helpers
# ---------------------------------------------------------------------------


def _write_gradient_jpeg(path: Path, index: int) -> None:
    """Write a small gradient JPEG to path, varied by index."""
    img = Image.new("RGB", (GRADIENT_SIZE, GRADIENT_SIZE))
    pixels = img.load()
    for row in range(GRADIENT_SIZE):
        for col in range(GRADIENT_SIZE):
            value = (row * index + col) % COLOR_CHANNEL_RANGE
            pixels[col, row] = (
                value,
                (value + COLOR_SHIFT_HALF) % COLOR_CHANNEL_RANGE,
                (value + COLOR_SHIFT_HALF) % COLOR_CHANNEL_RANGE,
            )
    img.save(path, "JPEG")


def _build_corpus(tmp_path: Path) -> None:
    """Write CORPUS_COUNT gradient JPEG files into tmp_path."""
    for idx in range(CORPUS_COUNT):
        _write_gradient_jpeg(tmp_path / f"photo_{idx:03d}.jpg", idx + 1)


def _make_run_input(config: CullConfig, source_path: Path) -> _PipelineRunInput:
    """Build a _PipelineRunInput from config and source path."""
    return _PipelineRunInput(config=config, source_path=source_path)


# ---------------------------------------------------------------------------
# Common monkeypatching helper
# ---------------------------------------------------------------------------


def _patch_pipeline_infra(
    monkeypatch: pytest.MonkeyPatch,
    cm_factory: Any,
) -> None:
    """Patch the vlm_session CM and dashboard for pipeline tests."""
    monkeypatch.setattr("cull.vlm_session.vlm_session", cm_factory)
    monkeypatch.setattr("cull._pipeline.orchestrator.vlm_session", cm_factory)
    monkeypatch.setattr(
        "cull._pipeline.orchestrator._make_dashboard",
        _make_stub_dashboard,
    )
    monkeypatch.setattr(
        "cull._pipeline.orchestrator._run_s1",
        lambda ctx: _Stage1Output(results={}, survivors=ctx.paths, encodings={}),
    )
    monkeypatch.setattr(
        "cull._pipeline.orchestrator._run_s2",
        lambda run_in: _Stage2Output(
            results={},
            portraits={},
            ambiguous=list(run_in.s1_out.survivors),
            keepers=[],
            rejects=[],
            search_cache=None,
        ),
    )
    monkeypatch.setattr(
        "cull._pipeline.orchestrator._run_s2_reducer",
        lambda _run_in: None,
    )
    monkeypatch.setattr(
        "cull._pipeline.orchestrator._unload_stage2_models",
        lambda: None,
    )
    monkeypatch.setattr(
        "cull._pipeline.orchestrator._run_s3_if_configured",
        lambda run_in: {
            str(path): __import__("cull.stage3.vlm_scoring", fromlist=["score_photo"]).score_photo(
                SimpleNamespace(
                    request=SimpleNamespace(image_path=path),
                    session=run_in.ctx.vlm_session,
                )
            )
            for path in run_in.ctx.paths
        } if 3 in run_in.ctx.config.stages else {},
    )
    monkeypatch.setattr(
        "cull._pipeline.orchestrator._build_all_decisions",
        lambda _ctx: [
            PhotoDecision(
                photo=PhotoMeta(path=path, filename=path.name),
                decision="keeper",
            )
            for path in _ctx.paths
        ],
    )
    monkeypatch.setattr(
        "cull._pipeline.orchestrator._build_summary",
        lambda _decisions: SessionSummary(
            keepers=len(_decisions), rejected=0, duplicates=0, uncertain=0, selected=0
        ),
    )
    monkeypatch.setattr(
        "cull._pipeline.orchestrator._run_s4",
        lambda s4_in: (
            __import__("cull.stage4.vlm_tiebreak", fromlist=["compare_photos"]).compare_photos(
                CuratorTiebreakCallInput(
                    tiebreak_input=CuratorTiebreakInput(
                        photo_a=s4_in.ctx.paths[0],
                        photo_b=s4_in.ctx.paths[1],
                        context=PromptContext(),
                        context_b=PromptContext(),
                        model="mock-model",
                    ),
                    session=s4_in.ctx.vlm_session,
                )
            ),
            CurationResult(
                is_enabled=True,
                target_count=2,
                actual_count=0,
                cluster_count=0,
                vlm_tiebreakers=1,
                threshold_used=0.0,
                elapsed_seconds=0.0,
                selected=[],
            ),
        )[1] if s4_in.ctx.config.curate_target is not None else None,
    )


def _make_stub_dashboard(run_in: Any) -> Any:
    """Return a no-op dashboard context manager stub."""
    return _StubDashboard()


class _StubDashboard:
    """No-op test stub satisfying the Dashboard protocol. All methods are pass-through; no behaviour. Exempt from 3-method rule (protocol stub / data-class exception)."""

    def __enter__(self) -> "_StubDashboard":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def __getattr__(self, _name: str) -> Any:
        return lambda *args, **kwargs: None

    def set_photo_count(self, count: int) -> None:
        pass

    def begin_scan(self) -> None:
        pass

    def update_scan_progress(self, count: int, total_bytes: int) -> None:
        pass

    def end_scan(self) -> None:
        pass

    def show_results(self, result: Any) -> None:
        pass

    def update_stage(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update_progress(self, *args: Any, **kwargs: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stages_3_and_4_share_session(
    tmp_path: Path,
    mock_scorers: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 3 and Stage 4 receive the same VlmSession instance."""
    _build_corpus(tmp_path)
    state = _SessionCmState()
    cm_factory = _make_fake_vlm_session_cm(state)
    _patch_pipeline_infra(monkeypatch, cm_factory)

    score_recorder = _ScoreRecorder()
    tiebreak_recorder = _TiebreakRecorder()
    monkeypatch.setattr(
        "cull.stage3.vlm_scoring.score_photo",
        score_recorder.record,
    )
    monkeypatch.setattr(
        "cull.stage4.vlm_tiebreak.compare_photos",
        tiebreak_recorder.record,
    )

    config = CullConfig(stages=[1, 2, 3], curate_target=2, is_portrait=False)
    run_pipeline(_make_run_input(config, tmp_path))

    assert score_recorder.session is not None
    assert tiebreak_recorder.session is not None
    assert score_recorder.session is tiebreak_recorder.session
    assert state.session is not None
    assert state.session.is_unloaded


def test_no_vlm_no_curate_skips_session(
    tmp_path: Path,
    mock_scorers: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session CM is never entered when stages=[1,2] and curate_target=None."""
    _build_corpus(tmp_path)
    state = _SessionCmState()
    cm_factory = _make_fake_vlm_session_cm(state)
    _patch_pipeline_infra(monkeypatch, cm_factory)

    config = CullConfig(stages=[1, 2], curate_target=None, is_portrait=False)
    run_pipeline(_make_run_input(config, tmp_path))

    assert state.entry_count == 0


def test_stage3_only_releases_at_end_of_pipeline(
    tmp_path: Path,
    mock_scorers: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session is entered exactly once and released at pipeline end."""
    _build_corpus(tmp_path)
    state = _SessionCmState()
    cm_factory = _make_fake_vlm_session_cm(state)
    _patch_pipeline_infra(monkeypatch, cm_factory)

    score_recorder = _ScoreRecorder()
    monkeypatch.setattr(
        "cull.stage3.vlm_scoring.score_photo",
        score_recorder.record,
    )

    config = CullConfig(stages=[1, 2, 3], curate_target=None, is_portrait=False)
    run_pipeline(_make_run_input(config, tmp_path))

    assert state.entry_count == 1
    assert state.session is not None
    assert state.session.is_unloaded
