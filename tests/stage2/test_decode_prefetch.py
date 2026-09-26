"""Stage 2 decodes ahead on a thread pool without changing what each step sees."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from cull._pipeline import stage2_runner, stage2_scoring
from cull._pipeline.stage2_runner import (
    _BatchCtx,
    _DualPrefetch,
    _EmitInput,
    _score_chunks,
    _Stage2LoopInput,
    _Stage2Output,
)
from cull._pipeline.stage2_scoring import _DualPilBatch
from cull.config import CullConfig
from cull.stage2.portrait import PortraitResult

CHUNK_COUNT: int = 3


def _chunks() -> list[list[Path]]:
    """Return CHUNK_COUNT two-photo chunks of fake paths."""
    return [[Path(f"/fake/{c}_{i}.jpg") for i in range(2)] for c in range(CHUNK_COUNT)]


def test_each_chunk_is_scored_in_order_with_its_own_prefetched_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every chunk is decoded once, and _process_batch gets that chunk's decode."""
    loaded: list[list[Path]] = []
    scored: list[tuple[list[Path], list[Path]]] = []

    def fake_load(load_in):
        loaded.append(load_in.paths)
        return _DualPilBatch(pil_224=[], pil_1280=[], tensor_1280=[], paths=list(load_in.paths))

    def fake_process(chunk_in, batch_ctx):
        scored.append((chunk_in.paths, chunk_in.dual_pil.paths))
        return MagicMock(pairs=[], portraits={})

    monkeypatch.setattr(stage2_runner, "_load_dual_pil_batch", fake_load)
    monkeypatch.setattr(stage2_runner, "_process_batch", fake_process)
    monkeypatch.setattr(stage2_runner, "_emit_batch_results", lambda emit_in: None)
    loop_in = _Stage2LoopInput(survivors=[], config=CullConfig())
    emit_in = _EmitInput(pairs=[], loop_in=loop_in, output=_Stage2Output(), dashboard=None, device="cpu")
    with ThreadPoolExecutor(max_workers=2) as pool:
        _score_chunks(_DualPrefetch(chunks=_chunks(), pool=pool, device="cpu"), emit_in)

    assert sorted(map(tuple, loaded)) == sorted(map(tuple, _chunks()))
    assert scored == [(chunk, chunk) for chunk in _chunks()]


def test_portrait_uses_the_prefetched_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prefetched decode is consumed instead of decoding the photo again."""
    path = Path("/fake/a.jpg")
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    decode: Future = Future()
    decode.set_result(frame)
    seen: list[np.ndarray] = []

    def fail_decode(p: Path) -> None:
        raise AssertionError("decoded again")

    def fake_assess(image: np.ndarray, config: CullConfig) -> PortraitResult:
        seen.append(image)
        return PortraitResult(face_count=0)

    monkeypatch.setattr(stage2_scoring, "_decode_full_res_bgr", fail_decode)
    monkeypatch.setattr(stage2_scoring, "assess_portrait_from_array", fake_assess)
    ctx = _BatchCtx(loop_in=_Stage2LoopInput(survivors=[path], config=CullConfig()))
    ctx.full_res_bgr[str(path)] = decode

    stage2_scoring._portrait_for(path, ctx)

    assert seen == [frame]
    assert str(path) not in ctx.full_res_bgr
