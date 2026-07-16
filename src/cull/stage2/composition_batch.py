"""Composition batch — ThreadPoolExecutor dispatch for smartcrop crop calls."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, ConfigDict

from cull.models import CompositionScore, CropProposal

SMARTCROP_THREAD_WORKERS: int = 4


class _ScoreDispatchInput(BaseModel):
    """Bundle of one composition item plus its precomputed topiq_iaa score.

    item is typed Any (rather than CompositionInput) to avoid a circular
    import — composition.py imports from this module.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: Any
    topiq_iaa: float


_SMARTCROP_EXECUTOR: ThreadPoolExecutor | None = None


def _get_smartcrop_executor() -> ThreadPoolExecutor:
    """Return the module-level ThreadPoolExecutor for smartcrop dispatch."""
    global _SMARTCROP_EXECUTOR
    if _SMARTCROP_EXECUTOR is None:
        _SMARTCROP_EXECUTOR = ThreadPoolExecutor(max_workers=SMARTCROP_THREAD_WORKERS)
    return _SMARTCROP_EXECUTOR


def _resolve_future(
    fut: Future[CropProposal | None] | None,
) -> CropProposal | None:
    """Block on a crop future, returning its result or None."""
    if fut is None:
        return None
    return fut.result()


def _score_and_dispatch(
    dispatch_in: _ScoreDispatchInput, executor: ThreadPoolExecutor
) -> tuple[CompositionScore, Future[CropProposal | None] | None]:
    """Score one image using its precomputed topiq_iaa; dispatch its crop call."""
    from cull.stage2 import composition  # noqa: PLC0415 — lazy monkeypatch seam

    item = dispatch_in.item
    score = composition._score_image_with_topiq(item, dispatch_in.topiq_iaa)
    if item.skip_crop:
        return score, None
    crop_future = executor.submit(
        composition._build_crop, item.pil_1280, item.saliency_result
    )
    return score, crop_future
