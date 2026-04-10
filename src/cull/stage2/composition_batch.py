"""Composition batch — ThreadPoolExecutor dispatch for smartcrop crop calls."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from cull.models import CompositionScore, CropProposal

if TYPE_CHECKING:
    from cull.stage2.composition import CompositionInput

SMARTCROP_THREAD_WORKERS: int = 4

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
    item: CompositionInput, executor: ThreadPoolExecutor
) -> tuple[CompositionScore, Future[CropProposal | None] | None]:
    """Score one image synchronously; dispatch its crop call to the executor."""
    from cull.stage2 import composition  # noqa: PLC0415 — lazy monkeypatch seam

    score = composition._score_image(item.pil_1280, item.saliency_result)
    if item.skip_crop:
        return score, None
    crop_future = executor.submit(
        composition._build_crop, item.pil_1280, item.saliency_result
    )
    return score, crop_future
