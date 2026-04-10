"""Composition scoring — slim public hub over geometry/topiq/crop/batch siblings."""

# Torch/smartcrop/pyiqa weights are pinned via TORCH_HOME, set in
# cull.env_bootstrap. No loader kwargs are needed here — pyiqa resolves
# from $TORCH_HOME/hub/pyiqa; smartcrop reads from the same torch hub dir.

from __future__ import annotations

from concurrent.futures import Future

from PIL import Image
from pydantic import BaseModel, ConfigDict

from cull.models import CompositionScore, CropProposal
from cull.saliency import SaliencyResult
from cull.stage2.composition_batch import (
    _SMARTCROP_EXECUTOR,
    _get_smartcrop_executor,
    _resolve_future,
    _score_and_dispatch,
)
from cull.stage2.composition_crop import (
    SMARTCROP_DOWNSAMPLE_LONG_EDGE,
    _UpscaleParams,
    _build_crop,
    _downsample_for_smartcrop,
    _saliency_thirds_crop,
    _try_smartcrop,
    _upscale_box,
)
from cull.stage2.composition_geometry import (
    _GeometryMetrics,
    _balance_ratio,
    _composite_score,
    _compute_geometry_metrics,
    _edge_clearance,
    _negative_space_balance,
    _thirds_alignment,
)
from cull.stage2.composition_topiq import (
    _TOPIQ_IAA_METRIC,
    _get_topiq_iaa_metric,
    _score_topiq_iaa,
    unload_topiq_iaa,
    warmup_topiq_iaa,
)

# `__all__` re-exports sibling-module symbols so existing test_composition.py
# monkeypatches that target `composition._symbol` continue to resolve. The
# four locally-defined symbols (CompositionInput, score_one, _score_image, score_batch)
# are joined by 23 re-exports from composition_{geometry,topiq,crop,batch}.
__all__ = [
    "CompositionInput",
    "score_one",
    "score_batch",
    "_score_image",
    "_build_crop",
    "_try_smartcrop",
    "_get_topiq_iaa_metric",
    "_TOPIQ_IAA_METRIC",
    "_score_topiq_iaa",
    "warmup_topiq_iaa",
    "unload_topiq_iaa",
    "_GeometryMetrics",
    "_compute_geometry_metrics",
    "_thirds_alignment",
    "_edge_clearance",
    "_negative_space_balance",
    "_balance_ratio",
    "_composite_score",
    "_downsample_for_smartcrop",
    "_upscale_box",
    "_UpscaleParams",
    "_saliency_thirds_crop",
    "_SMARTCROP_EXECUTOR",
    "_get_smartcrop_executor",
    "_resolve_future",
    "_score_and_dispatch",
    "SMARTCROP_DOWNSAMPLE_LONG_EDGE",
]


class CompositionInput(BaseModel):
    """Bundle of pre-loaded PIL and pre-computed saliency for composition scoring."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pil_1280: Image.Image
    saliency_result: SaliencyResult
    skip_crop: bool = False


def score_one(score_input: CompositionInput) -> tuple[CompositionScore, CropProposal | None]:
    """Compute composition score and crop proposal for one image."""
    score = _score_image(score_input.pil_1280, score_input.saliency_result)
    if score_input.skip_crop:
        return score, None
    crop = _build_crop(score_input.pil_1280, score_input.saliency_result)
    return score, crop


def _score_image(
    image: Image.Image, saliency: SaliencyResult
) -> CompositionScore:
    """Run geometry + topiq scoring on an open image; return CompositionScore."""
    width, height = image.size
    metrics = _compute_geometry_metrics(saliency, (width, height))
    topiq_iaa = _score_topiq_iaa(image)
    composite = _composite_score(metrics, topiq_iaa)
    return CompositionScore(
        thirds_alignment=metrics.thirds_alignment,
        edge_clearance=metrics.edge_clearance,
        negative_space_balance=metrics.negative_space_balance,
        topiq_iaa=topiq_iaa,
        composite=composite,
    )


def score_batch(
    inputs: list[CompositionInput],
) -> list[tuple[CompositionScore, CropProposal | None]]:
    """Compute composition scores and crop proposals for a batch of images.

    Smartcrop calls are dispatched to a ThreadPoolExecutor so they overlap any
    GPU-bound work that runs in parallel. Photos with skip_crop=True bypass
    the crop step entirely. All futures are awaited before returning.
    """
    executor = _get_smartcrop_executor()
    pending: list[tuple[CompositionScore, Future[CropProposal | None] | None]] = []
    for item in inputs:
        score, crop_future = _score_and_dispatch(item, executor)
        pending.append((score, crop_future))
    return [(score, _resolve_future(fut)) for score, fut in pending]
