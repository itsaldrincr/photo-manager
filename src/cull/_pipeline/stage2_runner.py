"""Stage 2 runner — batched CLIP forward, fusion, routing, portrait assessment."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field

from cull.config import (
    CLIP_MODEL_ID,
    CLIP_PATCH_GRID,
    CullConfig,
    EMBEDDING_CACHE_FILENAME,
    EMBEDDING_INDEX_FILENAME,
    IQA_EXPOSURE_DEFAULT,
    STAGE2_BATCH_SIZE,
)
from cull.dashboard import (
    Dashboard,
    _Stage2UpdateInput,
)
from cull import clip_loader
from cull.models import Stage1Result
from cull.saliency import (
    SaliencyFromTokensRequest,
    SaliencyResult,
    compute_saliency_from_tokens,
)
from cull.stage2.aesthetic import (
    _get_head,
    score_from_embeddings,
    warmup_predictor,
)
from cull.stage2.composition import (
    warmup_topiq_iaa,
)
from cull.stage2.fusion import (
    FusionResult,
    IqaScores,
    compute_composite,
)
from cull.stage2.iqa import (
    score_clipiqa_batch,
    score_topiq_batch,
    select_device,
    warmup_metrics,
)
from cull.stage2.portrait import (
    PortraitResult,
    _get_face_landmarker,
    assess_portrait,
)

from cull._pipeline.stage1_runner import _Stage1Output
if TYPE_CHECKING:
    from cull.pipeline import _StageRunCtx
from cull._pipeline.stage2_scoring import (
    _CompositionApplyInput,
    _SubjectBlurApplyInput,
    _apply_composition_to_scores,
    _apply_exposure_to_scores,
    _apply_geometry_to_scores,
    _apply_stage1_blur_context_to_scores,
    _apply_subject_blur_to_scores,
    _apply_taste_to_scores,
    _DualLoadInput,
    _DualPilBatch,
    _load_dual_pil_batch,
    _load_search_cache,
    _load_tensor_batch,
    _load_tensor_only_batch,
    _SearchCache,
    _SubjectBlurCtx,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 2 data bundles
# ---------------------------------------------------------------------------


class _Stage2Output(BaseModel):
    """Aggregated Stage 2 results for routing."""

    model_config = {"arbitrary_types_allowed": True}

    results: dict[str, FusionResult] = Field(default_factory=dict)
    portraits: dict[str, PortraitResult] = Field(default_factory=dict)
    ambiguous: list[Path] = Field(default_factory=list)
    keepers: list[Path] = Field(default_factory=list)
    rejects: list[Path] = Field(default_factory=list)
    search_cache: _SearchCache | None = None


class _Stage2ScoreInput(BaseModel):
    """Input bundle for scoring one photo in Stage 2."""

    model_config = {"arbitrary_types_allowed": True}

    path: Path
    tensor: Any  # torch.Tensor — Any for Pydantic compat
    config: CullConfig
    device: str
    s1_result: Stage1Result | None = None


class _Stage2LoopInput(BaseModel):
    """Input bundle for Stage 2 loop."""

    survivors: list[Path]
    config: CullConfig
    s1_results: dict[str, Stage1Result] = Field(default_factory=dict)
    source_path: Path | None = None


class _Stage2BatchInput(BaseModel):
    """Bundle for one batch of Stage 2 scoring inputs."""

    model_config = {"arbitrary_types_allowed": True}

    tensor_batch: torch.Tensor
    pil_images: list[Image.Image] | None
    embeddings: torch.Tensor | None
    photo_paths: list[Path]


class _ChunkInput(BaseModel):
    """A chunk of paths plus the device to score on."""

    model_config = {"arbitrary_types_allowed": True}

    paths: list[Path]
    device: str


class _EmitInput(BaseModel):
    """Bundle for emitting one batch of results to output and dashboard."""

    model_config = {"arbitrary_types_allowed": True}

    pairs: list[tuple[Path, FusionResult]]
    loop_in: _Stage2LoopInput
    output: _Stage2Output
    dashboard: Any
    device: str


def _build_batch_input(
    chunk_in: _ChunkInput, embeddings: torch.Tensor | None
) -> _Stage2BatchInput:
    """Load tensors (and PIL on miss) and assemble a Stage 2 batch input bundle."""
    if embeddings is not None:
        tensor_batch = _load_tensor_only_batch(chunk_in.paths)
        return _Stage2BatchInput(
            tensor_batch=tensor_batch, pil_images=None,
            embeddings=embeddings, photo_paths=chunk_in.paths,
        )
    tensor_batch, pil_images = _load_tensor_batch(chunk_in.paths)
    return _Stage2BatchInput(
        tensor_batch=tensor_batch, pil_images=pil_images,
        embeddings=None, photo_paths=chunk_in.paths,
    )


class _BatchCtx(BaseModel):
    """Context carried into _process_batch alongside the chunk."""

    model_config = {"arbitrary_types_allowed": True}

    loop_in: _Stage2LoopInput
    cache: _SearchCache | None = None
    saliency_cache: dict[str, SaliencyResult | None] = Field(
        default_factory=dict
    )
    dual_pil: _DualPilBatch | None = None
    aesthetic_score_cache: dict[str, float] = Field(default_factory=dict)
    embedding_rows: list[Any] = Field(default_factory=list)


class _BatchOutcome(BaseModel):
    """Outputs of one Stage 2 batch: fusion pairs and a portrait cache."""

    model_config = {"arbitrary_types_allowed": True}

    pairs: list[tuple[Path, FusionResult]]
    portraits: dict[str, PortraitResult]


class _SharedClipOutput(BaseModel):
    """Output of one shared CLIP vision forward pass on a pil_224 batch."""

    model_config = {"arbitrary_types_allowed": True}

    patch_tokens_batch: torch.Tensor
    image_embeds: torch.Tensor


def _run_shared_clip_forward(
    pil_224: list[Image.Image], device: str
) -> _SharedClipOutput:
    """Run ONE CLIP vision forward; return patch tokens and pooled image embeds."""
    clip_model = clip_loader.get_clip_model()
    processor = clip_loader.get_clip_processor()
    inputs = processor(images=pil_224, return_tensors="pt").to(device)  # type: ignore[operator]
    with torch.no_grad():
        vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])  # type: ignore[attr-defined]
        pooled = vision_out.pooler_output
        image_embeds = clip_model.visual_projection(pooled)  # type: ignore[attr-defined]
    patch_tokens_batch = vision_out.last_hidden_state[:, 1:, :]
    return _SharedClipOutput(
        patch_tokens_batch=patch_tokens_batch, image_embeds=image_embeds
    )


def _populate_saliency_cache(
    patch_tokens_batch: torch.Tensor, batch_ctx: _BatchCtx
) -> None:
    """Fill saliency_cache from per-image patch tokens; one entry per path."""
    paths = batch_ctx.dual_pil.paths  # type: ignore[union-attr]
    for idx, path in enumerate(paths):
        tokens_request = SaliencyFromTokensRequest(
            patch_tokens=patch_tokens_batch[idx], grid_size=CLIP_PATCH_GRID
        )
        batch_ctx.saliency_cache[str(path)] = compute_saliency_from_tokens(tokens_request)


class _AestheticApplyInput(BaseModel):
    """Bundle of iqa_list + CLIP image embeddings + device for aesthetic application."""

    model_config = {"arbitrary_types_allowed": True}

    iqa_list: list[IqaScores]
    image_embeds: torch.Tensor
    device: str


def _normalize_embeddings(image_embeds: torch.Tensor) -> torch.Tensor:
    """L2-normalize CLIP embeddings before aesthetic scoring."""
    return image_embeds / image_embeds.norm(dim=-1, keepdim=True)


def _apply_precomputed_aesthetic(apply_in: _AestheticApplyInput) -> None:
    """Swap aesthetic scores in-place using pre-computed CLIP embeddings."""
    head = _get_head(apply_in.device)
    embeddings = _normalize_embeddings(apply_in.image_embeds)
    scores = score_from_embeddings(head, embeddings)
    for iqa, score in zip(apply_in.iqa_list, scores):
        iqa.laion_aesthetic = score


class _SharedBuildInput(BaseModel):
    """Bundle of chunk + batch_ctx + shared CLIP output for iqa assembly."""

    model_config = {"arbitrary_types_allowed": True}

    chunk_in: _ChunkInput
    batch_ctx: _BatchCtx
    shared: _SharedClipOutput


def _build_iqa_pyiqa_only(
    batch_input: _Stage2BatchInput, device: str
) -> list[IqaScores]:
    """Score topiq + clipiqa on tensor_1280; skip aesthetic (filled later)."""
    topiq_scores = score_topiq_batch(batch_input.tensor_batch, device)
    clipiqa_scores = score_clipiqa_batch(batch_input.tensor_batch, device)
    return [
        IqaScores(
            photo_path=p, topiq=t, laion_aesthetic=0.0,
            clipiqa=c, exposure=IQA_EXPOSURE_DEFAULT,
        )
        for p, t, c in zip(batch_input.photo_paths, topiq_scores, clipiqa_scores)
    ]


def _build_iqa_with_shared_embeds(
    build_in: _SharedBuildInput,
) -> list[IqaScores]:
    """Build iqa list from tensor_1280 + pre-computed aesthetic embeddings."""
    dual = build_in.batch_ctx.dual_pil  # type: ignore[union-attr]
    batch_input = _Stage2BatchInput(
        tensor_batch=dual.tensor_1280, pil_images=None,
        embeddings=None, photo_paths=build_in.chunk_in.paths,
    )
    iqa_list = _build_iqa_pyiqa_only(batch_input, build_in.chunk_in.device)
    _apply_precomputed_aesthetic(_AestheticApplyInput(
        iqa_list=iqa_list, image_embeds=build_in.shared.image_embeds,
        device=build_in.chunk_in.device,
    ))
    return iqa_list


def _compute_fusion_pairs(
    iqa_list: list[IqaScores], batch_ctx: _BatchCtx
) -> list[tuple[Path, FusionResult]]:
    """Compute (path, FusionResult) pairs for one batch."""
    config = batch_ctx.loop_in.config
    paths = batch_ctx.dual_pil.paths  # type: ignore[union-attr]
    return [(p, compute_composite(iqa, config)) for p, iqa in zip(paths, iqa_list)]


class _EmbedCollectInput(BaseModel):
    """Bundle for accumulating per-chunk CLIP embeddings into batch_ctx."""

    model_config = {"arbitrary_types_allowed": True}

    paths: list[Path]
    image_embeds: Any  # torch.Tensor — L2-normalised before append


def _accumulate_embedding_rows(
    collect_in: _EmbedCollectInput, batch_ctx: _BatchCtx
) -> None:
    """Append (path, L2-normalised embedding) tuples for each chunk member."""
    embeds = collect_in.image_embeds
    normed = embeds / embeds.norm(dim=-1, keepdim=True)
    rows = normed.cpu().numpy().astype(np.float32)
    for path, row in zip(collect_in.paths, rows):
        batch_ctx.embedding_rows.append((path, row))


def _score_one_chunk(
    chunk_in: _ChunkInput, batch_ctx: _BatchCtx
) -> list[IqaScores]:
    """Run shared CLIP forward + IQA assembly for one chunk; returns iqa list."""
    batch_ctx.dual_pil = _load_dual_pil_batch(
        _DualLoadInput(paths=chunk_in.paths, device=chunk_in.device)
    )
    shared = _run_shared_clip_forward(batch_ctx.dual_pil.pil_224, chunk_in.device)
    _populate_saliency_cache(shared.patch_tokens_batch, batch_ctx)
    _accumulate_embedding_rows(
        _EmbedCollectInput(paths=chunk_in.paths, image_embeds=shared.image_embeds),
        batch_ctx,
    )
    return _build_iqa_with_shared_embeds(_SharedBuildInput(
        chunk_in=chunk_in, batch_ctx=batch_ctx, shared=shared,
    ))


def _process_batch(
    chunk_in: _ChunkInput, batch_ctx: _BatchCtx
) -> _BatchOutcome:
    """Score one chunk of paths via shared CLIP forward; returns fusion pairs."""
    iqa_list = _score_one_chunk(chunk_in, batch_ctx)
    _apply_exposure_to_scores(iqa_list, batch_ctx.loop_in.s1_results)
    _apply_stage1_blur_context_to_scores(iqa_list, batch_ctx.loop_in.s1_results)
    _apply_geometry_to_scores(iqa_list, batch_ctx.loop_in.s1_results)
    _apply_composition_to_scores(
        _CompositionApplyInput(iqa_list=iqa_list, paths=chunk_in.paths, ctx=batch_ctx)
    )
    _apply_taste_to_scores(iqa_list, chunk_in.paths)
    sb_ctx = _SubjectBlurCtx(paths=chunk_in.paths, config=batch_ctx.loop_in.config)
    portraits = _apply_subject_blur_to_scores(
        _SubjectBlurApplyInput(iqa_list=iqa_list, ctx=sb_ctx, batch_ctx=batch_ctx)
    )
    pairs = _compute_fusion_pairs(iqa_list, batch_ctx)
    return _BatchOutcome(pairs=pairs, portraits=portraits)


def _emit_batch_results(emit_in: _EmitInput) -> None:
    """Route, portrait-assess, and update dashboard for each result in a batch."""
    for path, fusion in emit_in.pairs:
        emit_in.output.results[str(path)] = fusion
        _classify_s2_routing(fusion, emit_in.output)
        score_in = _Stage2ScoreInput(
            path=path, tensor=None, config=emit_in.loop_in.config,
            device=emit_in.device, s1_result=emit_in.loop_in.s1_results.get(str(path)),
        )
        _run_portrait_if_needed(score_in, emit_in.output)
        emit_in.dashboard.update_stage2(
            _Stage2UpdateInput(path=path, fusion=fusion, routing=fusion.routing)
        )


def _resolve_search_cache(loop_in: _Stage2LoopInput) -> _SearchCache | None:
    """Load the search cache for this loop's source dir, or return None."""
    if loop_in.source_path is None:
        return None
    return _load_search_cache(loop_in.source_path)


def _build_search_cache_from_rows(
    rows: list[tuple[Path, np.ndarray]],
) -> _SearchCache:
    """Stack accumulated (path, embedding) rows into a _SearchCache object."""
    paths = [p for p, _ in rows]
    matrix = np.stack([r for _, r in rows], axis=0).astype(np.float32)
    path_to_row = {str(p): i for i, p in enumerate(paths)}
    return _SearchCache(embeddings=matrix, path_to_row=path_to_row)


def _persist_search_cache(source_path: Path, cache: _SearchCache) -> None:
    """Write cache to .cull_embeddings.npy + index JSON atomically via rename."""
    emb_final = source_path / EMBEDDING_CACHE_FILENAME
    idx_final = source_path / EMBEDDING_INDEX_FILENAME
    emb_tmp = source_path / f"{EMBEDDING_CACHE_FILENAME}.tmp"
    idx_tmp = source_path / f"{EMBEDDING_INDEX_FILENAME}.tmp"
    with emb_tmp.open("wb") as handle:
        np.save(handle, cache.embeddings, allow_pickle=False)
    index_payload = {
        "model_id": CLIP_MODEL_ID,
        "paths": list(cache.path_to_row.keys()),
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    idx_tmp.write_text(json.dumps(index_payload))
    emb_tmp.replace(emb_final)
    idx_tmp.replace(idx_final)


def _maybe_write_search_cache(
    batch_ctx: _BatchCtx, output: _Stage2Output
) -> None:
    """Build + persist the search cache if none exists and we have embedding rows."""
    if output.search_cache is not None:
        return
    source_path = batch_ctx.loop_in.source_path
    if source_path is None or not batch_ctx.embedding_rows:
        return
    cache = _build_search_cache_from_rows(batch_ctx.embedding_rows)
    output.search_cache = cache
    try:
        _persist_search_cache(source_path, cache)
    except OSError as exc:
        logger.warning("Failed to persist search cache to %s: %s", source_path, exc)


def _prewarm_stage2_models(loop_in: _Stage2LoopInput, dashboard: Dashboard) -> None:
    """Eagerly load all Stage 2 models so the per-photo loop sees no download stalls."""
    device = select_device()
    dashboard.start_stage2_loading()
    try:
        warmup_metrics(device)
        warmup_topiq_iaa(device)
        warmup_predictor(device)
        if loop_in.config.is_portrait:
            _get_face_landmarker()
    finally:
        dashboard.clear_stage2_loading()


def _run_stage2_loop(loop_in: _Stage2LoopInput, dashboard: Dashboard) -> _Stage2Output:
    """Run Stage 2 IQA scoring on all survivors with dashboard."""
    _prewarm_stage2_models(loop_in, dashboard)
    device = select_device()
    output = _Stage2Output()
    dashboard.start_stage2(len(loop_in.survivors))
    search_cache = _resolve_search_cache(loop_in)
    output.search_cache = search_cache
    batch_ctx = _BatchCtx(loop_in=loop_in, cache=search_cache)
    for i in range(0, len(loop_in.survivors), STAGE2_BATCH_SIZE):
        chunk = loop_in.survivors[i: i + STAGE2_BATCH_SIZE]
        outcome = _process_batch(_ChunkInput(paths=chunk, device=device), batch_ctx)
        output.portraits.update(outcome.portraits)
        emit_in = _EmitInput(
            pairs=outcome.pairs, loop_in=loop_in, output=output,
            dashboard=dashboard, device=device,
        )
        _emit_batch_results(emit_in)
    _maybe_write_search_cache(batch_ctx, output)
    return output


def _classify_s2_routing(fusion: FusionResult, output: _Stage2Output) -> None:
    """Classify a photo by its fusion routing label into output lists."""
    path = fusion.stage2.photo_path
    if fusion.routing == "KEEPER":
        output.keepers.append(path)
    elif fusion.routing == "AMBIGUOUS":
        output.ambiguous.append(path)
    else:
        output.rejects.append(path)


def _run_portrait_if_needed(score_in: _Stage2ScoreInput, output: _Stage2Output) -> None:
    """Run portrait assessment if portrait mode is enabled and not already cached."""
    if not score_in.config.is_portrait:
        return
    key = str(score_in.path)
    if key in output.portraits:
        return
    output.portraits[key] = assess_portrait(score_in.path, score_in.config)


# ---------------------------------------------------------------------------
# Stage 2 run wrapper
# ---------------------------------------------------------------------------


class _S2RunInput(BaseModel):
    """Input bundle for Stage 2 execution."""

    s1_out: _Stage1Output
    ctx: "_StageRunCtx"


def _run_s2(run_in: _S2RunInput) -> _Stage2Output:
    """Execute Stage 2 and record timing."""
    t0 = time.monotonic()
    loop_in = _Stage2LoopInput(
        survivors=run_in.s1_out.survivors,
        config=run_in.ctx.config,
        s1_results=run_in.s1_out.results,
        source_path=run_in.ctx.source_path,
    )
    s2_out = _run_stage2_loop(loop_in, run_in.ctx.dashboard)
    elapsed = time.monotonic() - t0
    run_in.ctx.timings.stage2 = elapsed
    run_in.ctx.dashboard.complete_stage2(elapsed)
    return s2_out
