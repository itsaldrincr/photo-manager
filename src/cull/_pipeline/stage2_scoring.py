"""Stage 2 per-photo scoring helpers — IQA, composition, saliency, taste, tensor loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_tensor as tv_to_tensor

from cull.config import (
    CLIP_MODEL_ID,
    CullConfig,
    EMBEDDING_CACHE_FILENAME,
    EMBEDDING_INDEX_FILENAME,
    IMAGE_LONG_EDGE_PX,
    IQA_EXPOSURE_DEFAULT,
    KEYSTONE_PENALTY_DEGREES,
    PALETTE_CENTROID_DOWNSAMPLE_PX,
    SHARED_DECODE_CLIP_PX,
    SHARED_DECODE_PIXEL_PX,
    TILT_PENALTY_DEGREES,
)
from cull.saliency import (
    SaliencyRequest,
    SaliencyResult,
    compute_saliency,
)
from cull.stage2 import taste as taste_module
from cull.stage2.aesthetic import score_aesthetic_batch
from cull.stage2.composition import (
    CompositionInput,
    score_batch as score_composition_batch,
)
from cull.stage2.fusion import IqaScores, compute_composite
from cull.stage2.iqa import (
    score_clipiqa_batch,
    score_topiq_batch,
    select_device,
)
from cull.stage2.portrait import PortraitResult, assess_portrait_from_array
from cull.stage2.subject_blur import SubjectBlurInput, score_one as score_subject_blur
from cull.stage2.taste import TasteScoreInput
from cull.models import PortraitScores
from cull.taste_features import TasteFeatureInputs, build_taste_feature_row, flatten_stage1_scores

if TYPE_CHECKING:
    from cull._pipeline.stage2_runner import _BatchCtx, _Stage2BatchInput

logger = logging.getLogger(__name__)


class _SearchCache(BaseModel):
    """Pre-computed CLIP embeddings keyed by source path."""

    model_config = {"arbitrary_types_allowed": True}

    embeddings: np.ndarray
    path_to_row: dict[str, int]


def _validate_search_index(index: dict, source_dir: Path) -> bool:
    """Return True if the search index has a matching CLIP model_id."""
    index_model_id = index.get("model_id")
    if index_model_id != CLIP_MODEL_ID:
        logger.info(
            "Search cache rejected: model_id mismatch (expected %s, got %s) at %s",
            CLIP_MODEL_ID, index_model_id, source_dir,
        )
        return False
    return True


def _load_search_cache(source_dir: Path) -> _SearchCache | None:
    """Return parsed search cache for source_dir or None on miss/mismatch."""
    emb_path = source_dir / EMBEDDING_CACHE_FILENAME
    idx_path = source_dir / EMBEDDING_INDEX_FILENAME
    if not emb_path.exists() or not idx_path.exists():
        logger.info("Search cache rejected: required paths missing in %s", source_dir)
        return None
    index: dict = json.loads(idx_path.read_text())
    if not _validate_search_index(index, source_dir):
        return None
    embeddings: np.ndarray = np.load(str(emb_path))
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        logger.warning("Search cache rejected: embeddings invalid shape (ndim=%d, shape=%s) at %s", embeddings.ndim, embeddings.shape, emb_path)
        return None
    index_row_count = len(index.get("paths", []))
    if len(embeddings) != index_row_count:
        logger.info("Search cache rejected: row count mismatch (index %d vs parquet %d) at %s", index_row_count, len(embeddings), source_dir)
        return None
    path_to_row = {p: i for i, p in enumerate(index.get("paths", []))}
    return _SearchCache(embeddings=embeddings, path_to_row=path_to_row)


def _gather_chunk_embeddings(
    chunk: list[Path], cache: _SearchCache | None
) -> torch.Tensor | None:
    """Return stacked embedding tensor for chunk if all hit cache, else None."""
    if cache is None:
        return None
    rows: list[np.ndarray] = []
    for path in chunk:
        idx = cache.path_to_row.get(str(path))
        if idx is None:
            return None
        rows.append(cache.embeddings[idx])
    stacked = np.stack(rows, axis=0).astype(np.float32)
    return torch.from_numpy(stacked).to(select_device())


def _load_tensor(path: Path) -> torch.Tensor:
    """Load image as a torch tensor resized to IMAGE_LONG_EDGE_PX."""
    img = Image.open(path).convert("RGB")
    long_edge = max(img.size)
    if long_edge > IMAGE_LONG_EDGE_PX:
        scale = IMAGE_LONG_EDGE_PX / long_edge
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return ToTensor()(img).unsqueeze(0)


def _load_tensor_only_batch(paths: list[Path]) -> torch.Tensor:
    """Load a batch of images as a stacked (N,C,H,W) tensor (no PIL list)."""
    tensors = [_load_tensor(path) for path in paths]
    return torch.cat(tensors, dim=0)


def _load_tensor_batch(paths: list[Path]) -> tuple[torch.Tensor, list[Image.Image]]:
    """Load a batch of images; returns stacked (N,C,H,W) tensor and PIL list."""
    tensors: list[torch.Tensor] = []
    pil_images: list[Image.Image] = []
    for path in paths:
        tensors.append(_load_tensor(path))
        pil_images.append(Image.open(path).convert("RGB"))
    return torch.cat(tensors, dim=0), pil_images


class _DualLoadInput(BaseModel):
    """Input bundle for dual-resolution PIL batch loading."""

    paths: list[Path]
    device: str


class _DualPilBatch(BaseModel):
    """Dual-resolution PIL batch: 224 for CLIP, 1280 for pixel consumers."""

    model_config = {"arbitrary_types_allowed": True}

    pil_224: list[Image.Image]
    pil_1280: list[Image.Image]
    tensor_1280: torch.Tensor
    paths: list[Path]


def _make_pil_1280(pil: Image.Image) -> Image.Image:
    """Resize a PIL image via LANCZOS so its long edge equals SHARED_DECODE_PIXEL_PX."""
    w, h = pil.size
    long_edge = max(w, h)
    if long_edge <= SHARED_DECODE_PIXEL_PX:
        return pil
    scale = SHARED_DECODE_PIXEL_PX / long_edge
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return pil.resize(new_size, Image.LANCZOS)


def _make_pil_224(pil: Image.Image) -> Image.Image:
    """Resize-shortest-edge + center-crop to SHARED_DECODE_CLIP_PX square via BICUBIC."""
    w, h = pil.size
    scale = SHARED_DECODE_CLIP_PX / min(w, h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = pil.resize(new_size, Image.BICUBIC)
    left = (resized.size[0] - SHARED_DECODE_CLIP_PX) // 2
    top = (resized.size[1] - SHARED_DECODE_CLIP_PX) // 2
    return resized.crop((left, top, left + SHARED_DECODE_CLIP_PX, top + SHARED_DECODE_CLIP_PX))


def _stack_tensor_1280(pil_list: list[Image.Image]) -> torch.Tensor:
    """Stack per-image 1280-edge PIL tensors into a single (N,C,H,W) batch."""
    tensors = [tv_to_tensor(pil).unsqueeze(0) for pil in pil_list]
    return torch.cat(tensors, dim=0)


def _load_dual_pil_batch(load_in: _DualLoadInput) -> _DualPilBatch:
    """Open each path once; build pil_224, pil_1280, and stacked tensor_1280."""
    pil_224_list: list[Image.Image] = []
    pil_1280_list: list[Image.Image] = []
    for path in load_in.paths:
        full = Image.open(path).convert("RGB")
        pil_1280_list.append(_make_pil_1280(full))
        pil_224_list.append(_make_pil_224(full))
    return _DualPilBatch(
        pil_224=pil_224_list,
        pil_1280=pil_1280_list,
        tensor_1280=_stack_tensor_1280(pil_1280_list),
        paths=list(load_in.paths),
    )


def _palette_lab_centroid(pil_image: Image.Image) -> tuple[float, float, float]:
    """Compute a cheap LAB centroid from a small downsample of a decoded photo."""
    small = pil_image.resize(
        (PALETTE_CENTROID_DOWNSAMPLE_PX, PALETTE_CENTROID_DOWNSAMPLE_PX), Image.BILINEAR
    )
    rgb_array = np.asarray(small, dtype=np.uint8)
    lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    mean = lab_array.reshape(-1, 3).mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def score_palette_lab_batch(pil_images: list[Image.Image]) -> list[tuple[float, float, float]]:
    """Compute a LAB palette centroid per image for shoot-level outlier detection."""
    return [_palette_lab_centroid(img) for img in pil_images]


def apply_palette_lab_to_scores(
    iqa_list: list[IqaScores], pil_1280: list[Image.Image]
) -> None:
    """Compute and patch a LAB palette centroid onto each IqaScores in-place."""
    for iqa, centroid in zip(iqa_list, score_palette_lab_batch(pil_1280)):
        iqa.palette_lab = centroid


def _score_aesthetic_for_batch(batch_in: "_Stage2BatchInput") -> list[float]:
    """Dispatch aesthetic scoring on either the embedding or PIL path."""
    if batch_in.embeddings is not None:
        return score_aesthetic_batch(batch_in.pil_images, embeddings=batch_in.embeddings)
    return score_aesthetic_batch(batch_in.pil_images)


def _build_iqa_scores(batch_in: "_Stage2BatchInput", device: str) -> list[IqaScores]:
    """Score one batch through all IQA metrics; returns aligned IqaScores list."""
    topiq_scores = score_topiq_batch(batch_in.tensor_batch, device)
    clipiqa_scores = score_clipiqa_batch(batch_in.tensor_batch, device)
    aesthetic_scores = _score_aesthetic_for_batch(batch_in)
    return [
        IqaScores(photo_path=p, topiq=t, laion_aesthetic=a, clipiqa=c, exposure=IQA_EXPOSURE_DEFAULT)
        for p, t, c, a in zip(
            batch_in.photo_paths, topiq_scores, clipiqa_scores, aesthetic_scores
        )
    ]


def _apply_exposure_to_scores(
    iqa_list: list[IqaScores], s1_results: dict[str, Any]
) -> None:
    """Patch exposure score in-place from Stage 1 results."""
    for iqa in iqa_list:
        s1 = s1_results.get(str(iqa.photo_path))
        if s1 is not None:
            iqa.exposure = s1.exposure.dr_score


def _apply_stage1_blur_context_to_scores(
    iqa_list: list[IqaScores], s1_results: dict[str, Any]
) -> None:
    """Patch Stage 1 blur context in-place for preset-aware Stage 2 fusion."""
    for iqa in iqa_list:
        s1 = s1_results.get(str(iqa.photo_path))
        if s1 is not None:
            iqa.is_bokeh = s1.blur.is_bokeh


def _tilt_penalty_from_geometry(geometry: Any) -> float:
    """Map tilt + keystone degrees into a [0, 1] penalty value."""
    tilt_norm = abs(geometry.tilt_degrees) / TILT_PENALTY_DEGREES
    keystone_norm = abs(geometry.keystone_degrees) / KEYSTONE_PENALTY_DEGREES
    return float(min(1.0, max(tilt_norm, keystone_norm)))


def _apply_geometry_to_scores(
    iqa_list: list[IqaScores], s1_results: dict[str, Any]
) -> None:
    """Patch tilt_penalty in-place from Stage 1 geometry results."""
    for iqa in iqa_list:
        s1 = s1_results.get(str(iqa.photo_path))
        if s1 is not None and s1.geometry is not None:
            iqa.tilt_penalty = _tilt_penalty_from_geometry(s1.geometry)


def _compute_saliency_for_path(
    path: Path, ctx: "_BatchCtx"
) -> SaliencyResult | None:
    """Check cache; compute saliency on miss."""
    path_key = str(path)
    if path_key in ctx.saliency_cache:
        return ctx.saliency_cache[path_key]
    try:
        result = compute_saliency(SaliencyRequest(image_path=path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Saliency failed for %s: %s", path, exc)
        result = None
    ctx.saliency_cache[path_key] = result
    return result


class _CompositionBuildInput(BaseModel):
    """Bundle for composition input construction: paths + skip-flags."""

    model_config = {"arbitrary_types_allowed": True}

    paths: list[Path]
    skip_flags: list[bool]


class _CompositionApplyInput(BaseModel):
    """Bundle for applying composition scores to an IQA batch."""

    model_config = {"arbitrary_types_allowed": True}

    iqa_list: list[IqaScores]
    paths: list[Path]
    ctx: Any


def _build_composition_inputs(
    build_in: _CompositionBuildInput, ctx: "_BatchCtx"
) -> tuple[list[CompositionInput], list[int]]:
    """Build composition inputs for paths with successful saliency."""
    inputs: list[CompositionInput] = []
    valid_indices: list[int] = []
    for idx, path in enumerate(build_in.paths):
        saliency = _compute_saliency_for_path(path, ctx)
        if saliency is None:
            continue
        dual_idx = _path_to_dual_index(path, ctx)
        inputs.append(CompositionInput(
            pil_1280=ctx.dual_pil.pil_1280[dual_idx],  # type: ignore[union-attr]
            saliency_result=saliency,
            skip_crop=build_in.skip_flags[idx],
        ))
        valid_indices.append(idx)
    return inputs, valid_indices


def _photo_must_reject(iqa: IqaScores, config: CullConfig) -> bool:
    """Return True if even a perfect composition score cannot lift the photo above REJECT."""
    optimistic = iqa.model_copy()
    optimistic.composition = 1.0
    fusion = compute_composite(optimistic, config)
    return fusion.routing == "REJECT"


def _apply_composition_to_scores(apply_in: _CompositionApplyInput) -> None:
    """Compute composition for the chunk and patch iqa_list in-place."""
    config = apply_in.ctx.loop_in.config
    skip_flags = [_photo_must_reject(iqa, config) for iqa in apply_in.iqa_list]
    build_in = _CompositionBuildInput(paths=apply_in.paths, skip_flags=skip_flags)
    inputs, valid_indices = _build_composition_inputs(build_in, apply_in.ctx)
    if not inputs:
        return
    outputs = score_composition_batch(inputs)
    for local_idx, (score, crop) in enumerate(outputs):
        target = apply_in.iqa_list[valid_indices[local_idx]]
        target.composition = score.composite
        target.composition_score = score
        target.crop = crop


def _decode_full_res_bgr(path: Path) -> np.ndarray | None:
    """Decode a photo at full resolution as BGR for portrait analysis."""
    image = cv2.imread(str(path))
    if image is None:
        logger.warning("Could not read image for portrait analysis: %s", path)
    return image


def _portrait_or_none(path: Path, config: CullConfig) -> PortraitResult | None:
    """Decode full-res once and run portrait assessment; swallow failures.

    assess_portrait_from_array shares this single decode with the
    EmotiEffLib emotion pass (via portrait._assemble_result's
    landmark-derived face crop), eliminating the redundant cv2.imread
    that assess_portrait would otherwise perform. Full resolution is
    preserved (not the downscaled pil_1280) because eye-sharpness scoring
    is resolution-sensitive; see assess_portrait_from_array's docstring.
    """
    if not config.is_portrait:
        return None
    image = _decode_full_res_bgr(path)
    if image is None:
        return None
    try:
        return assess_portrait_from_array(image, config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Portrait assessment failed for %s: %s", path, exc)
        return None


def _path_to_dual_index(path: Path, ctx: "_BatchCtx") -> int:
    """Look up the dual_pil index for a given path; raises if dual_pil is unset."""
    if ctx.dual_pil is None:
        raise RuntimeError("_BatchCtx.dual_pil is unset; _load_dual_pil_batch was skipped")
    return ctx.dual_pil.paths.index(path)


def _build_subject_blur_input(
    path: Path, ctx: "_BatchCtx"
) -> tuple[SubjectBlurInput, PortraitResult | None]:
    """Compose a SubjectBlurInput from portrait + saliency fallbacks."""
    config = ctx.loop_in.config
    portrait = _portrait_or_none(path, config)
    needs_saliency = portrait is None or not portrait.has_face
    saliency = _compute_saliency_for_path(path, ctx) if needs_saliency else None
    idx = _path_to_dual_index(path, ctx)
    pil_1280 = ctx.dual_pil.pil_1280[idx]  # type: ignore[union-attr]
    return SubjectBlurInput(pil_1280=pil_1280, portrait=portrait, saliency=saliency), portrait


def _patch_subject_blur_scores(iqa: IqaScores, sb_input: SubjectBlurInput) -> None:
    """Run subject_blur scoring and patch iqa fields when has_subject."""
    sb_score = score_subject_blur(sb_input)
    if sb_score.has_subject:
        iqa.subject_blur = sb_score.tenengrad
        iqa.subject_blur_score = sb_score


def _to_portrait_scores(portrait: PortraitResult) -> PortraitScores:
    """Convert portrait analysis into the Stage 2 cross-stage model."""
    return PortraitScores(
        eye_sharpness_left=portrait.eye_sharpness_left,
        eye_sharpness_right=portrait.eye_sharpness_right,
        ear_left=portrait.ear_left,
        ear_right=portrait.ear_right,
        is_eyes_closed=portrait.eyes_closed,
        is_squinting=portrait.is_squinting,
        dominant_emotion=portrait.dominant_emotion,
        valence=portrait.valence,
        arousal=portrait.arousal,
        is_face_occluded=portrait.face_occluded,
        face_occlusion_ratio=portrait.occlusion_ratio,
    )


class _SubjectBlurCtx(BaseModel):
    """Bundle of paths + config carried into the subject_blur scoring pass."""

    paths: list[Path]
    config: CullConfig


class _SubjectBlurApplyInput(BaseModel):
    """Bundle for applying subject-blur scores to an IQA batch."""

    model_config = {"arbitrary_types_allowed": True}

    iqa_list: list[IqaScores]
    ctx: _SubjectBlurCtx
    batch_ctx: Any


def _apply_subject_blur_to_scores(apply_in: _SubjectBlurApplyInput) -> dict[str, PortraitResult]:
    """Score subject blur per path; patch iqa_list and return portrait cache."""
    portraits: dict[str, PortraitResult] = {}
    for iqa, path in zip(apply_in.iqa_list, apply_in.ctx.paths):
        sb_input, portrait = _build_subject_blur_input(path, apply_in.batch_ctx)
        if portrait is not None:
            portraits[str(path)] = portrait
            iqa.portrait = _to_portrait_scores(portrait)
        _patch_subject_blur_scores(iqa, sb_input)
    return portraits


class _TasteApplyInput(BaseModel):
    """Bundle for applying taste scores to an IQA batch."""

    model_config = {"arbitrary_types_allowed": True}

    iqa_list: list[IqaScores]
    paths: list[Path]
    batch_ctx: Any


class _TasteRowCtx(BaseModel):
    """Bundle for building one photo's canonical taste feature row mid-Stage-2."""

    model_config = {"arbitrary_types_allowed": True}

    path: Path
    iqa: IqaScores
    batch_ctx: Any


def _pre_taste_composite(iqa: IqaScores, config: CullConfig) -> float:
    """Composite as it stands before taste is scored (taste term is zero here).

    This is the analog of the OverrideEntry.stage2_composite historically
    logged for the 602 real overrides, all of which predate a real taste
    profile — their logged composite never carried a taste contribution.
    """
    return compute_composite(iqa, config).stage2.composite


def _stage1_scores_for(path: Path, s1_results: dict[str, Any]) -> dict[str, float]:
    """Look up Stage 1 scalar scores for a path, flattened via the canonical helper."""
    return flatten_stage1_scores(s1_results.get(str(path)))


def _taste_inputs_for(ctx: _TasteRowCtx) -> TasteFeatureInputs:
    """Assemble canonical taste feature inputs for one photo mid-Stage-2."""
    config = ctx.batch_ctx.loop_in.config
    composition = ctx.iqa.composition_score
    subject_blur = ctx.iqa.subject_blur_score
    return TasteFeatureInputs(
        stage1_scores=_stage1_scores_for(ctx.path, ctx.batch_ctx.loop_in.s1_results),
        stage2_composite=_pre_taste_composite(ctx.iqa, config),
        composition_composite=composition.composite if composition else None,
        thirds_alignment=composition.thirds_alignment if composition else None,
        negative_space_balance=composition.negative_space_balance if composition else None,
        subject_blur_tenengrad=subject_blur.tenengrad if subject_blur else None,
    )


def _apply_taste_to_scores(apply_in: _TasteApplyInput) -> None:
    """Score taste for each photo in the chunk and patch iqa_list in-place.

    Requires subject_blur to already be applied to iqa_list (see
    stage2_runner._process_batch ordering) so its tenengrad feeds the row.
    """
    taste_inputs = [
        TasteScoreInput(feature_row=build_taste_feature_row(
            _taste_inputs_for(_TasteRowCtx(path=path, iqa=iqa, batch_ctx=apply_in.batch_ctx))
        ))
        for path, iqa in zip(apply_in.paths, apply_in.iqa_list)
    ]
    taste_results = taste_module.score_batch(taste_inputs)
    for iqa, taste_result in zip(apply_in.iqa_list, taste_results):
        iqa.taste = taste_result.probability
        iqa.taste_score = taste_result
