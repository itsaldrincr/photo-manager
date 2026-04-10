"""LAION Aesthetics V2 scoring via simple-aesthetics-predictor."""

from __future__ import annotations

import logging

import torch
from pydantic import BaseModel, ConfigDict

from cull import clip_loader
from cull.config import ModelCacheConfig
from cull.io_silence import _silence_stdio
from cull.stage2.iqa import select_device

logger = logging.getLogger(__name__)

_HEADS: dict[str, "AestheticHead"] = {}
_CACHE: ModelCacheConfig = ModelCacheConfig.from_env()

AESTHETIC_SCORE_MIN: float = 1.0
AESTHETIC_SCORE_MAX: float = 10.0
AESTHETIC_MODEL_ID: str = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
AESTHETIC_EMBED_DIM: int = 768
AESTHETIC_DROPOUT_EARLY: float = 0.2
AESTHETIC_DROPOUT_LATE: float = 0.1
AESTHETIC_HEAD_FILENAME: str = "model.safetensors"
AESTHETIC_LAYER_PREFIX: str = "layers."


class AestheticHead(BaseModel):
    """Head-only wrapper around the shunk031 linear head; no CLIP backbone."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    layers: torch.nn.Module


def unload_predictor() -> None:
    """Release all cached aesthetic heads."""
    _HEADS.clear()
    logger.info("Aesthetic head unloaded")


def _build_laion_head() -> torch.nn.Sequential:
    """Construct the LAION Aesthetics V2 linearMSE MLP (no weights)."""
    return torch.nn.Sequential(
        torch.nn.Linear(AESTHETIC_EMBED_DIM, 1024),
        torch.nn.Dropout(AESTHETIC_DROPOUT_EARLY),
        torch.nn.Linear(1024, 128),
        torch.nn.Dropout(AESTHETIC_DROPOUT_EARLY),
        torch.nn.Linear(128, 64),
        torch.nn.Dropout(AESTHETIC_DROPOUT_LATE),
        torch.nn.Linear(64, 16),
        torch.nn.Linear(16, 1),
    )


def _locate_head_weights() -> str:
    """Return absolute path to the cached aesthetic model.safetensors file."""
    flat = AESTHETIC_MODEL_ID.replace("/", "--")
    repo_dir = _CACHE.hf_home / "hub" / f"models--{flat}" / "snapshots"
    weight_path = next(repo_dir.rglob(AESTHETIC_HEAD_FILENAME), None)
    if weight_path is None:
        raise FileNotFoundError(f"aesthetic head weights not found under {repo_dir}")
    return str(weight_path)


def _extract_head(device: str) -> AestheticHead:
    """Load linear-head weights directly from cached safetensors."""
    from safetensors.torch import load_file  # noqa: PLC0415

    logger.info("Loading LAION Aesthetics head on device '%s'", device)
    full_state = load_file(_locate_head_weights())
    head_state = {
        k.removeprefix(AESTHETIC_LAYER_PREFIX): v
        for k, v in full_state.items()
        if k.startswith(AESTHETIC_LAYER_PREFIX)
    }
    layers = _build_laion_head()
    layers.load_state_dict(head_state)
    layers.to(device).eval()
    return AestheticHead(layers=layers)


def _get_head(device: str) -> AestheticHead:
    """Return cached AestheticHead for the given device; load on miss."""
    if device in _HEADS:
        return _HEADS[device]
    head = _extract_head(device)
    _HEADS[device] = head
    return head


def warmup_predictor(device: str) -> None:
    """Warm shared CLIP singletons and the aesthetic linear head for the device."""
    clip_loader.get_clip_model()
    clip_loader.get_clip_processor()
    _get_head(device)


def _normalize_aesthetic(raw: float) -> float:
    """Normalize a 1-10 aesthetic score to [0, 1]."""
    clamped = max(AESTHETIC_SCORE_MIN, min(AESTHETIC_SCORE_MAX, raw))
    return (clamped - AESTHETIC_SCORE_MIN) / (AESTHETIC_SCORE_MAX - AESTHETIC_SCORE_MIN)


def _tensor_to_pil(image_tensor: torch.Tensor) -> object:
    """Convert a [1,C,H,W] tensor back to a PIL Image for CLIP processing."""
    from torchvision.transforms.functional import to_pil_image  # noqa: PLC0415

    return to_pil_image(image_tensor.squeeze(0).cpu())


class _PilEmbedRequest(BaseModel):
    """Bundle of PIL images and the target device for embedding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pil_images: list[object]
    device: str


def _embed_with_shared_clip(req: _PilEmbedRequest) -> torch.Tensor:
    """Use the shared CLIP vision encoder to produce image embeddings."""
    clip_model = clip_loader.get_clip_model()
    processor = clip_loader.get_clip_processor()
    with _silence_stdio():
        inputs = processor(images=req.pil_images, return_tensors="pt").to(req.device)
        with torch.no_grad():
            embeddings = clip_model.get_image_features(**inputs).pooler_output  # type: ignore[attr-defined]
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


def _score_pil_batch(pil_images: list[object], device: str) -> list[float]:
    """Run the PIL path: embed via shared CLIP, then feed the linear head."""
    head = _get_head(device)
    embeddings = _embed_with_shared_clip(
        _PilEmbedRequest(pil_images=pil_images, device=device)
    )
    return _score_embedding_batch(head, embeddings)


def _score_embedding_batch(
    head: AestheticHead, embeddings: torch.Tensor
) -> list[float]:
    """Run the embedding path: feed pre-computed embeddings through the head."""
    with torch.no_grad():
        raw = head.layers(embeddings).squeeze(-1).cpu().tolist()
    if not isinstance(raw, list):
        raw = [raw]
    return [_normalize_aesthetic(s) for s in raw]


def score_from_embeddings(head: AestheticHead, embeddings: torch.Tensor) -> list[float]:
    """Score pre-computed CLIP embeddings with the aesthetic linear head."""
    return _score_embedding_batch(head, embeddings)


def score_aesthetic_batch(
    pil_images: list[object], embeddings: torch.Tensor | None = None
) -> list[float]:
    """Score a batch; returns list of values in [0, 1]. Accepts pre-computed embeddings."""
    device = select_device()
    if isinstance(embeddings, torch.Tensor):
        head = _get_head(device)
        return _score_embedding_batch(head, embeddings)
    return _score_pil_batch(pil_images, device)


def score_aesthetic(image_tensor: torch.Tensor, device: str) -> float:
    """Score an image tensor with LAION Aesthetics V2; returns value in [0, 1]."""
    pil_image = _tensor_to_pil(image_tensor)
    return score_aesthetic_batch([pil_image])[0]
