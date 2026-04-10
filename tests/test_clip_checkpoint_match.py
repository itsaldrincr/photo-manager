"""Verify the CLIP backbone embedded in the aesthetics predictor matches CLIP_MODEL_ID."""

from __future__ import annotations

from transformers import AutoConfig, CLIPVisionConfig

from cull.config import CLIP_MODEL_ID
from cull.stage2.aesthetic import AESTHETIC_MODEL_ID

EXPECTED_HIDDEN_SIZE: int = 1024
EXPECTED_IMAGE_SIZE: int = 224

_GEOMETRY_KEYS: tuple[str, ...] = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "patch_size",
    "projection_dim",
    "hidden_act",
)


def _load_aesthetic_config() -> AutoConfig:
    """Fetch aesthetics predictor config (JSON only, no model weights)."""
    return AutoConfig.from_pretrained(AESTHETIC_MODEL_ID, trust_remote_code=True)


def _load_clip_vision_config() -> CLIPVisionConfig:
    """Fetch CLIP ViT config (JSON only, no model weights)."""
    return CLIPVisionConfig.from_pretrained(CLIP_MODEL_ID)


def test_clip_backbone_matches_clip_model_id() -> None:
    """Assert the aesthetics predictor backbone geometry matches CLIP_MODEL_ID."""
    aesthetic_cfg = _load_aesthetic_config()
    clip_cfg = _load_clip_vision_config()
    aesthetic_dict = aesthetic_cfg.to_dict()
    clip_dict = clip_cfg.to_dict()
    for key in _GEOMETRY_KEYS:
        aesthetic_val = aesthetic_dict.get(key)
        clip_val = clip_dict.get(key)
        assert aesthetic_val == clip_val, (
            f"Backbone mismatch on '{key}': "
            f"aesthetics predictor={aesthetic_val!r}, "
            f"{CLIP_MODEL_ID}={clip_val!r}"
        )


def test_vit_l14_geometry() -> None:
    """Assert ViT-L/14 geometry: hidden_size==1024 and image_size==224."""
    config = _load_aesthetic_config()
    assert config.hidden_size == EXPECTED_HIDDEN_SIZE, (
        f"Expected hidden_size {EXPECTED_HIDDEN_SIZE}, got {config.hidden_size}"
    )
    assert config.image_size == EXPECTED_IMAGE_SIZE, (
        f"Expected image_size {EXPECTED_IMAGE_SIZE}, got {config.image_size}"
    )
