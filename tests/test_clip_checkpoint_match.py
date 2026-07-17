"""Verify the CLIP backbone embedded in the aesthetics predictor matches CLIP_MODEL_ID."""

from __future__ import annotations

import json

from huggingface_hub import hf_hub_download
from transformers import CLIPVisionConfig

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


def _load_aesthetic_config() -> dict:
    """Fetch the aesthetics predictor's raw config.json.

    Read as plain JSON (no AutoConfig): the repo's remote code passes
    positional args to CLIPVisionConfig, which transformers>=5 rejects.
    Production never executes that remote code either.
    """
    config_path = hf_hub_download(AESTHETIC_MODEL_ID, "config.json")
    return json.loads(open(config_path, encoding="utf-8").read())


def _load_clip_vision_config() -> CLIPVisionConfig:
    """Fetch CLIP ViT config (JSON only, no model weights)."""
    return CLIPVisionConfig.from_pretrained(CLIP_MODEL_ID)


def test_clip_backbone_matches_clip_model_id() -> None:
    """Assert the aesthetics predictor backbone geometry matches CLIP_MODEL_ID."""
    aesthetic_dict = _load_aesthetic_config()
    clip_dict = _load_clip_vision_config().to_dict()
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
    assert config.get("hidden_size") == EXPECTED_HIDDEN_SIZE, (
        f"Expected hidden_size {EXPECTED_HIDDEN_SIZE}, got {config.get('hidden_size')}"
    )
    assert config.get("image_size") == EXPECTED_IMAGE_SIZE, (
        f"Expected image_size {EXPECTED_IMAGE_SIZE}, got {config.get('image_size')}"
    )
