"""Stage 3 VLM re-export shim — all logic lives in vlm_scoring.py."""

from cull.stage3.vlm_scoring import (  # noqa: F401
    SYSTEM_MSG,
    VLM_IMAGE_MAX_PX,
    VLM_JPEG_QUALITY,
    VlmRequest,
    VlmScoreCallInput,
    load_image_b64,
    resize_for_vlm,
    score_photo,
)

__all__ = [
    "VlmRequest",
    "VlmScoreCallInput",
    "score_photo",
    "load_image_b64",
    "resize_for_vlm",
    "SYSTEM_MSG",
    "VLM_IMAGE_MAX_PX",
    "VLM_JPEG_QUALITY",
]
