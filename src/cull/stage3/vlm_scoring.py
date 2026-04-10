"""Stage 3 VLM scoring — in-process mlx_vlm inference path."""

from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from cull.config import VLM_IMAGE_MAX_PX, VLM_JPEG_QUALITY, VLM_MAX_RETRIES
from cull.models import Stage3Result
from cull.stage3.parser import parse_vlm_response
from cull.stage3.prompt import PromptContext, build_prompt
from cull.vlm_session import VlmGenerateInput, VlmSession

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

RETRY_BASE_DELAY: float = 1.0
RETRY_BACKOFF_FACTOR: float = 2.0

SYSTEM_MSG: str = (
    "You are a photo quality scoring assistant. "
    "You MUST respond with ONLY a single JSON object. "
    "No markdown code fences. No explanation. No text before or after the JSON. "
    "Start your response with { and end with }."
)


class VlmRequest(BaseModel):
    """Input bundle for a single VLM scoring call."""

    image_path: Path
    context: PromptContext
    model: str = ""


class VlmScoreCallInput(BaseModel):
    """Bundle of request + live session for score_photo."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    request: VlmRequest
    session: VlmSession


def resize_for_vlm(img: PILImage) -> PILImage:
    """Resize image so long edge is at most VLM_IMAGE_MAX_PX."""
    from PIL import Image  # noqa: PLC0415

    long_edge = max(img.size)
    if long_edge <= VLM_IMAGE_MAX_PX:
        return img
    scale = VLM_IMAGE_MAX_PX / long_edge
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    return img.resize(new_size, Image.LANCZOS)


def load_image_b64(image_path: Path) -> str:
    """Resize image to VLM_IMAGE_MAX_PX and return base64 JPEG string."""
    from PIL import Image  # noqa: PLC0415

    img = Image.open(image_path).convert("RGB")
    img = resize_for_vlm(img)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=VLM_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


def _build_prompt_text(request: VlmRequest) -> str:
    """Build the full scoring prompt from context hints."""
    return f"{SYSTEM_MSG}\n\n{build_prompt(request.context)}"


def _run_attempt(call_in: VlmScoreCallInput) -> Stage3Result:
    """Run one generate() call and parse the result."""
    prompt = _build_prompt_text(call_in.request)
    gen_in = VlmGenerateInput(prompt=prompt, images=[call_in.request.image_path])
    raw_text = call_in.session.generate(gen_in)
    result = parse_vlm_response(raw_text)
    result.photo_path = call_in.request.image_path
    result.model_used = call_in.request.model
    return result


def _retry_loop(call_in: VlmScoreCallInput) -> Stage3Result:
    """Retry _run_attempt on parse errors with exponential backoff."""
    delay = RETRY_BASE_DELAY
    for attempt in range(1, VLM_MAX_RETRIES + 1):
        logger.info("VLM attempt %d/%d", attempt, VLM_MAX_RETRIES)
        result = _run_attempt(call_in)
        if not result.is_parse_error:
            return result
        if attempt < VLM_MAX_RETRIES:
            logger.warning("parse_error on attempt %d, retry in %.1fs", attempt, delay)
            time.sleep(delay)
            delay *= RETRY_BACKOFF_FACTOR
    logger.error("All %d VLM attempts failed for %s", VLM_MAX_RETRIES, call_in.request.image_path)
    return Stage3Result(
        photo_path=call_in.request.image_path,
        model_used=call_in.request.model,
        is_parse_error=True,
    )


def score_photo(call_in: VlmScoreCallInput) -> Stage3Result:
    """Check image exists, then run the retry loop and return a Stage3Result."""
    if not call_in.request.image_path.exists():
        logger.warning("Image not found, skipping: %s", call_in.request.image_path)
        return Stage3Result(photo_path=call_in.request.image_path, is_parse_error=True)
    return _retry_loop(call_in)
