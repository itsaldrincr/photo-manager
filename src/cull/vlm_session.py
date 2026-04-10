# stdlib
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

# third-party
from pydantic import BaseModel, ConfigDict

# local
from cull.config import (
    VLM_IMAGE_MAX_PX,
    VLM_JPEG_QUALITY,
    VLM_MAX_TOKENS,
    VLM_TEMPERATURE,
)
from cull.vlm_registry import VLMEntry, resolve_alias

logger = logging.getLogger(__name__)

VLM_CONFIG_FILENAME: str = "config.json"
VLM_TEMP_PREFIX: str = "cull_vlm_"
VLM_TEMP_SUFFIX: str = ".jpg"


class VlmLoadError(Exception):
    """Raised when mlx_vlm.load() fails for a resolved VLMEntry."""


class VlmGenerateInput(BaseModel):
    """Arguments for one generation call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: str
    images: list[Path]
    max_tokens: int = VLM_MAX_TOKENS
    temperature: float = VLM_TEMPERATURE


class VlmSession(BaseModel):
    """Thin stateful wrapper around a loaded mlx_vlm model + processor."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    entry: VLMEntry
    model: Any = None
    processor: Any = None
    config: Any = None

    def generate(self, call_in: VlmGenerateInput) -> str:
        """Resize each input image, run one mlx_vlm.generate() call, return raw text."""
        if self.model is None:
            raise RuntimeError("VlmSession.generate() called after unload()")
        resized = [_resize_image_for_vlm(p) for p in call_in.images]
        try:
            return self._invoke_generate(call_in, resized)
        finally:
            for path in resized:
                path.unlink(missing_ok=True)

    def _invoke_generate(
        self, call_in: VlmGenerateInput, image_paths: list[Path]
    ) -> str:
        """Apply chat template and call mlx_vlm.generate with pre-resized image paths."""
        mlx_vlm = _get_mlx_vlm()
        formatted = mlx_vlm.apply_chat_template(
            self.processor, self.config, call_in.prompt,
            num_images=len(image_paths),
        )
        result = mlx_vlm.generate(
            self.model, self.processor, formatted,
            image=[str(p) for p in image_paths],
            max_tokens=call_in.max_tokens, temperature=call_in.temperature,
        )
        return result.text

    def unload(self) -> None:
        """Drop references and call gc.collect() + mx.clear_cache()."""
        import gc  # noqa: PLC0415

        self.model = None
        self.processor = None
        self.config = None
        gc.collect()
        try:
            import mlx.core as mx  # noqa: PLC0415

            mx.clear_cache()
        except ImportError:
            logger.debug("mlx.core not available; skipping clear_cache")


@contextmanager
def vlm_session(alias: str) -> Iterator[VlmSession]:
    """Context-manager that owns one mlx_vlm load/unload cycle.

    Raises:
        VLMResolutionError: if alias does not resolve to a discovered VLM.
        VlmLoadError: if mlx_vlm.load() or config.json parse fails.
    """
    entry = resolve_alias(alias)
    session = _load_session(entry)
    try:
        yield session
    finally:
        session.unload()


def _get_mlx_vlm() -> Any:
    """Lazy-import mlx_vlm and return the module."""
    import mlx_vlm  # noqa: PLC0415

    return mlx_vlm


def _load_session(entry: VLMEntry) -> VlmSession:
    """Call mlx_vlm.load() and wrap the result in a VlmSession."""
    mlx_vlm = _get_mlx_vlm()
    try:
        model, processor = mlx_vlm.load(str(entry.directory))
        config = _load_raw_config(entry.directory)
    except Exception as exc:
        raise VlmLoadError(
            f"mlx_vlm.load failed for alias '{entry.alias}' at {entry.directory}: {exc}"
        ) from exc
    return VlmSession(entry=entry, model=model, processor=processor, config=config)


def _load_raw_config(directory: Path) -> dict:
    """Parse config.json so apply_chat_template can access the model_type."""
    import json  # noqa: PLC0415

    return json.loads((directory / VLM_CONFIG_FILENAME).read_text())


def _resize_image_for_vlm(path: Path) -> Path:
    """Resize image to VLM_IMAGE_MAX_PX long edge, save to temp JPEG, return temp path."""
    from PIL import Image  # noqa: PLC0415

    img = Image.open(path).convert("RGB")
    long_edge = max(img.size)
    if long_edge > VLM_IMAGE_MAX_PX:
        scale = VLM_IMAGE_MAX_PX / long_edge
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    fd, tmp = tempfile.mkstemp(prefix=VLM_TEMP_PREFIX, suffix=VLM_TEMP_SUFFIX)
    os.close(fd)
    img.save(tmp, "JPEG", quality=VLM_JPEG_QUALITY)
    return Path(tmp)
