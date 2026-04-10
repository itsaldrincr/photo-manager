"""CLI-facing config construction helpers and default constants.

Extracted from ``cull.cli`` as part of the 600-series CLI hub split.
Owns ``_build_config``, ``_build_stages``, and the default-value
constants referenced by the Click command decorators.
"""

from __future__ import annotations

from cull.config import (
    CURATE_VLM_TIEBREAK_THRESHOLD,
    CullConfig,
    STAGE_VLM,
)

DEFAULT_THRESHOLD: float = 0.65
DEFAULT_BURST_GAP: float = 0.5
DEFAULT_STAGES: tuple[int, ...] = (1, 2, 3)
PRESET_CHOICES: list[str] = [
    "general", "wedding", "documentary", "wildlife", "landscape", "street", "holiday"
]


def _build_stages(stage: tuple[int, ...], should_skip_vlm: bool) -> list[int]:
    """Return the list of pipeline stages to run."""
    base = list(stage) if stage else list(DEFAULT_STAGES)
    return [s for s in base if s != STAGE_VLM] if should_skip_vlm else base


def _build_config(kwargs: dict) -> CullConfig:
    """Construct CullConfig from parsed CLI keyword arguments."""
    stages = _build_stages(kwargs["stage"], kwargs["no_vlm"])
    return CullConfig(
        threshold=kwargs["threshold"],
        burst_gap=kwargs["burst_gap"],
        preset=kwargs["preset"],
        model=kwargs["model"],
        is_portrait=kwargs["portrait"],
        is_dry_run=kwargs["dry_run"],
        stages=stages,
        curate_target=kwargs.get("curate"),
        curate_vlm_threshold=kwargs.get(
            "curate_vlm_threshold", CURATE_VLM_TIEBREAK_THRESHOLD,
        ),
        is_sidecars=not kwargs.get("no_sidecars", False),
    )
