"""Subprocess worker: weak-label a face-crop corpus with the production VLM.

Loads exactly ONE heavy model (gemma-4-12b via the production vlm_session),
scores every image serially, then exits — the one-heavy-model-resident rule
for this 18 GB host. Each double-pass is two separate invocations with
paraphrased prompts (see weaklabel_models.TASK_PROMPTS); generation runs at
the production temperature (0.0), so paraphrase — not sampling — provides
the independence between passes.

Usage:
    python3 benchmarks/gemma_weaklabel_runner.py <task> <manifest.json> <output.json>
    with <task> one of occlusion_v1 | occlusion_v2 | expression_v1 | expression_v2
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from weaklabel_models import (
    EXPRESSION_VOCAB,
    TASK_PROMPTS,
    WEAKLABEL_MAX_TOKENS,
    WeaklabelManifest,
    WeaklabelReading,
    WeaklabelRunOutput,
)

logger = logging.getLogger(__name__)


class RunnerArgs(BaseModel):
    """Parsed CLI arguments."""

    task: str
    manifest_path: Path
    output_path: Path


def _parse_args(argv: list[str]) -> RunnerArgs:
    """Validate and bundle the three positional arguments."""
    if len(argv) != 4 or argv[1] not in TASK_PROMPTS:
        raise SystemExit(
            f"usage: gemma_weaklabel_runner.py {{{'|'.join(sorted(TASK_PROMPTS))}}} "
            "<manifest.json> <output.json>"
        )
    return RunnerArgs(task=argv[1], manifest_path=Path(argv[2]), output_path=Path(argv[3]))


def _extract_payload(text: str) -> dict | None:
    """Pull the first balanced JSON object out of raw model text."""
    from cull.stage3.parser import _clean_json_text, _extract_json_text  # noqa: PLC0415

    json_text = _extract_json_text(text)
    if json_text is None:
        return None
    try:
        payload = json.loads(_clean_json_text(json_text))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _confidence_of(payload: dict) -> float | None:
    """Return the payload's confidence as a float when present and numeric."""
    value = payload.get("confidence")
    return float(value) if isinstance(value, (int, float)) else None


def _occlusion_fields(payload: dict) -> dict | None:
    """Validate an occlusion answer; None when the schema is not met."""
    occluded = payload.get("face_occluded")
    if not isinstance(occluded, bool):
        return None
    occluder = payload.get("occluder")
    return {
        "face_occluded": occluded,
        "occluder": occluder if isinstance(occluder, str) else None,
        "confidence": _confidence_of(payload),
    }


def _expression_fields(payload: dict) -> dict | None:
    """Validate an expression answer against the closed vocabulary."""
    label = payload.get("expression")
    if not isinstance(label, str):
        return None
    normalized = label.strip().lower()
    if normalized not in EXPRESSION_VOCAB:
        return None
    return {"expression": normalized, "confidence": _confidence_of(payload)}


def _parse_reading(task: str, text: str) -> dict | None:
    """Parse raw model text into validated fields for the given task."""
    payload = _extract_payload(text)
    if payload is None:
        return None
    if task.startswith("occlusion"):
        return _occlusion_fields(payload)
    return _expression_fields(payload)


class _LabelCall(BaseModel):
    """One image's labeling call against the loaded session."""

    task: str
    image_path: Path


def _label_one(call: _LabelCall, session: object) -> WeaklabelReading:
    """Ask the loaded VLM one strict-JSON question about one image."""
    from cull.vlm_session import VlmGenerateInput  # noqa: PLC0415

    started = time.monotonic()
    try:
        text = session.generate(VlmGenerateInput(
            prompt=TASK_PROMPTS[call.task], images=[call.image_path],
            max_tokens=WEAKLABEL_MAX_TOKENS,
        ))
    except Exception:
        logger.exception("generate failed for %s", call.image_path.name)
        text = ""
    latency = time.monotonic() - started
    fields = _parse_reading(call.task, text) if text else None
    base = {
        "name": call.image_path.name, "path": str(call.image_path),
        "raw_text": text[:500], "latency_seconds": latency,
    }
    if fields is None:
        return WeaklabelReading(**base, is_parse_error=True)
    return WeaklabelReading(**base, **fields)


def _run_task(args: RunnerArgs) -> WeaklabelRunOutput:
    """Load the VLM once, label every manifest image serially, unload on exit."""
    from cull.vlm_session import vlm_session  # noqa: PLC0415

    manifest = WeaklabelManifest.model_validate_json(args.manifest_path.read_text())
    readings: list[WeaklabelReading] = []
    with vlm_session("gemma-4-12b") as session:
        for index, image_path in enumerate(manifest.image_paths):
            readings.append(_label_one(_LabelCall(task=args.task, image_path=image_path), session))
            logger.info("[%d/%d] %s", index + 1, len(manifest.image_paths), image_path.name)
    return WeaklabelRunOutput(task=args.task, prompt=TASK_PROMPTS[args.task], readings=readings)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args(sys.argv)
    output = _run_task(args)
    args.output_path.write_text(output.model_dump_json(indent=2))
    errors = sum(1 for r in output.readings if r.is_parse_error)
    logger.info("done: %d readings, %d parse errors", len(output.readings), errors)


if __name__ == "__main__":
    main()
