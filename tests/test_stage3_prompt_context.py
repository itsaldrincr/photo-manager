"""Tests for Stage 3 prompt/context enrichment without invoking a real VLM."""

from __future__ import annotations

from pathlib import Path

from cull._pipeline.stage3_runner import _PromptContextInput, _build_prompt_context
from cull.config import CullConfig
from cull.models import (
    BlurScores,
    CompositionScore,
    ExposureScores,
    PortraitScores,
    Stage1Result,
    Stage2Result,
)
from cull.stage2.fusion import FusionResult
from cull.stage3.prompt import PromptContext, build_prompt


PHOTO_PATH = Path("/tmp/test_stage3.jpg")


def _stage1() -> Stage1Result:
    return Stage1Result(
        photo_path=PHOTO_PATH,
        blur=BlurScores(
            tenengrad=10.0,
            fft_ratio=1.0,
            blur_tier=2,
            is_bokeh=True,
            is_motion_blur=True,
        ),
        exposure=ExposureScores(
            dr_score=0.6,
            clipping_highlight=0.1,
            clipping_shadow=0.2,
            midtone_pct=0.5,
            color_cast_score=0.4,
            has_highlight_clip=True,
            has_shadow_clip=True,
            has_color_cast=True,
        ),
        noise_score=0.1,
    )


def _fusion() -> FusionResult:
    return FusionResult(
        routing="AMBIGUOUS",
        stage2=Stage2Result(
            photo_path=PHOTO_PATH,
            topiq=0.55,
            laion_aesthetic=0.60,
            clipiqa=0.58,
            composite=0.62,
            portrait=PortraitScores(
                dominant_emotion="joy",
                is_eyes_closed=True,
                is_face_occluded=True,
            ),
            composition=CompositionScore(
                thirds_alignment=0.7,
                edge_clearance=0.6,
                negative_space_balance=0.5,
                topiq_iaa=0.65,
                composite=0.66,
            ),
        ),
    )


def test_build_prompt_context_includes_stage1_and_stage2_signals() -> None:
    """PromptContext should carry preset, Stage 2, portrait, and Stage 1 hints."""
    prompt_in = _PromptContextInput(
        s1=_stage1(),
        s2_fusion=_fusion(),
        config=CullConfig(preset="holiday"),
    )
    context = _build_prompt_context(prompt_in)
    assert context == PromptContext(
        preset="holiday",
        stage2_composite=0.62,
        composition_score=0.66,
        motion_blur_detected=True,
        dominant_emotion="joy",
        has_face=True,
        eyes_closed=True,
        face_occluded=True,
        is_bokeh=True,
        has_highlight_clip=True,
        has_shadow_clip=True,
        has_color_cast=True,
    )


def test_build_prompt_renders_preset_and_quality_hints() -> None:
    """Prompt text should mention preset guidance and enriched Stage 2 context."""
    prompt = build_prompt(
        PromptContext(
            preset="holiday",
            stage2_composite=0.62,
            composition_score=0.66,
            dominant_emotion="joy",
            has_face=True,
            eyes_closed=True,
            face_occluded=True,
            is_bokeh=True,
            motion_blur_detected=True,
            has_highlight_clip=True,
        )
    )
    assert "Balance people, atmosphere, and scene storytelling" in prompt
    assert "Stage 2 composite score was 0.62" in prompt
    assert "prior composition analysis scored 0.66" in prompt
    assert "face was detected" in prompt
    assert "expression classifier detected 'joy'" in prompt
    assert "face occlusion was detected" in prompt
    assert "shallow depth of field may be intentional" in prompt
