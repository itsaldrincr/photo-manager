"""Tests for src/cull/sidecar.py — no ML imports."""

from __future__ import annotations

from pathlib import Path

import pytest
from lxml import etree

from cull.config import CullConfig, SIDECAR_NAMESPACE_CRS
from cull.models import (
    BlurScores,
    BurstInfo,
    CropProposal,
    DecisionLabel,
    ExposureScores,
    GeometryScore,
    PhotoDecision,
    PhotoMeta,
    Stage1Result,
    Stage2Result,
)
from cull.sidecar import SidecarWriteInput, write_for_decision


def _make_decision(source: Path, decision: DecisionLabel = "keeper") -> PhotoDecision:
    """Build a synthetic PhotoDecision pointing at source."""
    meta = PhotoMeta(path=source, filename=source.name)
    blur = BlurScores(tenengrad=0.8, fft_ratio=0.9, blur_tier=1)
    exposure = ExposureScores(
        dr_score=0.7,
        clipping_highlight=0.01,
        clipping_shadow=0.01,
        midtone_pct=0.5,
        color_cast_score=0.1,
    )
    geometry = GeometryScore(
        tilt_degrees=1.5,
        keystone_degrees=2.0,
        confidence=0.9,
        has_horizon=True,
        has_verticals=False,
    )
    stage1 = Stage1Result(
        photo_path=source,
        blur=blur,
        exposure=exposure,
        noise_score=0.1,
        geometry=geometry,
    )
    crop = CropProposal(top=10, left=20, bottom=100, right=200, source="smartcrop")
    stage2 = Stage2Result(
        photo_path=source,
        topiq=0.8,
        laion_aesthetic=0.75,
        clipiqa=0.7,
        composite=0.77,
        crop=crop,
    )
    return PhotoDecision(
        photo=meta,
        decision=decision,
        stage1=stage1,
        stage2=stage2,
    )


def test_write_for_decision_creates_xmp(tmp_path: Path) -> None:
    """Sidecar file must exist, parse, and carry the crs namespace."""
    source = tmp_path / "IMG_0001.jpg"
    source.write_bytes(b"fake-jpeg")
    decision = _make_decision(source)
    config = CullConfig(is_sidecars=True)
    sidecar_input = SidecarWriteInput(decision=decision, config=config)

    result = write_for_decision(sidecar_input)

    assert result is not None
    assert result == source.with_suffix(".xmp")
    assert result.exists()

    tree = etree.parse(str(result))
    root = tree.getroot()
    namespaces = {v: k for k, v in root.nsmap.items() if v}
    assert SIDECAR_NAMESPACE_CRS in namespaces


def test_write_for_decision_skip_when_disabled(tmp_path: Path) -> None:
    """When is_sidecars=False, returns None and writes no file."""
    source = tmp_path / "IMG_0002.jpg"
    source.write_bytes(b"fake-jpeg")
    decision = _make_decision(source)
    config = CullConfig(is_sidecars=False)
    sidecar_input = SidecarWriteInput(decision=decision, config=config)

    result = write_for_decision(sidecar_input)

    assert result is None
    assert not source.with_suffix(".xmp").exists()
