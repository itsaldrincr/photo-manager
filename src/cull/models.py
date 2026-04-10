"""Pydantic data models for all objects crossing stage boundaries."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from cull.config import SEARCH_TOP_K_DEFAULT


# ---------------------------------------------------------------------------
# Scoring sub-models (new expansion types)
# ---------------------------------------------------------------------------


class GeometryScore(BaseModel):
    """Geometric correction metrics for a photo."""

    tilt_degrees: float
    keystone_degrees: float
    confidence: float
    has_horizon: bool
    has_verticals: bool


class CompositionScore(BaseModel):
    """Composition quality metrics from IQA Stage 2."""

    thirds_alignment: float
    edge_clearance: float
    negative_space_balance: float
    topiq_iaa: float
    composite: float


class CropProposal(BaseModel):
    """Suggested crop bounding box with pixel coordinates."""

    top: int
    left: int
    bottom: int
    right: int
    source: Literal["smartcrop", "saliency_thirds"]


class SubjectBlurScore(BaseModel):
    """Subject-region blur measurement."""

    tenengrad: float
    subject_region_source: Literal["face", "saliency_peak", "global"]
    has_subject: bool


class TasteScore(BaseModel):
    """Aesthetic taste model output."""

    probability: float
    label_count_at_score: int
    weight_applied: float
    model_version: str


class ShootStatsScore(BaseModel):
    """Shoot-level statistical outlier scores."""

    palette_outlier_score: float
    exposure_drift_score: float
    exif_anomaly_score: float
    scene_start_bonus: float
    scene_id: int


class PeakMomentScore(BaseModel):
    """Peak-moment detection scores for portrait and action photos."""

    eyes_open_score: float
    smile_score: float
    gaze_score: float
    motion_peak_score: float
    peak_type: Literal["portrait", "action", "composite_fallback"]


# ---------------------------------------------------------------------------
# Stage 0 — Photo metadata (input)
# ---------------------------------------------------------------------------


class PhotoMeta(BaseModel):
    """File identity and optional EXIF metadata for a single photo."""

    path: Path
    filename: str
    exif_datetime: datetime | None = None
    exif_subsec: str | None = None
    fs_mtime: float | None = None


# ---------------------------------------------------------------------------
# Stage 1 — Classical filter outputs
# ---------------------------------------------------------------------------


class BlurScores(BaseModel):
    """Blur detection metrics from Stage 1."""

    tenengrad: float
    fft_ratio: float
    blur_tier: int = Field(ge=1, le=3)
    subject_sharpness: float | None = None
    background_sharpness: float | None = None
    is_bokeh: bool = False
    is_motion_blur: bool = False


class ExposureScores(BaseModel):
    """Exposure quality metrics from Stage 1."""

    dr_score: float
    clipping_highlight: float
    clipping_shadow: float
    midtone_pct: float
    color_cast_score: float
    has_highlight_clip: bool = False
    has_shadow_clip: bool = False
    has_color_cast: bool = False
    has_low_dr: bool = False


class BurstInfo(BaseModel):
    """Burst group membership for a photo."""

    group_id: int
    rank: int
    group_size: int
    is_burst_winner: bool = False


class Stage1Result(BaseModel):
    """Complete output of Stage 1 for a single photo."""

    photo_path: Path
    blur: BlurScores
    exposure: ExposureScores
    noise_score: float
    burst: BurstInfo | None = None
    is_duplicate: bool = False
    dhash: str | None = None
    is_pass: bool = True
    reject_reason: str | None = None
    geometry: GeometryScore | None = None


# ---------------------------------------------------------------------------
# Stage 2 — IQA scoring outputs
# ---------------------------------------------------------------------------


class PortraitScores(BaseModel):
    """Portrait-mode IQA metrics from Stage 2b."""

    eye_sharpness_left: float | None = None
    eye_sharpness_right: float | None = None
    ear_left: float | None = None
    ear_right: float | None = None
    is_eyes_closed: bool = False
    is_squinting: bool = False
    dominant_emotion: str | None = None
    is_face_occluded: bool = False
    face_occlusion_ratio: float | None = None


class Stage2Result(BaseModel):
    """Complete output of Stage 2 for a single photo."""

    photo_path: Path
    topiq: float
    laion_aesthetic: float
    clipiqa: float
    composite: float
    portrait: PortraitScores | None = None
    preset_used: str = "general"
    composition: CompositionScore | None = None
    subject_blur: SubjectBlurScore | None = None
    taste: TasteScore | None = None
    shoot_stats: ShootStatsScore | None = None
    crop: CropProposal | None = None


# ---------------------------------------------------------------------------
# Stage 3 — VLM tiebreaker outputs
# ---------------------------------------------------------------------------


class Stage3Result(BaseModel):
    """Parsed VLM response from Stage 3."""

    photo_path: Path
    sharpness: float | None = None
    exposure: float | None = None
    composition: float | None = None
    expression: float | None = None
    is_keeper: bool | None = None
    confidence: float = 0.0
    flags: list[str] = Field(default_factory=list)
    model_used: str = ""
    is_parse_error: bool = False


# ---------------------------------------------------------------------------
# Final decision
# ---------------------------------------------------------------------------

DecisionLabel = Literal["keeper", "rejected", "duplicate", "uncertain", "select"]


class PhotoDecision(BaseModel):
    """Final per-photo culling decision with full stage audit trail."""

    photo: PhotoMeta
    decision: DecisionLabel
    destination: Path | None = None
    stage1: Stage1Result | None = None
    stage2: Stage2Result | None = None
    stage3: Stage3Result | None = None
    is_override: bool = False
    override_from: DecisionLabel | None = None
    override_by: str | None = None
    stage_reached: int = 1


# ---------------------------------------------------------------------------
# Stage 4 — Curation outputs
# ---------------------------------------------------------------------------


class CuratorSelection(BaseModel):
    """A single photo chosen by the Stage 4 curator."""

    path: Path
    cluster_id: int
    cluster_size: int
    composite: float
    is_vlm_winner: bool
    reason: str | None = None


class CurationResult(BaseModel):
    """Aggregate result of a Stage 4 curation pass."""

    is_enabled: bool
    target_count: int
    actual_count: int
    cluster_count: int
    vlm_tiebreakers: int
    threshold_used: float
    elapsed_seconds: float
    selected: list[CuratorSelection]
    narrative_flow_score: float | None = None


# ---------------------------------------------------------------------------
# Session-level summary
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Feature Extensions
# ---------------------------------------------------------------------------


class RejectBreakdown(BaseModel):
    """Percentage breakdown of reject reasons for a session."""

    blur_pct: float = 0.0
    exposure_pct: float = 0.0
    noise_pct: float = 0.0
    burst_pct: float = 0.0
    duplicate_pct: float = 0.0
    vlm_pct: float = 0.0


class ExifPattern(BaseModel):
    """Typical EXIF settings for a group of photos."""

    aperture: str | None = None
    shutter: str | None = None
    iso: int | None = None
    lens: str | None = None
    focal_length_mm: float | None = None


class ExifPatterns(BaseModel):
    """EXIF patterns observed across keepers and rejects."""

    keepers_typical: ExifPattern = Field(default_factory=ExifPattern)
    rejects_typical: ExifPattern = Field(default_factory=ExifPattern)
    motion_risk_pct: float = 0.0


class PortraitStats(BaseModel):
    """Aggregate portrait-quality metrics for a session."""

    unique_subjects: int = 0
    eye_sharpness_median: float = 0.0
    eyes_closed_rate: float = 0.0


class SessionTiming2(BaseModel):
    """Timing statistics for a cull session."""

    duration_seconds: float = 0.0
    photos_per_minute: float = 0.0
    first_capture: datetime | None = None
    last_capture: datetime | None = None


class OverrideEntry(BaseModel):
    """A single user override event recorded to the override log."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    photo_path: str
    filename: str
    original_decision: DecisionLabel
    user_decision: DecisionLabel
    stage1_scores: dict[str, float] = Field(default_factory=dict)
    stage2_composite: float | None = None
    stage3_result: dict | None = None
    session_source: str
    override_origin: str
    stage2_composition: CompositionScore | None = None
    stage2_taste: TasteScore | None = None
    stage2_subject_blur: SubjectBlurScore | None = None
    tilt_degrees: float | None = None
    keystone_degrees: float | None = None
    stage2_shoot_outliers: ShootStatsScore | None = None


class SearchResult(BaseModel):
    """A single result from a semantic search query."""

    path: Path
    similarity: float
    rank: int


class SearchRequest(BaseModel):
    """Parameters for a semantic search operation."""

    query_text: str | None = None
    reference_path: Path | None = None
    source: Path
    top_k: int = SEARCH_TOP_K_DEFAULT


class ExplainRequest(BaseModel):
    """Input bundle for a photo explanation request."""

    image_path: Path
    stage1_result: Stage1Result | None = None
    stage2_composite: float | None = None
    stage3_result: Stage3Result | None = None
    model: str


class ExplainResult(BaseModel):
    """Structured explanation of a culling decision from the VLM."""

    photo_path: Path
    explanation: str = ""
    weaknesses: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    is_parse_error: bool = False
    model_used: str = ""


class ReportCard(BaseModel):
    """Session-level performance report card."""

    source_path: str
    keep_rate: float
    keep_count: int
    reject_count: int
    breakdown: RejectBreakdown = Field(default_factory=RejectBreakdown)
    exif_patterns: ExifPatterns = Field(default_factory=ExifPatterns)
    portrait_stats: PortraitStats | None = None
    timing: SessionTiming2 = Field(default_factory=SessionTiming2)
    advice: list[str] = Field(default_factory=list)
