"""Event preset: rating parse, cheap score, routing, stack representatives, report compatibility."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cull.cli_review import _load_session_from_report
from cull.config import EVENT_VLM_ALIAS, VLM_DEFAULT_ALIAS, CullConfig
from cull.models import (
    BlurScores,
    BurstInfo,
    CompositionScore,
    ExposureScores,
    PortraitScores,
    Stage1Result,
    Stage2Result,
    Stage3Result,
)
from cull.pipeline import SessionResult, _build_all_decisions, _DecisionCtx, _Stage1Output, _Stage2Output
from cull._pipeline.event_routing import EventRouteInput, collect_event_scores, label_by_rank, route_event
from cull._pipeline.orchestrator import _needs_vlm
from cull._pipeline.stack_resolution import resolve_event_stacks
from cull._pipeline.stage3_runner import _event_queue
from cull.stage2.fusion import FusionResult
from cull.stage3.event_prompt import EVENT_PROMPT, parse_event_response
from cull.stage3.event_score import cheap_prob, event_score, rating_label
from cull.stage3.event_scoring import rate_photo
from cull.stage3.prompt import PromptContext
from cull.stage3.vlm_scoring import VlmRequest, VlmScoreCallInput
from cull.vlm_registry import VLMEntry
from cull.vlm_session import VlmSession

GOOD_REPLY = '{"rating": 5, "keep": true, "flags": ["great_expression", "connection"]}'
JUNK_REPLY = "I think this photo is lovely."
TOLERANCE = 1e-9


class FakeVlmSession(VlmSession):
    """VlmSession whose generate() returns canned replies and records prompts."""

    _replies: list[str] = []
    _prompts: list[str] = []

    def configure(self, replies: list[str]) -> None:
        """Set the canned reply queue."""
        object.__setattr__(self, "_replies", list(replies))
        object.__setattr__(self, "_prompts", [])

    def generate(self, call_in: Any) -> str:  # noqa: ANN401
        """Return the next canned reply."""
        self._prompts.append(call_in.prompt)
        index = min(len(self._prompts) - 1, len(self._replies) - 1)
        return self._replies[index]


def _fake_session(replies: list[str]) -> FakeVlmSession:
    """Return a FakeVlmSession primed with replies."""
    entry = VLMEntry(alias="fake", directory=Path(tempfile.gettempdir()), display_name="Fake")
    session = FakeVlmSession(entry=entry)
    session.configure(replies)
    return session


def _call_in(session: FakeVlmSession, image: Path) -> VlmScoreCallInput:
    """Return a scoring call bundle for one image."""
    request = VlmRequest(image_path=image, context=PromptContext(), model="fake")
    return VlmScoreCallInput(request=request, session=session)


# ---------------------------------------------------------------------------
# Prompt and parser
# ---------------------------------------------------------------------------


def test_prompt_is_generic_and_keeps_rejection_criteria() -> None:
    """The prompt names no singles mixer and keeps the measured criteria word for word."""
    assert "singles mixer" not in EVENT_PROMPT
    assert "an event for the client" in EVENT_PROMPT
    assert "closed or half-closed eyes\non a main subject (a squint from genuine laughter is fine)" in EVENT_PROMPT
    assert '"obstructed_fg","cluttered","no_clear_subject","great_expression","connection"' in EVENT_PROMPT


def test_parse_good_reply() -> None:
    """A well-formed reply yields rating and flags."""
    result = parse_event_response(f"Sure:\n{GOOD_REPLY}\n")
    assert result.is_parse_error is False
    assert result.rating == 5
    assert result.flags == ["great_expression", "connection"]


@pytest.mark.parametrize("reply", [JUNK_REPLY, "{not json", '{"keep": true}', "[5]"])
def test_parse_junk_is_parse_error(reply: str) -> None:
    """Replies without a readable rating are parse errors."""
    result = parse_event_response(reply)
    assert result.is_parse_error is True
    assert result.rating is None


@pytest.mark.parametrize("rating", ["0", "6", "4.5", "true", '"high"', "null", "-1"])
def test_parse_out_of_range_or_non_integer_rating(rating: str) -> None:
    """Ratings outside 1-5, fractional, boolean or non-numeric are rejected."""
    assert parse_event_response(f'{{"rating": {rating}, "keep": true}}').is_parse_error is True


@pytest.mark.parametrize(("rating", "expected"), [('"4"', 4), ("3.0", 3), ("1", 1)])
def test_parse_accepts_integral_ratings(rating: str, expected: int) -> None:
    """Numeric strings and whole floats are read as ints."""
    assert parse_event_response(f'{{"rating": {rating}}}').rating == expected


def test_parse_drops_unknown_flags() -> None:
    """Flags outside the prompt's list are dropped."""
    result = parse_event_response('{"rating": 2, "flags": ["blur", "sunset", 3]}')
    assert result.flags == ["blur"]


def test_rate_photo_retries_junk_then_reads_rating(tmp_path: Path) -> None:
    """A junk reply retries; the next good reply sets rating, flags and keep verdict."""
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    session = _fake_session([JUNK_REPLY, GOOD_REPLY])
    with patch("cull.stage3.vlm_scoring.time.sleep"):
        result = rate_photo(_call_in(session, image))
    assert result.rating == 5
    assert result.is_keeper is True
    assert result.photo_path == image
    assert session._prompts == [EVENT_PROMPT, EVENT_PROMPT]


def test_rate_photo_marks_parse_error_after_retries(tmp_path: Path) -> None:
    """After every retry fails the result is a parse error with no rating."""
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    with patch("cull.stage3.vlm_scoring.time.sleep"):
        result = rate_photo(_call_in(_fake_session([JUNK_REPLY]), image))
    assert result.is_parse_error is True
    assert result.rating is None


# ---------------------------------------------------------------------------
# Cheap probability and event score
# ---------------------------------------------------------------------------


def _stage2(path: Path, **overrides: Any) -> Stage2Result:  # noqa: ANN401
    """Return a Stage2Result with fixed IQA scores."""
    values: dict[str, Any] = {"topiq": 0.60, "clipiqa": 0.65, "laion_aesthetic": 0.50, "composite": 0.5}
    values.update(overrides)
    return Stage2Result(photo_path=path, **values)


def test_cheap_prob_matches_hand_computed_value() -> None:
    """The logistic reproduces a value worked out by hand from the fitted constants.

    logit = -1.143 + 0.242*(0.60-0.4915)/0.0749 + 0.047*(0.65-0.6062)/0.0535
          + 0.131*(0.50-0.4607)/0.0274 - 0.063*(1.0-1.2097)/0.3717
          + 0.727*(0.5-0.1619)/0.3286 + 0.118*(0.2-0.0529)/0.1318
          - 0.53*(1-0.4406)/0.4965 - 0.08*(2000-1085.8869)/1528.8111
          - 0.347*(0-0.1107)/0.3137 = -0.173335; sigmoid = 0.456774.
    """
    stage2 = _stage2(
        Path("a.jpg"),
        composition=CompositionScore(
            thirds_alignment=0, edge_clearance=0, negative_space_balance=0, topiq_iaa=0, composite=1.0,
        ),
        portrait=PortraitScores(valence=0.5, arousal=0.2, eye_sharpness_left=2000.0),
    )
    assert cheap_prob(stage2) == pytest.approx(0.45677439584736096, abs=TOLERANCE)


def test_cheap_prob_counts_missing_values_as_zero() -> None:
    """No composition and no face: those features are 0 before standardising (logit -0.116982)."""
    assert cheap_prob(_stage2(Path("a.jpg"))) == pytest.approx(0.470787775457477, abs=TOLERANCE)


def test_event_score_is_rating_plus_tiebreak() -> None:
    """A rating dominates; cheap_prob adds 0.01 of itself; no rating means cheap_prob alone."""
    stage2 = _stage2(Path("a.jpg"))
    prob = cheap_prob(stage2)
    rated = Stage3Result(photo_path=Path("a.jpg"), rating=4)
    failed = Stage3Result(photo_path=Path("a.jpg"), is_parse_error=True)
    assert event_score(stage2, rated) == pytest.approx(4 + 0.01 * prob)
    assert event_score(stage2, failed) == pytest.approx(prob)
    assert event_score(stage2, None) == pytest.approx(prob)


# ---------------------------------------------------------------------------
# Routing and stack representatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("rating", "label"), [(5, "keeper"), (4, "uncertain"), (3, "rejected"), (1, "rejected")])
def test_rating_thresholds(rating: int, label: str) -> None:
    """5 keeps, 4 goes to human review, 3 or less rejects."""
    assert rating_label(rating) == label


def test_no_vlm_rank_fractions() -> None:
    """Top 20% keep, next 25% review, the rest reject."""
    keys = [f"p{i:02d}" for i in range(20)]
    labels = label_by_rank(keys)
    assert [labels[k] for k in keys].count("keeper") == 4
    assert [labels[k] for k in keys[4:9]] == ["uncertain"] * 5
    assert labels["p09"] == "rejected"


def _s1(path: Path, tenengrad: float) -> Stage1Result:
    """Return a passing Stage1Result."""
    return Stage1Result(
        photo_path=path,
        blur=BlurScores(tenengrad=tenengrad, fft_ratio=0.5, blur_tier=1),
        exposure=ExposureScores(
            dr_score=0.5, clipping_highlight=0.0, clipping_shadow=0.0, midtone_pct=0.5, color_cast_score=0.0,
        ),
        noise_score=0.0,
    )


STACK = ("best_composite.jpg", "vlm_hero.jpg", "sharpest.jpg")
LONE = ("lone_keep.jpg", "lone_review.jpg", "lone_reject.jpg")
RATINGS = {"best_composite.jpg": 3, "vlm_hero.jpg": 5, "sharpest.jpg": 2,
           "lone_keep.jpg": 5, "lone_review.jpg": 4, "lone_reject.jpg": 3}
COMPOSITES = {"best_composite.jpg": 0.99, "vlm_hero.jpg": 0.40, "sharpest.jpg": 0.60}


def _event_outputs(tmp_path: Path) -> tuple[list[Path], EventRouteInput]:
    """Return one three-frame stack plus three singletons, all rated."""
    paths = [tmp_path / name for name in (*STACK, *LONE)]
    for path in paths:
        path.write_bytes(b"")
    s1_out = _Stage1Output(
        results={str(p): _s1(p, 900.0 if p.name == "sharpest.jpg" else 100.0) for p in paths},
        survivors=paths, stacks=[sorted(str(tmp_path / n) for n in STACK)],
    )
    for name in STACK:
        s1_out.results[str(tmp_path / name)].burst = BurstInfo(group_id=0, rank=0, group_size=3)
    s2_out = _Stage2Output(results={
        str(p): FusionResult(stage2=_stage2(p, composite=COMPOSITES.get(p.name, 0.5)), routing="AMBIGUOUS")
        for p in paths
    })
    s3 = {str(p): Stage3Result(photo_path=p, rating=RATINGS[p.name]) for p in paths}
    return paths, EventRouteInput(s1_out=s1_out, s2_out=s2_out, s3_results=s3)


def test_representative_is_highest_event_score(tmp_path: Path) -> None:
    """The VLM's rating-5 frame leads the stack, not the best composite or the sharpest."""
    _, route_in = _event_outputs(tmp_path)
    resolve_event_stacks(route_in.s1_out, collect_event_scores(route_in))
    hero = route_in.s1_out.results[str(tmp_path / "vlm_hero.jpg")]
    assert hero.burst.is_burst_winner is True
    assert route_in.s1_out.duplicate_paths == {
        str(tmp_path / "best_composite.jpg"), str(tmp_path / "sharpest.jpg"),
    }


def test_event_decisions_route_by_rating(tmp_path: Path) -> None:
    """Representatives route by rating; non-representatives stay duplicates with their scores."""
    paths, route_in = _event_outputs(tmp_path)
    resolve_event_stacks(route_in.s1_out, collect_event_scores(route_in))
    dec_ctx = _DecisionCtx(
        paths=paths, s1_out=route_in.s1_out, s2_out=route_in.s2_out,
        s3_results=route_in.s3_results, event_labels=route_event(route_in),
    )
    by_name = {d.photo.filename: d for d in _build_all_decisions(dec_ctx)}
    assert by_name["vlm_hero.jpg"].decision == "keeper"
    assert by_name["lone_keep.jpg"].decision == "keeper"
    assert by_name["lone_review.jpg"].decision == "uncertain"
    assert by_name["lone_reject.jpg"].decision == "rejected"
    assert by_name["best_composite.jpg"].decision == "duplicate"
    assert by_name["best_composite.jpg"].stage3.rating == 3


def test_parse_error_representative_routes_by_cheap_rank(tmp_path: Path) -> None:
    """A representative whose rating failed routes by cheap_prob rank instead of by rating."""
    _, route_in = _event_outputs(tmp_path)
    for key in route_in.s3_results:
        route_in.s3_results[key] = Stage3Result(photo_path=Path(key), is_parse_error=True)
    resolve_event_stacks(route_in.s1_out, collect_event_scores(route_in))
    labels = route_event(route_in)
    assert len(labels) == 4
    assert sorted(labels.values()).count("keeper") == 1


def test_event_queue_rates_every_stack_member(tmp_path: Path) -> None:
    """Stage 3 in event mode sends every Stage-2-scored photo to the VLM."""
    paths, route_in = _event_outputs(tmp_path)
    assert _event_queue(route_in.s2_out) == sorted(paths)


# ---------------------------------------------------------------------------
# Config, VLM lifecycle, reports
# ---------------------------------------------------------------------------


def test_event_model_defaults() -> None:
    """--preset event defaults to EVENT_VLM_ALIAS; an explicit --model wins."""
    assert CullConfig(preset="event").model == EVENT_VLM_ALIAS
    assert CullConfig(preset="event", model=None).model == EVENT_VLM_ALIAS
    assert CullConfig(preset="event", model="gemma-4-12b").model == "gemma-4-12b"
    assert CullConfig(preset="wedding", model=None).model == VLM_DEFAULT_ALIAS


def test_event_no_vlm_curation_loads_no_vlm() -> None:
    """The event curator never calls the VLM, so --no-vlm --curate loads none."""
    assert _needs_vlm(CullConfig(preset="event", stages=[1, 2], curate_target=10)) is False
    assert _needs_vlm(CullConfig(preset="general", stages=[1, 2], curate_target=10)) is True


def test_old_report_without_rating_still_loads(tmp_path: Path) -> None:
    """A report written before the rating field parses, with rating None."""
    image = tmp_path / "a.jpg"
    image.write_bytes(b"")
    s1_out = _Stage1Output(results={str(image): _s1(image, 1.0)}, survivors=[image])
    s3 = {str(image): Stage3Result(photo_path=image, is_keeper=True, confidence=0.9)}
    decisions = _build_all_decisions(_DecisionCtx(paths=[image], s1_out=s1_out, s3_results=s3))
    data = json.loads(SessionResult(source_path=str(tmp_path), decisions=decisions).model_dump_json())
    data["decisions"][0]["stage3"].pop("rating")
    report = tmp_path / "session_report.json"
    report.write_text(json.dumps(data), encoding="utf-8")
    loaded = _load_session_from_report(report)
    assert loaded.decisions[0].stage3.rating is None
    assert loaded.decisions[0].stage3.is_keeper is True


def test_event_run_picks_representative_after_stage3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """In an event run, stacks resolve after Stage 3 rates every member, by event_score."""
    from cull._pipeline import orchestrator  # noqa: PLC0415

    paths, route_in = _event_outputs(tmp_path)
    monkeypatch.setattr(orchestrator, "_run_s1", lambda ctx: route_in.s1_out)
    monkeypatch.setattr(orchestrator, "_run_s2", lambda run_in: route_in.s2_out)
    monkeypatch.setattr(orchestrator, "_run_s2_reducer", lambda run_in: None)
    monkeypatch.setattr(orchestrator, "_unload_stage2_models", lambda: None)
    monkeypatch.setattr(orchestrator, "_run_s3_if_configured", lambda run_in: route_in.s3_results)
    ctx = orchestrator._StageRunCtx(config=CullConfig(preset="event"), paths=paths)
    stages = orchestrator._execute_stages_inline(ctx)
    assert stages.s1_out.results[str(tmp_path / "vlm_hero.jpg")].burst.is_burst_winner is True
    assert str(tmp_path / "best_composite.jpg") in stages.s1_out.duplicate_paths
