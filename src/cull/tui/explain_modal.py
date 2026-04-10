"""Modal screen that runs VLM explanation off-thread and renders the result."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.panel import Panel
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from cull.models import ExplainRequest, ExplainResult
from cull.stage3.vlm_explain import (
    EXPLAIN_MAX_TOKENS,
    EXPLAIN_TEMPERATURE,
    ExplainCallInput,
    _build_explain_prompt,
    _parse_explain_response,
    explain_photo,
)
from cull.vlm_session import VlmGenerateInput, vlm_session

logger = logging.getLogger(__name__)

LOADING_MESSAGE: str = "[dim]Analyzing photo...[/]"
ESCAPE_BINDING_LABEL: str = "Close"
QUIT_BINDING_LABEL: str = "Close"
CONTENT_WIDGET_ID: str = "explain-content"
CONTAINER_WIDGET_ID: str = "explain-container"
EMPTY_PLACEHOLDER: str = "(none noted)"
ERROR_TITLE: str = "Error"
ERROR_BODY_MARKUP: str = "[red]Unable to parse VLM response.[/]"
ERROR_BORDER_STYLE: str = "red"
RESULT_BORDER_STYLE: str = "bright_blue"
EXPLAIN_PANEL_ID: str = "explain-panel"
STRENGTHS_HEADER: str = "[bold green]Strengths[/]"
WEAKNESSES_HEADER: str = "[bold red]Weaknesses[/]"
SUMMARY_HEADER: str = "[bold bright_white]Summary[/]"
STRENGTH_BULLET: str = "  [green]●[/]"
WEAKNESS_BULLET: str = "  [red]○[/]"


def _bullet_lines(items: list[str], bullet: str) -> list[str]:
    """Render a list of items as bullet-prefixed markup lines."""
    source = items if items else [EMPTY_PLACEHOLDER]
    return [f"{bullet} {item}" for item in source]


def _result_body_lines(result: ExplainResult) -> list[str]:
    """Build the markup lines for the result body."""
    lines: list[str] = [STRENGTHS_HEADER]
    lines.extend(_bullet_lines(result.strengths, STRENGTH_BULLET))
    lines.append("")
    lines.append(WEAKNESSES_HEADER)
    lines.extend(_bullet_lines(result.weaknesses, WEAKNESS_BULLET))
    lines.append("")
    lines.append(SUMMARY_HEADER)
    lines.append(f"  {result.explanation}")
    lines.append("")
    lines.append(f"[dim]confidence: {result.confidence:.2f}[/]")
    return lines


def _build_error_panel(message: str) -> Panel:
    """Return the panel shown when the VLM response fails to parse."""
    body = Text(message)
    return Panel(body, title=ERROR_TITLE, border_style=ERROR_BORDER_STYLE)


def _build_result_panel(result: ExplainResult) -> Panel:
    """Build a Rich Panel displaying strengths, weaknesses, and summary."""
    if result.is_parse_error:
        msg = result.explanation if result.explanation else ERROR_BODY_MARKUP
        return _build_error_panel(msg)
    if not result.explanation and not result.strengths and not result.weaknesses:
        return _build_error_panel(
            f"[yellow]VLM returned empty response[/]\n[dim]model: {result.model_used}[/]"
        )
    body = Text.from_markup("\n".join(_result_body_lines(result)))
    return Panel(body, title=result.photo_path.name, border_style=RESULT_BORDER_STYLE)


def fetch_explanation_result(request: ExplainRequest) -> ExplainResult:
    """Run the blocking explain request and return a parsed result."""
    raw_text: str = ""
    try:
        with vlm_session(request.model) as session:
            prompt = _build_explain_prompt(request)
            gen_in = VlmGenerateInput(
                prompt=prompt,
                images=[request.image_path],
                max_tokens=EXPLAIN_MAX_TOKENS,
                temperature=EXPLAIN_TEMPERATURE,
            )
            raw_text = session.generate(gen_in)
            logger.debug("Raw VLM response for %s: %s", request.image_path, raw_text)
            result = _parse_explain_response(raw_text, request.image_path)
            result.model_used = request.model
    except Exception as exc:
        logger.debug("Exception in explain fetch: %s | raw: %s", exc, raw_text)
        result = ExplainResult(
            photo_path=request.image_path,
            is_parse_error=True,
            explanation=f"Explain failed: {exc}\n\nRaw output:\n{raw_text[:500]}",
            model_used=request.model,
        )
    if result.is_parse_error and not result.explanation:
        result.explanation = f"Parse failed (empty).\n\nRaw output:\n{raw_text[:500]}"
    return result


class ExplainPanel(Static):
    """Bottom-docked explain panel that keeps the photo visible above it."""

    DEFAULT_CSS = f"""
    ExplainPanel {{
        height: 16;
        width: 1fr;
        border: round $accent;
        background: $panel 92%;
        display: none;
    }}

    ExplainPanel.visible {{
        display: block;
    }}

    ExplainPanel > #{CONTENT_WIDGET_ID} {{
        width: 1fr;
        height: 1fr;
        overflow-y: auto;
    }}
    """

    def __init__(self) -> None:
        super().__init__(id=EXPLAIN_PANEL_ID)
        self._photo_path: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id=CONTENT_WIDGET_ID)

    @property
    def photo_path(self) -> str | None:
        return self._photo_path

    def show_loading(self, photo_path: str) -> None:
        """Show the panel with a loading placeholder."""
        self._photo_path = photo_path
        self.add_class("visible")
        self.query_one(f"#{CONTENT_WIDGET_ID}", Static).update(
            Panel(Text.from_markup(LOADING_MESSAGE), title=Path(photo_path).name, border_style=RESULT_BORDER_STYLE)
        )

    def show_result(self, result: ExplainResult) -> None:
        """Show the explain result."""
        self._photo_path = str(result.photo_path)
        self.add_class("visible")
        self.query_one(f"#{CONTENT_WIDGET_ID}", Static).update(_build_result_panel(result))

    def hide_panel(self) -> None:
        """Hide the panel and clear its contents."""
        self._photo_path = None
        self.remove_class("visible")
        self.query_one(f"#{CONTENT_WIDGET_ID}", Static).update("")


class ExplainModal(ModalScreen):
    """Modal overlay that runs VLM explanation off-thread and displays result."""

    DEFAULT_CSS = f"""
    ExplainModal {{
        align: center bottom;
        padding: 0 2 3 2;
        background: $surface 35%;
    }}

    ExplainModal:ansi {{
        background: $surface 35%;
    }}

    ExplainModal > #{CONTAINER_WIDGET_ID} {{
        width: 92%;
        max-width: 120;
        height: 52%;
        min-height: 12;
        max-height: 24;
        padding: 1 2 0 2;
        border: round $accent;
        background: $panel 92%;
    }}

    ExplainModal > #{CONTAINER_WIDGET_ID} > #{CONTENT_WIDGET_ID} {{
        width: 1fr;
        height: 1fr;
        overflow-y: auto;
    }}
    """

    BINDINGS = [
        Binding("escape", "dismiss", ESCAPE_BINDING_LABEL),
        Binding("q", "dismiss", QUIT_BINDING_LABEL),
    ]

    def __init__(self, request: ExplainRequest) -> None:
        super().__init__()
        self._request = request
        self._result: ExplainResult | None = None

    def compose(self) -> ComposeResult:
        """Build the modal layout with a loading placeholder."""
        yield Vertical(
            Static(LOADING_MESSAGE, id=CONTENT_WIDGET_ID),
            id=CONTAINER_WIDGET_ID,
        )

    def on_mount(self) -> None:
        """Kick off the VLM call in a background worker."""
        self.run_worker(self._fetch_explanation, thread=True, exclusive=True)

    def _fetch_explanation(self) -> None:
        """Run the blocking VLM call off the UI thread."""
        result = fetch_explanation_result(self._request)
        self.app.call_from_thread(self._show_result, result)

    def _show_result(self, result: ExplainResult) -> None:
        """Update the static widget with the rendered panel."""
        if not self.is_mounted:
            return
        self._result = result
        panel = _build_result_panel(result)
        self.query_one(f"#{CONTENT_WIDGET_ID}", Static).update(panel)
