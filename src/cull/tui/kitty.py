"""Kitty graphics protocol: escape sequences, placeholder cells, cell geometry.

Images are shown with Unicode-placeholder virtual placements (``U=1``). The
terminal stores the image and a virtual placement; Textual then paints
ordinary text cells (U+10EEEE plus row/column diacritics, foreground colour =
image id). Because the image lives in cells Textual owns, modals, panels and
screen switches clip or hide it like any other text, and nothing we write
has to race Textual's own output.
"""

from __future__ import annotations

import base64
import fcntl
import os
import struct
import sys
import termios
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from rich.color import Color
from rich.style import Style

PLACEHOLDER: str = "\U0010EEEE"
KITTY_CHUNK_SIZE: int = 4096
ESC_APC: str = "\x1b_G"
ESC_ST: str = "\x1b\\"

# Row/column diacritics from kitty's rowcolumn-diacritics.txt: Unicode 6.0
# marks of combining class 230 that appear in no canonical decomposition, so
# NFC normalisation cannot rewrite them. Index n encodes row/column n.
_DIACRITIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0305, 0x0305), (0x030D, 0x030E), (0x0310, 0x0310), (0x0312, 0x0312),
    (0x033D, 0x033F), (0x0346, 0x0346), (0x034A, 0x034C), (0x0350, 0x0352),
    (0x0357, 0x0357), (0x035B, 0x035B), (0x0363, 0x036F), (0x0483, 0x0487),
    (0x0592, 0x0595), (0x0597, 0x0599), (0x059C, 0x05A1), (0x05A8, 0x05A9),
    (0x05AB, 0x05AC), (0x05AF, 0x05AF), (0x05C4, 0x05C4), (0x0610, 0x0617),
    (0x0657, 0x065B), (0x065D, 0x065E), (0x06D6, 0x06DC), (0x06DF, 0x06E2),
    (0x06E4, 0x06E4), (0x06E7, 0x06E8), (0x06EB, 0x06EC), (0x0730, 0x0730),
    (0x0732, 0x0733), (0x0735, 0x0736), (0x073A, 0x073A), (0x073D, 0x073D),
    (0x073F, 0x0741), (0x0743, 0x0743), (0x0745, 0x0745), (0x0747, 0x0747),
    (0x0749, 0x074A), (0x07EB, 0x07F1), (0x07F3, 0x07F3), (0x0816, 0x0819),
    (0x081B, 0x0823), (0x0825, 0x0827), (0x0829, 0x082D), (0x0951, 0x0951),
    (0x0953, 0x0954), (0x0F82, 0x0F83), (0x0F86, 0x0F87), (0x135D, 0x135F),
    (0x17DD, 0x17DD), (0x193A, 0x193A), (0x1A17, 0x1A17), (0x1A75, 0x1A7C),
    (0x1B6B, 0x1B6B), (0x1B6D, 0x1B73), (0x1CD0, 0x1CD2), (0x1CDA, 0x1CDB),
    (0x1CE0, 0x1CE0), (0x1DC0, 0x1DC1), (0x1DC3, 0x1DC9), (0x1DCB, 0x1DCC),
    (0x1DD1, 0x1DE6), (0x1DFE, 0x1DFE), (0x20D0, 0x20D1), (0x20D4, 0x20D7),
    (0x20DB, 0x20DC), (0x20E1, 0x20E1), (0x20E7, 0x20E7), (0x20E9, 0x20E9),
    (0x20F0, 0x20F0), (0x2CEF, 0x2CF1), (0x2DE0, 0x2DFF), (0xA66F, 0xA66F),
    (0xA67C, 0xA67D), (0xA6F0, 0xA6F1), (0xA8E0, 0xA8F1), (0xAAB0, 0xAAB0),
    (0xAAB2, 0xAAB3), (0xAAB7, 0xAAB8), (0xAABE, 0xAABF), (0xAAC1, 0xAAC1),
    (0xFE20, 0xFE26), (0x10A0F, 0x10A0F), (0x10A38, 0x10A38),
    (0x1D185, 0x1D189), (0x1D1AA, 0x1D1AD), (0x1D242, 0x1D244),
)
ROW_COLUMN_DIACRITICS: tuple[str, ...] = tuple(
    chr(code) for low, high in _DIACRITIC_RANGES for code in range(low, high + 1)
)
MAX_PLACEHOLDER_CELLS: int = len(ROW_COLUMN_DIACRITICS)
# Ids stay below 2**16 so the red channel is always 0. Textual's modal tint
# blends every foreground towards the backdrop, which lifts red above 0 and
# turns covered placeholders into ids the terminal does not know: hidden.
MAX_IMAGE_ID: int = 0xFFFF

FALLBACK_CELL_WIDTH_PX: float = 10.0
FALLBACK_CELL_HEIGHT_PX: float = 20.0
SSH_ENV_VARS: tuple[str, ...] = ("SSH_TTY", "SSH_CONNECTION", "SSH_CLIENT")
TRANSMIT_ENV_VAR: str = "CULL_TUI_TRANSMIT"


class CellSize(BaseModel):
    """Pixel size of one terminal cell."""

    model_config = ConfigDict(frozen=True)

    width: float
    height: float


class CellBox(BaseModel):
    """A rectangle measured in terminal cells."""

    model_config = ConfigDict(frozen=True)

    cols: int
    rows: int


class FitRequest(BaseModel):
    """Image pixel size to fit inside a cell box."""

    model_config = ConfigDict(frozen=True)

    image_width: int
    image_height: int
    box: CellBox
    cell: CellSize


class DirectTransmit(BaseModel):
    """PNG bytes to send inline (``t=d``) under one image id."""

    image_id: int
    png_bytes: bytes


def query_cell_size() -> CellSize:
    """Return the cell pixel size from TIOCGWINSZ, or a fallback when unknown."""
    try:
        packed = fcntl.ioctl(sys.__stdout__.fileno(), termios.TIOCGWINSZ, bytes(8))
    except (OSError, AttributeError, ValueError):
        return CellSize(width=FALLBACK_CELL_WIDTH_PX, height=FALLBACK_CELL_HEIGHT_PX)
    rows, cols, x_pixels, y_pixels = struct.unpack("HHHH", packed)
    if not (rows and cols and x_pixels and y_pixels):
        return CellSize(width=FALLBACK_CELL_WIDTH_PX, height=FALLBACK_CELL_HEIGHT_PX)
    return CellSize(width=x_pixels / cols, height=y_pixels / rows)


def prefers_file_transmit() -> bool:
    """Return True when the terminal can read our cache files directly (``t=f``).

    Over SSH the terminal runs on another machine and cannot see our paths.
    """
    forced = os.environ.get(TRANSMIT_ENV_VAR, "").lower()
    if forced in ("file", "direct"):
        return forced == "file"
    return not any(os.environ.get(name) for name in SSH_ENV_VARS)


def fit_cells(request: FitRequest) -> CellBox:
    """Return the largest cell box inside ``request.box`` that keeps the image aspect."""
    box, cell = request.box, request.cell
    scale = min(
        box.cols * cell.width / request.image_width,
        box.rows * cell.height / request.image_height,
    )
    cols = round(request.image_width * scale / cell.width)
    rows = round(request.image_height * scale / cell.height)
    return CellBox(
        cols=max(1, min(cols, box.cols, MAX_PLACEHOLDER_CELLS)),
        rows=max(1, min(rows, box.rows, MAX_PLACEHOLDER_CELLS)),
    )


def transmit_file_sequence(image_id: int, path: Path) -> str:
    """Build a store-only (``a=t``) transmit that makes the terminal read a PNG file."""
    payload = base64.standard_b64encode(str(path).encode("utf-8")).decode("ascii")
    return f"{ESC_APC}a=t,t=f,f=100,i={image_id},q=2;{payload}{ESC_ST}"


def transmit_direct_sequence(transmit: DirectTransmit) -> str:
    """Build chunked store-only (``a=t``) transmits carrying PNG bytes inline."""
    data = base64.standard_b64encode(transmit.png_bytes).decode("ascii")
    chunks = [data[i:i + KITTY_CHUNK_SIZE] for i in range(0, len(data), KITTY_CHUNK_SIZE)]
    parts: list[str] = []
    for index, chunk in enumerate(chunks):
        more = 1 if index < len(chunks) - 1 else 0
        keys = f"a=t,t=d,f=100,i={transmit.image_id},q=2,m={more}" if index == 0 else f"m={more},q=2"
        parts.append(f"{ESC_APC}{keys};{chunk}{ESC_ST}")
    return "".join(parts)


def virtual_placement_sequence(image_id: int, box: CellBox) -> str:
    """Build a virtual placement (``U=1``) that placeholder cells will draw from."""
    return f"{ESC_APC}a=p,U=1,i={image_id},c={box.cols},r={box.rows},q=2{ESC_ST}"


def delete_image_sequence(image_id: int) -> str:
    """Build a delete that frees the stored image and all its placements."""
    return f"{ESC_APC}a=d,d=I,i={image_id},q=2{ESC_ST}"


def placeholder_style(image_id: int) -> Style:
    """Return the style whose 24-bit foreground encodes ``image_id``."""
    return Style(color=Color.from_rgb((image_id >> 16) & 0xFF, (image_id >> 8) & 0xFF, image_id & 0xFF))


class PlaceholderRow(BaseModel):
    """One row of placeholder cells (the image id travels in the style, not the text)."""

    model_config = ConfigDict(frozen=True)

    row: int
    cols: int


@lru_cache(maxsize=4096)
def placeholder_text(spec: PlaceholderRow) -> str:
    """Return one row of placeholder cells as a single string.

    Every cell carries explicit row and column diacritics, so the terminal
    maps each cell to its tile even when a row is cut by an overlay. Each
    cell is exactly three code points (U+10EEEE, row mark, column mark).
    """
    row_mark = ROW_COLUMN_DIACRITICS[spec.row]
    return "".join(f"{PLACEHOLDER}{row_mark}{ROW_COLUMN_DIACRITICS[col]}" for col in range(spec.cols))
