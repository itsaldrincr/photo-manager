"""Escape-sequence and geometry tests for the kitty placeholder protocol helpers."""

from __future__ import annotations

import base64
import re
from pathlib import Path

from cull.tui import kitty
from cull.tui.kitty import CellBox, CellSize, DirectTransmit, FitRequest, PlaceholderRow

CELL: CellSize = CellSize(width=10.0, height=20.0)
APC_RE = re.compile(r"\x1b_G([^;\x1b]*)(?:;([^\x1b]*))?\x1b\\")


def _keys(sequence: str) -> list[dict[str, str]]:
    """Parse every APC command in ``sequence`` into its key=value map."""
    return [dict(pair.split("=") for pair in m.group(1).split(",")) for m in APC_RE.finditer(sequence)]


def test_file_transmit_is_store_only() -> None:
    """Upload uses a=t (store only), never a=T (store and display at the cursor)."""
    sequence = kitty.transmit_file_sequence(7, Path("/tmp/x.png"))
    (keys,) = _keys(sequence)
    assert keys["a"] == "t"
    assert keys["t"] == "f"
    assert keys["i"] == "7"
    assert "a=T" not in sequence
    payload = APC_RE.search(sequence).group(2)
    assert base64.standard_b64decode(payload).decode() == "/tmp/x.png"


def test_direct_transmit_is_store_only_and_chunked() -> None:
    """Inline upload: first chunk carries a=t,t=d; chunks reassemble to the PNG bytes."""
    png = bytes(range(256)) * 40
    sequence = kitty.transmit_direct_sequence(DirectTransmit(image_id=3, png_bytes=png))
    commands = _keys(sequence)
    assert commands[0]["a"] == "t" and commands[0]["t"] == "d"
    assert [c["m"] for c in commands] == ["1"] * (len(commands) - 1) + ["0"]
    assert "a=T" not in sequence
    joined = "".join(m.group(2) for m in APC_RE.finditer(sequence))
    assert base64.standard_b64decode(joined) == png


def test_virtual_placement_uses_unicode_placeholders() -> None:
    """Placement is virtual (U=1) and sized in cells."""
    (keys,) = _keys(kitty.virtual_placement_sequence(9, CellBox(cols=40, rows=12)))
    assert keys == {"a": "p", "U": "1", "i": "9", "c": "40", "r": "12", "q": "2"}


def test_delete_frees_image_and_placements() -> None:
    """Delete uses d=I so stored data and placements both go."""
    (keys,) = _keys(kitty.delete_image_sequence(5))
    assert keys["a"] == "d" and keys["d"] == "I" and keys["i"] == "5"


def test_fit_landscape_keeps_aspect() -> None:
    """A 3:2 frame in a 100x50 cell box (1000x1000 px) fills the width, not the height."""
    box = kitty.fit_cells(FitRequest(image_width=3000, image_height=2000, box=CellBox(cols=100, rows=50), cell=CELL))
    assert box == CellBox(cols=100, rows=33)


def test_fit_portrait_keeps_aspect() -> None:
    """A 2:3 frame in the same box fills the height and narrows."""
    box = kitty.fit_cells(FitRequest(image_width=2000, image_height=3000, box=CellBox(cols=100, rows=50), cell=CELL))
    assert box == CellBox(cols=67, rows=50)


def test_fit_is_clamped_to_box_and_diacritic_limit() -> None:
    """Fit never exceeds the box nor the 297 encodable rows/columns."""
    box = kitty.fit_cells(FitRequest(image_width=10, image_height=10, box=CellBox(cols=500, rows=500), cell=CellSize(width=1, height=1)))
    assert box.cols <= kitty.MAX_PLACEHOLDER_CELLS and box.rows <= kitty.MAX_PLACEHOLDER_CELLS


def test_diacritic_table_matches_kitty() -> None:
    """297 row/column diacritics, starting U+0305, U+030D, U+030E, ending U+1D244."""
    table = kitty.ROW_COLUMN_DIACRITICS
    assert len(table) == 297
    assert [ord(c) for c in table[:3]] == [0x0305, 0x030D, 0x030E]
    assert ord(table[-1]) == 0x1D244


def test_placeholder_text_encodes_row_and_column() -> None:
    """Each cell is U+10EEEE + row mark + column mark (three code points)."""
    text = kitty.placeholder_text(PlaceholderRow(row=2, cols=4))
    assert len(text) == 12
    for column in range(4):
        cell = text[column * 3:column * 3 + 3]
        assert cell == kitty.PLACEHOLDER + kitty.ROW_COLUMN_DIACRITICS[2] + kitty.ROW_COLUMN_DIACRITICS[column]


def test_placeholder_style_encodes_image_id_in_24_bit_foreground() -> None:
    """The foreground colour's RGB is the image id."""
    assert kitty.placeholder_style(0x010203).color.triplet == (1, 2, 3)
