"""Stdio silencer — Python-level redirect that leaves Rich Live untouched."""

from __future__ import annotations

import contextlib
import io
from typing import Iterator


@contextlib.contextmanager
def _silence_stdio() -> Iterator[None]:
    """Redirect stdout/stderr at Python level — Rich Live is unaffected."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield
