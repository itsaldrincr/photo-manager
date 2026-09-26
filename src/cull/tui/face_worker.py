"""Face-analysis worker process: one JSON request per stdin line, one report per stdout line.

Started by ``cull.tui.faces.FaceWorkerClient`` in its own session with
stderr on a log file. Native libraries that print to fd 1 would corrupt the
protocol, so the reply stream is a private dup of fd 1 and fd 1 is pointed
at the log (fd 2) before MediaPipe is imported.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def handle_line(line: str) -> str:
    """Answer one request line with one FaceReport JSON line."""
    from cull.tui import faces, previews  # noqa: PLC0415

    request = json.loads(line)
    previews.PREVIEW_CACHE_DIR = Path(request["cache_dir"])
    source = Path(request["source"])
    try:
        report = faces.analyse_faces(source)
    except Exception as exc:  # noqa: BLE001 - one bad photo must not kill the worker
        report = faces.FaceReport(source=source, error=str(exc))
    return report.model_dump_json() + "\n"


def main() -> None:
    """Serve requests until stdin closes."""
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    for line in sys.stdin:
        if line.strip():
            replies.write(handle_line(line))
            replies.flush()


if __name__ == "__main__":
    main()
