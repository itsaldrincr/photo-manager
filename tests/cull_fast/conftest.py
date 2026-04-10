"""Sub-conftest for cull_fast tests: provides mock_musiq_scorers fixture.

Mirrors the pattern in tests/_mock_scorers.py: deterministic, hash-derived
[0, 1]-bounded scores keyed by str(photo_path). Never invokes pyiqa.
"""

from __future__ import annotations

import hashlib

import pytest

from cull_fast.musiq import MusiQScorePair, _MusiQBatchRequest

PATH_DIGEST_BYTES: int = 4
_DIGEST_BIT_WIDTH: int = 31
PATH_DIGEST_MODULUS: int = 2 ** _DIGEST_BIT_WIDTH
SCORE_DIVISOR: float = float(PATH_DIGEST_MODULUS)
TECHNICAL_SALT: str = "musiq_technical"
AESTHETIC_SALT: str = "musiq_aesthetic"


def _path_score(photo_path: object, salt: str) -> float:
    """Map (str(path), salt) to a deterministic float in [0.0, 1.0]."""
    digest = hashlib.sha256(f"{salt}:{photo_path}".encode()).digest()
    raw = int.from_bytes(digest[:PATH_DIGEST_BYTES], "big") % PATH_DIGEST_MODULUS
    return raw / SCORE_DIVISOR


def _mock_score_musiq_batch(req: _MusiQBatchRequest) -> list[MusiQScorePair]:
    """Return deterministic MusiQScorePair list keyed by photo path."""
    return [
        MusiQScorePair(
            photo_path=path,
            technical=_path_score(path, TECHNICAL_SALT),
            aesthetic=_path_score(path, AESTHETIC_SALT),
        )
        for path in req.photo_paths
    ]


def _noop_musiq_metric(_name: str) -> object:
    """Stub replacement for _get_musiq_metric — never loads pyiqa."""
    return lambda batch: batch


@pytest.fixture
def mock_musiq_scorers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch cull_fast.musiq scorers with deterministic stubs."""
    monkeypatch.setattr("cull_fast.musiq.score_musiq_batch", _mock_score_musiq_batch)
    monkeypatch.setattr("cull_fast.musiq._get_musiq_metric", _noop_musiq_metric)
