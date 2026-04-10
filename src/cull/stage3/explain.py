"""Stage 3 explain — thin re-export shim preserving existing import paths."""

from cull.config import EXPLAIN_MAX_TOKENS  # noqa: F401
from cull.stage3.vlm_explain import (  # noqa: F401
    ExplainCallInput,
    _build_hint_block,
    _parse_explain_response,
    explain_photo,
)
