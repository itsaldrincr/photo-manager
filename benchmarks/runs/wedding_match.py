"""Match Edited exports back to Backup originals via DINOv2 embeddings.

Edited filenames are renamed sequential exports (no DSCF/HIEP token), so
filename matching is impossible; content matching is required. Edited files
are crops/tone-edits of their originals — DINOv2 was promoted precisely for
crop robustness (benchmarks/LOG.md 2026-07-17), so nearest-neighbor cosine
in DINOv2 space with a best-vs-second margin check gives reliable pairs.

Usage: python3 wedding_match.py <backup_dir> <edited_dir> <out_json>
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BATCH_SIZE = 8
ACCEPT_MIN_SIMILARITY = 0.60
ACCEPT_MIN_MARGIN = 0.03


def _embed_dir(directory: Path) -> tuple[list[str], np.ndarray]:
    """Embed every jpeg in a directory with DINOv2; return (names, vectors)."""
    import torch
    from PIL import Image
    from cull import dinov2_loader
    from cull.stage2.iqa import select_device

    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    processor = dinov2_loader.get_dinov2_processor()
    model = dinov2_loader.get_dinov2_model()
    device = select_device()
    vectors = []
    for start in range(0, len(paths), BATCH_SIZE):
        images = [Image.open(p).convert("RGB") for p in paths[start:start + BATCH_SIZE]]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        vectors.append(out.pooler_output.cpu().numpy())
        logger.info("%s: %d/%d", directory.name, min(start + BATCH_SIZE, len(paths)), len(paths))
    return [p.name for p in paths], np.concatenate(vectors, axis=0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    backup_dir, edited_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    backup_names, backup_vecs = _embed_dir(backup_dir)
    edited_names, edited_vecs = _embed_dir(edited_dir)
    b = backup_vecs / np.linalg.norm(backup_vecs, axis=1, keepdims=True)
    e = edited_vecs / np.linalg.norm(edited_vecs, axis=1, keepdims=True)
    sim = e @ b.T  # edited x backup
    matches = []
    for i, name in enumerate(edited_names):
        order = np.argsort(-sim[i])
        best, second = order[0], order[1]
        accepted = bool(
            sim[i, best] >= ACCEPT_MIN_SIMILARITY
            and (sim[i, best] - sim[i, second]) >= ACCEPT_MIN_MARGIN
        )
        matches.append({
            "edited": name, "backup": backup_names[best],
            "backup_second": backup_names[second],
            "similarity": float(sim[i, best]),
            "second_similarity": float(sim[i, second]),
            "margin": float(sim[i, best] - sim[i, second]),
            "accepted": accepted,
        })
    accepted_n = sum(1 for m in matches if m["accepted"])
    out = {
        "backup_dir": str(backup_dir), "edited_dir": str(edited_dir),
        "matcher": "dinov2-small pooled CLS cosine NN",
        "accept_min_similarity": ACCEPT_MIN_SIMILARITY, "accept_min_margin": ACCEPT_MIN_MARGIN,
        "matches": matches,
    }
    Path(out_path).write_text(json.dumps(out, indent=1))
    logger.info("matched %d/%d edited (accepted)", accepted_n, len(edited_names))


if __name__ == "__main__":
    main()
