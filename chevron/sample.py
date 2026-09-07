"""Image sampling from a root folder — random and smart (low-confidence first)."""
from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

# The one definition of what counts as an image file; the engine imports it rather than keeping
# literal sets of its own. When the two drifted, `.webp` files were counted and ingested by the
# engine but never sampled, because sampling goes through `list_images` and its list lacked the
# suffix.
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})


def list_images(root: str | Path, exts: Iterable[str] = IMAGE_EXTS) -> list[str]:
    """Every file under `root` (recursively) whose suffix is in `exts`, sorted. Matches on the
    suffix only, case-insensitively; nothing is opened."""
    root = Path(root)
    if not root.exists():
        return []
    exts = {e.lower() for e in exts}
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(str(p))
    return sorted(out)


def sample_random(all_files: list[str], n: int, exclude: set[str] | None = None,
                  seed: int | None = None) -> list[str]:
    exclude = exclude or set()
    pool = [f for f in all_files if f not in exclude]
    if n >= len(pool):
        return pool
    rng = random.Random(seed)
    return rng.sample(pool, n)


def pick_lowest(scored: list[tuple[str, float]], n: int) -> list[str]:
    """Given (path, confidence) pairs, return the n lowest-confidence paths
    (smart sampling: prioritize images the model is least sure about)."""
    return [p for p, _ in sorted(scored, key=lambda t: t[1])[:n]]
