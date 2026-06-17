"""Image sampling from a root folder — random and smart (low-confidence first)."""
from __future__ import annotations

import random
from pathlib import Path

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(root: str | Path, exts: tuple[str, ...] = _EXTS) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
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
