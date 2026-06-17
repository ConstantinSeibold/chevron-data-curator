"""Stable identity primitives for the curator.

The playground's per-instance `inst_id` is a positional running counter
(`len(records)` at collect time) — it is NOT stable across additive re-sampling.
So the curator keys every instance by an immutable `iuid` (uuid4 hex). The
positional row index (`row` / `inst_id`) is derived and rewritten on every
load/append; it is only used to index the row-aligned `feats` matrices.
"""
from __future__ import annotations

import uuid


def new_uid() -> str:
    """Immutable per-instance key."""
    return uuid.uuid4().hex


def batch_id() -> str:
    """Identifies one additive sampling batch."""
    return "b_" + uuid.uuid4().hex[:12]


def class_id() -> str:
    """Stable taxonomy class id (the display name is mutable; this is not)."""
    return "c_" + uuid.uuid4().hex[:12]
