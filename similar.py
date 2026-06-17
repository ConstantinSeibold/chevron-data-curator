"""kNN 'find similar instances' over a fusion-spec feature space (cosine)."""
from __future__ import annotations

import numpy as np

from .cluster import fused_matrix, normalize_spec
from .state import CuratorState


def find_similar(collection: dict, state: CuratorState, iuid: str, *, k: int = 20, spec=None,
                 only_unassigned: bool = True) -> list[tuple[str, float]]:
    spec = normalize_spec(spec or {"decoder": 1.0})
    X = fused_matrix(collection, spec)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    q_row = state.meta[iuid].row
    sims = Xn @ Xn[q_row]
    cand = []
    for u, m in state.meta.items():
        if u == iuid:
            continue
        if only_unassigned and (m.assigned_class is not None or m.is_background or m.merged_into is not None):
            continue
        cand.append((u, float(sims[m.row])))
    cand.sort(key=lambda t: -t[1])
    return cand[:k]
