"""Persisted dimensionality reduction, and projecting a NEW point into an existing map.

The map is only useful as a place if a query can be put *on* it — "here is an image, where does it
sit", "type a phrase, show me what is near it". That needs three things kept from fit time, not two:

  1. the fitted reducer                    (obvious)
  2. the coordinate normalisation          (Spacewalker stores `scale_val` for exactly this reason)
  3. the FEATURE-space statistics          (the one that is easy to miss)

(3) matters because `fuse_features` z-scores every block against the collection's own mean and std
before the reducer ever sees it. Transform a query with fresh statistics and it lands somewhere
meaningless, however correct the reducer is. So the fusion stats are part of the fitted artefact.

Not every reducer can place a new point: PCA, UMAP, Isomap and openTSNE implement `.transform`;
sklearn's MDS and TSNE do not. That is checked on the fitted object rather than guessed from a name.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FittedProjection:
    """Everything needed to place a new point exactly where the map would have put it."""
    reducer: Any                       # the fitted DR object (None when the embedding was trivial)
    method: str                        # what was actually used (may differ from what was asked)
    dims: int
    spec: dict                         # {feature_method: weight} the fusion used
    block_stats: dict                  # method -> {"mean", "std", "weight", "dim"} at fit time
    coord_min: np.ndarray              # normalisation captured AT FIT TIME, not recomputed per call
    coord_max: np.ndarray
    n_fit: int
    truncated: int                     # instances left out by the projection cap — be honest about it

    @property
    def queryable(self) -> bool:
        return self.reducer is not None and hasattr(self.reducer, "transform")

    def normalise(self, Y: np.ndarray) -> np.ndarray:
        """Map raw reducer output into the [0,1] frame the UI draws in."""
        return (Y - self.coord_min) / np.maximum(self.coord_max - self.coord_min, 1e-9)

    def fuse_query(self, by_method: dict[str, np.ndarray]) -> np.ndarray:
        """Build a query vector in the SAME fused space, using the fit-time statistics.

        Mirrors `core.collection.fuse_features` exactly: L2-normalise the block, z-score it, scale by
        its weight, concatenate in spec order.
        """
        missing = [m for m in self.spec if m not in by_method]
        if missing:
            raise ValueError(f"the projection space needs {sorted(self.spec)}; "
                             f"cannot build {missing} for this query")
        parts = []
        for m in self.spec:
            st = self.block_stats[m]
            v = np.asarray(by_method[m], np.float32).reshape(1, -1)
            if v.shape[1] != st["dim"]:
                raise ValueError(f"'{m}' query vector is {v.shape[1]}-d, the map was fitted on {st['dim']}-d")
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            v = (v - st["mean"]) / (st["std"] + 1e-9)
            parts.append(v * float(st["weight"]))
        return np.concatenate(parts, axis=1).astype(np.float32)

    # ---- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import joblib
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"reducer": self.reducer, "method": self.method, "dims": self.dims,
                     "spec": self.spec, "block_stats": self.block_stats,
                     "coord_min": self.coord_min, "coord_max": self.coord_max,
                     "n_fit": self.n_fit, "truncated": self.truncated}, p)

    @classmethod
    def load(cls, path: str | Path) -> "FittedProjection | None":
        import joblib
        p = Path(path)
        if not p.is_file():
            return None
        try:
            return cls(**joblib.load(p))
        except Exception:
            return None                        # a stale/incompatible artefact just means "refit"


def block_stats(X_blocks: dict[str, np.ndarray], spec: dict) -> dict:
    """Per-block fit-time statistics, computed the way fuse_features computes them."""
    out = {}
    for m, w in spec.items():
        B = np.asarray(X_blocks[m], np.float32)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        out[m] = {"mean": B.mean(0, keepdims=True), "std": B.std(0, keepdims=True),
                  "weight": float(w), "dim": int(B.shape[1])}
    return out
