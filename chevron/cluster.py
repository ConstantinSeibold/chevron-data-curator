"""FINCH clustering across feature types/combinations.

Partitions live only in memory on the engine: what gets clustered is the unassigned pool, which
changes with every assignment, so there is no on-disk key worth caching them under. Partition
INDICES are never persisted in assignments either (FINCH renumbers on recompute); the engine
resolves a (level, pid) to a concrete iuid set at click time.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .state import CuratorState


def normalize_spec(spec) -> dict:
    """list[str] | dict[str,float] -> {method: weight}."""
    if isinstance(spec, (list, tuple)):
        return {m: 1.0 for m in spec}
    return dict(spec)


def cache_key(spec: dict, distance: str, per_image: bool, coll_version: int) -> str:
    payload = json.dumps({"spec": sorted(normalize_spec(spec).items()), "distance": distance,
                          "per_image": bool(per_image), "cv": int(coll_version)}, sort_keys=True)
    return hashlib.blake2b(payload.encode(), digest_size=12).hexdigest()


def fused_matrix(collection: dict, spec: dict) -> np.ndarray:
    from ._bootstrap import get_P
    P = get_P()
    X, _ = P.fuse_features(collection, normalize_spec(spec))
    return X


def finch_partitions(X: np.ndarray, distance: str = "cosine"):
    from ._bootstrap import get_P
    P = get_P()
    return P.finch_hierarchy(X, distance=distance)         # (partitions [N,P], counts [P])


def labels_at_level(partitions: np.ndarray, level: int) -> np.ndarray:
    level = max(0, min(level, partitions.shape[1] - 1))
    return partitions[:, level]


def partition_iuids(state: CuratorState, labels: np.ndarray, pid: int) -> list[str]:
    return [state.order[r] for r in np.where(labels == pid)[0]]
