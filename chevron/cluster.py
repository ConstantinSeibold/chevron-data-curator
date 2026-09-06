"""FINCH clustering across feature types/combinations, with a coll_version-keyed cache.

Partition INDICES are never persisted in assignments (FINCH renumbers on recompute);
the engine resolves a (level, pid) to a concrete iuid set at click time.
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


def cluster_with_cache(collection: dict, state: CuratorState, store, spec, *,
                       distance: str = "cosine", per_image: bool = False, algo: str = "agglomerative"):
    """Returns (partitions [N, P], counts [P]). Cached by (spec, distance, per_image, coll_version);
    cache hit is validated against the current state.order."""
    spec = normalize_spec(spec)
    key = cache_key(spec, distance, per_image, state.coll_version)
    cached = store.load_cache(key)
    if cached is not None:
        partitions, counts, order = cached
        if list(order) == state.order and partitions.shape[0] == len(state.order):
            return partitions, counts
    X = fused_matrix(collection, spec)
    if per_image:
        from ._bootstrap import get_P
        P = get_P()
        labels = P.cluster_per_image_X(collection, X, algo=algo)   # 1 level
        partitions, counts = labels.reshape(-1, 1), [int(len(set(labels.tolist())))]
    else:
        partitions, counts = finch_partitions(X, distance)
    store.save_cache(key, np.asarray(partitions), counts, state.order)
    return partitions, counts


def labels_at_level(partitions: np.ndarray, level: int) -> np.ndarray:
    level = max(0, min(level, partitions.shape[1] - 1))
    return partitions[:, level]


def partition_iuids(state: CuratorState, labels: np.ndarray, pid: int) -> list[str]:
    return [state.order[r] for r in np.where(labels == pid)[0]]
