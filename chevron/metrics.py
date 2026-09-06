"""Partition quality, instance filtering/sorting, and audit-log access.
Pure-python (numpy); operates on a CuratorState + the heavy collection.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .state import CuratorState


def partition_summary(labels: np.ndarray, state: CuratorState, scores: np.ndarray) -> dict[int, dict]:
    """labels/scores are row-aligned to state.order. Returns pid -> summary dict:
    size, mean_score, purity (majority fraction among ASSIGNED), entropy,
    frac_unassigned, majority_class (class_id or None)."""
    out: dict[int, dict] = {}
    order = state.order
    for pid in sorted(set(int(x) for x in labels)):
        rows = np.where(labels == pid)[0]
        size = int(len(rows))
        assigned = []
        for r in rows:
            m = state.meta.get(order[r])
            if m and m.assigned_class and not m.is_background:
                assigned.append(m.assigned_class)
        cnt = Counter(assigned)
        purity = (max(cnt.values()) / len(assigned)) if assigned else None
        ent = _entropy(cnt) if assigned else None
        out[pid] = {
            "pid": pid, "size": size,
            "mean_score": float(scores[rows].mean()) if size else 0.0,
            "purity": purity, "entropy": ent,
            "frac_unassigned": float((size - len(assigned)) / size) if size else 0.0,
            "majority_class": (cnt.most_common(1)[0][0] if assigned else None),
            "n_assigned": len(assigned),
        }
    return out


def _entropy(counter: Counter) -> float:
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    h = 0.0
    for v in counter.values():
        p = v / tot
        h -= p * math.log(p + 1e-12)
    return float(h)


def _score_of(state: CuratorState, collection: dict, iuid: str) -> float:
    m = state.meta.get(iuid)
    if m is None:
        return 0.0
    return float(collection["records"][m.row]["score"])


def filter_instances(state: CuratorState, collection: dict, *, classes=None, only_unassigned=False,
                     only_background=False, score_min=None, score_max=None, image_id=None,
                     refined=None, is_merge_child=None) -> list[str]:
    out = []
    for iuid, m in state.meta.items():
        if only_unassigned and (m.assigned_class is not None or m.is_background):
            continue
        if only_background and not m.is_background:
            continue
        if classes is not None and m.assigned_class not in classes:
            continue
        if image_id is not None and m.image_id != image_id:
            continue
        if refined is not None and m.refined != refined:
            continue
        if is_merge_child is not None and (m.merged_into is not None) != is_merge_child:
            continue
        if score_min is not None or score_max is not None:
            s = _score_of(state, collection, iuid)
            if score_min is not None and s < score_min:
                continue
            if score_max is not None and s > score_max:
                continue
        out.append(iuid)
    return out


def sort_instances(iuids: list[str], state: CuratorState, collection: dict, *,
                   by: str = "score", desc: bool = True) -> list[str]:
    def key(iuid: str):
        m = state.meta[iuid]
        rec = collection["records"][m.row]
        if by == "score":
            return float(rec["score"])
        if by == "area":
            return float(rec.get("mask_area_frac", 0.0))
        if by == "assign_score":
            return float(m.assign_score or 0.0)
        if by == "class":
            return state.class_name(m.assigned_class) or "~"
        if by == "image":
            return m.image_id
        return 0.0
    return sorted(iuids, key=key, reverse=desc)


def audit_log(store, limit: int | None = None) -> list[dict]:
    return store.read_history(limit=limit)
