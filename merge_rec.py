"""Learned merge recommender: a PAIRWISE P(merge | A, B) model trained on the user's past in-image
merges, with candidate >2-instance merges formed as connected components of high-probability edges.

Reuses notebooks/qseg_playground.py:pair_features (per-encoder cosine+L2, centroid dist, bbox IoU/gap,
class agreement, score, shape ratios) — it takes pairs of collection ROW indices and returns (Xp, names).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .state import CuratorState


def _P():
    from ._bootstrap import get_P
    return get_P()


def _methods(spec) -> tuple:
    return tuple(spec.keys()) if isinstance(spec, dict) else tuple(spec)


def _within_pairs(rows: list[int]) -> list[tuple[int, int]]:
    return [(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]


_NEG_RATIO = 10            # n_neg <= _NEG_RATIO * n_pos: bound class imbalance (was a flat 4000 -> ~80:1)
_HARD_FRAC = 0.8           # fraction of negatives mined from the HARDEST (most confusable) pairs; rest random
_MIN_POS = 8              # below this many positive pairs the model is undertrained -> warn in the report


def _make(algo: str):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    if algo == "rf":
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1)
    else:
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=2000, class_weight="balanced")
    # RobustScaler (median/IQR), not StandardScaler: pair_features has heavy-tailed columns (area/elongation
    # ratio ~1e6 on degenerate masks, unbounded L2) that would otherwise dominate the fit.
    return make_pipeline(RobustScaler(), est)


def _finite(X: np.ndarray) -> np.ndarray:
    """Sanitize a PAIR-feature matrix to float32 with no NaN/inf. pair_features' GEOMETRY columns
    (orientation diff, elongation/area ratios, bbox gap) divide by per-instance shape quantities, so a
    degenerate instance (tiny / zero-area mask, a split child) yields NaN/inf there — regardless of the
    feature spec (so it bites even with clean raddino feats). Map those rare degenerate cells to 0 (the
    StandardScaler then centres them) rather than letting sklearn raise 'Input contains NaN'."""
    return np.nan_to_num(np.asarray(X, np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _group_rows(state: CuratorState, groups) -> list[list[int]]:
    out = []
    for g in groups:
        rows = [state.meta[u].row for u in g if u in state.meta]
        if len(rows) >= 2:
            out.append(rows)
    return out


def _confusability(collection: dict, pairs: np.ndarray) -> np.ndarray:
    """Per candidate ROW-index pair, a cheap 'how confusable a negative is this' score (HIGHER = harder):
    bbox IoU + centroid proximity + (if present) appearance cosine. Drives hard-negative mining so the model
    learns the boundary between 'looks mergeable but isn't' and 'should merge' — not 'random far pair'."""
    recs = collection["records"]
    cx = np.array([float(r.get("cx", 0.0)) for r in recs], np.float32)
    cy = np.array([float(r.get("cy", 0.0)) for r in recs], np.float32)
    bx = np.array([list(r.get("box_xyxy", (0, 0, 0, 0))) for r in recs], np.float32)
    I, J = pairs[:, 0], pairs[:, 1]
    a, b = bx[I], bx[J]
    inter = (np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
             * np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None))
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / (aa + ab - inter + 1e-6)
    dc = np.hypot(cx[I] - cx[J], cy[I] - cy[J])                       # normalized centroid distance
    score = iou + np.exp(-dc / 0.05)                                 # overlap + spatial proximity (~5% of image)
    feats = collection.get("feats", {})
    if "decoder" in feats and len(feats["decoder"]):
        F = np.asarray(feats["decoder"], np.float32)
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        score = score + np.clip((Fn[I] * Fn[J]).sum(1), 0, None)     # + appearance similarity
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def build_pair_xy(collection: dict, state: CuratorState, pos_groups, rejected_groups, spec, *,
                  max_neg: int = 4000, seed: int = 0):
    """Positives = within-merge-group pairs. Negatives = HARD same-image non-merged pairs (the most
    confusable, by `_confusability`) + a small random tail + all user-rejected pairs, bounded to
    `_NEG_RATIO * n_pos`. Returns (X, y, pairs[P,2 row idx], report)."""
    pos_pairs, rej_pairs = set(), set()
    for rows in _group_rows(state, pos_groups):
        for a, b in _within_pairs(rows):
            pos_pairs.add((min(a, b), max(a, b)))
    for rows in _group_rows(state, rejected_groups):
        for a, b in _within_pairs(rows):
            rej_pairs.add((min(a, b), max(a, b)))
    rej_pairs -= pos_pairs

    by_img = defaultdict(list)                          # current instances per image (negative-pair source)
    for u, m in state.meta.items():
        if not m.is_background and m.merged_into is None:
            by_img[m.image_id].append(m.row)
    cand = []
    for rows in by_img.values():
        for a, b in _within_pairs(rows):
            key = (min(a, b), max(a, b))
            if key not in pos_pairs:
                cand.append(key)

    rng = np.random.default_rng(seed)
    neg_pairs = set(rej_pairs)                          # user rejects = the hardest negatives -> keep ALL
    target = min(int(_NEG_RATIO * max(len(pos_pairs), 1)), int(max_neg))   # bounded imbalance
    need = max(0, target - len(neg_pairs))
    if cand and need:
        ca = np.array(cand, int)
        order = np.argsort(-_confusability(collection, ca))           # hardest (most confusable) first
        n_hard = int(need * _HARD_FRAC)
        pick = list(order[:n_hard])
        rest = order[n_hard:]
        if len(rest):                                                 # random tail covers the easy region
            pick += list(rng.choice(rest, min(need - n_hard, len(rest)), replace=False))
        for idx in pick:
            if len(neg_pairs) >= target:
                break
            neg_pairs.add((int(ca[idx, 0]), int(ca[idx, 1])))
    neg_pairs -= pos_pairs

    pairs = list(pos_pairs) + list(neg_pairs)
    y = np.r_[np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))].astype(np.int64)
    rep = {"n_pos": int(len(pos_pairs)), "n_neg": int(len(neg_pairs)),
           "n_rejected_neg": int(len(rej_pairs)), "undertrained": int(len(pos_pairs)) < _MIN_POS}
    if not pairs:
        return np.zeros((0, 0), np.float32), y, np.zeros((0, 2), int), rep
    parr = np.array(pairs, dtype=int)
    Xp, names = _P().pair_features(collection, parr, methods=_methods(spec) or ("decoder",))
    rep["names"] = names
    return _finite(Xp), y, parr, rep


def train(X: np.ndarray, y: np.ndarray, *, algo: str = "logreg"):
    clf = _make(algo)
    clf.fit(X, y)
    return clf


def pr_youden(X: np.ndarray, y: np.ndarray, *, algo: str = "logreg", n_splits: int = 3) -> dict:
    """CV-OOF precision/recall-vs-threshold + Youden-J recommended threshold for the single merge class."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    if len(set(y.tolist())) < 2 or int(min(np.bincount(y))) < 2:
        return {"curve": {}, "youden": 0.5}
    k = int(min(n_splits, int(min(np.bincount(y)))))
    oof = cross_val_predict(_make(algo), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y, oof)
    fpr, tpr, rt = roc_curve(y, oof)
    jt = rt[int(np.argmax(tpr - fpr))]
    youden = float(min(max(jt, 0.0), 1.0)) if np.isfinite(jt) else 0.5
    return {"curve": {"precision": p.tolist(), "recall": r.tolist(), "thresholds": t.tolist()}, "youden": youden}


def candidate_groups(collection: dict, state: CuratorState, clf, spec, thresh: float, *,
                     max_groups: int = 20, max_images: int = 400, max_comp: int = 10,
                     only_image: int | None = None):
    """Per image: score all current-instance pairs, keep edges >= thresh, return connected components
    (size in [2, max_comp]). A component is scored by its WEAKEST connecting edge (min, not mean) so a weak
    transitive chain ranks low / is filtered by raising the threshold, and a component is dropped if it grows
    past `max_comp` (a giant component is a low-threshold blow-up, not a real merge). Connected components
    (not cliques) so genuine LINE chains A-B-C survive while the min-edge score stays honest. `only_image`
    restricts to one image (the In-image-tab suggestions)."""
    by_img = defaultdict(list)
    for u, m in state.meta.items():
        if not m.is_background and m.merged_into is None:
            if only_image is not None and int(m.image_id) != int(only_image):
                continue
            by_img[m.image_id].append(u)
    groups = []
    for iid, ius in list(by_img.items())[:max_images]:
        if len(ius) < 2:
            continue
        rows = [state.meta[u].row for u in ius]
        idx_pairs = [(i, j) for i in range(len(ius)) for j in range(i + 1, len(ius))]
        parr = np.array([(rows[i], rows[j]) for i, j in idx_pairs], int)
        Xp, _ = _P().pair_features(collection, parr, methods=_methods(spec) or ("decoder",))
        prob = clf.predict_proba(_finite(Xp))[:, 1]
        edge = {(i, j): float(pr) for (i, j), pr in zip(idx_pairs, prob)}
        parent = list(range(len(ius)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for (i, j), pr in edge.items():
            if pr >= thresh:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
        comp = defaultdict(list)
        for k in range(len(ius)):
            comp[find(k)].append(k)
        for members in comp.values():
            if not (2 <= len(members) <= max_comp):                    # giant component = low-threshold blow-up
                continue
            ms = set(members)
            conn = [edge[(i, j)] for (i, j) in idx_pairs if i in ms and j in ms and edge[(i, j)] >= thresh]
            groups.append({"iuids": [ius[k] for k in members], "image_id": int(iid),
                           "prob": float(min(conn)) if conn else 0.0})  # weakest connecting link = real cohesion
    groups.sort(key=lambda g: -g["prob"])
    return groups[:max_groups]
