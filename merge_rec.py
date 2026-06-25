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


def _make(algo: str):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if algo == "rf":
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1)
    else:
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=2000, class_weight="balanced")
    return make_pipeline(StandardScaler(), est)        # pair_features columns differ wildly in scale


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


def build_pair_xy(collection: dict, state: CuratorState, pos_groups, rejected_groups, spec, *,
                  max_neg: int = 4000, seed: int = 0):
    """Positives = within-merge-group pairs; negatives = same-image pairs NOT co-merged (sampled) +
    rejected groups' pairs. Returns (X, y, pairs[P,2 row idx], report)."""
    pos_pairs, rej_pairs = set(), set()
    for rows in _group_rows(state, pos_groups):
        for a, b in _within_pairs(rows):
            pos_pairs.add((min(a, b), max(a, b)))
    for rows in _group_rows(state, rejected_groups):
        for a, b in _within_pairs(rows):
            rej_pairs.add((min(a, b), max(a, b)))
    rej_pairs -= pos_pairs

    by_img = defaultdict(list)                          # current instances per image for negative sampling
    for u, m in state.meta.items():
        if not m.is_background and m.merged_into is None:
            by_img[m.image_id].append(m.row)
    cand = []
    for rows in by_img.values():
        for a, b in _within_pairs(rows):
            key = (min(a, b), max(a, b))
            if key not in pos_pairs:
                cand.append(key)
    rng = np.random.default_rng(seed); rng.shuffle(cand)
    neg_pairs = set(rej_pairs)
    cap = max(int(max_neg), len(pos_pairs))             # keep classes from being wildly imbalanced
    for key in cand:
        if len(neg_pairs) >= cap:
            break
        neg_pairs.add(key)
    neg_pairs -= pos_pairs

    pairs = list(pos_pairs) + list(neg_pairs)
    y = np.r_[np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))].astype(np.int64)
    rep = {"n_pos": int(len(pos_pairs)), "n_neg": int(len(neg_pairs))}
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
                     max_groups: int = 20, max_images: int = 400, only_image: int | None = None):
    """Per image: score all current-instance pairs, keep edges >= thresh, return connected components
    (size >= 2) sorted by mean within-component edge probability. With ``only_image`` set, restrict to
    that single image (the In-image-tab in-context suggestions)."""
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
        parent = list(range(len(ius)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for (i, j), pr in zip(idx_pairs, prob):
            if pr >= thresh:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
        comp = defaultdict(list)
        for k in range(len(ius)):
            comp[find(k)].append(k)
        for members in comp.values():
            if len(members) < 2:
                continue
            ms = set(members)
            ep = [pr for (i, j), pr in zip(idx_pairs, prob) if i in ms and j in ms and pr >= thresh]
            groups.append({"iuids": [ius[k] for k in members], "image_id": int(iid),
                           "prob": float(np.mean(ep)) if ep else 0.0})
    groups.sort(key=lambda g: -g["prob"])
    return groups[:max_groups]
