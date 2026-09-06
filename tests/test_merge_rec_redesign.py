"""Merge-recommender redesign: hard-negative mining + bounded imbalance (Tier 1), min-edge / size-capped
connected-component grouping that keeps LINE chains but doesn't transitively over-merge (Tier 2), and the new
mask-free `collinearity` feature + clipped ratios in pair_features (Tier 3). Model-free (pair_features/clf
stubbed for the merge_rec logic; pair_features imported directly for the feature test).
Run: pytest tests/test_merge_rec_redesign.py -q
"""
from __future__ import annotations

import numpy as np


def _state(specs):
    """specs: [(iuid, image_id, box_xyxy, cx, cy)] -> (CuratorState, collection) with minimal records."""
    from chevron.state import CuratorState, InstanceMeta
    st = CuratorState(project_dir="/tmp/mr")
    recs = []
    for i, (u, img, box, cx, cy) in enumerate(specs):
        st.meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=img)
        recs.append({"cx": cx, "cy": cy, "box_xyxy": list(box), "score": 0.9, "pred_class": 0, "H": 100, "W": 100})
    st.order = [s[0] for s in specs]
    coll = {"records": recs, "feats": {"decoder": np.zeros((len(recs), 4), np.float32)}}
    return st, coll


# ---- Tier 1: hard negatives + bounded imbalance --------------------------------------------------------
def test_confusability_ranks_close_over_far():
    from chevron import merge_rec as MR
    _, coll = _state([("a", 1, [10, 10, 30, 30], .2, .2),
                      ("b", 1, [12, 12, 32, 32], .22, .22),   # overlaps a -> harder negative
                      ("c", 1, [80, 80, 90, 90], .85, .85)])   # far
    conf = MR._confusability(coll, np.array([[0, 1], [0, 2]]))
    assert conf[0] > conf[1]


def test_build_pair_xy_bounded_and_hard(monkeypatch):
    from chevron import merge_rec as MR

    class _FakeP:
        def pair_features(self, collection, parr, methods=("decoder",)):
            return np.ones((len(parr), 3), np.float32), ["f0", "f1", "f2"]
    monkeypatch.setattr(MR, "_P", lambda: _FakeP())

    specs = [("u0", 1, [10, 10, 30, 30], .2, .2), ("u1", 1, [40, 40, 60, 60], .5, .5),
             ("u2", 1, [12, 12, 32, 32], .22, .22)]            # u2 overlaps u0 -> hardest negative
    specs += [(f"u{k}", 1, [200 + k, k, 205 + k, 5 + k], .9, .05 * k) for k in range(3, 8)]
    st, coll = _state(specs)

    X, y, parr, rep = MR.build_pair_xy(coll, st, [["u0", "u1"]], [], {"decoder": 1.0})
    assert rep["n_pos"] == 1 and rep["undertrained"] is True               # 1 < _MIN_POS
    assert rep["n_neg"] <= MR._NEG_RATIO * rep["n_pos"]                     # bounded (<=10), not 4000
    negs = {tuple(sorted(map(int, p))) for p in parr[rep["n_pos"]:]}
    assert (0, 2) in negs                                                  # confusable pair mined as a negative


# ---- Tier 2: grouping --------------------------------------------------------------------------------
class _EdgeClf:
    """(pair_features, clf) fake: pair_features encodes a per-pair prob from `edge`; clf echoes it."""
    def __init__(self, edge):
        self.edge = edge

    def pf(self):
        edge = self.edge

        class _PF:
            def pair_features(self, collection, parr, methods=("decoder",)):
                X = np.array([[edge.get((min(int(a), int(b)), max(int(a), int(b))), 0.0)] for a, b in parr], np.float32)
                return X, ["p"]
        return _PF()

    def predict_proba(self, X):
        p = np.asarray(X)[:, 0]
        return np.column_stack([1 - p, p])


def _cg(monkeypatch, n, edge, thresh, **kw):
    from chevron import merge_rec as MR
    specs = [(chr(ord("a") + k), 1, [0, 0, 1, 1], 0.0, 0.0) for k in range(n)]
    st, coll = _state(specs)
    h = _EdgeClf(edge)
    monkeypatch.setattr(MR, "_P", lambda: h.pf())
    return MR.candidate_groups(coll, st, h, {"decoder": 1.0}, thresh, **kw)


def test_line_chain_survives_with_min_edge(monkeypatch):
    # A-B, B-C strong; A-C weak. Connected components keep all 3 (a line); score = weakest STRONG link.
    g = _cg(monkeypatch, 3, {(0, 1): 0.9, (1, 2): 0.8, (0, 2): 0.2}, 0.5)
    assert len(g) == 1 and len(g[0]["iuids"]) == 3
    assert abs(g[0]["prob"] - 0.8) < 1e-6                                   # min(0.9, 0.8); the weak 0.2 excluded


def test_no_overmerge_of_unrelated(monkeypatch):
    g = _cg(monkeypatch, 3, {(0, 1): 0.9, (0, 2): 0.1, (1, 2): 0.1}, 0.5)
    assert len(g) == 1 and set(g[0]["iuids"]) == {"a", "b"}                 # unrelated c left out


def test_giant_component_dropped(monkeypatch):
    edge = {(i, j): 0.9 for i in range(5) for j in range(i + 1, 5)}          # 5-clique
    assert _cg(monkeypatch, 5, edge, 0.5, max_comp=3) == []                 # > max_comp -> dropped
    assert len(_cg(monkeypatch, 5, edge, 0.5, max_comp=10)) == 1            # within cap -> kept


# ---- Tier 3: pair_features collinearity + clipped ratios ------------------------------------------------
def test_pair_features_collinearity_and_clip():
    import importlib
    pf = importlib.import_module("chevron.core.collection").pair_features
    recs = [{"cx": c, "cy": c, "box_xyxy": [0, 0, 10, 10], "score": .9, "pred_class": 0, "H": 100, "W": 100}
            for c in (0.2, 0.4, 0.6)]                                        # 3 fragments along the 45° diagonal
    recs.append({"cx": 0.4, "cy": 0.6, "box_xyxy": [0, 0, 10, 10], "score": .9, "pred_class": 0, "H": 100, "W": 100})
    S = np.array([[45, 6, 100], [45, 6, 100], [45, 6, 100], [135, 6, 1e-3]], np.float32)  # orient,elong,area
    coll = {"records": recs, "feats": {"decoder": np.zeros((4, 4), np.float32), "shape": S,
                                       "_shape_cols": ["orientation", "elongation", "area"]}}
    Xp, names = pf(coll, np.array([[0, 1], [0, 3]]), methods=("decoder",))
    ci, ar = names.index("collinearity"), names.index("area_ratio")
    assert Xp[0, ci] > 0.7                                                  # collinear elongated fragments
    assert Xp[1, ci] < 0.3                                                  # #3 perpendicular -> low
    assert Xp[1, ar] <= 50.0                                                # huge area ratio clipped, not ~1e6
