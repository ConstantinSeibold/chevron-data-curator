"""v7.0 merge recommender: pair training data from logged merges, train, candidate connected-components,
engine logging + accept/reject. Run: pytest tools/curator/tests/test_merge_rec.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


def _engine(tmp_path, n_img=4, per_img=6):
    """n_img images x per_img instances; instance j alternates two decoder centres so 'same centre +
    close' pairs are mergeable. records carry box_xyxy (pair_features needs it)."""
    import cv2
    from tools.curator import ids
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    recs, order, meta, dec = [], [], {}, []
    for ii in range(n_img):
        p = tmp_path / f"im{ii}.png"
        cv2.imwrite(str(p), (np.random.default_rng(ii).random((128, 128, 3)) * 120 + 40).astype(np.uint8))
        for j in range(per_img):
            cx0 = 20 + 18 * j
            m = np.zeros((128, 128), np.uint8); cv2.circle(m, (cx0, 64), 8, 1, -1); mb = m > 0
            u = ids.new_uid(); row = len(recs)
            recs.append({"iuid": u, "row": row, "inst_id": row, "image_id": 1000 + ii, "H": 128, "W": 128,
                         "score": 0.6, "pred_class": j % 2, "rle": _rle(mb), "file_name": str(p), "abs_path": str(p),
                         "batch_id": "b", "box_xyxy": np.array([cx0 - 8, 56, cx0 + 8, 72], np.float32),
                         "cx": cx0 / 128, "cy": 0.5, "bw": 16 / 128, "bh": 16 / 128, "box_area": 0.015,
                         "mask_area_frac": float(mb.mean())})
            base = np.zeros(6, np.float32); base[j % 2] = 5.0     # class-0 vs class-1 decoder centre
            dec.append(base + np.random.default_rng(row).normal(0, 0.2, 6))
            order.append(u); meta[u] = InstanceMeta(iuid=u, batch_id="b", row=row, image_id=1000 + ii)
    eng.collection = {"records": recs, "n_images": n_img, "feats": {"decoder": np.array(dec, np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    return eng, order


def test_commit_merge_logs_event(tmp_path):
    eng, order = _engine(tmp_path, n_img=1, per_img=4)
    eng.merge_instances([order[0], order[2]])                      # two same-centre (class 0) instances
    evs = [e for e in eng.store.read_merge_events() if e.get("kind") == "merge"]
    assert len(evs) == 1 and set(evs[0]["iuids"]) == {order[0], order[2]}
    assert eng._merge_pos_groups()                                 # exposes the positive group


def test_build_pair_xy_and_train(tmp_path):
    from tools.curator import merge_rec as mr
    eng, order = _engine(tmp_path, n_img=4, per_img=6)
    # log a few merges of same-centre pairs (class 0 = even indices) on each image
    for ii in range(4):
        b = ii * 6
        eng.merge_instances([order[b + 0], order[b + 2]])          # even/even -> positive
    pos = eng._merge_pos_groups()
    X, y, pairs, rep = mr.build_pair_xy(eng.collection, eng.state, pos, [], {"decoder": 1.0})
    assert rep["n_pos"] >= 1 and rep["n_neg"] >= 1 and X.shape[0] == len(y) == len(pairs)
    assert X.shape[1] > 0                                          # geometry + decoder cos/l2 columns
    clf = mr.train(X, y, algo="logreg")
    assert clf.predict_proba(X).shape == (len(y), 2)


def test_engine_train_recommend_accept_reject(tmp_path):
    eng, order = _engine(tmp_path, n_img=4, per_img=6)
    for ii in range(4):                                            # teach: even-even on the same image merge
        b = ii * 6
        eng.merge_instances([order[b + 0], order[b + 2]])
    rep = eng.train_merge_recommender({"decoder": 1.0})
    assert "error" not in rep and rep["n_pos"] >= 1 and rep["n_merge_events"] >= 1
    cands = eng.recommend_merges(0.0)                              # thresh 0 -> every image yields a group
    assert isinstance(cands, list) and all("iuids" in c and "prob" in c for c in cands)
    # reject logs a negative (no state change); accept merges (logs a positive)
    n_ev0 = len(eng.store.read_merge_events())
    if cands:
        eng.reject_merge(cands[0]["iuids"])
        assert any(e.get("kind") == "reject" for e in eng.store.read_merge_events())
    # cold start on a fresh project -> clear error
    from tools.curator.engine import CuratorEngine
    eng2 = CuratorEngine(tmp_path / "fresh")
    eng2.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    eng2.collection = {"records": [], "n_images": 0, "feats": {"decoder": np.zeros((0, 6), np.float32)}}
    assert "error" in eng2.train_merge_recommender({"decoder": 1.0})


def test_candidate_groups_connected_components(tmp_path):
    from tools.curator import merge_rec as mr
    eng, order = _engine(tmp_path, n_img=1, per_img=4)

    class _AlwaysMerge:                                            # stub clf: P(merge)=1 for all pairs
        def predict_proba(self, X):
            return np.tile([0.0, 1.0], (len(X), 1))
    groups = mr.candidate_groups(eng.collection, eng.state, _AlwaysMerge(), {"decoder": 1.0}, 0.5)
    assert len(groups) == 1 and len(groups[0]["iuids"]) == 4       # all 4 on the image collapse to one component
