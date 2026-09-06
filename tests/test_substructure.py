"""Within-class substructure: feature-space contrastive (SimCLR/NT-Xent) + FINCH sub-clustering.
Run: pytest tests/test_substructure.py -q
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


def _two_modes(n=40, d=12, sep=10.0, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (n, d)); a[:, 0] += sep
    b = rng.normal(0, 1, (n, d)); b[:, 0] -= sep
    return np.vstack([a, b]).astype(np.float32), np.array([0] * n + [1] * n)


def _purity(group_of, truth):
    """group_of: list of cluster ids; truth: list of true labels (parallel)."""
    by: dict = {}
    for g, t in zip(group_of, truth):
        by.setdefault(g, []).append(t)
    return sum(Counter(v).most_common(1)[0][1] for v in by.values()) / max(1, len(truth))


def test_train_embeddings_normalized_and_separates():
    from chevron.contrastive import train_embeddings
    X, y = _two_modes()
    emb = train_embeddings(X, dim=32, epochs=120, seed=0)
    assert emb.shape == (len(y), 32)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)     # L2-normalized
    assert np.isfinite(emb).all()
    from sklearn.cluster import KMeans
    km = KMeans(2, n_init=5, random_state=0).fit_predict(emb)
    assert _purity(km.tolist(), y.tolist()) > 0.85                      # the two modes are recovered


def test_train_embeddings_small_n_fallback():
    from chevron.contrastive import train_embeddings
    X = np.random.default_rng(0).normal(0, 1, (5, 8)).astype(np.float32)
    emb = train_embeddings(X, dim=16)                                   # < min_n -> normalized raw features
    assert emb.shape[0] == 5 and np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def _bimodal_engine(tmp_path):
    import cv2
    from chevron import ids
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    p = tmp_path / "im.png"; cv2.imwrite(str(p), np.zeros((64, 64, 3), np.uint8))
    mb = np.zeros((64, 64), np.uint8); cv2.circle(mb, (32, 32), 12, 1, -1); mb = mb > 0
    dec, y = _two_modes(n=30, d=8, sep=10.0, seed=1)
    recs, order, meta = [], [], {}
    for j in range(len(y)):
        u = ids.new_uid()
        recs.append({"iuid": u, "row": j, "inst_id": j, "image_id": 7, "H": 64, "W": 64, "score": 0.6,
                     "rle": _rle(mb), "file_name": str(p), "abs_path": str(p), "batch_id": "b",
                     "cx": .5, "cy": .5, "bw": .3, "bh": .3, "box_area": .09, "mask_area_frac": float(mb.mean())})
        order.append(u); meta[u] = InstanceMeta(iuid=u, batch_id="b", row=j, image_id=7)
    eng.collection = {"records": recs, "n_images": 1, "feats": {"decoder": dec}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    return eng, order, {order[i]: int(y[i]) for i in range(len(y))}


def test_subcluster_finds_substructure(tmp_path):
    """Contrastive + FINCH on a 2-mode class splits it into pure sub-clusters, browsable + assignable."""
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, order, mode = _bimodal_engine(tmp_path)
    c = TestClient(create_app(engine=eng))
    c.post("/api/assign", json={"iuids": order, "cls": "mix"})         # one coarse class with 2 hidden modes
    cid = eng.state.class_id_by_name("mix")

    r = c.post("/api/subcluster", json={"target": f"class:{cid}", "features": ["decoder"], "epochs": 120}).json()
    assert r["ok"] and r["n"] == len(order) and r["n_levels"] >= 1

    levels = c.get("/api/subclusters").json()["levels"]
    fine = max(levels, key=lambda l: l["n"])["i"]                      # finest level (most sub-clusters)
    c.post("/api/subcluster_level", json={"level": fine})
    sc = c.get("/api/subclusters").json()
    assert sc["active"] and len(sc["rows"]) >= 2                       # substructure surfaced

    groups, truth = [], []
    for row in sc["rows"]:
        for it in c.get(f"/api/subcluster_instances?subpid={row['subpid']}&limit=1000").json()["items"]:
            groups.append(row["subpid"]); truth.append(mode[it["iuid"]])
    assert _purity(groups, truth) > 0.8                               # sub-clusters align with the true modes

    # a sub-cluster is assignable to a new sub-class (split the coarse class)
    sub0 = sc["rows"][0]["subpid"]
    ius = [it["iuid"] for it in c.get(f"/api/subcluster_instances?subpid={sub0}&limit=1000").json()["items"]]
    ar = c.post("/api/assign", json={"iuids": ius, "cls": "mix_a"}).json()
    assert ar["ok"] and "mix_a" in ar["classes"] and all(eng.state.meta[u].assigned_class is not None for u in ius)

    # the Substructure Reject/Unassign actions reuse /api/reject /api/unassign on sub-cluster instances
    other = [it["iuid"] for row in sc["rows"][1:]
             for it in c.get(f"/api/subcluster_instances?subpid={row['subpid']}&limit=1000").json()["items"]
             if eng.state.meta[it["iuid"]].assigned_class is not None]
    assert other                                                      # remaining 'mix' instances exist
    assert c.post("/api/reject", json={"iuids": other[:1]}).json()["ok"] and eng.state.meta[other[0]].is_background
    if len(other) > 1:
        c.post("/api/unassign", json={"iuids": other[1:2]})
        assert eng.state.meta[other[1]].assigned_class is None        # back to the unassigned pool
