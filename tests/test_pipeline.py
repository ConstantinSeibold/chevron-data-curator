"""Model-free tests for refine / classify / export-import / cluster-cache.
Run: pytest chevron/tests/test_pipeline.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np

from chevron import classify, cluster, export_coco, ids, refine
from chevron.state import CuratorState, InstanceMeta


# ---- refine ---------------------------------------------------------------
def _gray_mask():
    import cv2
    g = np.zeros((64, 64), np.uint8)
    cv2.circle(g, (32, 32), 12, 200, -1)
    m = np.zeros((64, 64), bool)
    cv2.circle(m_u := np.zeros((64, 64), np.uint8), (32, 32), 10, 1, -1)
    return g, (m_u > 0)


def test_refine_ops():
    g, m = _gray_mask()
    assert np.array_equal(refine.apply_ops(g, m, []), m)              # no-op
    grown = refine.apply_ops(g, m, [{"name": "dilate", "kw": {"k": 2, "max_contrast": 0.5}}])
    assert grown.sum() >= m.sum()                                    # dilation grows (gated)
    shrunk = refine.apply_ops(g, m, [{"name": "erode", "kw": {"k": 2, "min_contrast": 0.01}}])
    assert shrunk.sum() <= m.sum()
    assert refine.apply_ops(g, m, [{"name": "otsu"}]).dtype == bool
    assert refine.apply_ops(g, m, [{"name": "fill"}, {"name": "largest_cc"}, {"name": "smooth"}]).shape == m.shape
    assert m.sum() > 0                                               # input not mutated


# ---- classify / similar (uses P.fuse_features via _bootstrap) -------------
def _classif_collection():
    """2 well-separated classes in 'decoder' feature space, 4 samples each + 4 unassigned."""
    rng = np.random.default_rng(0)
    a = rng.normal([5, 5, 5, 5], 0.2, (4, 4))
    b = rng.normal([-5, -5, -5, -5], 0.2, (4, 4))
    un = np.vstack([rng.normal([5, 5, 5, 5], 0.2, (2, 4)), rng.normal([-5, -5, -5, -5], 0.2, (2, 4))])
    feats = {"decoder": np.vstack([a, b, un]).astype(np.float32)}
    recs = [{"score": 0.6, "row": i} for i in range(12)]
    col = {"records": recs, "feats": feats, "n_images": 1}
    st = CuratorState(project_dir="/tmp/x")
    cA, cB = st.add_class("A"), st.add_class("B")
    for i in range(12):
        u = f"u{i}"; st.order.append(u)
        st.meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=1)
    for i in range(4):
        st.meta[f"u{i}"].assigned_class = cA
    for i in range(4, 8):
        st.meta[f"u{i}"].assigned_class = cB
    return col, st, cA, cB


def test_classify_and_similar():
    col, st, cA, cB = _classif_collection()
    X, y, iu = classify.build_xy(col, st, {"decoder": 1.0})
    assert X.shape == (8, 4) and len(y) == 8
    clf, report = classify.train(X, y, algo="logreg")
    assert report["n_classes"] == 2
    iuids, proba, classes = classify.predict_unassigned(clf, col, st, {"decoder": 1.0})
    assert len(iuids) == 4 and proba.shape == (4, 2)
    assigned = classify.threshold_assign(iuids, proba, classes, 0.5)
    assert len(assigned) == 4                                        # all 4 confidently assigned
    # u8,u9 near A; u10,u11 near B
    amap = {u: c for u, c, _ in assigned}
    assert amap["u8"] == cA and amap["u11"] == cB
    from chevron import similar
    sims = similar.find_similar(col, st, "u0", k=3, spec={"decoder": 1.0}, only_unassigned=True)
    assert sims[0][0] in ("u8", "u9")                               # nearest unassigned to an A-sample is an A-like


# ---- export / import round-trip -------------------------------------------
def _export_collection():
    import cv2
    from pycocotools import mask as mu
    st = CuratorState(project_dir="/tmp/x")
    cid = st.add_class("ett")
    recs = []
    for i in range(3):
        m = np.zeros((48, 48), np.uint8)
        cv2.circle(m, (24, 24), 8 + i, 1, -1)
        r = mu.encode(np.asfortranarray(m)); r["counts"] = r["counts"].decode("ascii")
        recs.append({"rle": r, "image_id": 1000 + (i % 2), "H": 48, "W": 48, "score": 0.7,
                     "file_name": f"/imgs/x{i}.png", "row": i,
                     "keypoints": np.array([[10, 10], [20, 20]], float),
                     "keypoint_vis": np.array([1.0, 0.9])})
        u = f"u{i}"; st.order.append(u)
        st.meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=1000 + (i % 2), assigned_class=cid)
    return {"records": recs, "feats": {}, "n_images": 2}, st, cid


def test_export_import_roundtrip(tmp_path):
    col, st, cid = _export_collection()
    coco = export_coco.assemble_curated_coco(col, st, with_keypoints=True)
    assert len(coco["annotations"]) == 3 and len(coco["images"]) == 2
    a0 = coco["annotations"][0]
    assert a0["iuid"] == "u0" and "keypoints" in a0 and a0["num_keypoints"] == 2
    assert isinstance(a0["segmentation"], dict)                     # RLE
    p = export_coco.export(col, st, tmp_path / "out.json")
    # import into a FRESH state -> assignments restored by iuid
    st2 = CuratorState(project_dir="/tmp/x")
    for i in range(3):
        u = f"u{i}"; st2.order.append(u)
        st2.meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=1000 + (i % 2))
    rep = export_coco.import_coco(p, col, st2)
    assert rep["matched"] == 3 and rep["by_iuid"] == 3
    assert all(st2.meta[f"u{i}"].assigned_class is not None for i in range(3))
    assert st2.class_names() == ["ett"]
    # polygon export
    cocop = export_coco.assemble_curated_coco(col, st, polygon=True)
    assert isinstance(cocop["annotations"][0]["segmentation"], list)


def test_export_carries_provenance(tmp_path):
    """Released COCO annotations carry per-mask provenance (assign_source/assign_score/src_score/detector_ckpt);
    None fields are omitted so hand-labeled masks stay clean."""
    col, st, cid = _export_collection()
    st.meta["u0"].assign_source = "manual"                          # no score/provenance
    st.meta["u1"].assign_source = "classifier"; st.meta["u1"].assign_score = 0.83
    st.meta["u1"].provenance = {"src_score": 0.71, "ckpt": "out/model_best.pth", "file": "/imgs/x1.png"}
    st.meta["u2"].assign_source = "partition"
    by_iuid = {a["iuid"]: a for a in export_coco.assemble_curated_coco(col, st)["annotations"]}
    assert by_iuid["u0"]["assign_source"] == "manual" and "assign_score" not in by_iuid["u0"]   # None omitted
    assert "src_score" not in by_iuid["u0"] and "detector_ckpt" not in by_iuid["u0"]
    a1 = by_iuid["u1"]
    assert a1["assign_source"] == "classifier" and abs(a1["assign_score"] - 0.83) < 1e-6
    assert abs(a1["src_score"] - 0.71) < 1e-6 and a1["detector_ckpt"] == "out/model_best.pth"
    assert by_iuid["u2"]["assign_source"] == "partition"
    # partial-label export carries it too (alongside curator_status)
    pa = {a["iuid"]: a for a in export_coco.assemble_curated_coco(col, st, partial_labels=True)["annotations"]}
    assert pa["u1"]["assign_source"] == "classifier" and "curator_status" in pa["u1"]


# ---- cluster cache key ----------------------------------------------------
def test_cache_key():
    k1 = cluster.cache_key({"decoder": 1.0}, "cosine", False, 5)
    assert k1 == cluster.cache_key({"decoder": 1.0}, "cosine", False, 5)   # deterministic
    assert k1 != cluster.cache_key({"decoder": 1.0}, "cosine", False, 6)   # coll_version sensitive
    assert k1 != cluster.cache_key({"decoder": 1.0, "shape": 0.5}, "cosine", False, 5)
    assert cluster.normalize_spec(["a", "b"]) == {"a": 1.0, "b": 1.0}
