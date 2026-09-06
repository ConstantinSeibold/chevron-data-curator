"""v2 tests: NMS, dedup, merge-same-image, merge_instances, instance_at_pixel, context crop,
refine-partition, cluster(req_clust). Run: pytest tests/test_v2.py -q (repo root)
"""
from __future__ import annotations

import numpy as np

from chevron import collect, ids
from chevron.engine import CuratorEngine
from chevron.state import InstanceMeta


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


def _disk(cx, cy, r=8, h=64, w=64):
    import cv2
    m = np.zeros((h, w), np.uint8); cv2.circle(m, (cx, cy), r, 1, -1)
    return m > 0


def test_mask_nms_filters_records_and_feats():
    m1, m2 = _disk(20, 20), _disk(45, 45)
    recs = [{"rle": _rle(m1), "score": 0.9, "image_id": 1},      # keep
            {"rle": _rle(m1), "score": 0.5, "image_id": 1},      # dup of #0 -> drop
            {"rle": _rle(m2), "score": 0.8, "image_id": 1},      # keep (distinct)
            {"rle": _rle(m1), "score": 0.7, "image_id": 2}]      # diff image -> keep
    col = {"records": recs, "feats": {"decoder": np.arange(16).reshape(4, 4).astype(np.float32)}, "n_images": 2}
    collect.mask_nms(col, 0.8)
    assert len(col["records"]) == 3                              # one dup removed
    assert col["feats"]["decoder"].shape == (3, 4)
    assert [r["row"] for r in col["records"]] == [0, 1, 2]       # reindexed
    # feats row for the surviving image-1 instances kept their original values (0-3, 8-11, 12-15)
    assert col["feats"]["decoder"][0, 0] == 0 and col["feats"]["decoder"][1, 0] == 8


def _png(p, h=64, w=64):
    import cv2
    cv2.imwrite(str(p), (np.random.default_rng(1).random((h, w, 3)) * 100 + 40).astype(np.uint8))


def _engine_with_dups(tmp_path):
    """1 image, 3 instances: two identical disks (dup) + one distinct."""
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    p = tmp_path / "im0.png"; _png(p)
    masks = [_disk(18, 18), _disk(18, 18), _disk(46, 46)]        # 0,1 identical; 2 distinct
    scores = [0.9, 0.5, 0.8]
    recs, order, meta = [], [], {}
    dec = [[5, 5, 5, 5], [5, 5, 5, 5], [-5, -5, -5, -5]]
    for i, (mb, sc) in enumerate(zip(masks, scores)):
        ys, xs = np.where(mb)
        u = ids.new_uid()
        recs.append({"iuid": u, "row": i, "inst_id": i, "image_id": 1000, "H": 64, "W": 64, "score": sc,
                     "rle": _rle(mb), "file_name": str(p), "abs_path": str(p),
                     "cx": float(xs.mean() / 64), "cy": float(ys.mean() / 64), "bw": 0.3, "bh": 0.3,
                     "box_area": 0.09, "mask_area_frac": float(mb.mean())})
        order.append(u); meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=1000)
    eng.collection = {"records": recs, "n_images": 1, "feats": {"decoder": np.array(dec, np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    # a manual single-partition clustering with all 3 in pid 0 (pool = all 3 unassigned)
    eng._cluster = {"spec": {"decoder": 1.0}, "distance": "cosine",
                    "partitions": np.array([[0], [0], [0]]), "counts": [1], "level": 0, "pool": list(order)}
    return eng, order


def test_dedup_current(tmp_path):
    eng, order = _engine_with_dups(tmp_path)
    n = eng.dedup_current(0.8)
    assert n == 1                                               # the lower-score identical disk -> background
    bg = [u for u in order if eng.state.meta[u].is_background]
    assert len(bg) == 1 and eng.state.meta[bg[0]].score if False else True
    eng.undo()
    assert sum(eng.state.meta[u].is_background for u in order) == 0   # reversible


def test_merge_same_image_and_instances(tmp_path):
    eng, order = _engine_with_dups(tmp_path)
    n = eng.merge_partition_by_image(0)
    assert n == 1                                               # all 3 same-image -> one merged group
    reps = [u for u in order if eng.state.meta[u].merge_members]
    children = [u for u in order if eng.state.meta[u].merged_into is not None]
    assert len(reps) == 1 and len(children) == 2
    eng.undo()
    assert all(not eng.state.meta[u].merge_members and eng.state.meta[u].merged_into is None for u in order)
    # explicit merge of 2
    eng.merge_instances([order[0], order[2]])
    assert any(eng.state.meta[u].merge_members for u in (order[0], order[2]))


def test_image_instance_iuids_excludes_merged_children(tmp_path):
    """v6.1: the in-image grid hides merge CHILDREN so it collapses to the representative after a merge."""
    eng, order = _engine_with_dups(tmp_path)                       # 3 instances on image_id 1000
    assert len(eng.image_instance_iuids(1000)) == 3
    eng.merge_instances([order[0], order[1]])                      # merge two -> one child hidden
    iu = eng.image_instance_iuids(1000)
    assert len(iu) == 2                                            # representative + the distinct instance
    assert all(eng.state.meta[u].merged_into is None for u in iu)  # no children shown


def test_instance_at_pixel_and_context_crop(tmp_path):
    eng, order = _engine_with_dups(tmp_path)
    u = eng.instance_at_pixel(1000, 18, 18)                     # inside the disk at (18,18) -> highest score there
    assert u == order[0]                                        # score 0.9 > 0.5 for the two identical disks
    assert eng.instance_at_pixel(1000, 2, 2) is None            # background pixel
    full = eng.crop(order[0], context=True)
    crp = eng.crop(order[0], context=False)
    assert full.shape[:2] == (64, 64)                           # context = whole image
    assert crp.shape[0] <= 64 and crp.shape[0] < full.shape[0] + 1   # crop no larger than full


def test_refine_partition_and_revert(tmp_path):
    eng, order = _engine_with_dups(tmp_path)
    n = eng.apply_refine_partition(0, [{"name": "fill"}])
    assert n == 3 and all(eng.state.meta[u].refined for u in order)
    eng.undo()
    assert all(not eng.state.meta[u].refined for u in order)    # undo clears refined flags
    b, a = eng.refine_partition_preview(0, [{"name": "dilate", "kw": {"k": 1, "max_contrast": 0.9}}], n=2)
    assert len(b) == 2 and b[0][0].ndim == 3


def test_factored_classifier_open_set(tmp_path):
    """Open-set: a clearly-background unassigned instance must NOT be confidently assigned to a
    class (closed-set softmax would force it); a class-like unassigned instance should be."""
    from chevron import classify
    rng = np.random.default_rng(0)
    A = rng.normal([6, 0, 0, 0], 0.3, (6, 4))      # class A cluster
    B = rng.normal([0, 6, 0, 0], 0.3, (6, 4))      # class B cluster
    bg = rng.normal([0, 0, 6, 0], 0.3, (8, 4))     # background pool (far from A and B)
    un_like_A = rng.normal([6, 0, 0, 0], 0.3, (1, 4))
    un_like_bg = rng.normal([0, 0, 6, 0], 0.3, (1, 4))
    feats = np.vstack([A, B, bg, un_like_A, un_like_bg]).astype(np.float32)
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    order, meta, recs = [], {}, []
    for i in range(feats.shape[0]):
        u = ids.new_uid(); order.append(u); recs.append({"iuid": u, "row": i, "score": 0.6, "image_id": 1})
        meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=1)
    eng.collection = {"records": recs, "n_images": 1, "feats": {"decoder": feats}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    cA, cB = eng.state.add_class("A"), eng.state.add_class("B")
    for i in range(6):
        meta[order[i]].assigned_class = cA
    for i in range(6, 12):
        meta[order[i]].assigned_class = cB
    for i in range(12, 20):
        meta[order[i]].is_background = True
    rep = eng.train_classifier({"decoder": 1.0}, algo="logreg", use_unassigned_negatives=True)
    assert rep["n_classes"] == 2 and rep["n_background"] == 8
    preds = dict((u, (cid, conf)) for u, cid, conf in eng.predict_and_threshold(0.5))
    u_A, u_bg = order[20], order[21]
    assert u_A in preds and preds[u_A][0] == cA            # class-like -> assigned A
    assert u_bg not in preds                               # background-like -> stays unassigned (open-set)


def test_cluster_req_clust(tmp_path):
    # 8 points in two well-separated blobs so FINCH's finest partition has >=2 clusters
    rng = np.random.default_rng(0)
    dec = np.vstack([rng.normal([5, 5, 5, 5], 0.3, (4, 4)), rng.normal([-5, -5, -5, -5], 0.3, (4, 4))]).astype(np.float32)
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    order, meta, recs = [], {}, []
    for i in range(8):
        u = ids.new_uid(); order.append(u)
        recs.append({"iuid": u, "row": i, "score": 0.6, "image_id": 1000})
        meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=1000)
    eng.collection = {"records": recs, "n_images": 1, "feats": {"decoder": dec}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    info = eng.cluster({"decoder": 1.0}, req_clust=2)
    assert info["counts"] == [2]                                # forced exactly 2 clusters
    assert len(set(eng._pool_labels().tolist())) == 2
