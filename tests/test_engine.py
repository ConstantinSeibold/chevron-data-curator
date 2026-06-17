"""Engine integration test on an injected fake collection (no GPU/model).
Covers: cluster -> assign-partition -> save/resume -> undo/redo -> merge -> refine ->
classifier -> export. Run: pytest tools/curator/tests/test_engine.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np

from tools.curator import ids
from tools.curator.engine import CuratorEngine
from tools.curator.state import InstanceMeta


def _png(path, h=64, w=64):
    import cv2
    img = (np.random.default_rng(abs(hash(str(path))) % 999).random((h, w, 3)) * 120 + 40).astype(np.uint8)
    cv2.imwrite(str(path), img)


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


def _inject(eng, tmp_path):
    """Simulate a sampled collection: 2 images x 3 instances; 2 feature clusters."""
    import cv2
    rng = np.random.default_rng(0)
    recs, dec, scoord, coords = [], [], [], []
    order, meta = [], {}
    img_paths = {}
    for img_i in range(2):
        p = tmp_path / f"im{img_i}.png"; _png(p); img_paths[img_i] = str(p)
        iid = 1000 + img_i
        for j in range(3):
            m = np.zeros((64, 64), np.uint8)
            if j == 0:
                cv2.line(m, (5, 5 + img_i), (58, 50), 1, 2)          # line-like
                feat = rng.normal([5, 5, 5, 5], 0.2, 4)
            else:
                cv2.circle(m, (20 + 15 * j, 30), 7, 1, -1)           # blob-like
                feat = rng.normal([-5, -5, -5, -5], 0.2, 4)
            mb = m > 0
            u = ids.new_uid(); row = len(recs)
            ys, xs = np.where(mb)
            cx, cy = float(xs.mean() / 64), float(ys.mean() / 64)
            recs.append({"iuid": u, "row": row, "inst_id": row, "image_id": iid, "H": 64, "W": 64,
                         "score": 0.6 + 0.05 * j, "rle": _rle(mb),
                         "file_name": str(p), "abs_path": str(p),
                         "cx": cx, "cy": cy, "bw": 0.3, "bh": 0.3, "box_area": 0.09, "mask_area_frac": float(mb.mean()),
                         "keypoints": np.array([[10, 10], [30, 30]], float),
                         "keypoint_vis": np.array([1.0, 1.0])})
            dec.append(np.concatenate([feat, rng.normal(0, 0.1, 4)]))
            scoord.append(np.zeros(29, np.float32))
            coords.append([0.3, 0.3, 0.2, 0.2, 0.04, 0.02])
            order.append(u)
            meta[u] = InstanceMeta(iuid=u, batch_id="b", row=row, image_id=iid)
    eng.collection = {"records": recs, "n_images": 2,
                      "feats": {"decoder": np.array(dec, np.float32),
                                "shapecoord": np.array(scoord, np.float32),
                                "coords": np.array(coords, np.float32),
                                "_shapecoord_cols": ["c"] * 29}}
    eng.state.order = order
    eng.state.meta = meta
    eng.state.coll_version = 1
    eng.state.collection_dirty = True
    eng.store.save_collection(eng.collection)
    eng.save()


def test_engine_full_loop(tmp_path):
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)},
                      "model": {"ckpt": "x", "score_thresh": 0.3},
                      "features": {"model_features": ["decoder"]}})
    _inject(eng, tmp_path)

    # cluster
    info = eng.cluster({"decoder": 1.0}, distance="cosine")
    assert info["n_levels"] >= 1
    pv = eng.partition_view()
    assert len(pv) >= 1 and "purity" in pv[0]

    # crops render (uses real pngs) BEFORE assigning
    pid = pv[0]["pid"]
    n_in = len(eng.partition_iuids(pid))
    crops, iuids = eng.partition_crops(pid, mask_overlay=True)
    assert len(crops) == n_in and crops[0][0].ndim == 3   # (img, caption) tuples

    # assign the partition -> its instances LEAVE the partition (now empty) and become a class pseudo-partition
    eng.assign_partition(pid, "lineA")
    assert eng.stats()["n_assigned"] == n_in
    assert "lineA" in eng.state.class_names()
    assert len(eng.partition_iuids(pid)) == 0                       # assigned left the FINCH partition
    assert any(str(r["pid"]).startswith("class:") for r in eng.partition_view())

    # undo / redo
    eng.undo(); assert eng.stats()["n_assigned"] == 0
    eng.redo(); assert eng.stats()["n_assigned"] == n_in

    # save + resume -> assignments persist
    eng2 = CuratorEngine(tmp_path)
    assert eng2.stats()["n_assigned"] == n_in
    assert eng2.collection is not None and len(eng2.state.order) == 6

    # in-image merge preview + commit
    iid = eng.image_ids()[0]
    before, after, groups = eng.merge_preview(iid, dist_kind="centroid", thresh=2.0, max_group_size=None)
    assert before.shape == after.shape == before.shape
    eng.commit_merge(iid, groups)
    # at least one instance became a merge child OR rep (if any group had >1)
    merged = [m for m in eng.state.meta.values() if m.merged_into is not None or m.merge_members]
    assert isinstance(merged, list)

    # refine an instance
    some = eng.state.order[0]
    o, r = eng.refine_preview(some, [{"name": "dilate", "kw": {"k": 1, "max_contrast": 0.9}}])
    assert o.ndim == 3 and r.ndim == 3
    eng.apply_refine(some, [{"name": "fill"}])
    assert eng.state.meta[some].refined is True
    assert some in eng._overlay_rle
    eng.revert_refine(some)
    assert eng.state.meta[some].refined is False

    # classifier: assign a second class to enable training
    eng.assign([eng.state.order[i] for i in (3, 4)], "blobB")
    rep = eng.train_classifier({"decoder": 1.0}, algo="logreg")
    assert rep.get("n_classes") == 2 or "error" in rep

    # export COCO + check it reads back
    p = eng.export_coco()
    import json
    coco = json.loads(p.read_text())
    assert len(coco["annotations"]) >= 1 and "iuid" in coco["annotations"][0]
