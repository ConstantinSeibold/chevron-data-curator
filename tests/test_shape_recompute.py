"""shape_descriptors must never emit NaN/inf (degenerate masks made cv2.fitEllipse NaN), and the Config
'Recompute shape features' action re-derives `shape`+`shapecoord` from masks, sanitizes, persists, and
un-flags `shape` from feature_nan_methods so it's selectable again.
Run: pytest tools/curator/tests/test_shape_recompute.py -q
"""
from __future__ import annotations

import numpy as np


def test_shape_descriptors_never_nan():
    import importlib
    sd = importlib.import_module("notebooks.qseg_playground").shape_descriptors
    masks = [np.zeros((20, 20), bool)]                                  # empty
    one = np.zeros((20, 20), bool); one[10, 10] = True; masks.append(one)        # single pixel
    diag = np.zeros((20, 20), bool)
    for k in range(2, 18):
        diag[k, k] = True
    masks.append(diag)                                                 # 1-px diagonal (collinear -> fitEllipse NaN)
    bar = np.zeros((20, 20), bool); bar[10, 2:18] = True; masks.append(bar)      # 1-px-wide bar
    for m in masks:
        out = sd(m)
        bad = [k for k, v in out.items() if not np.isfinite(v)]
        assert not bad, f"non-finite shape descriptors {bad} for mask sum={int(m.sum())}"


def _eng_with_nan_shape(tmp_path, n=4):
    import cv2
    from pycocotools import mask as mu
    from tools.curator import ids
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    p = tmp_path / "im.png"
    cv2.imwrite(str(p), (np.random.default_rng(0).random((64, 64, 3)) * 120 + 40).astype(np.uint8))
    recs, order, meta = [], [], {}
    for j in range(n):
        m = np.zeros((64, 64), np.uint8); cv2.circle(m, (16 + 10 * j, 32), 6, 1, -1); mb = m > 0
        r = mu.encode(np.asfortranarray(mb.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
        u = ids.new_uid()
        recs.append({"iuid": u, "row": j, "inst_id": j, "image_id": 1000, "H": 64, "W": 64, "score": .8,
                     "pred_class": 0, "rle": r, "file_name": str(p), "abs_path": str(p), "batch_id": "b",
                     "box_xyxy": [16 + 10 * j - 6, 26, 16 + 10 * j + 6, 38], "cx": (16 + 10 * j) / 64, "cy": .5,
                     "bw": 12 / 64, "bh": 12 / 64, "box_area": .03, "mask_area_frac": float(mb.mean())})
        order.append(u); meta[u] = InstanceMeta(iuid=u, batch_id="b", row=j, image_id=1000)
    shp = np.ones((n, 3), np.float32); shp[1, 1] = np.nan                # a NaN-tainted stored shape matrix
    eng.collection = {"records": recs, "n_images": 1,
                      "feats": {"decoder": np.zeros((n, 4), np.float32), "shape": shp, "_shape_cols": ["a", "b", "c"]}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    return eng


def test_recompute_shape_replaces_nan_and_persists(tmp_path):
    eng = _eng_with_nan_shape(tmp_path, 4)
    assert "shape" in eng.feature_nan_methods()                        # NaN -> flagged (disabled in selectors)
    v0 = eng.state.coll_version

    rep = eng.recompute_shape_features()
    assert rep["ok"] and rep["n"] == 4
    feats = eng.collection["feats"]
    assert np.isfinite(feats["shape"]).all() and np.isfinite(feats["shapecoord"]).all()
    assert "shape" not in eng.feature_nan_methods()                    # clean now -> selectable again
    assert eng.state.coll_version > v0
    eng.state.assert_aligned(feats["shape"].shape[0])
    reloaded = eng.store.load_collection()                             # persisted clean
    assert np.isfinite(reloaded["feats"]["shape"]).all()


def test_recompute_shape_no_collection(tmp_path):
    from tools.curator.engine import CuratorEngine
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    assert "error" in eng.recompute_shape_features()
