"""v5.5: feature-selection robustness + in-image overlay crash fix.
- available_features() reports present feats keys
- cluster() / train_classifier() return-or-raise a CLEAR error on an absent-only spec (no np.concatenate crash)
- image_overlay() runs without the removed self._labels() (AttributeError regression)
Run: pytest tools/curator/tests/test_feature_select.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np
import pytest


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


def _engine(tmp_path, *, with_decoder: bool):
    """1 image, 3 instances; feats always have coords+shape, decoder only if with_decoder."""
    import cv2
    from tools.curator import ids
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    p = tmp_path / "im0.png"
    cv2.imwrite(str(p), (np.random.default_rng(0).random((64, 64, 3)) * 120 + 40).astype(np.uint8))
    recs, order, meta = [], [], {}
    for j in range(3):
        m = np.zeros((64, 64), np.uint8); cv2.circle(m, (16 + 16 * j, 32), 7, 1, -1)
        mb = m > 0; ys, xs = np.where(mb); u = ids.new_uid()
        recs.append({"iuid": u, "row": j, "inst_id": j, "image_id": 1000, "H": 64, "W": 64, "score": 0.6,
                     "rle": _rle(mb), "file_name": str(p), "abs_path": str(p), "batch_id": "b",
                     "cx": float(xs.mean() / 64), "cy": float(ys.mean() / 64), "bw": 0.2, "bh": 0.2,
                     "box_area": 0.04, "mask_area_frac": float(mb.mean())})
        order.append(u); meta[u] = InstanceMeta(iuid=u, batch_id="b", row=j, image_id=1000)
    feats = {"coords": np.random.default_rng(1).normal(0, 1, (3, 6)).astype(np.float32),
             "shape": np.random.default_rng(2).normal(0, 1, (3, 5)).astype(np.float32), "_shape_cols": ["c"] * 5}
    if with_decoder:
        feats["decoder"] = np.random.default_rng(3).normal(0, 1, (3, 8)).astype(np.float32)
    eng.collection = {"records": recs, "n_images": 1, "feats": feats}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    return eng, order


def test_available_features(tmp_path):
    eng, _ = _engine(tmp_path, with_decoder=False)
    assert eng.available_features() == ["coords", "shape"]          # no decoder; '_shape_cols' excluded
    eng2, _ = _engine(tmp_path / "p2", with_decoder=True)
    assert "decoder" in eng2.available_features()


def test_cluster_absent_feature_raises_clear(tmp_path):
    eng, _ = _engine(tmp_path, with_decoder=False)
    with pytest.raises(ValueError, match="none of the selected features"):
        eng.cluster({"raddino": 1.0})                               # absent -> clear error, not np.concatenate crash
    info = eng.cluster({"decoder": 1.0, "coords": 1.0})             # decoder absent, coords present -> survives
    assert info["counts"]


def test_train_classifier_absent_feature_returns_error(tmp_path):
    eng, order = _engine(tmp_path, with_decoder=False)
    cA, cB = eng.state.add_class("A"), eng.state.add_class("B")
    for i in (0, 1):
        eng.state.meta[order[i]].assigned_class = cA
    eng.state.meta[order[2]].assigned_class = cB
    rep = eng.train_classifier({"decoder": 1.0, "raddino": 1.0})    # both absent -> error report, no crash
    assert "error" in rep and "available" in rep["error"]


def test_image_overlay_no_labels_attribute_error(tmp_path):
    eng, _ = _engine(tmp_path, with_decoder=True)
    eng.cluster({"decoder": 1.0})                                   # sets _cluster so color_by='partition' path runs
    ov = eng.image_overlay(1000, color_by="partition")             # would AttributeError on the old self._labels()
    assert ov.shape == (64, 64, 3) and ov.dtype == np.uint8
    assert eng.image_overlay(1000, color_by="class").shape == (64, 64, 3)
