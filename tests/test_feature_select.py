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


def test_train_skips_singleton_classes(tmp_path):
    """≥2 classes have ≥2 instances + a singleton class -> train on the qualifying ones, skip the
    singleton (regression: old `min(bincount) < 2` rejected the whole thing)."""
    eng, order = _engine(tmp_path, with_decoder=True)               # 3 instances over 1 image
    # need >=2 instances in each of 2 classes + a singleton -> add 3 more instances by re-using rows
    cA, cB, cC = eng.state.add_class("A"), eng.state.add_class("B"), eng.state.add_class("C")
    # A: order[0], order[1]; B: order[2] + a duplicate-meta trick isn't possible, so widen the fixture:
    import cv2
    from tools.curator import ids
    from tools.curator.state import InstanceMeta
    feats = eng.collection["feats"]["decoder"]
    base = eng.collection["records"][0]
    for _ in range(3):                                              # append 3 more instances (rows 3,4,5)
        u = ids.new_uid(); row = len(eng.collection["records"])
        r = dict(base); r["iuid"] = u; r["row"] = row
        eng.collection["records"].append(r)
        eng.collection["feats"]["decoder"] = np.vstack([eng.collection["feats"]["decoder"], feats[0:1]])
        eng.state.order.append(u); eng.state.meta[u] = InstanceMeta(iuid=u, batch_id="b", row=row, image_id=1000)
    eng.state.rebuild_rows()
    o = eng.state.order
    eng.state.meta[o[0]].assigned_class = cA; eng.state.meta[o[1]].assigned_class = cA   # A:2
    eng.state.meta[o[2]].assigned_class = cB; eng.state.meta[o[3]].assigned_class = cB   # B:2
    eng.state.meta[o[4]].assigned_class = cC                                             # C:1 (singleton)
    rep = eng.train_classifier({"decoder": 1.0})
    assert "error" not in rep                                       # NOT rejected by the singleton
    assert rep["n_classes"] == 2 and set(rep["classes"]) == {cA, cB}
    assert rep["skipped_classes"] == [cC] and rep["skipped_names"] == ["C"]


def test_train_needs_two_trainable_classes(tmp_path):
    eng, order = _engine(tmp_path, with_decoder=True)
    cA, cB = eng.state.add_class("A"), eng.state.add_class("B")
    eng.state.meta[order[0]].assigned_class = cA; eng.state.meta[order[1]].assigned_class = cA  # A:2
    eng.state.meta[order[2]].assigned_class = cB                                                # B:1
    rep = eng.train_classifier({"decoder": 1.0})                    # only 1 trainable class -> clear error
    assert "error" in rep and "counts" in rep["error"]


def test_classifier_preview_renders_and_pr_names(tmp_path):
    """do_predict returns (msg, rows, preds); the classifier-preview @gr.render body builds >=1 gr.Image
    from preds (the reliable path, vs the gr.Gallery that didn't render); the P/R legend uses class NAMES."""
    from gradio.context import LocalContext
    from tools.curator import app
    eng, order = _engine(tmp_path, with_decoder=True)             # 3 instances; widen to 2 classes x 2+
    from tools.curator import ids
    from tools.curator.state import InstanceMeta
    feats = eng.collection["feats"]["decoder"]; base = eng.collection["records"][0]
    for _ in range(3):
        u = ids.new_uid(); row = len(eng.collection["records"]); r = dict(base); r["iuid"] = u; r["row"] = row
        eng.collection["records"].append(r)
        eng.collection["feats"]["decoder"] = np.vstack([eng.collection["feats"]["decoder"], feats[0:1]])
        eng.state.order.append(u); eng.state.meta[u] = InstanceMeta(iuid=u, batch_id="b", row=row, image_id=1000)
    eng.state.rebuild_rows(); o = eng.state.order
    cA, cB = eng.state.add_class("letters"), eng.state.add_class("leads")
    eng.state.meta[o[0]].assigned_class = cA; eng.state.meta[o[1]].assigned_class = cA
    eng.state.meta[o[2]].assigned_class = cB; eng.state.meta[o[3]].assigned_class = cB
    app.ENG = eng
    try:
        rep = eng.train_classifier({"decoder": 1.0})
        fig = app._pr_fig(rep.get("pr", {}))
        leg = fig.axes[0].get_legend()
        labels = [t.get_text() for t in leg.get_texts()] if leg else []
        assert any("letters" in lbl or "leads" in lbl for lbl in labels)            # NAMES in legend
        assert not any(c.startswith("c_") for lbl in labels for c in lbl.split())   # no raw cids
        msg, rows, preds, _excl = app.do_predict(0.0)                                # (msg, rows, preds, excluded-reset)
        assert isinstance(rows, list) and isinstance(preds, list) and preds
        # the classifier-preview @gr.render body must build gr.Image cells from preds (the reliable path).
        # Two renderables take 2 inputs (in-image grid: image_id,nonce; classifier preview: pred_state,pred_n);
        # apply(preds, 12) builds images for the classifier one and errors-out (caught) for the in-image one.
        import gradio as gr
        demo = app.build_app(str(tmp_path))
        tok = LocalContext.blocks_config.set(demo.default_config)
        try:
            n_images = 0
            with demo:
                for r in (r for r in demo.renderables if len(r.inputs) == 2):
                    before = len(demo.blocks)
                    try:
                        r.apply(preds, 12)
                    except Exception:
                        continue
                    n_images += sum(1 for b in list(demo.blocks.values())[before:] if type(b).__name__ == "Image")
            assert n_images >= 1                                     # classifier preview rendered >=1 image
        finally:
            LocalContext.blocks_config.reset(tok)
    finally:
        app.ENG = None


def test_per_class_apply_youden_and_unassigned_only(tmp_path):
    """v6.0: per-class apply assigns ONLY the chosen class; predict scores unassigned-only; report
    carries a Youden-J recommended threshold per class."""
    import cv2
    from tools.curator import ids
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    p = tmp_path / "im0.png"
    cv2.imwrite(str(p), (np.random.default_rng(0).random((64, 64, 3)) * 120 + 40).astype(np.uint8))
    centers = [[6, 0, 0, 0], [0, 6, 0, 0]]                         # two well-separated clusters A / B
    recs, order, meta, dec = [], [], {}, []
    for j in range(16):
        m = np.zeros((64, 64), np.uint8); cv2.circle(m, (8 + 3 * j, 32), 4, 1, -1); mb = m > 0; u = ids.new_uid()
        recs.append({"iuid": u, "row": j, "inst_id": j, "image_id": 1000, "H": 64, "W": 64, "score": 0.6,
                     "rle": _rle(mb), "file_name": str(p), "abs_path": str(p), "batch_id": "b",
                     "cx": 0.3, "cy": 0.5, "bw": 0.1, "bh": 0.1, "box_area": 0.01, "mask_area_frac": float(mb.mean())})
        dec.append(np.random.default_rng(j).normal(centers[j % 2], 0.3, 4)); order.append(u)
        meta[u] = InstanceMeta(iuid=u, batch_id="b", row=j, image_id=1000)
    eng.collection = {"records": recs, "n_images": 1, "feats": {"decoder": np.array(dec, np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    cA, cB = eng.state.add_class("A"), eng.state.add_class("B")
    for j in (0, 2, 4):
        eng.state.meta[order[j]].assigned_class = cA              # A examples (even -> centre A)
    for j in (1, 3, 5):
        eng.state.meta[order[j]].assigned_class = cB              # B examples (odd -> centre B)
    rep = eng.train_classifier({"decoder": 1.0})
    cur = rep["pr"]["curves"]
    assert cur and all("youden" in v and 0.0 <= v["youden"] <= 1.0 for v in cur.values())   # Youden-J per class

    # predict scores ONLY unassigned
    preds = eng.predict_and_threshold(0.0)
    assert preds and all(eng.state.meta[u].assigned_class is None for u, _, _ in preds)

    # per-class apply: only class A gets assigned; no classifier-assigned B
    n = eng.apply_predictions(0.0, only_class=cA)
    clf_assigned = [(u, m.assigned_class) for u, m in eng.state.meta.items() if m.assign_source == "classifier"]
    assert n > 0 and clf_assigned and all(c == cA for _, c in clf_assigned)   # ONLY A applied

    # after apply, those instances are no longer scored (unassigned-only)
    preds2 = eng.predict_and_threshold(0.0)
    assigned_now = {u for u, _ in clf_assigned}
    assert assigned_now.isdisjoint({u for u, _, _ in preds2})


def test_knn_classifier_works_with_one_per_class_and_exclude(tmp_path):
    """v7.3: kNN trains with a SINGLE sample per class (factored needs >=2); apply respects exclude."""
    import cv2
    from tools.curator import ids
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    p = tmp_path / "im0.png"
    cv2.imwrite(str(p), (np.random.default_rng(0).random((64, 64, 3)) * 120 + 40).astype(np.uint8))
    centers = [[6, 0, 0, 0], [0, 6, 0, 0]]
    recs, order, meta, dec = [], [], {}, []
    for j in range(12):
        m = np.zeros((64, 64), np.uint8); cv2.circle(m, (6 + 4 * j, 32), 3, 1, -1); mb = m > 0; u = ids.new_uid()
        recs.append({"iuid": u, "row": j, "inst_id": j, "image_id": 1000, "H": 64, "W": 64, "score": 0.6,
                     "rle": _rle(mb), "file_name": str(p), "abs_path": str(p), "batch_id": "b",
                     "cx": 0.3, "cy": 0.5, "bw": 0.1, "bh": 0.1, "box_area": 0.01, "mask_area_frac": float(mb.mean())})
        dec.append(np.random.default_rng(j).normal(centers[j % 2], 0.2, 4)); order.append(u)
        meta[u] = InstanceMeta(iuid=u, batch_id="b", row=j, image_id=1000)
    eng.collection = {"records": recs, "n_images": 1, "feats": {"decoder": np.array(dec, np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    cA, cB = eng.state.add_class("A"), eng.state.add_class("B")
    eng.state.meta[order[0]].assigned_class = cA           # ONE sample of A (even centre)
    eng.state.meta[order[1]].assigned_class = cB           # ONE sample of B (odd centre)
    # with 1/class the LOGREG path errors (needs >=2); kNN is the way around it
    assert "error" in eng.train_classifier({"decoder": 1.0}, algo="logreg")
    rep = eng.train_classifier({"decoder": 1.0}, algo="knn", knn_k=1, knn_metric="cosine")
    assert "error" not in rep and rep["algo"] == "knn" and rep["n_classes"] == 2   # trained on 1/class!
    preds = eng.predict_and_threshold(0.0)
    assert preds and all(c in (cA, cB) for _, c, _ in preds)
    # exclude: applying with an instance excluded must not assign it
    cand = preds[0][0]
    eng.apply_predictions(0.0, exclude={cand})
    assert eng.state.meta[cand].assigned_class is None      # excluded -> stayed unassigned


def test_knn_confidence_tracks_distance_to_nearest_class_sample():
    """v7.4: kNN confidence = exp(-distance-to-nearest-class-sample / margin) → monotonically
    decreasing with distance; background gates open-set rejection."""
    from tools.curator.classify import KNNClassifier
    A = np.array([[5, 0], [5.1, 0.1], [4.9, -0.1]], np.float32)
    B = np.array([[0, 5], [0.1, 5.1], [-0.1, 4.9]], np.float32)
    bg = np.array([[0, 0], [0.2, 0.1]], np.float32)
    clf = KNNClassifier(["A", "B"], [A, B], bg, k=1, metric="euclidean")
    conf = lambda q: float(clf.proba(np.array([q], np.float32))[0].max())
    on, near, mid = conf([5, 0]), conf([4.3, 0.4]), conf([3.0, 0.3])
    assert on == 1.0 and on > near > mid > 0.0                  # closer -> higher confidence
    assert clf.proba(np.array([[0.1, 0.1]], np.float32))[0].max() == 0.0  # on background -> rejected (open-set)
    assert int(clf.proba(np.array([[0.2, 4.8]], np.float32))[0].argmax()) == 1  # near B -> predicts B


def test_image_overlay_no_labels_attribute_error(tmp_path):
    eng, _ = _engine(tmp_path, with_decoder=True)
    eng.cluster({"decoder": 1.0})                                   # sets _cluster so color_by='partition' path runs
    ov = eng.image_overlay(1000, color_by="partition")             # would AttributeError on the old self._labels()
    assert ov.shape == (64, 64, 3) and ov.dtype == np.uint8
    assert eng.image_overlay(1000, color_by="class").shape == (64, 64, 3)
    bare = eng.image_overlay(1000, color_by="partition", show_masks=False)   # toggle off => no mask fills/contours
    assert bare.shape == ov.shape and bare.dtype == np.uint8
    assert not np.array_equal(bare, ov)                            # masks were actually drawn on `ov`
