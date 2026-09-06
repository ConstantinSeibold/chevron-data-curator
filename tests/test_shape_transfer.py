"""Few-shot shape-transfer refinement: build a k-shot template from reference mask(s), warp into each
partition member's bbox, SAM/SAM-HQ-decode toward it, gate, preview, commit (undoable).
SAM is not in CI, so the non-SAM logic is unit-tested and the decode is stubbed for orchestration.
Run: pytest tests/test_shape_transfer.py -q
"""
from __future__ import annotations

import numpy as np


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


def _circle(H=128, r=18, cx=64, cy=64):
    import cv2
    m = np.zeros((H, H), np.uint8); cv2.circle(m, (cx, cy), r, 1, -1)
    return m > 0


def _line(H=128):
    import cv2
    m = np.zeros((H, H), np.uint8); cv2.line(m, (40, 40), (90, 90), 1, 3)
    return m > 0


def _engine(tmp_path, masks, draw=False):
    """One instance per mask, each on its own 128×128 PNG, with feats incl. shapecoord (so the write path's
    shape-feature recompute runs). draw=True paints the mask bright into the image (so the vessel trace has a
    ridge to follow)."""
    import cv2
    from chevron import ids
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    recs, order, meta, dec, shp = [], [], {}, [], []
    for j, mb in enumerate(masks):
        u = ids.new_uid()
        p = tmp_path / f"im{j}.png"
        img = (np.random.default_rng(j).random((128, 128, 3)) * 80 + 20).astype(np.uint8)
        if draw:
            img[mb] = 220
        cv2.imwrite(str(p), img)
        ys, xs = np.where(mb)
        recs.append({"iuid": u, "row": j, "inst_id": j, "image_id": 1000 + j, "H": 128, "W": 128, "score": 0.9,
                     "rle": _rle(mb), "file_name": str(p), "abs_path": str(p), "batch_id": "b",
                     "cx": float(xs.mean() / 128), "cy": float(ys.mean() / 128),
                     "bw": float((xs.max() - xs.min() + 1) / 128), "bh": float((ys.max() - ys.min() + 1) / 128),
                     "box_area": 0.1, "mask_area_frac": float(mb.mean())})
        order.append(u); meta[u] = InstanceMeta(u, "b", j, 1000 + j)
        dec.append(np.zeros(4, np.float32)); shp.append(np.zeros(29, np.float32))
    eng.collection = {"records": recs, "n_images": len(masks),
                      "feats": {"decoder": np.array(dec, np.float32), "shapecoord": np.array(shp, np.float32),
                                "_shapecoord_cols": ["c"] * 29}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    return eng, order


# ---- pure pieces (no SAM) --------------------------------------------------
def test_shape_template_and_warp():
    from chevron.engine import CuratorEngine
    T = CuratorEngine._shape_template([_circle(), _circle(r=12)])
    assert T.shape == (256, 256) and T.min() >= 0.0 and T.max() <= 1.0 and T.sum() > 0
    assert CuratorEngine._shape_template([np.zeros((128, 128), bool)]) is None    # all-empty -> None
    E = CuratorEngine._warp_template_to_box(T, (40, 40, 90, 90), 128, 128)
    assert E.dtype == bool and E.shape == (128, 128)
    assert E[40:90, 40:90].any() and not E[:38, :].any()                          # inside the box, not outside


def test_members_resolution_and_radino_gate(tmp_path, monkeypatch):
    eng, order = _engine(tmp_path, [_circle()] * 4)
    eng.assign(order, "A")
    cid = eng.state.class_id_by_name("A")
    pid, members, sk = eng.shape_transfer_members([order[0]])
    assert pid == f"class:{cid}" and members == order[1:] and sk == 0           # refs excluded
    emb = {order[0]: [1, 0], order[1]: [1, 0], order[2]: [0, 1], order[3]: [0, 1]}  # only order[1] resembles ref
    monkeypatch.setattr(eng, "_instance_ref_embeddings",
                        lambda iuids: np.array([emb[u] for u in iuids], np.float32))
    pid, members, sk = eng.shape_transfer_members([order[0]], match_thresh=0.9)
    assert members == [order[1]] and sk == 2


def test_set_mask_nohist_and_undo(tmp_path):
    eng, order = _engine(tmp_path, [_circle(), _circle()])
    eng.assign(order, "A")
    u = order[1]; orig = int(eng._mask(u).sum())
    new = _circle(r=8)
    tok = eng.history.begin(eng.state, [u], [])
    eng._set_mask_nohist(u, new, op={"name": "shape_transfer", "kw": {"refs": [order[0]]}})
    eng.history.commit(eng.state, tok, "shape_transfer", "x"); eng._after_mutation()
    assert eng.state.meta[u].refined is True and eng.state.meta[u].rule_ops[0]["name"] == "shape_transfer"
    assert int(eng._mask(u).sum()) == int(new.sum()) != orig                    # effective mask is the new one
    eng.undo()
    assert eng.state.meta[u].refined is False and int(eng._mask(u).sum()) == orig  # reverts


# ---- orchestration with the SAM decode stubbed (decode == warped template) --
def test_shape_transfer_applies_with_stubbed_sam(tmp_path, monkeypatch):
    from chevron import refine
    monkeypatch.setattr(refine, "sam_refine", lambda gray, E, **kw: E)
    eng, order = _engine(tmp_path, [_circle()] * 4)
    eng.assign(order, "A")
    res = eng.shape_transfer([order[0]])
    assert res["applied"] == 3 and res["gated_out"] == 0 and res["pid"].startswith("class:")
    for u in order[1:]:
        assert eng.state.meta[u].refined is True and eng.state.meta[u].rule_ops[0]["name"] == "shape_transfer"
    assert eng.state.meta[order[0]].refined is False                            # the reference is untouched


def test_shape_transfer_agreement_gate_drops_mismatch(tmp_path, monkeypatch):
    from chevron import refine
    monkeypatch.setattr(refine, "sam_refine", lambda gray, E, **kw: E)
    eng, order = _engine(tmp_path, [_circle(), _circle(), _line()])           # ref + matching circle + a line
    eng.assign(order, "A")
    res = eng.shape_transfer([order[0]], agree_iou=0.5)
    assert res["applied"] == 1 and res["gated_out"] == 1                       # disk-into-line-bbox IoU < 0.5
    assert eng.state.meta[order[1]].refined is True and eng.state.meta[order[2]].refined is False


def test_shape_transfer_preview_no_writes(tmp_path, monkeypatch):
    from chevron import refine
    monkeypatch.setattr(refine, "sam_refine", lambda gray, E, **kw: E)
    eng, order = _engine(tmp_path, [_circle()] * 4)
    eng.assign(order, "A")
    res = eng.shape_transfer_preview([order[0]], sample=2)
    assert res["n_members"] == 3 and res["shown"] == 2 and res["truncated"] == 1
    assert len(res["items"]) == 2 and all(it["before"].shape[2] == 3 for it in res["items"])  # rgb panels
    assert all(eng.state.meta[u].refined is False for u in order)              # preview wrote nothing


# ---- line partitions: auto-route to the vessel trace (no SAM) --------------
def _vline(H=128, x=64, y0=20, y1=108, w=2):
    import cv2
    m = np.zeros((H, H), np.uint8); cv2.line(m, (x, y0), (x, y1), 1, w)
    return m > 0


def test_partition_kind_and_reference_line_ops():
    from chevron.engine import CuratorEngine
    assert CuratorEngine._partition_shape_kind.__name__                       # method exists
    # the width calibration: a 2-px line -> small tube width fed to vessel_extend/line_centerline
    import types
    eng = types.SimpleNamespace(_mask=lambda u: _vline(w=2))
    ops, w = CuratorEngine._reference_line_ops(eng, ["r0", "r1"])
    assert [o["name"] for o in ops] == ["vessel_extend", "line_centerline"] and 1 <= w <= 6
    assert ops[0]["kw"]["max_width"] == w


def test_line_partition_routes_to_vessel_trace_no_sam(tmp_path, monkeypatch):
    from chevron import refine
    def _boom(*a, **k):
        raise AssertionError("SAM must NOT be called for a line partition")
    monkeypatch.setattr(refine, "sam_refine", _boom)
    eng, order = _engine(tmp_path, [_vline(x=60), _vline(x=64), _vline(x=68)], draw=True)
    eng.assign(order, "L")
    res = eng.shape_transfer([order[0]])
    assert res["kind"] == "line" and res["width"] >= 1 and res["applied"] >= 1   # routed to vessel trace
    for u, _ in zip(order[1:], range(res["applied"])):
        assert eng.state.meta[order[1]].refined is True
    # provenance records the line mode
    assert eng.state.meta[order[1]].rule_ops[0]["kw"]["kind"] == "line"


def test_blob_partition_still_routes_to_sam(tmp_path, monkeypatch):
    from chevron import refine
    monkeypatch.setattr(refine, "sam_refine", lambda gray, E, **kw: E)
    eng, order = _engine(tmp_path, [_circle()] * 3)
    eng.assign(order, "B")
    res = eng.shape_transfer([order[0]])
    assert res["kind"] == "blob" and res["applied"] == 2


# ---- server endpoints ------------------------------------------------------
def _client(eng):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    return TestClient(create_app(engine=eng))


def test_endpoints_preview_and_commit(tmp_path, monkeypatch):
    from chevron import refine
    monkeypatch.setattr(refine, "sam_refine", lambda gray, E, **kw: E)
    eng, order = _engine(tmp_path, [_circle()] * 4)
    eng.assign(order, "A")
    c = _client(eng)
    pv = c.post("/api/shape_transfer_preview", json={"ref_iuids": [order[0]]}).json()
    assert pv["n_members"] == 3 and len(pv["items"]) == 3
    assert pv["items"][0]["before"].startswith("data:image/png") and "iou" in pv["items"][0]
    ap = c.post("/api/shape_transfer", json={"ref_iuids": [order[0]]}).json()
    assert ap["ok"] and ap["applied"] == 3


def test_endpoint_sam_missing_returns_400(tmp_path, monkeypatch):
    from chevron import refine
    monkeypatch.setattr(refine, "find_sam_checkpoint", lambda ckpt=None, family=None: (None, None))
    monkeypatch.setattr(refine, "sam_available", lambda: True)
    eng, order = _engine(tmp_path, [_circle(), _circle()])
    eng.assign(order, "A")
    r = _client(eng).post("/api/shape_transfer", json={"ref_iuids": [order[0]]})
    assert r.status_code == 400 and "checkpoint" in r.json()["detail"].lower()


def test_endpoint_no_partition_400(tmp_path):
    eng, order = _engine(tmp_path, [_circle(), _circle()])                     # never assigned -> no partition
    r = _client(eng).post("/api/shape_transfer", json={"ref_iuids": [order[0]]})
    assert r.status_code == 400


# ---- hand-draw mask editor (edit_view + set_mask) --------------------------
def _png_of(mask):
    import cv2
    ok, buf = cv2.imencode(".png", (mask.astype(np.uint8) * 255))
    return buf.tobytes()


def test_edit_view_returns_crop_and_mapping(tmp_path):
    eng, order = _engine(tmp_path, [_circle()])
    v = eng.edit_view(order[0])
    assert {"img", "mask", "box", "w", "h"} <= set(v)
    assert v["img"].shape[2] == 3 and v["mask"].shape == (v["h"], v["w"])
    x1, y1, x2, y2 = v["box"]
    assert 0 <= x1 < x2 <= 128 and 0 <= y1 < y2 <= 128                          # bbox crop within the image
    assert eng.edit_view(order[0], context=True)["box"] == [0, 0, 128, 128]     # context = whole image


def test_set_mask_writes_box_preserves_outside_and_undo(tmp_path):
    eng, order = _engine(tmp_path, [_circle(cx=40, cy=40), _circle(cx=90, cy=90)])
    eng.assign(order, "A")
    u = order[0]
    # current mask is a circle at (40,40); draw a NEW filled square in a box on the OTHER side
    box = [70, 70, 110, 110]
    drawn = np.ones((40, 40), bool)                                            # canvas-res: fully painted box
    before = eng._mask(u).copy()
    res = eng.set_mask(u, _png_of(drawn), box)
    assert res["area"] > 0 and eng.state.meta[u].refined is True
    m = eng._mask(u)
    assert m[80:100, 80:100].all()                                             # the drawn box is on
    assert m[35:45, 35:45].any() == before[35:45, 35:45].any()                 # original circle (outside box) preserved
    assert eng.state.meta[u].rule_ops[0]["name"] == "draw"
    eng.undo()
    assert eng.state.meta[u].refined is False and np.array_equal(eng._mask(u), before)


def test_edit_view_and_set_mask_endpoints(tmp_path):
    import base64
    eng, order = _engine(tmp_path, [_circle()])
    eng.assign(order, "A")
    c = _client(eng)
    v = c.get(f"/api/edit_view?iuid={order[0]}").json()
    assert v["img"].startswith("data:image/png") and v["mask"].startswith("data:image/png") and "box" in v
    png = "data:image/png;base64," + base64.b64encode(_png_of(np.ones((v["h"], v["w"]), bool))).decode()
    r = c.post("/api/set_mask", json={"iuid": order[0], "png": png, "box": v["box"]}).json()
    assert r["ok"] and r["area"] > 0
