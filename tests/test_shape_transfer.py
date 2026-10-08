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


def test_edit_view_seeded_with_ops_is_the_chain_result(tmp_path):
    """'touch up result': edit_view(ops=...) seeds the editor with the refine chain's output (same base as
    refine_preview), windowed on the union with the current mask — not the current mask itself."""
    eng, order = _engine(tmp_path, [_circle()])
    u = order[0]
    plain = eng.edit_view(u)
    grow = [{"name": "dilate", "kw": {"k": 9, "max_contrast": 1.0}}]
    seeded = eng.edit_view(u, ops=grow)
    assert (seeded["mask"] > 0).sum() > (plain["mask"] > 0).sum() * (seeded["w"] * seeded["h"]) / (plain["w"] * plain["h"])
    sx1, sy1, sx2, sy2 = seeded["box"]; px1, py1, px2, py2 = plain["box"]
    assert sx1 <= px1 and sy1 <= py1 and sx2 >= px2 and sy2 >= py2              # window covers the current mask too
    assert eng.state.meta[u].refined is False                                  # read-only: nothing written


def test_mask_state_tracks_hand_drawn(tmp_path):
    eng, order = _engine(tmp_path, [_circle()])
    eng.assign(order, "A")
    u = order[0]
    assert eng.mask_state(u) == "original"
    v = eng.edit_view(u)
    eng.set_mask(u, _png_of(np.ones((v["h"], v["w"]), bool)), v["box"])
    assert eng.mask_state(u) == "hand-drawn"
    eng.apply_refine(u, [{"name": "fill"}])                                     # ops on top of a drawing: still hand-drawn
    assert eng.mask_state(u) == "hand-drawn"
    eng.undo(); eng.undo()
    assert eng.mask_state(u) == "original"


def test_edit_view_endpoint_accepts_ops_and_preview_reports_state(tmp_path):
    import json
    eng, order = _engine(tmp_path, [_circle()])
    c = _client(eng)
    ops = json.dumps([{"name": "dilate", "kw": {"k": 5, "max_contrast": 1.0}}])
    assert "box" in c.get(f"/api/edit_view?iuid={order[0]}&ops={ops}").json()
    assert c.get(f"/api/edit_view?iuid={order[0]}&ops=notjson").status_code == 400
    r = c.post("/api/refine_preview", json={"iuid": order[0], "ops": []}).json()
    assert r["mask_state"] == "original"


def test_refine_pane_is_queue_stage_and_fix_panel():
    """One instance in focus: queue → stage → fix panel, with hand editing in the Draw mode and the bulk
    step (apply to others) behind a preview. Each control exists exactly once."""
    import re
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "chevron/web/index.html").read_text()
    tab = html[html.index('id="tab-refine"'):html.index("<!-- CLASSIFIER -->")]
    order = [tab.index(f'id="{i}"') for i in ("rq", "rs", "rp")]
    assert order == sorted(order)
    draw = re.search(r'data-pane="draw">(.*?)</div>\s*<div class="rpPane"', tab, re.S).group(1)
    assert 'id="rfEditMask"' in draw and 'id="rfTouchUp"' in draw
    assert tab.index('id="raPreview"') < tab.index('id="raApply"') and "disabled" in \
        tab[tab.index('id="raApply"'):tab.index('id="raApply"') + 60]
    for i in ("rfEditMask", "rpAccept", "raApply", "mcView"):
        assert html.count(f'id="{i}"') == 1


# ---- editor: SAM click correction, revert --------------------------------------------------------
class _ClickPredictor:
    """SamPredictor double: records prompts; decodes a disc around the first include-click."""
    def __init__(self):
        self.n_set, self.calls, self.features, self.is_image_set = 0, [], None, False

    def set_image(self, img):
        self.n_set += 1; self.shape = img.shape[:2]; self.features = object(); self.is_image_set = True

    def predict(self, point_coords, point_labels, box=None, mask_input=None, multimask_output=True):
        import cv2
        self.calls.append(dict(pts=np.asarray(point_coords), lbl=np.asarray(point_labels), box=box,
                               mask_input=mask_input, multi=multimask_output))
        m = np.zeros(self.shape, np.uint8)
        x, y = point_coords[list(point_labels).index(1)]
        cv2.circle(m, (int(x), int(y)), 10, 1, -1)
        n = 3 if multimask_output else 1
        return np.stack([m > 0] * n), np.linspace(0.5, 0.9, n), None


def test_sam_clicks_prompts_reuse_embedding_and_refuse_medsam(monkeypatch):
    from chevron import refine as rf
    fp = _ClickPredictor()
    monkeypatch.setattr(rf, "_resolve_predictor", lambda ckpt=None, mt=None, model="auto": (fp, model))
    rf._CLICK_EMB.update(pred=None, feat=None, key=None)
    img = np.zeros((100, 200, 3), np.uint8)
    out = rf.sam_clicks(img, [[50, 40]], [1], prior=None, key="a", model="sam")
    assert out[40, 50] and fp.calls[-1]["multi"] and fp.calls[-1]["box"] is None      # 1 click, no prior → multimask
    prior = out.copy()
    rf.sam_clicks(img, [[50, 40], [60, 40]], [1, 0], prior=prior, key="a", model="sam")
    c = fp.calls[-1]
    assert fp.n_set == 1                                               # same image key → embedding reused
    assert not c["multi"] and c["mask_input"].shape == (1, 256, 256) and c["box"] is not None
    assert c["mask_input"][0, 200, 200] == -8                          # 100×200 image: bottom rows are padding
    rf.sam_clicks(img, [[50, 40]], [1], key="b", model="sam")
    assert fp.n_set == 2                                               # new image → new embedding
    import pytest
    with pytest.raises(RuntimeError, match="MedSAM"):
        rf.sam_clicks(img, [[50, 40]], [1], key="b", model="medsam")


def test_edit_sam_maps_canvas_clicks_into_the_image(tmp_path, monkeypatch):
    from chevron import refine as rf
    eng, order = _engine(tmp_path, [_circle()])
    u = order[0]
    seen = {}

    def fake(img, pts, labels, *, prior, key, model):
        seen.update(pts=pts, labels=labels, key=key, prior=prior.copy())
        m = np.zeros(img.shape[:2], bool); m[60:70, 60:70] = True
        return m
    monkeypatch.setattr(rf, "sam_clicks", fake)
    v = eng.edit_view(u)
    x1, y1, x2, y2 = v["box"]
    res = eng.edit_sam(u, _png_of(v["mask"] > 0), v["box"], [[0, 0], [v["w"] - 1, v["h"] - 1]], [1, 0])
    (ax, ay), (bx, by) = seen["pts"]
    assert x1 <= ax < x1 + 2 and y1 <= ay < y1 + 2 and x2 - 2 < bx <= x2 and y2 - 2 < by <= y2
    assert seen["key"] == u and seen["labels"] == [1, 0] and seen["prior"].sum() == eng._mask(u).sum()
    assert res["mask"].shape == (v["h"], v["w"]) and res["mask"].max() == 255
    assert eng.state.meta[u].refined is False                                  # nothing written before Save


def test_revert_mask_endpoint(tmp_path):
    eng, order = _engine(tmp_path, [_circle(), _circle(cx=40, cy=40)])
    eng.assign(order, "A")
    u = order[0]
    before = eng._mask(u).copy()
    v = eng.edit_view(u)
    eng.set_mask(u, _png_of(np.ones((v["h"], v["w"]), bool)), v["box"])
    c = _client(eng)
    r = c.post("/api/revert_mask", json={"iuid": u}).json()
    assert r["mask_state"] == "original" and np.array_equal(eng._mask(u), before)
    assert c.post("/api/revert_mask", json={"iuid": "nope"}).status_code == 404
    eng.state.meta[order[1]].merge_members = [order[0]]
    assert c.post("/api/revert_mask", json={"iuid": order[1]}).status_code == 400


# ---- "apply to others": scope → sampled preview → commit -----------------------------------------
def test_refine_scope_preview_samples_the_group_without_writing(tmp_path):
    eng, order = _engine(tmp_path, [_circle(), _circle(cx=40, cy=40), _circle(cx=90, cy=90)])
    eng.assign(order, "A")
    grow = [{"name": "dilate", "kw": {"k": 7, "max_contrast": 1.0}}]
    sc = eng.refine_scope_members(order[0], "group")
    assert sc["members"] == order[1:] and sc["label"] == "class A" and sc["cls"] == "A"
    r = eng.refine_scope_preview(order[0], method="recipe", scope="group", ops=grow, sample=8)
    assert r["n_members"] == 2 and r["shown"] == 2 and {it["iuid"] for it in r["items"]} == set(order[1:])
    assert r["summary"]["changed"] == 2 and all(it["iou"] < 0.98 for it in r["items"])
    assert not any(eng.state.meta[u].refined for u in order)                     # dry-run
    assert eng.refine_scope_preview(order[0], method="recipe", scope="group", ops=grow, sample=1,
                                    seed=3)["shown"] == 1


def test_refine_scope_selection_and_errors(tmp_path):
    eng, order = _engine(tmp_path, [_circle(), _circle(cx=40, cy=40), _circle(cx=90, cy=90)])
    eng.assign(order, "A")
    sc = eng.refine_scope_members(order[0], "selection", iuids=[order[0], order[2], "nope"])
    assert sc["members"] == [order[2]]                                           # never the reference itself
    assert "error" in eng.refine_scope_members(order[0], "similar")               # needs τ
    assert "error" in eng.refine_scope_preview(order[0], method="recipe", scope="group", ops=[])
    assert "error" in eng.refine_scope_preview(order[0], method="transfer", scope="selection",
                                               iuids=[order[1]])


def test_refine_scope_apply_recipe_saves_rule_and_undoes(tmp_path):
    eng, order = _engine(tmp_path, [_circle(), _circle(cx=40, cy=40), _circle(cx=90, cy=90)])
    eng.assign(order, "A")
    grow = [{"name": "dilate", "kw": {"k": 7, "max_contrast": 1.0}}]
    before = {u: int(eng._mask(u).sum()) for u in order}
    r = eng.refine_scope_apply(order[0], method="recipe", scope="group", ops=grow, save_rule=True)
    assert r["applied"] == 2 and r["rule_saved"]
    assert all(int(eng._mask(u).sum()) > before[u] for u in order[1:])
    assert int(eng._mask(order[0]).sum()) == before[order[0]]                   # the reference is Accept's job
    assert [o["name"] for o in eng.class_rule_for("A")] == ["dilate"]
    eng.undo()
    assert all(int(eng._mask(u).sum()) == before[u] for u in order)


def test_refine_scope_endpoints(tmp_path):
    eng, order = _engine(tmp_path, [_circle(), _circle(cx=40, cy=40)])
    eng.assign(order, "A")
    c = _client(eng)
    body = {"ref": order[0], "method": "recipe", "scope": "group",
            "ops": [{"name": "dilate", "kw": {"k": 5, "max_contrast": 1.0}}]}
    r = c.post("/api/refine_scope/preview", json=body).json()
    assert r["items"][0]["before"].startswith("data:image/png") and r["n_members"] == 1
    assert c.post("/api/refine_scope/preview", json={**body, "scope": "bogus"}).status_code == 400
    a = c.post("/api/refine_scope/apply", json=body).json()
    assert a["applied"] == 1 and "stats" in a
    info = c.post("/api/instances_info", json={"iuids": [order[1], "nope"]}).json()
    assert [it["iuid"] for it in info["items"]] == [order[1]]
    peers = c.get(f"/api/instance_peers?iuid={order[0]}").json()
    assert peers["label"] == "class A"
