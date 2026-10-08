"""Image view "+ Add instance": a box drawn on the image becomes a NEW instance (SAM-HQ / SAM segment it,
or the box itself), with a full feature row, for objects the proposals missed or rejected by mistake.
Run: pytest tests/test_add_instance.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np

from test_merge_rec import _engine


def test_box_becomes_a_reviewed_instance_with_a_feature_row(tmp_path):
    eng, o = _engine(tmp_path, n_img=1, per_img=4)
    n = len(eng.state.order)
    r = eng.add_instance(1000, box=[10, 20, 50, 60], method="box", cls="lead")
    assert r["ok"] and r["reviewed"] and len(eng.state.order) == n + 1
    u = r["iuid"]
    m = eng.state.meta[u]
    assert eng.state.class_name(m.assigned_class) == "lead" and eng.mask_reviewed(u)
    assert int(eng._mask(u).sum()) == 40 * 40 and u in eng._image_members(1000)
    assert all(np.isfinite(f).all() and f.shape[0] == n + 1
               for k, f in eng.collection["feats"].items() if not k.startswith("_"))
    assert eng._method_of(u) == "manual"


def test_samhq_segments_the_box_and_stays_a_prediction(tmp_path, monkeypatch):
    from chevron import refine as rf
    eng, o = _engine(tmp_path, n_img=1, per_img=4)
    seen = {}

    def fake(img, boxes, family="sam", **k):
        seen.update(family=family, box=boxes[0])
        m = np.zeros(img.shape[:2], bool); m[30:40, 30:45] = True
        return [(m, 0.9)]
    monkeypatch.setattr(rf, "sam_boxes", fake)
    r = eng.add_instance(1000, box_frac=[0.25, 0.25, 0.5, 0.5], method="samhq")
    assert seen["family"] == "samhq" and seen["box"] == [32.0, 32.0, 64.0, 64.0]   # fractions -> pixels
    assert not r["reviewed"] and not eng.mask_reviewed(r["iuid"]) and r["box"] == [30, 30, 45, 40]


def test_bad_input_and_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, o = _engine(tmp_path, n_img=1, per_img=4)
    assert "too small" in eng.add_instance(1000, box=[5, 5, 6, 6], method="box")["error"]
    assert "unknown image" in eng.add_instance(42, box=[5, 5, 60, 60], method="box")["error"]
    c = TestClient(create_app(engine=eng))
    eng.assign([o[0]], "X")
    cid = eng.state.meta[o[0]].assigned_class
    r = c.post("/api/add_instance", json={"image_id": "1000", "box_frac": [0.1, 0.1, 0.4, 0.4],
                                          "method": "box", "cid": cid}).json()
    assert eng.state.meta[r["iuid"]].assigned_class == cid
    assert c.post("/api/add_instance", json={"image_id": "1000", "box": [0, 0, 1, 1], "method": "box"}).status_code == 400
