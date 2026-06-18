"""v8.0 custom (FastAPI) frontend: the engine is reused unchanged; verify the windowed/lazy API.
Run: pytest tools/curator/tests/test_server.py -q
"""
from __future__ import annotations

import numpy as np


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


def _engine(tmp_path, *, n=400, nimg=80):
    import cv2
    from tools.curator import ids
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    p = tmp_path / "im0.png"
    cv2.imwrite(str(p), (np.random.default_rng(0).random((128, 128, 3)) * 200).astype(np.uint8))
    mb = np.zeros((128, 128), np.uint8); cv2.circle(mb, (64, 64), 18, 1, -1); mb = mb > 0
    recs, order, meta, dec = [], [], {}, []
    rng = np.random.default_rng(1)
    cent = rng.normal(0, 1, (12, 8))
    for j in range(n):
        u = ids.new_uid()
        recs.append({"iuid": u, "row": j, "inst_id": j, "image_id": 1000 + (j % nimg), "H": 128, "W": 128,
                     "score": 0.6, "rle": _rle(mb), "file_name": str(p), "abs_path": str(p), "batch_id": "b",
                     "cx": 0.5, "cy": 0.5, "bw": 0.3, "bh": 0.3, "box_area": 0.09, "mask_area_frac": float(mb.mean())})
        dec.append(cent[j % 12] + rng.normal(0, 0.2, 8)); order.append(u)
        meta[u] = InstanceMeta(iuid=u, batch_id="b", row=j, image_id=1000 + (j % nimg))
    eng.collection = {"records": recs, "n_images": nimg, "feats": {"decoder": np.array(dec, np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    return eng, order


def _client(tmp_path):
    from fastapi.testclient import TestClient
    from tools.curator.server import create_app
    eng, order = _engine(tmp_path)
    return TestClient(create_app(engine=eng)), eng, order


def test_core_loop(tmp_path):
    c, eng, order = _client(tmp_path)
    assert c.get("/").status_code == 200                              # serves the page

    st = c.get("/api/state").json()
    assert st["stats"]["n_instances"] == len(order) and not st["clustered"]

    rep = c.post("/api/cluster", json={"features": ["decoder"]}).json()
    assert rep["ok"] and rep["n_levels"] >= 1

    # partitions are WINDOWED: limit caps the payload regardless of partition count
    pg = c.get("/api/partitions?offset=0&limit=10").json()
    assert pg["total"] >= 1 and len(pg["rows"]) <= 10
    pid = pg["rows"][0]["pid"]

    inst = c.get(f"/api/instances?pid={pid}&offset=0&limit=20").json()
    assert inst["total"] >= 1 and len(inst["items"]) <= 20
    iuid = inst["items"][0]["iuid"]

    # lazy crop endpoint returns a real PNG
    crop = c.get(f"/api/crop?iuid={iuid}&max_side=128")
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/png" and crop.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert c.get("/api/crop?iuid=nope").status_code == 404

    # assign -> instance leaves the unassigned pool; a 'class:' pseudo-partition appears
    ar = c.post("/api/assign", json={"iuids": [iuid], "cls": "lead"}).json()
    assert ar["ok"] and "lead" in ar["classes"] and eng.state.meta[iuid].assigned_class is not None
    assert any(r["pid"].startswith("class:") for r in c.get("/api/partitions?limit=500").json()["rows"])

    # reject + unassign reach the engine
    iu2 = inst["items"][1]["iuid"]
    assert c.post("/api/reject", json={"iuids": [iu2]}).json()["ok"] and eng.state.meta[iu2].is_background
    assert c.post("/api/unassign", json={"iuids": [iuid]}).json()["ok"] and eng.state.meta[iuid].assigned_class is None


def test_partition_window_caps_payload(tmp_path):
    """Even with many partitions, the API ships only the requested window (the whole point vs Gradio)."""
    c, eng, order = _client(tmp_path)
    n = len(order)
    eng._cluster = {"spec": {"decoder": 1.0}, "distance": "cosine", "level": 0,
                    "partitions": np.arange(n).reshape(-1, 1), "counts": [n], "pool": list(order)}
    eng._pv_cache = None
    pg = c.get("/api/partitions?offset=0&limit=50").json()
    assert pg["total"] == n and len(pg["rows"]) == 50                 # total reported, payload bounded
    assert c.get("/api/partitions?offset=50&limit=50").json()["rows"][0]["pid"] != pg["rows"][0]["pid"]
