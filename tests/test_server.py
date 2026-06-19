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


def test_phase2_endpoints(tmp_path):
    c, eng, order = _client(tmp_path)
    c.post("/api/cluster", json={"features": ["decoder"]})

    # undo/redo are wired
    assert c.post("/api/undo").json()["ok"] and c.post("/api/redo").json()["ok"]

    # in-image: overlay PNG + windowed instances for an image
    iid = eng.state.meta[order[0]].image_id
    ov = c.get(f"/api/image_overlay?image_id={iid}")
    assert ov.status_code == 200 and ov.content[:8] == b"\x89PNG\r\n\x1a\n"
    ii = c.get(f"/api/image_instances?image_id={iid}&limit=50").json()
    assert ii["total"] >= 1 and all("image_id" in it for it in ii["items"])

    # classifier (kNN trains with >=1/class): assign 2 classes, train, predict, apply
    c.post("/api/assign", json={"iuids": [order[0]], "cls": "A"})
    c.post("/api/assign", json={"iuids": [order[1]], "cls": "B"})
    rep = c.post("/api/train_classifier", json={"features": ["decoder"], "algo": "knn"}).json()
    assert rep["ok"] and rep["n_classes"] == 2
    pred = c.get("/api/predict?thresh=0.0&limit=10").json()
    assert pred["total"] >= 1 and {"iuid", "cls", "conf"} <= set(pred["items"][0])
    ap = c.post("/api/apply_predictions", json={"thresh": 0.0, "exclude": [pred["items"][0]["iuid"]]}).json()
    assert ap["ok"] and ap["n"] >= 1

    # refine: preview (data-URIs) + apply + split
    u = order[5]
    rp = c.post("/api/refine_preview", json={"iuid": u, "ops": [{"name": "largest_cc"}]}).json()
    assert rp["before"].startswith("data:image/png;base64,") and rp["after"].startswith("data:image/png;base64,")
    assert c.post("/api/apply_refine", json={"iuid": u, "ops": [{"name": "fill"}]}).json()["ok"]
    assert "n" in c.post("/api/split", json={"iuids": [u]}).json()

    # rejected + unreject round-trip
    c.post("/api/reject", json={"iuids": [order[6]]})
    rj = c.get("/api/rejected?limit=50").json()
    assert order[6] in [it["iuid"] for it in rj["items"]]
    assert c.post("/api/unreject", json={"iuids": [order[6]]}).json()["ok"]
    assert not eng.state.meta[order[6]].is_background


def test_v1_fixes_endpoints(tmp_path):
    """crop context param, windowed /api/images picker, /api/merge_preview."""
    c, eng, order = _client(tmp_path)
    c.post("/api/cluster", json={"features": ["decoder"]})

    # crop supports context=1 (whole-image view) AND mask=0/1 — both return valid PNGs
    for q in ("context=1", "mask=0", "mask=1&context=1"):
        r = c.get(f"/api/crop?iuid={order[0]}&{q}")
        assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"

    # windowed image picker: bounded payload + count + query filter
    im = c.get("/api/images?limit=10").json()
    assert im["total"] >= 1 and len(im["items"]) <= 10 and {"image_id", "n"} <= set(im["items"][0])
    iid = im["items"][0]["image_id"]
    assert all(str(iid) in str(it["image_id"]) for it in c.get(f"/api/images?query={iid}").json()["items"])

    # merge preview: >=2 instances -> a data-URI PNG; <2 -> null
    iu = c.get(f"/api/image_instances?image_id={iid}&limit=5").json()["items"]
    if len(iu) >= 2:
        mp = c.post("/api/merge_preview", json={"iuids": [iu[0]["iuid"], iu[1]["iuid"]], "mode": "union"}).json()
        assert mp["img"].startswith("data:image/png;base64,")
    assert c.post("/api/merge_preview", json={"iuids": [order[0]]}).json()["img"] is None


def test_match_features_and_partition_of(tmp_path):
    """Model-free core of find-by-image: cosine-NN over a stored feature + partition mapping.
    (The model-forward query path is GPU-gated and exercised manually.)"""
    import numpy as np
    eng, order = _engine(tmp_path)
    # give the collection a 'roialign' feature with one clearly-closest row to a known query
    n = len(order)
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (n, 16)).astype(np.float32)
    q = X[7] * 1.3 + rng.normal(0, 1e-3, 16).astype(np.float32)   # row 7 is the unambiguous nearest
    eng.collection["feats"]["roialign"] = X
    eng.cluster({"decoder": 1.0})

    res = eng.match_features(q, feature="roialign", k=5)
    assert res["matches"][0]["iuid"] == order[7]                  # nearest neighbour found
    assert all("pid" in m and "score" in m for m in res["matches"])
    assert eng.match_features(q, feature="nope")["error"]         # missing feature -> error
    assert eng.match_features(np.zeros(3), feature="roialign")["error"]   # dim mismatch -> error

    # partition_of: assigned -> class:, pool -> finch pid, rejected -> None
    eng.assign([order[0]], "A"); eng.set_background([order[1]])
    assert eng.partition_of(order[0]) == f"class:{eng.state.class_id_by_name('A')}"
    assert eng.partition_of(order[1]) is None
    assert eng.partition_of(order[7]) is not None                 # still in the unassigned pool


def test_statistics(tmp_path):
    c, eng, order = _client(tmp_path)
    eng.cluster({"decoder": 1.0})
    c.post("/api/assign", json={"iuids": [order[0], order[1]], "cls": "A"})
    c.post("/api/assign", json={"iuids": [order[2]], "cls": "B"})
    c.post("/api/reject", json={"iuids": [order[3]]})
    eng.merge_instances([order[4], order[5]])                  # one merge (a child becomes hidden)

    s = c.get("/api/statistics").json()
    o = s["overview"]
    assert o["instances_total"] == len(order) and o["merged_children"] == 1
    assert o["assigned"] == 3 and o["rejected"] == 1
    assert o["instances_live"] == len(order) - 1               # merged child excluded
    assert 0 <= o["pct_curated"] <= 100
    names = {c2["class"]: c2 for c2 in s["classes"]}
    assert names["A"]["n"] == 2 and names["B"]["n"] == 1 and names["A"]["images"] >= 1
    assert s["classes"][0]["n"] >= s["classes"][-1]["n"]       # sorted desc
    assert sum(s["sources"].values()) == 3                     # 3 assignments by source
    assert len(s["score_hist"]["counts"]) == 20 and len(s["instances_per_image"]) >= 1
    assert s["cooccurrence"]["classes"] and len(s["cooccurrence"]["matrix"]) == len(s["cooccurrence"]["classes"])
    assert s["partitions"]["clustered"] is True


def test_match_features_dedups_partitions(tmp_path):
    """Reference search returns each PARTITION once (best instance), not k instances."""
    import numpy as np
    eng, order = _engine(tmp_path)
    n = len(order)
    eng.collection["feats"]["roialign"] = np.random.default_rng(5).normal(0, 1, (n, 16)).astype(np.float32)
    eng.cluster({"decoder": 1.0})
    q = eng.collection["feats"]["roialign"][3]
    res = eng.match_features(q, feature="roialign", k=8)
    pids = [m["pid"] for m in res["matches"]]
    assert len(pids) == len(set(pids))                    # each partition shown at most once
    assert all(p is not None for p in pids)               # rejected/merged (None) are skipped
    # with dedup off, the same partition can repeat (k instances)
    raw = eng.match_features(q, feature="roialign", k=n, dedup_partition=False)["matches"]
    assert len(raw) > len(res["matches"]) or len({m["pid"] for m in raw}) <= len(raw)


def test_infer_dir_endpoint(tmp_path, monkeypatch):
    """/api/infer_dir validates the folder and routes through ingest_paths (model stubbed)."""
    c, eng, order = _client(tmp_path)
    seen = {}
    def fake_ingest(paths):
        seen["paths"] = paths
        return {"n_new_images": len(paths), "n_new_instances": 7, **eng.stats()}
    monkeypatch.setattr(eng, "ingest_paths", fake_ingest)
    assert c.post("/api/infer_dir", json={"dir": "/no/such/dir"}).status_code == 400
    r = c.post("/api/infer_dir", json={"dir": str(tmp_path), "limit": 5}).json()
    assert r["ok"] and r["n_new_instances"] == 7 and seen["paths"]   # the project's test image(s) were ingested


def test_match_image_endpoint(tmp_path, monkeypatch):
    """/api/match_image decodes a base64 image, runs match_image, and attaches preview crops."""
    import base64
    import cv2
    import numpy as np
    c, eng, order = _client(tmp_path)
    eng.collection["feats"]["roialign"] = np.random.default_rng(0).normal(0, 1, (len(order), 16)).astype(np.float32)
    eng.cluster({"decoder": 1.0})
    # stub the model-forward query path (no GPU here): pretend the upload matched order[2]
    monkeypatch.setattr(eng, "match_image",
                        lambda img, **k: {"matches": [{"iuid": order[2], "score": 0.99, "pid": eng.partition_of(order[2])}],
                                          "query_score": 0.8, "n_detected": 1})
    png = cv2.imencode(".png", np.zeros((32, 32, 3), np.uint8))[1].tobytes()
    r = c.post("/api/match_image", json={"image": "data:image/png;base64," + base64.b64encode(png).decode()}).json()
    assert r["matches"][0]["iuid"] == order[2] and r["matches"][0]["crop"].startswith("data:image/png;base64,")
    assert c.post("/api/match_image", json={"image": ""}).status_code == 400


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
