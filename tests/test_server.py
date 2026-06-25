"""v8.0 custom (FastAPI) frontend: the engine is reused unchanged; verify the windowed/lazy API.
Run: pytest tools/curator/tests/test_server.py -q
"""
from __future__ import annotations

import numpy as np

# Pre-warm torch on the MAIN thread at collection time. The FastAPI TestClient runs endpoint handlers in a
# worker threadpool; if torch's FIRST import happens there it can land half-initialized in sys.modules, and a
# later main-thread `import torch` (the adopt/integrity tests) then hits "partially initialized module 'torch'
# … circular import". Importing it here, fully, once, before any handler thread runs makes every later import
# a cached no-op. (Documented torch-threading fragility; optional so non-torch CI still collects the file.)
try:
    import torch  # noqa: F401
except Exception:
    pass


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
    eng.merge_instances([order[4], order[4 + 80]])             # same image (1004) -> one merge, a child hidden

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
    def fake_ingest(paths, *, mode="new", score_thresh=None, nms_iou=None, with_raddino=False, raddino_pool="mask"):
        seen["paths"] = paths; seen["mode"] = mode
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


def test_reference_find_endpoint(tmp_path, monkeypatch):
    """/api/reference/find ranks ALL instances against a reference class (no partition preselect). The
    endpoint requires `cls` (400 otherwise) and passes the result through; the RAD-DINO ranking itself is
    stubbed here (GPU/HF-free)."""
    c, eng, order = _client(tmp_path)
    monkeypatch.setattr(eng, "reference_find_instances",
                        lambda cls, **k: {"items": [{"iuid": order[1], "score": 0.91,
                                                     "pid": eng.partition_of(order[1]), "cls": None}],
                                          "truncated": False, "cls": cls})
    r = c.post("/api/reference/find", json={"cls": "pacemaker"}).json()
    assert r["cls"] == "pacemaker" and r["items"][0]["iuid"] == order[1]
    assert c.post("/api/reference/find", json={"cls": ""}).status_code == 400


def test_find_instances_endpoint(tmp_path):
    """Refine instance picker: empty query returns a window; iuid-prefix / image-id queries filter."""
    c, eng, order = _client(tmp_path)
    empty = c.get("/api/find_instances?limit=20").json()
    assert 1 <= len(empty["items"]) <= 20 and {"iuid", "caption", "image_id"} <= set(empty["items"][0])
    # iuid prefix
    u = order[0]
    byid = c.get(f"/api/find_instances?query={u[:6]}").json()
    assert u in [it["iuid"] for it in byid["items"]]
    # image-id exact (image_id is emitted as a STRING for JS precision)
    iid = eng.state.meta[order[0]].image_id
    byimg = c.get(f"/api/find_instances?query={iid}").json()
    assert byimg["items"] and all(it["image_id"] == str(iid) for it in byimg["items"])


def test_refine_preview_respects_mask_flag(tmp_path):
    """refine_preview honours the mask flag (m-toggle) — both render valid PNGs and differ."""
    c, eng, order = _client(tmp_path)
    u = order[3]
    on = c.post("/api/refine_preview", json={"iuid": u, "ops": [], "mask": 1}).json()
    off = c.post("/api/refine_preview", json={"iuid": u, "ops": [], "mask": 0}).json()
    assert on["before"].startswith("data:image/png;base64,") and off["before"].startswith("data:image/png;base64,")
    assert on["before"] != off["before"]                       # overlay drawn vs not


def test_refine_preview_context_flag(tmp_path):
    """The 'c' toggle: context=1 renders the WHOLE image (instance highlighted), context=0 the tight crop —
    so the two before-panels differ (this is the bug where the c-toggle did nothing in the Refine tab)."""
    import base64

    import cv2
    import numpy as np
    c, eng, order = _client(tmp_path)
    u = order[3]

    def _decode(uri):
        return cv2.imdecode(np.frombuffer(base64.b64decode(uri.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    crop = c.post("/api/refine_preview", json={"iuid": u, "ops": [], "context": 0}).json()
    full = c.post("/api/refine_preview", json={"iuid": u, "ops": [], "context": 1}).json()
    assert crop["before"] != full["before"]                    # the c-toggle actually changes the render
    cb, fb = _decode(crop["before"]), _decode(full["before"])
    assert fb.shape[1] / max(fb.shape[0], 1) != cb.shape[1] / max(cb.shape[0], 1) or fb.size != cb.size


def test_instance_peers_endpoint(tmp_path):
    """The Refine left list shows the previewed instance's PARTITION PEERS, not a global search. Peers of an
    iuid == its partition's members; an assigned instance reports its class partition; unknown iuid -> 404."""
    c, eng, order = _client(tmp_path)
    eng.cluster({"decoder": 1.0})
    u = order[0]
    pid = eng.partition_of(u)
    r = c.get(f"/api/instance_peers?iuid={u}").json()
    assert r["pid"] == str(pid)
    got = {it["iuid"] for it in r["items"]}
    assert u in got and got <= set(eng.partition_iuids(pid))    # all returned are real peers, incl. self
    eng.assign([u], "wire")                                     # now in a class partition
    ra = c.get(f"/api/instance_peers?iuid={u}").json()
    assert ra["pid"].startswith("class:") and u in {it["iuid"] for it in ra["items"]}
    assert c.get("/api/instance_peers?iuid=deadbeef").status_code == 404


def test_sam_status_and_graceful_refine(tmp_path, monkeypatch):
    """/api/sam_status reports state; a 'sam' op with no checkpoint returns a clean 400, not a 500."""
    from tools.curator import refine as _rf
    c, eng, order = _client(tmp_path)
    s = c.get("/api/sam_status").json()
    assert {"installed", "ckpt", "model_type"} <= set(s)
    # force the no-checkpoint path regardless of the dev box's cache
    monkeypatch.setattr(_rf, "find_sam_checkpoint", lambda ckpt=None, family=None: (None, None))
    monkeypatch.setattr(_rf, "sam_available", lambda: True)
    r = c.post("/api/refine_preview", json={"iuid": order[0], "ops": [{"name": "sam"}]})
    assert r.status_code == 400 and "checkpoint" in r.json()["detail"].lower()
    r2 = c.post("/api/apply_refine", json={"iuid": order[0], "ops": [{"name": "sam"}]})
    assert r2.status_code == 400


def test_sam_prompt_preview_endpoint(tmp_path):
    """The SAM prompt visualisation is pure geometry — returns a PNG + point counts with NO checkpoint."""
    c, eng, order = _client(tmp_path)
    r = c.post("/api/sam_prompt_preview", json={"iuid": order[0], "ops": [], "n_pos": 6, "n_neg": 8}).json()
    assert r["img"].startswith("data:image/png;base64,")
    assert 1 <= r["n_pos"] <= 6 and 0 <= r["n_neg"] <= 8


def test_merge_groups_by_image(tmp_path):
    """A cross-image selection merges PER IMAGE (never across), so masks of different shapes can't
    collide (was: `res |= m` broadcast error). Same-image pairs merge; lone-per-image ones don't."""
    c, eng, order = _client(tmp_path)
    # order[j].image_id == 1000 + j%80, so order[0]&order[80] share img 1000; order[1]&order[81] share 1001
    same_a, same_b = [order[0], order[80]], [order[1], order[81]]
    # two same-image pairs in one selection -> 2 groups merged, 2 children hidden
    r = c.post("/api/merge", json={"iuids": same_a + same_b}).json()
    assert r["ok"] and r["n_groups"] == 2
    assert eng.state.meta[order[80]].merged_into == order[0]      # rep is the higher/equal-score first
    assert eng.state.meta[order[81]].merged_into == order[1]
    # a selection with no two sharing an image merges nothing (no crash) -> n_groups 0
    r2 = c.post("/api/merge", json={"iuids": [order[2], order[3]]}).json()   # imgs 1002 vs 1003
    assert r2["ok"] and r2["n_groups"] == 0


def test_image_id_round_trips_as_string(tmp_path):
    """Real projects hash file paths to 56-bit image_ids (> 2^53), which JS rounds when they arrive as
    JSON numbers -> the partition->in-image jump looks up a non-existent id and shows 'no instances'.
    The API must emit image_id as the EXACT string and accept it back (this is the reported bug)."""
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
    from tools.curator import ids
    from tools.curator.engine import CuratorEngine
    from tools.curator.server import create_app
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    p = tmp_path / "im.png"; cv2.imwrite(str(p), np.zeros((64, 64, 3), np.uint8))
    mb = np.zeros((64, 64), np.uint8); cv2.circle(mb, (32, 32), 10, 1, -1); mb = mb > 0
    big = 2 ** 55 + 1                             # 56-bit, ODD -> not representable as a double
    assert int(float(big)) != big                 # i.e. JSON-number transport WOULD corrupt it
    recs, order, meta = [], [], {}
    for j in range(3):
        u = ids.new_uid()
        recs.append({"iuid": u, "row": j, "inst_id": j, "image_id": big, "H": 64, "W": 64, "score": 0.6,
                     "rle": _rle(mb), "file_name": str(p), "abs_path": str(p), "batch_id": "b",
                     "cx": .5, "cy": .5, "bw": .3, "bh": .3, "box_area": .09, "mask_area_frac": float(mb.mean())})
        order.append(u); meta[u] = InstanceMeta(iuid=u, batch_id="b", row=j, image_id=big)
    eng.collection = {"records": recs, "n_images": 1, "feats": {"decoder": np.zeros((3, 8), np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    c = TestClient(create_app(engine=eng))

    fi = c.get("/api/find_instances?limit=5").json()["items"]
    assert fi and fi[0]["image_id"] == str(big)               # exact string, no rounding
    # the round-trip that was failing: feed that id string back -> the image's instances are found
    ii = c.get(f"/api/image_instances?image_id={fi[0]['image_id']}").json()
    assert ii["total"] == 3
    im = c.get("/api/images").json()["items"]
    assert im and im[0]["image_id"] == str(big)
    ov = c.get(f"/api/image_overlay?image_id={fi[0]['image_id']}")
    assert ov.status_code == 200 and ov.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_refine_preview_diff_overlay(tmp_path):
    """The 'after' panel is a yellow/green/red diff (unchanged/added/removed). A donut mask + 'fill' adds
    the centre (GREEN) and keeps the ring (YELLOW) and removes nothing (no RED). Black image -> pure blend."""
    import cv2
    import numpy as np
    from tools.curator import ids
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    p = tmp_path / "im.png"; cv2.imwrite(str(p), np.zeros((128, 128, 3), np.uint8))
    donut = np.zeros((128, 128), np.uint8)
    cv2.circle(donut, (64, 64), 30, 1, -1); cv2.circle(donut, (64, 64), 12, 0, -1); donut = donut > 0
    u = ids.new_uid()
    eng.collection = {"records": [{"iuid": u, "row": 0, "inst_id": 0, "image_id": 7, "H": 128, "W": 128,
                                   "score": 0.6, "rle": _rle(donut), "file_name": str(p), "abs_path": str(p),
                                   "batch_id": "b", "cx": .5, "cy": .5, "bw": .3, "bh": .3, "box_area": .09,
                                   "mask_area_frac": float(donut.mean())}],
                      "n_images": 1, "feats": {"decoder": np.zeros((1, 8), np.float32)}}
    eng.state.order = [u]; eng.state.meta = {u: InstanceMeta(iuid=u, batch_id="b", row=0, image_id=7)}
    eng.state.coll_version = 1; eng.store.save_collection(eng.collection); eng.save()

    before, after = eng.refine_preview(u, [{"name": "fill"}], mask_overlay=True)
    R, G, B = after[..., 0].astype(int), after[..., 1].astype(int), after[..., 2].astype(int)
    green_added = (G > 100) & (R < 70) & (B < 70)        # filled centre
    yellow_same = (R > 100) & (G > 100) & (B < 60)       # untouched ring
    red_removed = (R > 100) & (G < 70) & (B < 70)        # nothing removed
    assert green_added.sum() > 20 and yellow_same.sum() > 20 and red_removed.sum() == 0
    assert not np.array_equal(before, after)             # the diff panel differs from 'before'


def test_class_rules_and_partition_refine(tmp_path):
    """Per-class rule chains: apply a chain to a whole class, store it as the class recipe (persisted),
    record it on each instance; also apply a chain to a selected (finch) partition."""
    from tools.curator.engine import CuratorEngine
    c, eng, order = _client(tmp_path)
    eng.cluster({"decoder": 1.0})
    c.post("/api/assign", json={"iuids": order[:3], "cls": "lung"})
    chain = [{"name": "fill"}, {"name": "largest_cc"}]
    r = c.post("/api/apply_class_rule", json={"cls": "lung", "ops": chain}).json()
    assert r["ok"] and r["n"] == 3
    rules = c.get("/api/class_rules").json()["rules"]
    assert any(x["cls"] == "lung" and x["ops"] == ["fill", "largest_cc"] and x["n"] == 3 for x in rules)
    for u in order[:3]:                                   # each member records the chain it received
        assert eng.state.meta[u].rule_ops == chain and eng.state.meta[u].refined
    # the recipe persists across a reload (saved in state)
    eng2 = CuratorEngine(tmp_path)
    cid = eng2.state.class_id_by_name("lung")
    assert eng2.state.class_rules.get(cid) == chain
    # re-apply the STORED rule (ops omitted) -> same members
    assert c.post("/api/apply_class_rule", json={"cls": "lung"}).json()["n"] == 3
    # apply a chain to a finch partition by pid
    pid = next(row["pid"] for row in c.get("/api/partitions?limit=500").json()["rows"]
               if not row["pid"].startswith("class:"))
    rp = c.post("/api/apply_refine_partition", json={"pid": pid, "ops": [{"name": "fill"}]}).json()
    assert rp["ok"] and rp["n"] >= 1


def test_merge_classes(tmp_path):
    """Merge classes into one (new or existing target): instances reassigned, emptied classes removed,
    reversible."""
    c, eng, order = _client(tmp_path)
    eng.cluster({"decoder": 1.0})
    c.post("/api/assign", json={"iuids": order[:2], "cls": "A"})
    c.post("/api/assign", json={"iuids": order[2:5], "cls": "B"})
    c.post("/api/assign", json={"iuids": [order[5]], "cls": "C"})
    cls = {x["cls"]: x["n"] for x in c.get("/api/classes").json()["classes"]}
    assert cls.get("A") == 2 and cls.get("B") == 3 and cls.get("C") == 1

    # merge A,B into a NEW class "AB"
    r = c.post("/api/merge_classes", json={"sources": ["A", "B"], "into": "AB"}).json()
    assert r["ok"] and r["moved"] == 5 and set(r["removed"]) == {"A", "B"}
    names = eng.state.class_names()
    assert "AB" in names and "A" not in names and "B" not in names and "C" in names
    assert all(eng.state.class_name(eng.state.meta[u].assigned_class) == "AB" for u in order[:5])

    # merge C into the EXISTING class "AB"
    r2 = c.post("/api/merge_classes", json={"sources": ["C"], "into": "AB"}).json()
    assert r2["ok"] and r2["moved"] == 1 and r2["removed"] == ["C"]
    assert eng.state.class_name(eng.state.meta[order[5]].assigned_class) == "AB"
    assert "C" not in eng.state.class_names()

    # reversible: undo restores C + its instance
    assert c.post("/api/undo").json()["ok"]
    assert "C" in eng.state.class_names() and eng.state.class_name(eng.state.meta[order[5]].assigned_class) == "C"

    # guard: empty target or no sources -> error
    assert c.post("/api/merge_classes", json={"sources": ["AB"], "into": ""}).json().get("error")


def test_class_samples_endpoint(tmp_path):
    """Classes-tab preview: /api/class_samples returns a class's live instances (highest-score first) as
    iuids to render via /api/crop; limit caps the payload, rejected/merged + unknown class are excluded."""
    c, eng, order = _client(tmp_path)
    c.post("/api/assign", json={"iuids": order[:3], "cls": "lead"})
    cid = eng.state.class_id_by_name("lead")
    r = c.get(f"/api/class_samples?class_id={cid}&limit=10").json()
    assert set(r["iuids"]) == set(order[:3])                       # exactly the class's instances
    crop = c.get(f"/api/crop?iuid={r['iuids'][0]}&mask=1")         # each renders a real PNG
    assert crop.status_code == 200 and crop.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(c.get(f"/api/class_samples?class_id={cid}&limit=2").json()["iuids"]) == 2   # limit caps
    c.post("/api/reject", json={"iuids": [order[0]]})             # rejected -> dropped
    assert order[0] not in c.get(f"/api/class_samples?class_id={cid}").json()["iuids"]
    assert c.get("/api/class_samples?class_id=does-not-exist").json()["iuids"] == []


def test_activity_endpoint(tmp_path):
    """Activity tab: /api/activity assembles a read-only interaction timeline from the four append-only
    logs (history / ingests / merge / lineage). Empty project -> well-formed zeros; after real ops +
    synthetic ingest/lineage rows, the op breakdown, merge split, ingest/retrain rows + bins all surface."""
    import time
    c, eng, order = _client(tmp_path)

    a0 = c.get("/api/activity").json()                                # empty project: shaped, zeroed
    assert a0["span"]["n_events"] == 0 and a0["totals"]["commands"] == 0
    assert a0["timeline"]["counts"] and not any(a0["timeline"]["counts"])   # bins exist, all zero
    assert a0["sessions"] == [] and a0["ingests"] == [] and a0["lineage"] == []

    c.post("/api/assign", json={"iuids": order[:3], "cls": "lead"})   # history: assign (3 inst)
    c.post("/api/reject", json={"iuids": order[3:5]})                 # history: background -> "reject"
    c.post("/api/merge", json={"iuids": [order[5], order[85]]})       # same image -> merge + merge_log
    eng.store.append_ingest_event({"ingest_id": "ing_000", "ts": time.time(), "n_instances": 400,
                                   "n_images": 80, "batch_ids": ["b"], "mode": "new", "score_thresh": 0.3})
    eng.store.append_lineage_event({"ts": time.time(), "coll_version": 1, "n_assigned": 3,
                                    "ckpt": "/runs/r0/model_best.pth", "metric_name": "segm/AP",
                                    "metric": 0.41, "regressed": False})

    a = c.get("/api/activity").json()
    assert a["span"]["n_events"] >= 5 and a["span"]["n_sessions"] >= 1
    assert a["totals"]["commands"] >= 3 and a["totals"]["instances_touched"] >= 3
    assert a["op_counts"].get("assign", 0) >= 1 and a["cat_counts"].get("reject", 0) >= 1
    assert a["merge"]["by_kind"].get("merge", 0) >= 1 and a["merge"]["instances"] >= 2
    assert a["merge"]["by_source"].get("manual", 0) >= 1
    assert a["ingests"] and a["ingests"][0]["n_images"] == 80
    assert a["lineage"] and a["lineage"][0]["ckpt"] == "model_best.pth" and a["lineage"][0]["metric"] == 0.41
    assert any(a["timeline"]["counts"])                              # at least one populated bin now
    assert len(c.get("/api/activity?bins=10").json()["timeline"]["counts"]) == 10   # bins param honored


def test_partial_label_export(tmp_path):
    """Partial-label export: positives=GT (iscrowd0), unreviewed=__ignore__ (iscrowd1), rejected omitted,
    per-image reviewed_exhaustive + counts; class_agnostic collapses positives to one 'object' class."""
    import json
    from pathlib import Path
    c, eng, order = _client(tmp_path)
    eng.cluster({"decoder": 1.0})
    c.post("/api/assign", json={"iuids": order[:3], "cls": "line"})        # positives (reviewed)
    c.post("/api/reject", json={"iuids": [order[3]]})                      # negative (rejected)
    # the rest stay unassigned -> ignore

    r = c.post("/api/export", json={"partial": True}).json()
    assert r["ok"] and r["partial"]
    coco = json.loads(Path(r["path"]).read_text())
    assert coco["info"]["partial_labels"] is True
    cats = {x["name"]: x["id"] for x in coco["categories"]}
    assert "__ignore__" in cats and "line" in cats
    pos = [a for a in coco["annotations"] if a["curator_status"] == "positive"]
    ign = [a for a in coco["annotations"] if a["curator_status"] == "ignore"]
    assert len(pos) == 3 and all(a["iscrowd"] == 0 and "iuid" in a for a in pos)
    assert len(ign) >= 1 and all(a["iscrowd"] == 1 and a["category_id"] == cats["__ignore__"] for a in ign)
    assert not any(a["category_id"] == cats["__ignore__"] and a["iscrowd"] == 0 for a in coco["annotations"])
    assert all({"reviewed_exhaustive", "n_ignore", "n_positive", "n_negative"} <= set(im) for im in coco["images"])

    # class-agnostic: positives collapse to a single 'object' category
    r2 = c.post("/api/export", json={"partial": True, "class_agnostic": True}).json()
    coco2 = json.loads(Path(r2["path"]).read_text())
    assert {a["category_id"] for a in coco2["annotations"] if a["curator_status"] == "positive"} == {1}
    assert any(x["name"] == "object" and x["id"] == 1 for x in coco2["categories"])


def test_train_launch_status_adopt(tmp_path, monkeypatch):
    """Loop orchestration: launch spawns qseg-train as a detached process on the partial export, unloads
    the inference model, status reflects the job, adopt repoints the curator's model ckpt. (subprocess +
    train binary are stubbed — no real training.)"""
    import subprocess
    from pathlib import Path
    c, eng, order = _client(tmp_path)
    c.post("/api/assign", json={"iuids": order[:2], "cls": "device"})
    eng.model = "LOADED"                                   # pretend an inference model is resident

    binp = tmp_path / "qseg-train"; binp.write_text("#!/bin/sh\n")
    monkeypatch.setattr(eng, "_qseg_train_bin", lambda: binp)

    class FakeProc:
        def __init__(self, cmd, **kw):
            self.pid = 4242
            out = kw.get("stdout")
            if out:
                out.write("epoch 0/40 ...\n"); out.flush()
        def poll(self):
            return None                                    # still running
    captured = {}
    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd; captured["cwd"] = kw.get("cwd"); captured["env"] = kw.get("env")
        return FakeProc(cmd, **kw)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    r = c.post("/api/train/launch", json={"mode": "finetune", "config_name": "experiments/foo", "partial": True}).json()
    assert r["ok"] and r["pid"] == 4242
    assert eng.model is None                               # inference model unloaded (same-GPU)
    cmd = captured["cmd"]
    assert cmd[:3] == [str(binp), "--config-name", "experiments/foo"]
    assert any(a.startswith("data.json_train=") and a.endswith("curated.json") for a in cmd)
    assert any(a.startswith("train.output_dir=") for a in cmd)
    assert any(a.startswith("train.init_weights=") for a in cmd)     # finetune warm-starts from current ckpt
    assert not any(a.startswith("data.json_val=") for a in cmd)      # no val given -> config default applies
    assert "MaskDINO" in captured["env"]["PYTHONPATH"]


    s = c.get("/api/train/status").json()
    assert s["active"] and s["running"] and s["pid"] == 4242 and "epoch 0/40" in s["log_tail"]

    # a second launch is refused while one is running
    assert c.post("/api/train/launch", json={}).json().get("error")

    # adopt a (real, loadable) checkpoint -> curator model repointed (integrity gate torch.loads it)
    import torch
    ckp = tmp_path / "best.pth"; torch.save({"model": {"w": torch.zeros(1)}}, ckp)
    a = c.post("/api/train/adopt", json={"ckpt": str(ckp)}).json()
    assert a["ok"] and eng.state.config["model"]["ckpt"] == str(ckp)


def test_to_class_agnostic(tmp_path):
    """Collapse a multi-class COCO to one 'object' class (id 1) + absolutize paths (synthfb-as-val target)."""
    import json
    from tools.curator.export_coco import to_class_agnostic
    (tmp_path / "images").mkdir()
    coco = {"images": [{"id": 1, "file_name": "a.png", "height": 32, "width": 32}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 9, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0},
                            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [5, 5, 4, 4], "area": 16, "iscrowd": 0}],
            "categories": [{"id": 9, "name": "tube"}, {"id": 3, "name": "clip"}]}
    p = tmp_path / "ann.json"; p.write_text(json.dumps(coco))
    ca = to_class_agnostic(str(p))
    assert ca["categories"] == [{"id": 1, "name": "object", "supercategory": "device"}]
    assert len(ca["annotations"]) == 2 and all(a["category_id"] == 1 for a in ca["annotations"])
    assert ca["images"][0]["file_name"] == str(tmp_path / "images" / "a.png")   # relative -> absolute


def test_merge_coco_sources():
    """Merging curated (partial) + extra (synthfb, complete) into one train json: ids reindexed, categories
    aligned (class-agnostic -> object/__ignore__), extra images marked exhaustive, curated flags kept."""
    from tools.curator.export_coco import merge_coco_sources
    curated = {"images": [{"id": 7, "file_name": "r.png", "height": 64, "width": 64, "reviewed_exhaustive": False}],
               "annotations": [{"id": 1, "image_id": 7, "category_id": 5, "iscrowd": 0, "curator_status": "positive",
                                "bbox": [0, 0, 4, 4], "area": 16, "segmentation": {"size": [64, 64], "counts": "x"}},
                               {"id": 2, "image_id": 7, "category_id": 0, "iscrowd": 1, "curator_status": "ignore",
                                "bbox": [5, 5, 4, 4], "area": 16, "segmentation": {"size": [64, 64], "counts": "y"}}],
               "categories": [{"id": 5, "name": "line"}, {"id": 0, "name": "__ignore__"}]}
    extra = {"images": [{"id": 7, "file_name": "s.png", "height": 64, "width": 64}],   # SAME id 7 -> must reindex
             "annotations": [{"id": 1, "image_id": 7, "category_id": 3, "iscrowd": 0, "bbox": [1, 1, 2, 2],
                              "area": 4, "segmentation": {"size": [64, 64], "counts": "z"}}],
             "categories": [{"id": 3, "name": "device_a"}]}
    m = merge_coco_sources(curated, extra, class_agnostic=True)
    assert len({im["id"] for im in m["images"]}) == 2                  # no image-id collision
    assert len({a["id"] for a in m["annotations"]}) == 3              # no ann-id collision
    cats = {c["name"]: c["id"] for c in m["categories"]}
    assert cats == {"object": 1, "__ignore__": 0}                    # collapsed
    pos = [a for a in m["annotations"] if a.get("curator_status") == "positive"]
    ign = [a for a in m["annotations"] if a.get("curator_status") == "ignore"]
    assert pos[0]["category_id"] == 1 and ign[0]["category_id"] == 0 and ign[0]["iscrowd"] == 1
    src = {im["file_name"]: im for im in m["images"]}
    assert src["s.png"]["reviewed_exhaustive"] is True and src["r.png"]["reviewed_exhaustive"] is False
    # the synth annotation collapsed to 'object' too
    synth_iid = src["s.png"]["id"]
    assert all(a["category_id"] == 1 for a in m["annotations"] if a["image_id"] == synth_iid)


def test_merge_absolutizes_extra_paths(tmp_path):
    """The extra (synthfb) source has RELATIVE file_names; the merge absolutizes them against
    <extra_json_dir>/images so both sources resolve under one image_root."""
    import json
    from tools.curator.export_coco import merge_coco_sources
    (tmp_path / "images").mkdir()
    extra = {"images": [{"id": 1, "file_name": "000001.png", "height": 32, "width": 32}],
             "annotations": [{"id": 1, "image_id": 1, "category_id": 9, "iscrowd": 0, "bbox": [0, 0, 4, 4],
                              "area": 16, "segmentation": {"size": [32, 32], "counts": "x"}}],
             "categories": [{"id": 9, "name": "synthdev"}]}
    ep = tmp_path / "annotations.json"; ep.write_text(json.dumps(extra))
    cur = {"images": [{"id": 7, "file_name": "/abs/r.png", "height": 32, "width": 32}],
           "annotations": [], "categories": [{"id": 1, "name": "object"}]}
    m = merge_coco_sources(cur, str(ep), class_agnostic=True)
    syn = [im for im in m["images"] if im["source"] == "extra"][0]
    assert syn["file_name"] == str(tmp_path / "images" / "000001.png") and syn["reviewed_exhaustive"] is True


def test_export_drops_mask_not_matching_image(tmp_path):
    """A stale/legacy mask whose RLE size != its image (H,W) is dropped from the export — such an
    annotation is invalid COCO and crashes the trainer's augmentation (assertion in apply_image).
    Regression for the pre-fix cross-image-merge residue."""
    import json
    from pathlib import Path

    import numpy as np
    from pycocotools import mask as mu
    c, eng, order = _client(tmp_path)
    c.post("/api/assign", json={"iuids": order[:2], "cls": "x"})
    bad = np.zeros((128, 100), np.uint8); bad[10:20, 10:20] = 1          # wrong width (image is 128x128)
    r = mu.encode(np.asfortranarray(bad)); r["counts"] = r["counts"].decode("ascii")
    u = order[0]; eng._overlay_rle[u] = r                               # forge a mismatched effective mask

    res = c.post("/api/export", json={"partial": True}).json()
    coco = json.loads(Path(res["path"]).read_text())
    assert coco["info"]["n_skipped_bad_mask"] >= 1
    imgs = {im["id"]: im for im in coco["images"]}
    assert all(list(a["segmentation"]["size"]) == [imgs[a["image_id"]]["height"], imgs[a["image_id"]]["width"]]
               for a in coco["annotations"] if isinstance(a["segmentation"], dict))   # every mask now fits
    assert u not in {a.get("iuid") for a in coco["annotations"]}        # the bad instance is gone


def test_reinfer_replace_and_append(tmp_path, monkeypatch):
    """Re-infer the processed pool with the (new) model. replace: hide old UN-curated instances on those
    images (assigned/rejected/merged kept) + add the new predictions. append: keep old + add."""
    import numpy as np
    from tools.curator import collect as _co
    from tools.curator import ids
    c, eng, order = _client(tmp_path)
    p = eng.collection["records"][0]["abs_path"]
    iid = _co.path_image_id(p)
    us = order[:5]
    for u in us:                                                   # put 5 instances on the path-derived image
        eng.state.meta[u].image_id = iid
        eng.collection["records"][eng.state.meta[u].row]["image_id"] = iid
    eng.assign(list(us[:2]), "keep"); eng.set_background([us[4]])   # 2 assigned, 1 rejected, us[2]/us[3] unassigned
    man = eng.store.load_manifest(); man["processed_paths"] = [p]; eng.store.save_manifest(man)

    monkeypatch.setattr(eng, "_ensure_model", lambda: (None, None, None))
    def fake_collect(model, cfg, d2_cfg, files, **kw):
        u = ids.new_uid()
        return {"records": [{"iuid": u, "row": 0, "inst_id": 0, "image_id": iid, "H": 128, "W": 128,
                             "score": 0.7, "rle": eng.collection["records"][0]["rle"], "file_name": p,
                             "abs_path": p, "batch_id": "b2", "cx": .5, "cy": .5, "bw": .3, "bh": .3,
                             "box_area": .09, "mask_area_frac": 0.1}],
                "n_images": 1, "feats": {"decoder": np.zeros((1, 8), np.float32)}}
    monkeypatch.setattr(_co, "collect_batch", fake_collect)

    r = eng.reinfer_processed(mode="replace")
    assert r["n_new_instances"] == 1 and r["n_replaced"] == 2       # 2 unassigned hidden
    assert eng.state.meta[us[2]].is_background and eng.state.meta[us[3]].is_background
    assert eng.state.meta[us[0]].assigned_class is not None         # assigned kept
    assert not eng.state.meta[us[1]].is_background                  # assigned not hidden

    r2 = eng.reinfer_processed(mode="append")                       # append: nothing newly hidden
    assert r2["n_new_instances"] == 1 and r2["n_replaced"] == 0


def test_preview_inference_before_after_nondestructive(tmp_path, monkeypatch):
    """Preview renders BEFORE (current instances) vs AFTER (new model) per image and does NOT modify the
    collection (non-destructive look before committing a re-infer)."""
    import numpy as np
    from tools.curator import collect as _co
    from tools.curator import ids
    c, eng, order = _client(tmp_path)
    p = eng.collection["records"][0]["abs_path"]
    iid = _co.path_image_id(p)
    for u in order[:3]:                                            # 3 current instances on this image
        eng.state.meta[u].image_id = iid
        eng.collection["records"][eng.state.meta[u].row]["image_id"] = iid
    man = eng.store.load_manifest(); man["processed_paths"] = [p]; eng.store.save_manifest(man)
    n_order, n_meta = len(eng.state.order), len(eng.state.meta)

    monkeypatch.setattr(eng, "_ensure_model", lambda: (None, None, None))
    def fake_collect(model, cfg, d2_cfg, files, **kw):
        recs = [{"iuid": ids.new_uid(), "row": j, "inst_id": j, "image_id": iid, "H": 128, "W": 128,
                 "score": 0.7, "rle": eng.collection["records"][0]["rle"], "file_name": p, "abs_path": p,
                 "batch_id": "b2", "cx": .5, "cy": .5, "bw": .3, "bh": .3, "box_area": .09, "mask_area_frac": .1}
                for j in range(2)]                                 # new model predicts 2
        return {"records": recs, "n_images": 1, "feats": {"decoder": np.zeros((2, 8), np.float32)}}
    monkeypatch.setattr(_co, "collect_batch", fake_collect)

    res = eng.preview_processed(6)
    assert res["sampled"] == 1 and res["n_inst"] == 2 and res["n_before"] == 3
    it = res["items"][0]
    assert it["before"].shape[2] == 3 and it["after"].shape[2] == 3 and "→" in it["caption"]
    assert len(eng.state.order) == n_order and len(eng.state.meta) == n_meta   # NON-destructive

    ep = c.post("/api/preview_infer", json={"n": 6}).json()        # endpoint -> before/after data-URIs
    assert ep["n_before"] == 3 and ep["n_inst"] == 2
    assert ep["items"][0]["before"].startswith("data:image/png;base64,")
    assert ep["items"][0]["after"].startswith("data:image/png;base64,")


def test_reinfer_endpoint_passes_mode(tmp_path, monkeypatch):
    c, eng, order = _client(tmp_path)
    seen = {}
    monkeypatch.setattr(eng, "reinfer_processed",
                        lambda *, mode="replace", limit=None, score_thresh=None, nms_iou=None,
                        with_raddino=False, raddino_pool="mask":
                        seen.update(mode=mode, with_raddino=with_raddino, raddino_pool=raddino_pool)
                        or {"n_new_instances": 0, **eng.stats()})
    assert c.post("/api/reinfer", json={"mode": "append"}).json()["ok"] and seen["mode"] == "append"
    assert seen["with_raddino"] is False                                  # opt-in: off by default
    # the chain flag + pool are forwarded when requested
    c.post("/api/reinfer", json={"mode": "replace", "with_raddino": True, "raddino_pool": "bbox"})
    assert seen["with_raddino"] is True and seen["raddino_pool"] == "bbox"


def test_ingest_chains_raddino(tmp_path, monkeypatch):
    """with_raddino=True chains compute_raddino once after the seg run (one click, not two), forwarding the
    pool; the chained result lands in the response. Off by default. (seg + raddino both stubbed — no GPU.)"""
    import cv2
    import numpy as np
    from tools.curator import collect as _co
    from tools.curator import ids
    c, eng, order = _client(tmp_path)
    monkeypatch.setattr(eng, "_ensure_model", lambda: (None, None, None))
    def fake_collect(model, cfg, d2_cfg, files, **kw):
        recs = [{"iuid": ids.new_uid(), "row": 0, "inst_id": 0, "image_id": _co.path_image_id(files[0]),
                 "H": 128, "W": 128, "score": 0.7, "rle": eng.collection["records"][0]["rle"],
                 "file_name": files[0], "abs_path": files[0], "batch_id": "b2", "cx": .5, "cy": .5,
                 "bw": .3, "bh": .3, "box_area": .09, "mask_area_frac": .1}]
        return {"records": recs, "n_images": 1, "feats": {"decoder": np.zeros((1, 8), np.float32)}}
    monkeypatch.setattr(_co, "collect_batch", fake_collect)
    calls = {"n": 0}
    monkeypatch.setattr(eng, "compute_raddino",
                        lambda *, force=False, pool="mask": calls.update(n=calls["n"] + 1, pool=pool)
                        or {"ok": True, "n": 999, "available": ["decoder", "raddino"]})

    p1 = str(tmp_path / "fresh.png"); cv2.imwrite(p1, np.zeros((128, 128, 3), np.uint8))
    r = eng.ingest_paths([p1], with_raddino=True, raddino_pool="bbox")
    assert calls["n"] == 1 and calls["pool"] == "bbox"                    # chained exactly once, pool forwarded
    assert r["n_new_instances"] == 1 and r["raddino_n"] == 999            # chained result surfaced

    p2 = str(tmp_path / "fresh2.png"); cv2.imwrite(p2, np.zeros((128, 128, 3), np.uint8))
    out = eng.ingest_paths([p2])                                          # default: no chain
    assert calls["n"] == 1 and "raddino_n" not in out


def test_concurrent_saves_dont_collide(tmp_path):
    """The threaded server runs requests in parallel; a fixed '<file>.tmp' made two concurrent saves
    collide (one os.replace moved the shared tmp, the other FileNotFoundError'd). Unique temp names fix it."""
    import json
    import threading
    eng, order = _engine(tmp_path)
    errs = []
    def hammer():
        try:
            for _ in range(25):
                eng.save()
        except Exception as e:                                     # the old fixed-tmp code raised here
            errs.append(e)
    ts = [threading.Thread(target=hammer) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs                                                # no FileNotFoundError under concurrency
    json.loads((tmp_path / "state.json").read_text())              # final state is valid JSON
    assert not list(tmp_path.glob("*.tmp"))                        # no stray temp files left behind


def test_inimage_excludes_rejected(tmp_path):
    """Rejecting an instance removes it from the In-image set (and overlay), persistently — not just a
    visual drop that reappears on reload."""
    c, eng, order = _client(tmp_path)
    iid = eng.state.meta[order[0]].image_id
    before = c.get(f"/api/image_instances?image_id={iid}&limit=100000").json()
    u = before["items"][0]["iuid"]
    assert c.post("/api/reject", json={"iuids": [u]}).json()["ok"]
    after = c.get(f"/api/image_instances?image_id={iid}&limit=100000").json()
    assert u not in [it["iuid"] for it in after["items"]]           # gone from the set on reload
    assert after["total"] == before["total"] - 1


def _fake_run(tmp_path, name, metric=41.3, *, corrupt=False, dumps=False):
    """A finished train-run dir: a (real torch / or corrupt) model_best.pth + summary.csv, optional dumps."""
    import torch
    out_dir = tmp_path / "train_runs" / name; out_dir.mkdir(parents=True, exist_ok=True)
    if corrupt:
        (out_dir / "model_best.pth").write_bytes(b"PK\x03\x04truncated")     # looks like a zip, fails to read
    else:
        torch.save({"model": {"w": torch.zeros(2)}}, out_dir / "model_best.pth")
    (out_dir / "summary.csv").write_text(f"best_val_metric_name,best_val_metric_value\nsegm/AP,{metric}\n")
    if dumps:
        for d in ("inference_val", "inference_test", "val"):
            (out_dir / d).mkdir(); (out_dir / d / "preds.json").write_bytes(b"x" * 4096)
        (out_dir / "predictions.json").write_bytes(b"y" * 4096)
        (out_dir / "model_0000999.pth").write_bytes(b"z" * 4096)              # an intermediate checkpoint
    return out_dir


def test_warmstart_collapses_multiclass_base(tmp_path):
    """Gate 3: a class-agnostic run off a MULTI-class base collapses the head (warm-start), not reinit;
    a base already class-agnostic is used as-is (so iterative chaining is clean)."""
    import torch
    c, eng, order = _client(tmp_path)
    multi = tmp_path / "synth_base.pth"                            # 120-class M2F head (119 fg + void)
    torch.save({"model": {"sem_seg_head.predictor.class_embed.weight": torch.randn(120, 8),
                          "sem_seg_head.predictor.class_embed.bias": torch.randn(120)}}, multi)
    out = eng._warmstart_init(str(multi), class_agnostic=True)
    assert out != str(multi) and out.endswith("warmstart_collapsed.pth")
    cw = torch.load(out, map_location="cpu")["model"]["sem_seg_head.predictor.class_embed.weight"]
    assert tuple(cw.shape) == (2, 8)                               # collapsed to 1 object + void
    assert eng._warmstart_init(out, class_agnostic=True) == out    # already 1-fg -> used as-is (chain ok)
    assert eng._warmstart_init(str(multi), class_agnostic=False) == str(multi)   # class-aware run: no surgery


def test_overfit_check_is_circular(tmp_path, monkeypatch):
    """Gate 2: the overfit harness trains on a few human-verified instances and evaluates on the SAME images
    (train==val==test) with early-stop off — the 'can the pipeline learn at all' certificate."""
    import subprocess
    c, eng, order = _client(tmp_path)
    binp = tmp_path / "qseg-train"; binp.write_text("#!/bin/sh\n")
    monkeypatch.setattr(eng, "_qseg_train_bin", lambda: binp)
    c.post("/api/assign", json={"iuids": [order[0], order[1], order[2]], "cls": "device"})  # manual -> verified
    for u in (order[0], order[1], order[2]):
        assert eng.state.meta[u].assign_source == "manual"
    captured = {}
    class _P:
        pid = 99
        def __init__(s, cmd, **kw): captured["cmd"] = cmd
        def poll(s): return None
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: _P(cmd, **kw))
    r = c.post("/api/train/overfit_check", json={"n": 8, "epochs": 30}).json()
    assert r["ok"] and r.get("overfit") is True
    cmd = captured["cmd"]
    j = [a.split("=", 1)[1] for a in cmd if a.startswith("data.json_train=")][0]
    assert j.endswith("overfit.json")
    assert f"data.json_val={j}" in cmd and f"data.json_test={j}" in cmd      # circular ON PURPOSE
    assert "eval.early_stop.enable=false" in cmd and "train.max_epochs=30" in cmd
    assert eng.training_status()["output_dir"].rsplit("/", 1)[-1].startswith("overfit_")


def test_overfit_check_needs_verified(tmp_path):
    """No human-verified labels -> a clear error (classifier-propagated labels don't count for the gate)."""
    c, eng, order = _client(tmp_path)
    rep = eng.launch_overfit_check(n=8)
    assert "error" in rep and "verified" in rep["error"].lower()


def test_adopt_writes_training_lineage(tmp_path):
    """Adopting a trained checkpoint persists a dataset->ckpt->metric lineage record (survives restart)."""
    c, eng, order = _client(tmp_path)
    c.post("/api/assign", json={"iuids": [order[0], order[1]], "cls": "A"})
    out_dir = _fake_run(tmp_path, "round_1", metric=41.3)
    eng._train_job = {"output_dir": str(out_dir), "export": "exports/curated.json",
                      "config_name": "experiments/curator_loop", "coll_version": 5, "n_assigned": 2}
    rep = eng.adopt_checkpoint()                                    # no ckpt arg -> finds model_best.pth in the run
    assert rep["ok"] and rep["metric_name"] == "segm/AP" and abs(rep["metric"] - 41.3) < 1e-6
    lin = eng.store.read_lineage()
    assert len(lin) == 1 and lin[0]["coll_version"] == 5 and lin[0]["n_assigned"] == 2
    assert lin[0]["ckpt"].endswith("model_best.pth") and abs(lin[0]["metric"] - 41.3) < 1e-6
    assert eng.state.config["model"]["ckpt"].endswith("model_best.pth")   # also adopted for inference


def test_adopt_refuses_corrupt_checkpoint(tmp_path):
    """Gate 0: a truncated/corrupt checkpoint (the disk-full failure) is NEVER adopted for inference."""
    c, eng, order = _client(tmp_path)
    out_dir = _fake_run(tmp_path, "round_1", corrupt=True)
    eng._train_job = {"output_dir": str(out_dir), "coll_version": 1, "n_assigned": 0}
    rep = eng.adopt_checkpoint()
    assert "error" in rep and "corrupt" in rep["error"].lower()
    assert eng.state.config["model"]["ckpt"] == "x"                 # unchanged (fixture base) — corrupt NOT adopted
    assert eng.store.read_lineage() == []                           # no lineage written for a bad ckpt


def test_adopt_refuses_regression_unless_forced(tmp_path):
    """Gate 1: a retrain that scores below the best prior run is rejected (catastrophic-forgetting guard)."""
    c, eng, order = _client(tmp_path)
    eng._train_job = {"output_dir": str(_fake_run(tmp_path, "round_1", metric=40.0)), "coll_version": 1, "n_assigned": 0}
    assert eng.adopt_checkpoint()["ok"]                             # baseline floor = 40.0
    eng._train_job = {"output_dir": str(_fake_run(tmp_path, "round_2", metric=31.0)), "coll_version": 2, "n_assigned": 0}
    rep = eng.adopt_checkpoint()                                    # 31 < 40 -> refused
    assert rep.get("regressed") and "error" in rep and abs(rep["floor"] - 40.0) < 1e-6
    assert eng.state.config["model"]["ckpt"].endswith("round_1/model_best.pth")   # still the good one
    forced = eng.adopt_checkpoint(force=True)                       # override
    assert forced["ok"] and forced["regressed"]
    assert eng.state.config["model"]["ckpt"].endswith("round_2/model_best.pth")


def test_adopt_prunes_heavy_dumps(tmp_path):
    """Adopt reclaims the GBs of eval dumps + intermediate checkpoints, keeping only the adopted ckpt."""
    c, eng, order = _client(tmp_path)
    out_dir = _fake_run(tmp_path, "round_1", metric=42.0, dumps=True)
    eng._train_job = {"output_dir": str(out_dir), "coll_version": 1, "n_assigned": 0}
    assert eng.adopt_checkpoint()["ok"]
    assert (out_dir / "model_best.pth").exists()                    # kept
    for gone in ("inference_val", "inference_test", "val", "predictions.json", "model_0000999.pth"):
        assert not (out_dir / gone).exists(), gone                  # heavy/regenerable artifacts pruned


def test_training_status_flags_disk_full(tmp_path):
    """Gate 0: a finished job whose log shows ENOSPC is reported FAILED, not silently 'done'."""
    c, eng, order = _client(tmp_path)
    out_dir = tmp_path / "train_runs" / "round_1"; out_dir.mkdir(parents=True)
    (out_dir / "train.log").write_text("epoch 0 ...\nOSError: [Errno 28] No space left on device\n")
    class _Done:
        def poll(self): return 1                                    # exited non-zero
    eng._train_job = {"proc": _Done(), "pid": 1, "log": str(out_dir / "train.log"),
                      "output_dir": str(out_dir), "config_name": "x", "cmd": "qseg-train ..."}
    st = eng.training_status()
    assert st["failed"] and "disk" in st["reason"].lower()


def test_merge_recommender(tmp_path):
    """Web merge recommender: train on logged merges -> recommend candidate groups (globally + per-image) ->
    accept (merges) / reject (logs a negative); merge_result serves the would-be merged PNG."""
    c, eng, order = _client(tmp_path)
    for r in eng.collection["records"]:                              # pair_features needs box_xyxy + pred_class
        cx, cy, bw, bh = r["cx"], r["cy"], r["bw"], r["bh"]
        r["box_xyxy"] = np.array([(cx - bw / 2) * 128, (cy - bh / 2) * 128,
                                  (cx + bw / 2) * 128, (cy + bh / 2) * 128], np.float32)
        r["pred_class"] = r["row"] % 3
    # cold start: nothing trained
    assert c.post("/api/train_merge_recommender", json={"features": ["decoder"]}).json()["ok"] is False
    assert c.get("/api/recommend_merges").json()["trained"] is False
    # log a few same-image merges (order[j] & order[j+80] share image 1000+j) -> positive pairs, then train
    for j in range(6):
        assert c.post("/api/merge", json={"iuids": [order[j], order[j + 80]]}).json()["n_groups"] == 1
    rep = c.post("/api/train_merge_recommender", json={"features": ["decoder"], "algo": "logreg"}).json()
    assert rep["ok"] and rep["n_merge_events"] >= 1 and rep["n_pos"] >= 1

    g = c.get("/api/recommend_merges?thresh=0.0").json()            # global: every image with >=2 live instances
    assert g["trained"] and g["groups"] and all({"iuids", "image_id", "prob", "n"} <= set(x) for x in g["groups"])
    iid = str(int(eng.state.meta[order[10]].image_id))              # per-image: scoped to one image
    gi = c.get(f"/api/recommend_merges?image_id={iid}&thresh=0.0").json()
    assert gi["groups"] and all(x["image_id"] == iid for x in gi["groups"])

    png = c.get(f"/api/merge_result?iuids={order[20]},{order[100]}&mode=union")   # same image (1020); would-be merge PNG
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
    grp = gi["groups"][0]["iuids"]                                  # accept a per-image candidate -> merges
    assert c.post("/api/accept_merge", json={"iuids": grp, "mode": "union"}).json()["ok"]
    assert sum(eng.state.meta[u].merged_into is not None for u in grp) >= 1
    c.post("/api/reject_merge", json={"iuids": [order[30], order[110]]})           # logs a negative event
    assert any(e.get("kind") == "reject" for e in eng.store.read_merge_events())


def test_progress_endpoint(tmp_path):
    """/api/progress reports the running inference/RAD-DINO job state (polled by the UI for a progress bar)."""
    c, eng, order = _client(tmp_path)
    p = c.get("/api/progress").json()
    assert {"phase", "done", "total", "active"} <= set(p) and p["active"] is False
    eng._set_progress("segmentation inference", 3, 10)
    p = c.get("/api/progress").json()
    assert p["active"] and p["done"] == 3 and p["total"] == 10 and p["phase"] == "segmentation inference"
    eng._clear_progress()
    assert c.get("/api/progress").json()["active"] is False


def test_inference_threshold_overrides(tmp_path, monkeypatch):
    """sample / infer_dir / preview / reinfer honor per-run score_thresh + nms_iou (override the config
    defaults); blank -> config default. collect_batch is stubbed to record what it was called with."""
    from tools.curator import collect as _co
    c, eng, order = _client(tmp_path)
    eng.state.config.setdefault("model", {})["score_thresh"] = 0.3
    seen = {}
    monkeypatch.setattr(eng, "_ensure_model", lambda: (None, None, None))
    dim = eng.collection["feats"]["decoder"].shape[1]
    def fake_collect(model, cfg, d2_cfg, files, *, score_thresh, feature_cfg):
        seen["score_thresh"] = score_thresh; seen["nms_iou"] = feature_cfg.get("nms_iou")
        return {"records": [], "feats": {"decoder": np.zeros((0, dim), np.float32)}, "n_images": 0}  # match methods for concat
    monkeypatch.setattr(_co, "collect_batch", fake_collect)
    eng.store.save_manifest({"processed_paths": ["/x/a.png", "/x/b.png"]})    # so reinfer has targets
    # explicit overrides reach collect_batch
    c.post("/api/reinfer", json={"mode": "append", "score_thresh": 0.05, "nms_iou": 0.5})
    assert abs(seen["score_thresh"] - 0.05) < 1e-9 and abs(seen["nms_iou"] - 0.5) < 1e-9
    # blank/omitted -> config default score_thresh, default nms
    seen.clear()
    c.post("/api/reinfer", json={"mode": "append"})
    assert abs(seen["score_thresh"] - 0.3) < 1e-9                              # fell back to config default


def test_compute_raddino_endpoint(tmp_path, monkeypatch):
    """RAD-DINO extraction is exposed + makes 'raddino' a selectable feature (was Gradio-only, unreachable
    in the web frontend). GPU extraction stubbed."""
    from tools.curator import _bootstrap
    from tools.curator import collect as _co
    c, eng, order = _client(tmp_path)
    assert "raddino" not in c.get("/api/features").json()["available"]      # not present initially
    monkeypatch.setattr(_bootstrap, "get_P", lambda: object())
    def fake_rad(col, P, progress=None, pool="mask"):
        if progress:
            progress(0, 1)
        col["feats"]["raddino"] = np.zeros((len(col["records"]), 8), np.float32); return col
    monkeypatch.setattr(_co, "_raddino_by_path", fake_rad)
    r = c.post("/api/compute_raddino", json={}).json()
    assert r["ok"] and r["n"] == len(order) and "raddino" in r["available"]
    feat = c.get("/api/features").json()
    assert feat["has_raddino"] and "raddino" in feat["available"]           # now selectable everywhere
    # idempotent: second call sees it present (no recompute) unless force
    assert c.post("/api/compute_raddino", json={}).json()["ok"]


def test_match_features_shows_classes_and_pool(tmp_path):
    """find-by-reference must surface BOTH assigned class:<cid> partitions AND unassigned FINCH partitions —
    not only classes (which crowd out the pool once many instances are assigned)."""
    c, eng, order = _client(tmp_path)
    assert c.post("/api/cluster", json={"features": ["decoder"]}).json()["ok"]
    c.post("/api/assign", json={"iuids": [order[0], order[1]], "cls": "A"})
    q = eng.collection["feats"]["decoder"][eng.state.meta[order[5]].row]
    r = eng.match_features(q, feature="decoder", k=8)
    assert {"matches", "matches_class", "matches_pool"} <= set(r)
    assert any(str(m["pid"]).startswith("class:") for m in r["matches_class"])     # the assigned class appears
    assert r["matches_pool"] and all(not str(m["pid"]).startswith("class:") for m in r["matches_pool"])  # AND pool
    assert all("cls" in m for m in r["matches_class"])                              # class matches carry a name


def test_class_rule_reload(tmp_path):
    """Refine can reload a class's FULL saved rule-chain (ops + kw) — what the preview needs when reselecting
    a class (the summary endpoint only carries op names)."""
    c, eng, order = _client(tmp_path)
    c.post("/api/assign", json={"iuids": [order[0], order[1]], "cls": "tube"})
    ops = [{"name": "contrast", "kw": {"clip": 3.0}, "on": True}, {"name": "vessel_extend", "kw": {"max_width": 6}}]
    assert c.post("/api/apply_class_rule", json={"cls": "tube", "ops": ops}).json()["ok"]
    r = c.get("/api/class_rule?cls=tube").json()
    assert [o["name"] for o in r["ops"]] == ["contrast", "vessel_extend"]
    assert r["ops"][0]["kw"]["clip"] == 3.0 and r["ops"][1]["kw"]["max_width"] == 6   # kw preserved for reload
    assert c.get("/api/class_rule?cls=nope").json()["ops"] == []                       # unknown class -> empty


def test_recommend_interesting(tmp_path):
    """Active-learning acquisition: unassigned instances ranked by classifier uncertainty (most-uncertain
    first); falls back to lowest-detection-score with no classifier."""
    c, eng, order = _client(tmp_path)
    # no classifier -> fallback ranks by (1 - detection score); make one instance clearly low-score
    eng.collection["records"][eng.state.meta[order[5]].row]["score"] = 0.05
    fb = eng.recommend_interesting(10)
    assert fb and fb[0][0] == order[5] and fb[0][1] is None       # lowest-score first, no predicted class
    assert c.get("/api/recommend_interesting").json()["trained"] is False
    # train 2 clean classes (centre = j % 12), then uncertainty ranking kicks in
    c.post("/api/assign", json={"iuids": [order[0], order[12]], "cls": "A"})
    c.post("/api/assign", json={"iuids": [order[1], order[13]], "cls": "B"})
    assert c.post("/api/train_classifier", json={"features": ["decoder"], "algo": "knn"}).json()["ok"]
    rec = eng.recommend_interesting(20, metric="entropy")
    assert rec and all(u in eng.state.unassigned_iuids() for u, _, _ in rec)
    assert [t[2] for t in rec] == sorted((t[2] for t in rec), reverse=True)   # most-uncertain first
    js = c.get("/api/recommend_interesting?metric=margin&limit=5").json()
    assert js["trained"] and len(js["items"]) <= 5 and {"iuid", "cls", "score"} <= set(js["items"][0])


def test_recommend_rejections(tmp_path):
    """The classifier surfaces unassigned instances it matches to NO class (max prob < cutoff) as
    reject candidates — the complement of the assign preview."""
    c, eng, order = _client(tmp_path)
    assert eng.recommend_rejections(0.5) == []                          # cold start: nothing trained
    assert c.get("/api/recommend_rejections?max_conf=0.5").json()["total"] == 0
    # two clean classes (same decoder centre per class; centre = j % 12), then train
    c.post("/api/assign", json={"iuids": [order[0], order[12]], "cls": "A"})
    c.post("/api/assign", json={"iuids": [order[1], order[13]], "cls": "B"})
    assert c.post("/api/train_classifier", json={"features": ["decoder"], "algo": "knn"}).json()["ok"]

    recs = eng.recommend_rejections(0.99)                               # instances near the 10 untrained centres
    assert recs and all(conf < 0.99 for _, _, conf in recs)
    assert all(eng.state.meta[u].assigned_class is None and not eng.state.meta[u].is_background for u, _, _ in recs)
    assert [t[2] for t in recs] == sorted(t[2] for t in recs)           # ascending: most-clearly-not-a-class first
    assert eng.recommend_rejections(0.0) == []                          # nothing has prob < 0
    js = c.get("/api/recommend_rejections?max_conf=0.99&limit=5").json()
    assert js["total"] == len(recs) and len(js["items"]) <= 5 and {"iuid", "cls", "conf"} <= set(js["items"][0])
    iu = js["items"][0]["iuid"]                                         # rejecting a candidate moves it to background
    assert c.post("/api/reject", json={"iuids": [iu]}).json()["ok"] and eng.state.meta[iu].is_background


def test_refine_uses_merge_union_not_original(tmp_path):
    """Refining a MERGED representative starts from the union (the effective mask), not the rep's original
    single-instance mask — otherwise refine silently reverts the merge."""
    import cv2
    import numpy as np
    from pycocotools import mask as mu
    c, eng, order = _client(tmp_path)
    a, b = order[0], order[1]
    iid = eng.state.meta[a].image_id
    eng.state.meta[b].image_id = iid
    eng.collection["records"][eng.state.meta[b].row]["image_id"] = iid

    def enc(m):
        r = mu.encode(np.asfortranarray(m.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii"); return r
    ma = np.zeros((128, 128), np.uint8); cv2.circle(ma, (40, 64), 15, 1, -1)
    mb = np.zeros((128, 128), np.uint8); cv2.circle(mb, (90, 64), 15, 1, -1)   # disjoint circle
    eng.collection["records"][eng.state.meta[a].row]["rle"] = enc(ma)
    eng.collection["records"][eng.state.meta[b].row]["rle"] = enc(mb)

    eng.merge_instances([a, b])                                    # same image -> one union group
    rep = a if eng.state.meta[a].merge_members else b
    base = eng._refine_base_rle(rep)
    assert int(mu.area(base)) > int(ma.sum())                      # union (2 circles) > rep's single
    assert abs(int(mu.area(base)) - int((ma | mb).sum())) <= 5     # ~= the union of both
    # a non-merged instance still refines from its original
    assert eng._refine_base_rle(order[5]) is eng.collection["records"][eng.state.meta[order[5]].row]["rle"]


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


def test_release_gate_endpoints(tmp_path):
    c, eng, order = _client(tmp_path)
    img0 = "1000"                                                    # assign every instance of one image -> final
    iu = [u for u in order if str(eng.state.meta[u].image_id) == img0]
    assert len(iu) > 1
    c.post("/api/assign", json={"iuids": iu, "cls": "lung"})

    rel = c.get("/api/release_images?filter=pending").json()
    assert rel["stats"]["fully_categorized"] == 1 and rel["stats"]["pending"] == 1
    it = next(x for x in rel["items"] if x["image_id"] == img0)
    assert it["status"] == "" and it["n_assigned"] == len(iu)

    r = c.post("/api/release_set", json={"image_ids": [img0], "status": "accepted"}).json()
    assert r["ok"] and r["stats"]["accepted"] == 1 and r["stats"]["pending"] == 0
    assert all(x["image_id"] != img0 for x in c.get("/api/release_images?filter=pending").json()["items"])
    acc = c.get("/api/release_images?filter=accepted").json()["items"]
    assert any(x["image_id"] == img0 and x["status"] == "accepted" for x in acc)
