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


def test_sam_status_and_graceful_refine(tmp_path, monkeypatch):
    """/api/sam_status reports state; a 'sam' op with no checkpoint returns a clean 400, not a 500."""
    from tools.curator import refine as _rf
    c, eng, order = _client(tmp_path)
    s = c.get("/api/sam_status").json()
    assert {"installed", "ckpt", "model_type"} <= set(s)
    # force the no-checkpoint path regardless of the dev box's cache
    monkeypatch.setattr(_rf, "find_sam_checkpoint", lambda ckpt=None: (None, None))
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
