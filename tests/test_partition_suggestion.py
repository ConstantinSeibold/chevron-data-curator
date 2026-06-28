"""Per-partition 1-NN "most likely class" suggestion (class / reject / none) + distance gate.
Run: pytest tools/curator/tests/test_partition_suggestion.py -q
"""
from __future__ import annotations

import numpy as np


def _engine(tmp_path):
    """4 moderately-separated blobs in 8-d decoder space: classes A,B + a REJECT blob + a FAR blob, each in
    a LABELED copy and an UNASSIGNED copy (the unassigned copies get clustered into FINCH partitions)."""
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    rng = np.random.default_rng(0)
    cent = {"A": [3, 1, 1, 0, 0, 0, 0, 0], "B": [1, 3, 1, 0, 0, 0, 0, 0],
            "BG": [1, 1, 3, 0, 0, 0, 0, 0], "FAR": [0, 0, 0, 0, 0, 5, 5, 0]}
    order, recs, meta, feats, grp = [], [], {}, [], {}
    i = 0
    # A/B/BG get a LABELED (or background) copy; ALL four get an UNASSIGNED copy. FAR has NO reference copy
    # -> it is far from everything labeled, so its partition must come back "no likely class".
    copies = [(g + "_lab", cent[g]) for g in ("A", "B", "BG")] + [(g + "_un", cent[g]) for g in cent]
    for tag, c in copies:
        grp[tag] = []
        for _ in range(12):
            u = f"u{i}"; order.append(u); recs.append({"iuid": u, "row": i, "score": 0.6})
            meta[u] = InstanceMeta(u, "b", i, 1000 + i)
            feats.append(np.asarray(c, float) + rng.normal(0, 0.2, 8)); grp[tag].append(u); i += 1
    eng.collection = {"records": recs, "n_images": i, "feats": {"decoder": np.array(feats, np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.assign(grp["A_lab"], "A"); eng.assign(grp["B_lab"], "B"); eng.set_background(grp["BG_lab"])
    return eng, grp


def _suggest_for(eng, member_iuid, **kw):
    pid = eng.partition_of(member_iuid)
    return eng.partition_class_suggestion(pid, **kw)


def test_suggestion_class_reject_none(tmp_path):
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)                              # FINCH over the unassigned copies
    near_a = _suggest_for(eng, grp["A_un"][0])
    assert near_a["verdict"] == "class" and near_a["top_class"] == "A" and near_a["confidence"] >= 0.7
    near_bg = _suggest_for(eng, grp["BG_un"][0])
    assert near_bg["verdict"] == "reject" and near_bg["reject_likelihood"] >= 0.7 and near_bg["top_class"] is None
    far = _suggest_for(eng, grp["FAR_un"][0])
    assert far["verdict"] == "none" and far["median_nearest_dist"] > far["threshold"]


def test_gate_mult_tightens(tmp_path):
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    pid = eng.partition_of(grp["A_un"][0])
    assert eng.partition_class_suggestion(pid, gate_mult=2.0)["verdict"] == "class"   # loose -> keeps A
    assert eng.partition_class_suggestion(pid, gate_mult=0.01)["verdict"] == "none"   # very strict -> none


def test_class_partition_excludes_self(tmp_path):
    eng, grp = _engine(tmp_path)
    cid = eng.state.class_id_by_name("A")
    r = eng.partition_class_suggestion(f"class:{cid}")
    assert r["top_class"] == "A" and r["verdict"] == "class"   # k=2 self-skip finds another A, not a 0-dist self
    assert 0.0 < r["median_nearest_dist"]                       # not the degenerate distance-0 artifact


def test_no_labels_and_empty(tmp_path):
    from tools.curator.engine import CuratorEngine
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    eng.collection = {"records": [], "n_images": 0, "feats": {"decoder": np.zeros((0, 8), np.float32)}}
    eng.state.coll_version = 1
    assert eng.partition_class_suggestion("class:nope")["verdict"] == "n/a"   # no labels -> n/a, no raise

    eng2, grp = _engine(tmp_path / "p2")
    assert eng2.partition_class_suggestion("999")["verdict"] == "n/a"          # unknown/empty partition


def test_reject_likelihood_reported_alongside_class(tmp_path):
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    r = eng.partition_class_suggestion(eng.partition_of(grp["A_un"][0]))
    assert "reject_likelihood" in r and "confidence" in r and r["has_reject"] is True   # always both


def test_cache_invalidates_on_mutation(tmp_path):
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    eng.partition_class_suggestion(eng.partition_of(grp["A_un"][0]))
    key1 = eng._psug_cache[0]
    eng.partition_class_suggestion(eng.partition_of(grp["B_un"][0]))
    assert eng._psug_cache[0] == key1                          # no mutation -> same reference index reused
    eng.assign([grp["A_un"][0]], "A")                          # a label changed
    eng.partition_class_suggestion(eng.partition_of(grp["B_un"][0]))
    assert eng._psug_cache[0] != key1                          # rebuilt


def test_partition_suggestion_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from tools.curator.server import create_app
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    c = TestClient(create_app(engine=eng))
    pid = eng.partition_of(grp["A_un"][0])
    r = c.get(f"/api/partition_suggestion?pid={pid}").json()
    assert {"verdict", "top_class", "confidence", "reject_likelihood", "threshold", "has_reject"} <= set(r)
    assert r["verdict"] == "class" and r["top_class"] == "A"
    strict = c.get(f"/api/partition_suggestion?pid={pid}&gate_mult=0.01").json()
    assert strict["verdict"] == "none"


def test_reject_partition(tmp_path):
    from fastapi.testclient import TestClient
    from tools.curator.server import create_app
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    pid = eng.partition_of(grp["FAR_un"][0])
    members = list(eng.partition_iuids(pid))
    assert len(members) >= 2
    c = TestClient(create_app(engine=eng))
    r = c.post("/api/reject_partition", json={"pid": pid}).json()
    assert r["ok"] and r["n"] == len(members)
    assert all(eng.state.meta[u].is_background for u in members)     # whole partition rejected
    assert eng.partition_iuids(pid) == []                            # partition now empty
    assert c.post("/api/reject_partition", json={}).json()["n"] == 0  # no pid -> no-op, not 500


def test_image_class_suggestion(tmp_path):
    from fastapi.testclient import TestClient
    from tools.curator.server import create_app
    eng, grp = _engine(tmp_path)
    # one image with: 2 instances near the labeled A/B region + 1 near the reject blob + 1 in the FAR region
    img = 7777
    mix = [grp["A_un"][0], grp["A_un"][1], grp["BG_un"][0], grp["FAR_un"][0]]
    for u in mix:
        eng.state.meta[u].image_id = img
    r = eng.image_class_suggestion(img)
    by = {it["iuid"]: it for it in r["items"]}
    assert len(r["items"]) == 4
    assert by[grp["BG_un"][0]]["label"] == "reject"                       # near the rejected blob -> reject
    assert by[grp["FAR_un"][0]]["label"] == "none"                        # far from everything labeled -> none
    assert by[grp["A_un"][0]]["label"] in ("A", "B")                      # near the labeled class region -> a class
    assert by[grp["A_un"][0]]["pred"] == by[grp["A_un"][0]]["label"]      # pred carries the class name
    assert r["summary"].get("reject") == 1 and r["summary"].get("none") == 1
    assert sum(v for k, v in r["summary"].items() if k not in ("reject", "none")) == 2

    c = TestClient(create_app(engine=eng))
    j = c.get(f"/api/image_suggestion?image_id={img}").json()
    assert {"items", "summary", "threshold", "has_reject"} <= set(j) and len(j["items"]) == 4


def test_image_suggestion_no_labels(tmp_path):
    from tools.curator.engine import CuratorEngine
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    eng.collection = {"records": [], "n_images": 0, "feats": {"decoder": np.zeros((0, 8), np.float32)}}
    eng.state.coll_version = 1
    assert eng.image_class_suggestion(1)["items"] == []   # no labels -> empty, no raise


def test_accept_partition_suggestion(tmp_path):
    from fastapi.testclient import TestClient
    from tools.curator.server import create_app
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    c = TestClient(create_app(engine=eng))
    # near-A FINCH partition -> Accept assigns the whole partition to A
    pid_a = eng.partition_of(grp["A_un"][0]); members = list(eng.partition_iuids(pid_a))
    r = c.post("/api/accept_partition_suggestion", json={"pid": pid_a}).json()
    assert r["action"] == "assign" and r["cls"] == "A" and r["n"] == len(members)
    cid = eng.state.class_id_by_name("A")
    assert all(eng.state.meta[u].assigned_class == cid for u in members)
    # FAR partition -> Accept does nothing (no likely class)
    pid_far = eng.partition_of(grp["FAR_un"][0])
    assert c.post("/api/accept_partition_suggestion", json={"pid": pid_far}).json()["action"] == "none"


def test_accept_image_predictions(tmp_path):
    from fastapi.testclient import TestClient
    from tools.curator.server import create_app
    eng, grp = _engine(tmp_path)
    img = 8888
    mix = [grp["A_un"][0], grp["A_un"][1], grp["BG_un"][0], grp["FAR_un"][0]]
    for u in mix:
        eng.state.meta[u].image_id = img
    c = TestClient(create_app(engine=eng))
    r = c.post("/api/accept_image_predictions", json={"image_id": img}).json()
    assert r["ok"] and r["rejected"] == 1 and r["skipped"] == 1                 # BG->reject, FAR->none(left)
    assert sum(r["assigned"].values()) == 2                                     # the two A/B-region instances assigned
    assert eng.state.meta[grp["BG_un"][0]].is_background                        # reject applied
    assert eng.state.meta[grp["A_un"][0]].assigned_class is not None            # class applied
    assert eng.state.meta[grp["FAR_un"][0]].assigned_class is None and not eng.state.meta[grp["FAR_un"][0]].is_background  # 'none' untouched
