"""Per-partition 1-NN "most likely class" suggestion (class / reject / none) + distance gate.
Run: pytest tests/test_partition_suggestion.py -q
"""
from __future__ import annotations

import numpy as np


def _engine(tmp_path):
    """4 moderately-separated blobs in 8-d decoder space: classes A,B + a REJECT blob + a FAR blob, each in
    a LABELED copy and an UNASSIGNED copy (the unassigned copies get clustered into FINCH partitions)."""
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
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
    from chevron.engine import CuratorEngine
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
    from chevron.server import create_app
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
    from chevron.server import create_app
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
    from chevron.server import create_app
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
    assert all("assigned" in it for it in r["items"])                    # every item reports its current category
    assert by[grp["A_un"][0]]["assigned"] is None and by[grp["FAR_un"][0]]["assigned"] is None  # all unassigned here
    assert r["summary"].get("reject") == 1 and r["summary"].get("none") == 1
    assert sum(v for k, v in r["summary"].items() if k not in ("reject", "none")) == 2

    c = TestClient(create_app(engine=eng))
    j = c.get(f"/api/image_suggestion?image_id={img}").json()
    assert {"items", "summary", "threshold", "has_reject"} <= set(j) and len(j["items"]) == 4


def test_image_suggestion_no_labels(tmp_path):
    from chevron.engine import CuratorEngine
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    eng.collection = {"records": [], "n_images": 0, "feats": {"decoder": np.zeros((0, 8), np.float32)}}
    eng.state.coll_version = 1
    assert eng.image_class_suggestion(1)["items"] == []   # no labels -> empty, no raise


def test_accept_partition_suggestion(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
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
    from chevron.server import create_app
    eng, grp = _engine(tmp_path)
    img = 8888
    pre = grp["B_lab"][0]; cid_b = eng.state.class_id_by_name("B")              # an ALREADY-categorized instance
    mix = [grp["A_un"][0], grp["A_un"][1], grp["BG_un"][0], grp["FAR_un"][0], pre]
    for u in mix:
        eng.state.meta[u].image_id = img
    c = TestClient(create_app(engine=eng))
    r = c.post("/api/accept_image_predictions", json={"image_id": img}).json()
    assert r["ok"] and r["rejected"] == 1 and r["skipped"] == 1                 # BG->reject, FAR->none(left)
    assert r["skipped_assigned"] == 1                                           # the already-assigned B instance is outside the gate
    assert sum(r["assigned"].values()) == 2                                     # only the two UNassigned A/B-region instances
    assert eng.state.meta[pre].assigned_class == cid_b                          # untouched — kept its original category
    assert eng.state.meta[grp["BG_un"][0]].is_background                        # reject applied
    assert eng.state.meta[grp["A_un"][0]].assigned_class is not None            # class applied
    assert eng.state.meta[grp["FAR_un"][0]].assigned_class is None and not eng.state.meta[grp["FAR_un"][0]].is_background  # 'none' untouched


def test_partition_predictions_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    c = TestClient(create_app(engine=eng))
    pid_a = eng.partition_of(grp["A_un"][0])
    r = c.get(f"/api/partition_predictions?pid={pid_a}").json()
    assert {"items", "threshold", "has_reject", "n_total"} <= set(r)
    assert set(r["items"]) == set(eng.partition_iuids(pid_a))               # EVERY member predicted
    assert all(it["label"] in ("A", "B") for it in r["items"].values())    # near the A/B region (1-NN noise)
    assert all(it["assigned"] is None for it in r["items"].values())       # FINCH members are unassigned
    rb = c.get(f"/api/partition_predictions?pid={eng.partition_of(grp['BG_un'][0])}").json()
    assert all(it["label"] == "reject" for it in rb["items"].values())     # near the rejected blob -> reject
    rf = c.get(f"/api/partition_predictions?pid={eng.partition_of(grp['FAR_un'][0])}").json()
    assert all(it["label"] == "none" for it in rf["items"].values())       # far from everything -> none


def test_instances_pred_filter(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    c = TestClient(create_app(engine=eng))
    pid_bg = eng.partition_of(grp["BG_un"][0]); members = set(eng.partition_iuids(pid_bg))
    assert c.get(f"/api/instances?pid={pid_bg}&limit=1000").json()["total"] == len(members)
    rej = c.get(f"/api/instances?pid={pid_bg}&pred=reject&limit=1000").json()
    assert rej["total"] == len(members) and {it["iuid"] for it in rej["items"]} == members   # all predicted reject
    assert c.get(f"/api/instances?pid={pid_bg}&pred=A&limit=1000").json()["total"] == 0       # none predicted A


def test_accept_partition_subset(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    c = TestClient(create_app(engine=eng))
    # reject the BG partition's predicted-reject subset (its whole self)
    pid_bg = eng.partition_of(grp["BG_un"][0]); members = list(eng.partition_iuids(pid_bg))
    r = c.post("/api/accept_partition_subset", json={"pid": pid_bg, "label": "reject"}).json()
    assert r["ok"] and r["action"] == "reject" and r["n"] == len(members)
    assert all(eng.state.meta[u].is_background for u in members)
    # assign the A partition's predicted-A subset to A (B-predicted noise stays unassigned)
    pid_a = eng.partition_of(grp["A_un"][0]); amem = list(eng.partition_iuids(pid_a))
    pred_a = set(eng.partition_iuids_predicted(pid_a, label="A"))
    cid_a = eng.state.class_id_by_name("A")
    ra = c.post("/api/accept_partition_subset", json={"pid": pid_a, "label": "A"}).json()
    assert ra["action"] == "assign" and ra["cls"] == "A" and ra["n"] == len(pred_a) and ra["n"] >= 1
    assert {u for u in amem if eng.state.meta[u].assigned_class == cid_a} == pred_a   # exactly the A-predicted ones


def test_class_partition_never_self_or_identical_copy_match(tmp_path):
    """A class partition's members must never match THEMSELVES: not their own row, and not a dist-0 identical
    COPY of themselves. Every member should match a DISTINCT neighbour (dist > 0 → score < 1)."""
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    rng = np.random.default_rng(1)
    order, recs, meta, feats = [], [], {}, []
    def add(vec):
        i = len(order); u = f"u{i}"; order.append(u); recs.append({"iuid": u, "row": i, "score": 0.6})
        meta[u] = InstanceMeta(u, "b", i, 1000 + i); feats.append(np.asarray(vec, float)); return u
    A = [add(np.asarray([3., 1., 0., 0., 0., 0.]) + rng.normal(0, 0.3, 6)) for _ in range(10)]
    A += [add(feats[0].copy()) for _ in range(3)]              # 3 EXACT copies of A[0] (dist-0 hazard)
    B = [add(np.asarray([1., 3., 0., 0., 0., 0.]) + rng.normal(0, 0.3, 6)) for _ in range(8)]
    eng.collection = {"records": recs, "n_images": len(order), "feats": {"decoder": np.array(feats, np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.assign(A, "A"); eng.assign(B, "B")
    items = eng._partition_member_preds(f"class:{eng.state.class_id_by_name('A')}")["items"]
    assert len(items) == len(A)
    assert all(it["score"] < 0.999 for it in items.values()), [round(it["score"], 4) for it in items.values()]
    assert all(it["label"] == "A" for it in items.values())   # still confidently its own class, via a DISTINCT member


def test_subset_gate_strict_shrinks(tmp_path):
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    pid_a = eng.partition_of(grp["A_un"][0])
    loose = eng._partition_member_preds(pid_a, gate_mult=2.0)["items"]
    strict = eng._partition_member_preds(pid_a, gate_mult=0.01)["items"]
    n_cls = lambda items: sum(1 for it in items.values() if it["label"] not in ("none", "reject"))
    assert n_cls(strict) < n_cls(loose)                                    # strict gate -> fewer class, more 'none'
    assert all(it["label"] == "none" for it in strict.values())            # very strict -> all 'none'


# ---- image workload ranking (estimated manual decisions left, from the 1-NN classifier) ------------------
def _img_of(eng, iuid):
    return str(1000 + eng.state.meta[iuid].row)        # the _engine fixture: one instance per image, id = 1000+row


def test_image_workload_ranking_buckets(tmp_path):
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    r = eng.image_workload_ranking(order="hard")
    assert not r["fallback"]
    it = {x["image_id"]: x for x in r["items"]}
    # FAR_un: unassigned, far from every label -> a real manual decision (NONE bucket)
    far = it[_img_of(eng, grp["FAR_un"][0])]
    assert far["n_uncat"] == 1 and far["n_none"] == 1 and far["work_est"] >= 1.0 and not far["done"]
    # A_un: unassigned near class A -> auto-resolvable (one Accept-all), no residual work
    a = it[_img_of(eng, grp["A_un"][0])]
    assert a["n_uncat"] == 1 and a["n_auto"] == 1 and a["work_est"] == 0.0 and not a["done"]
    # BG_un: nearest ref is the reject blob -> still AUTO (Accept-all backgrounds it), work_est 0
    bg = it[_img_of(eng, grp["BG_un"][0])]
    assert bg["n_auto"] == 1 and bg["work_est"] == 0.0
    # labeled / background images are fully categorized -> tagged done ('ready')
    assert it[_img_of(eng, grp["A_lab"][0])]["done"] and it[_img_of(eng, grp["A_lab"][0])]["n_uncat"] == 0
    assert it[_img_of(eng, grp["BG_lab"][0])]["done"]


def test_image_workload_ranking_orders(tmp_path):
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    for order, rev in (("hard", True), ("easy", False)):
        r = eng.image_workload_ranking(order=order)
        nd = [x for x in r["items"] if not x["done"]]
        work = [x["work_est"] for x in nd]
        assert work == sorted(work, reverse=rev)                       # most-work-first / least-work-first
        # done images ('ready') sorted to the END of either order
        assert all(r["items"][i]["done"] <= r["items"][i + 1]["done"] for i in range(len(r["items"]) - 1))


def test_image_ranking_gate_reuses_one_nn_pass(tmp_path):
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    a_img = _img_of(eng, grp["A_un"][0])
    loose = {x["image_id"]: x for x in eng.image_workload_ranking(order="hard", gate_mult=2.0)["items"]}
    cache = eng._iwl_dist_cache                                          # gate-independent NN pass, now cached
    strict = {x["image_id"]: x for x in eng.image_workload_ranking(order="hard", gate_mult=0.01)["items"]}
    assert eng._iwl_dist_cache is cache                                  # moving the gate did NOT re-query faiss
    assert loose[a_img]["work_est"] == 0.0                              # loose gate -> A_un auto-resolved
    assert strict[a_img]["work_est"] >= 1.0                             # strict gate -> now a manual decision


def test_image_ranking_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    c = TestClient(create_app(engine=eng))
    r = c.get("/api/image_ranking?order=hard&gate_mult=1.0").json()
    assert r["order"] == "hard" and not r["fallback"] and r["items"]
    assert {"image_id", "n_uncat", "n_auto", "n_none", "work_est", "done"} <= set(r["items"][0])


def test_image_ranking_fallback_no_labels(tmp_path):
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    order, recs, meta, feats = [], [], {}, []
    for i in range(6):
        u = f"u{i}"; order.append(u); recs.append({"iuid": u, "row": i, "score": 0.5})
        meta[u] = InstanceMeta(u, "b", i, 1000 + i); feats.append(np.ones(8, np.float32) * i)
    eng.collection = {"records": recs, "n_images": 6, "feats": {"decoder": np.array(feats, np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    r = eng.image_workload_ranking(order="easy")
    assert r["fallback"] and r["note"] == "no labels yet"               # no classifier yet -> most-populated order
    assert r["items"] and r["items"][0]["work_est"] is None and r["items"][0]["n_inst"] >= 1


def test_image_ranking_diversity_interleaves_classes(tmp_path):
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    # gate loose so A_un/B_un/BG_un all auto-resolve (work_est 0) -> they TIE on work; only their order differs.
    def head_classes(div, n=6):
        r = eng.image_workload_ranking(order="easy", gate_mult=2.0, diversity=div)
        nd = [x for x in r["items"] if not x["done"]]
        return [x["top_class"] for x in nd[:n]], r
    from collections import Counter
    plain, rp = head_classes(0.0)
    diverse, rd = head_classes(1.0)
    assert Counter(plain).most_common(1)[0][1] >= 5     # pure work order -> head dominated by ONE class (the bias)
    assert len(set(diverse)) > len(set(plain))          # variety re-rank brings in more distinct classes
    assert len(set(diverse[:3])) == 3                    # ...and spans all 3 (A/B/reject) right at the top
    assert rd["diversity"] == 1.0 and rp["diversity"] == 0.0
    # every non-done item still carries its dominant predicted class
    assert all(x["top_class"] in ("A", "B", "reject") for x in rd["items"] if not x["done"])


def test_image_ranking_diversity_gate_independent_cache(tmp_path):
    eng, grp = _engine(tmp_path)
    eng.cluster({"decoder": 1.0}, req_clust=4)
    eng.image_workload_ranking(order="easy", gate_mult=1.0, diversity=0.0)
    cache = eng._iwl_dist_cache                         # labels+dists computed once
    eng.image_workload_ranking(order="hard", gate_mult=0.5, diversity=1.0)
    assert eng._iwl_dist_cache is cache                 # diversity + gate only re-rank in python, no faiss re-query
