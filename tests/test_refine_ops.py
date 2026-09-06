"""New refine ops (within-mask threshold, grabcut, magic_wand, snap_edges) + engine.split_instances.
Run: pytest tests/test_refine_ops.py -q  (from repo root)
"""
from __future__ import annotations

import pytest
import numpy as np


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


# ---- refine.py ops ---------------------------------------------------------
def test_threshold_within_mask_vs_grow():
    import cv2
    from chevron import refine
    gray = np.full((64, 64), 30, np.uint8)
    cv2.rectangle(gray, (20, 20), (44, 44), 200, -1)          # bright square (interior)
    mask = np.zeros((64, 64), bool); mask[24:40, 24:40] = True  # smaller mask inside the square
    grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    within = refine.manual_threshold(gray, grown, 128, within_mask=True)
    # within-mask threshold can only KEEP pixels inside the (grown) input mask
    assert within.sum() > 0 and bool((within & ~grown).any()) is False
    free = refine.manual_threshold(gray, mask, 128, within_mask=False)
    # the grow variant may extend beyond the original small mask (it dilates the region)
    assert free.sum() >= mask.sum()


def test_grabcut_and_magic_wand_and_snap_return_valid_masks():
    import cv2
    from chevron import refine
    gray = np.full((80, 80), 40, np.uint8)
    cv2.circle(gray, (40, 40), 18, 220, -1)
    seed = np.zeros((80, 80), bool); seed[34:46, 34:46] = True
    for fn in (lambda m: refine.grabcut(gray, m, iters=3),
               lambda m: refine.magic_wand(gray, m, tol=0.1),
               lambda m: refine.active_contour_snap(gray, m, iters=10)):
        out = fn(seed)
        assert out.dtype == bool and out.shape == (80, 80)
    # magic wand grows the seed toward the similar-intensity bright disk (never shrinks below seed)
    mw = refine.magic_wand(gray, seed, tol=0.15)
    assert mw.sum() >= seed.sum()


def test_apply_ops_chain_dilate_then_threshold_within():
    import cv2
    from chevron import refine
    gray = np.full((64, 64), 30, np.uint8)
    cv2.rectangle(gray, (20, 20), (44, 44), 200, -1)
    mask = np.zeros((64, 64), bool); mask[26:38, 26:38] = True
    out = refine.apply_ops(gray, mask, [{"name": "dilate", "kw": {"k": 4, "max_contrast": 0.9}},
                                        {"name": "threshold", "kw": {"val": 128, "within_mask": True}}])
    assert out.dtype == bool and out.shape == (64, 64) and out.sum() > 0


def _skel_endpoints(mask):
    """Count 1-neighbor skeleton pixels — a simple path has exactly 2; a branch/mesh has >2."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    sk = skeletonize(mask > 0).astype(np.uint8)
    nb = ndi.convolve(sk, np.ones((3, 3), np.uint8), mode="constant") - sk
    return int(((nb == 1) & (sk > 0)).sum())


def test_line_centerline_prunes_branch_and_bridges_gap_vs_vessel_extend():
    import cv2
    from chevron import refine
    H = W = 96
    mask = np.zeros((H, W), bool)
    mask[19:22, 8:81] = True               # main horizontal line (3 px thick)
    mask[19:22, 40:48] = False             # GAP -> two fragments
    mask[6:21, 29:32] = True               # vertical BRANCH off the left fragment (-> Y / mesh)
    gray = (mask.astype(np.uint8) * 200)   # bright tube on dark bg (sato finds the ridge)

    # the synthetic input is genuinely branchy + fragmented
    assert cv2.connectedComponents(mask.astype(np.uint8))[0] - 1 == 2
    assert _skel_endpoints(mask) >= 3

    out = refine.line_centerline(gray, mask, alpha=0.7)
    assert out.dtype == bool and out.shape == (H, W)
    assert cv2.connectedComponents(out.astype(np.uint8))[0] - 1 == 1   # gap bridged -> single component
    assert _skel_endpoints(out) == 2                                   # single simple path: cannot mesh
    assert out[19:22, 8:12].any() and out[19:22, 76:81].any()          # both true line ends kept
    assert out[6:11, 29:32].sum() == 0                                 # off-axis branch PRUNED

    # determinism: same input -> byte-identical output (no run-to-run randomness)
    assert np.array_equal(out, refine.line_centerline(gray, mask, alpha=0.7))

    # A/B vs the grow op: vessel_extend is monotone (only adds) -> it RETAINS the branch, never prunes it
    ve = refine.apply_ops(gray, mask, [{"name": "vessel_extend", "kw": {}}])
    assert ve[6:11, 29:32].sum() >= mask[6:11, 29:32].sum()


def test_autorefine_search_line_and_blob():
    import cv2
    from chevron import autorefine as ar, refine

    # line: branchy + fragmented -> search should pick a line_centerline chain that yields one clean path
    m = np.zeros((96, 96), bool)
    m[19:22, 8:81] = True; m[19:22, 40:48] = False; m[6:21, 29:32] = True
    g = (m.astype(np.uint8) * 200)
    assert ar.classify_shape(m) == "line"
    res = ar.search(g, m, kind="auto")
    best = res["best"]
    assert res["kind"] == "line"
    assert any(o["name"] == "line_centerline" for o in best["chain"])
    assert best["ncc"] == 1                                              # collapsed to a single component
    noop = next(c for c in res["candidates"] if c["chain"] == [])
    assert best["score"] > noop["score"]                                # beats leaving the mess alone
    assert ar.search(g, m, kind="auto")["best"]["chain"] == best["chain"]  # deterministic

    # blob: holey disk + a stray speckle -> search should clean it (fill / largest_cc), not no-op
    b = np.zeros((80, 80), np.uint8)
    cv2.circle(b, (40, 40), 20, 1, -1); cv2.circle(b, (40, 40), 6, 0, -1); b[5:8, 5:8] = 1
    bm = b > 0; gb = (bm.astype(np.uint8) * 200)
    assert ar.classify_shape(bm) == "blob"
    rb = ar.search(gb, bm, kind="auto")
    assert rb["kind"] == "blob" and rb["best"]["ncc"] == 1 and len(rb["best"]["chain"]) >= 1
    assert rb["best"]["score"] > next(c for c in rb["candidates"] if c["chain"] == [])["score"]


def test_autorefine_leaves_clean_line_intact():
    from chevron import autorefine as ar, refine
    m = np.zeros((64, 100), bool); m[30:33, 10:90] = True               # one clean straight line
    g = (m.astype(np.uint8) * 200)
    best = ar.search(g, m, kind="line")["best"]
    assert best["ncc"] == 1                                             # stays a single component
    assert _skel_endpoints(refine.apply_ops(g, m, best["chain"])) == 2  # stays a simple 2-tip path (no damage)


def test_autorefine_partition_kind_robust_to_a_lying_member():
    from chevron import autorefine as ar
    lines = []
    for L in (60, 70, 80, 50):
        mm = np.zeros((96, 96), bool); mm[40:43, 8:8 + L] = True; lines.append(mm)
    stub = np.zeros((96, 96), bool); stub[40:52, 40:52] = True          # a 12x12 blob — a lying member
    assert ar.classify_shape(stub) == "blob"                           # alone it misclassifies
    assert ar.partition_kind(lines + [stub]) == "line"                 # in CONTEXT the class is a line


def test_autorefine_search_respects_injected_reward():
    from chevron import autorefine as ar
    m = np.zeros((64, 100), bool); m[30:33, 10:90] = True
    g = (m.astype(np.uint8) * 200)
    def biggest(orig, cand): return float(cand.sum()), {"area": int(cand.sum())}   # category reward stand-in
    res = ar.search(g, m, kind="line", reward_fn=biggest, reward_name="custom")
    assert res["reward"] == "custom"                                   # reported through
    assert res["best"]["area"] == max(c["area"] for c in res["candidates"])  # argmax of the injected reward


def test_autorefine_shape_prior_reward_smoke():
    import cv2
    import torch
    from qseg.evaluation.shape_prior_model import ConvDAE
    from chevron import autorefine as ar
    torch.manual_seed(0)
    rf = ar.shape_prior_reward(ConvDAE().eval(), device="cpu")
    b = np.zeros((80, 80), np.uint8); cv2.circle(b, (40, 40), 18, 1, -1); m = b > 0
    score, br = rf(m, m)
    assert 0.0 <= score <= 1.0 and "plausibility" in br


def test_write_behind_hot_path_persists_via_flush(tmp_path):
    from chevron import ids
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    u = ids.new_uid()
    eng.collection = {"records": [{"iuid": u, "row": 0, "inst_id": 0, "image_id": 1000, "H": 8, "W": 8,
                                   "score": 0.9, "rle": _rle(np.ones((8, 8), bool)), "file_name": "x",
                                   "abs_path": "x", "batch_id": "b", "cx": 0.5, "cy": 0.5, "bw": 1.0,
                                   "bh": 1.0, "box_area": 1.0, "mask_area_frac": 1.0}], "n_images": 1,
                      "feats": {"decoder": np.zeros((1, 4), np.float32),
                                "shapecoord": np.zeros((1, 29), np.float32), "_shapecoord_cols": ["c"] * 29}}
    eng.state.order = [u]; eng.state.meta = {u: InstanceMeta(iuid=u, batch_id="b", row=0, image_id=1000)}
    eng.state.coll_version = 1; eng.store.save_collection(eng.collection); eng.save()

    eng.assign([u], "lung")                              # hot path -> _after_mutation -> write-behind (deferred)
    assert eng.state.meta[u].assigned_class is not None  # in-memory state is correct immediately
    eng.flush()                                          # force the deferred write
    eng2 = CuratorEngine(tmp_path)                       # reopen from disk
    cid = eng2.state.class_id_by_name("lung")
    assert cid is not None and eng2.state.meta[u].assigned_class == cid   # the deferred mutation persisted
    eng.close(); eng2.close()


def test_partition_view_single_pass_and_class_name_dedup(tmp_path):
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta, TaxonomyClass
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    eng.state.taxonomy = {"cA": TaxonomyClass(class_id="cA", name="lung"),       # cA + cB share a name (dup)
                          "cB": TaxonomyClass(class_id="cB", name="lung"),
                          "cC": TaxonomyClass(class_id="cC", name="rib")}
    assign = ["cA", "cA", "cB", "cC", "cC"]
    recs = [{"iuid": f"u{i}", "row": i, "score": 0.5 + 0.1 * i} for i in range(len(assign))]
    eng.collection = {"records": recs, "n_images": len(assign),
                      "feats": {"decoder": np.zeros((len(assign), 4), np.float32)}}
    eng.state.order = [f"u{i}" for i in range(len(assign))]
    eng.state.meta = {f"u{i}": InstanceMeta(iuid=f"u{i}", batch_id="b", row=i, image_id=1000 + i,
                                            assigned_class=cid) for i, cid in enumerate(assign)}
    eng.state.coll_version = 1

    rows = eng.partition_view()
    by_pid = {r["pid"]: r for r in rows}
    assert by_pid["class:cA"]["size"] == 2 and by_pid["class:cB"]["size"] == 1 and by_pid["class:cC"]["size"] == 2
    assert len([r for r in rows if str(r["pid"]).startswith("class:")]) == 3   # one row per non-empty class id
    assert eng.state.class_names() == ["lung", "rib"]                          # dup-named ids collapse in pickers

    from chevron.server import _partition_rows                          # sidebar scope filter
    assert len(_partition_rows(eng, kind="class")) == 3                       # classes-only
    assert all(r["pid"].startswith("class:") for r in _partition_rows(eng, kind="class"))
    assert _partition_rows(eng, kind="part") == []                           # FINCH-only (none clustered here)
    assert len(_partition_rows(eng, kind="all")) == 3

    # instance loading reads the SAME memoized membership (O(1) lookup, not a fresh O(N) meta scan per page)
    assert eng.partition_iuids("class:cA") == ["u0", "u1"]
    assert eng.partition_iuids("class:cC") == ["u3", "u4"]
    assert eng.partition_iuids("class:cA") is eng.partition_iuids("class:cA")   # served from the cache
    assert eng.partition_iuids("class:missing") == [] and eng.partition_iuids("999") == []


def test_match_features_returns_labeled_and_unlabeled_groups(tmp_path):
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta, TaxonomyClass
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["roialign"]}})
    eng.state.taxonomy = {"cA": TaxonomyClass(class_id="cA", name="lung")}
    feats = np.array([[1, 0, 0, 0], [0.9, 0.1, 0, 0],                          # u0,u1 -> class cA
                      [0, 0, 1, 0], [0, 0, 0.9, 0.1]], np.float32)             # u2,u3 -> UNLABELED (no cluster)
    eng.collection = {"records": [{"iuid": f"u{i}", "row": i, "score": 0.9} for i in range(4)],
                      "n_images": 4, "feats": {"roialign": feats}}
    eng.state.order = [f"u{i}" for i in range(4)]
    eng.state.meta = {"u0": InstanceMeta("u0", "b", 0, 1000, assigned_class="cA"),
                      "u1": InstanceMeta("u1", "b", 1, 1001, assigned_class="cA"),
                      "u2": InstanceMeta("u2", "b", 2, 1002),
                      "u3": InstanceMeta("u3", "b", 3, 1003)}
    eng.state.coll_version = 1

    res = eng.match_features(np.array([0, 0, 1, 0], np.float32), feature="roialign", k=4)
    # UNLABELED group is populated even with NO cluster (the regression) — both unassigned surface
    assert {r["iuid"] for r in res["matches_pool"]} == {"u2", "u3"}
    # LABELED group present + deduped by class (cA appears once)
    assert len(res["matches_class"]) == 1 and res["matches_class"][0]["cls"] == "lung"
    # an unclustered unlabeled match is navigable (pid = its iuid -> singleton)
    assert res["matches_pool"][0]["pid"] in ("u2", "u3")
    assert eng.partition_iuids("u2") == ["u2"]


def test_release_gate_candidates_stats_and_set(tmp_path):
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta, TaxonomyClass
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    eng.state.taxonomy = {"cA": TaxonomyClass(class_id="cA", name="lung"),
                          "cB": TaxonomyClass(class_id="cB", name="rib")}
    # (iuid, class, background, image_id): img1001 final(2); img1002 has a pending inst; img1003 final(2 + ignored bg);
    # img1004 only 1 assigned -> not a candidate
    metas = [("u1", "cA", False, 1001), ("u2", "cB", False, 1001),
             ("u3", "cA", False, 1002), ("u4", None, False, 1002),
             ("u5", "cA", False, 1003), ("u6", "cB", False, 1003), ("u7", None, True, 1003),
             ("u8", "cA", False, 1004)]
    eng.state.order = [m[0] for m in metas]
    eng.state.meta = {u: InstanceMeta(u, "b", i, img, assigned_class=cid, is_background=bg)
                      for i, (u, cid, bg, img) in enumerate(metas)}
    eng.collection = {"records": [{"iuid": m[0], "row": i, "score": 0.9} for i, m in enumerate(metas)],
                      "n_images": 4, "feats": {"decoder": np.zeros((len(metas), 4), np.float32)}}
    eng.state.coll_version = 1

    assert eng.release_candidates() == [1001, 1003]                  # final = fully categorized AND >1 kept
    assert eng.release_stats() == {"fully_categorized": 2, "accepted": 0, "rejected": 0, "pending": 2}

    eng.set_release([1001], "accepted"); eng.set_release([1003], "rejected")
    assert eng.release_stats() == {"fully_categorized": 2, "accepted": 1, "rejected": 1, "pending": 0}
    va = eng.release_view(filter="accepted")
    assert [it["image_id"] for it in va["items"]] == ["1001"] and va["items"][0]["status"] == "accepted"
    assert va["items"][0]["n_assigned"] == 2
    assert eng.release_view(filter="pending")["total"] == 0

    eng.set_release([1001], "pending")                               # clear back to undecided
    assert eng.release_stats()["accepted"] == 0 and eng.release_stats()["pending"] == 1
    assert eng.state.release_gate == {"1003": "rejected"}            # only the live decision remains, persisted


def test_normed_feats_cache_and_ann_matches_brute(tmp_path, monkeypatch):
    # _ann_index degrades to None without faiss (by design — brute force stays correct, just slower),
    # so the engine swallows the ImportError and conftest never sees it. Skip explicitly.
    pytest.importorskip("faiss")
    from chevron import engine as eng_mod
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    rng = np.random.default_rng(0)
    N, D = 300, 16
    feats = rng.normal(0, 1, (N, D)).astype(np.float32)
    eng.collection = {"records": [{"iuid": f"u{i}", "row": i, "score": 0.9} for i in range(N)],
                      "n_images": N, "feats": {"decoder": feats}}
    eng.state.order = [f"u{i}" for i in range(N)]
    eng.state.meta = {f"u{i}": InstanceMeta(f"u{i}", "b", i, 1000 + i) for i in range(N)}
    eng.state.coll_version = 1

    a = eng._normed_feats("decoder")                       # cached by coll_version
    assert eng._normed_feats("decoder") is a
    eng.state.coll_version = 2
    assert eng._normed_feats("decoder") is not a

    q = feats[42]
    assert eng._ann_index("decoder") is None               # below _ANN_MIN -> exact brute path
    assert eng.match_features(q, feature="decoder", k=5)["matches"][0]["iuid"] == "u42"

    monkeypatch.setattr(eng_mod, "_ANN_MIN", 10)           # force the faiss HNSW path
    assert eng._ann_index("decoder") is not None
    assert eng.match_features(q, feature="decoder", k=5)["matches"][0]["iuid"] == "u42"   # ANN finds the exact match too


# ---- engine.split_instances ------------------------------------------------
def _png(p, h=64, w=64):
    import cv2
    cv2.imwrite(str(p), (np.random.default_rng(1).random((h, w, 3)) * 100 + 40).astype(np.uint8))


def test_split_instances(tmp_path):
    import cv2
    from chevron import ids
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    p = tmp_path / "im0.png"; _png(p)
    m = np.zeros((64, 64), np.uint8)                          # TWO disconnected blobs
    cv2.circle(m, (16, 16), 8, 1, -1); cv2.circle(m, (48, 48), 8, 1, -1)
    mb = m > 0; ys, xs = np.where(mb); u = ids.new_uid()
    rec = {"iuid": u, "row": 0, "inst_id": 0, "image_id": 1000, "H": 64, "W": 64, "score": 0.9,
           "rle": _rle(mb), "file_name": str(p), "abs_path": str(p), "batch_id": "b",
           "cx": float(xs.mean() / 64), "cy": float(ys.mean() / 64), "bw": 0.6, "bh": 0.6,
           "box_area": 0.36, "mask_area_frac": float(mb.mean())}
    eng.collection = {"records": [rec], "n_images": 1,
                      "feats": {"decoder": np.array([[1, 2, 3, 4]], np.float32),
                                "shapecoord": np.zeros((1, 29), np.float32), "_shapecoord_cols": ["c"] * 29}}
    eng.state.order = [u]; eng.state.meta = {u: InstanceMeta(iuid=u, batch_id="b", row=0, image_id=1000)}
    eng.state.coll_version = 1; eng.store.save_collection(eng.collection); eng.save()

    n = eng.split_instances([u])
    assert n == 2                                             # two connected components -> two new instances
    assert len(eng.state.order) == 3                         # parent + 2 children
    assert eng.state.meta[u].is_background is True           # parent sent to background
    children = [iu for iu in eng.state.order if iu != u]
    assert all(not eng.state.meta[c].is_background and eng.state.meta[c].assigned_class is None for c in children)
    # row-alignment invariant holds (every feats matrix has one row per record)
    eng.state.assert_aligned(eng.collection["feats"]["decoder"].shape[0])
    assert eng.collection["feats"]["decoder"].shape[0] == 3
    # each child mask is one of the two blobs (single connected component)
    for c in children:
        cm = eng._mask(c).astype(np.uint8)
        ncc, _ = cv2.connectedComponents(cm)
        assert ncc == 2                                      # background + exactly one component


def test_engine_auto_refine_search_apply_and_logs_demo(tmp_path):
    import cv2
    from chevron import ids
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    H = W = 96
    line = np.zeros((H, W), bool)                            # branchy + fragmented line
    line[19:22, 8:81] = True; line[19:22, 40:48] = False; line[6:21, 29:32] = True
    img = np.zeros((H, W, 3), np.uint8); img[line] = 220
    p = tmp_path / "im0.png"; cv2.imwrite(str(p), img)
    u = ids.new_uid(); ys, xs = np.where(line)
    rec = {"iuid": u, "row": 0, "inst_id": 0, "image_id": 1000, "H": H, "W": W, "score": 0.9,
           "rle": _rle(line), "file_name": str(p), "abs_path": str(p), "batch_id": "b",
           "cx": float(xs.mean() / W), "cy": float(ys.mean() / H), "bw": 0.7, "bh": 0.2,
           "box_area": 0.14, "mask_area_frac": float(line.mean())}
    eng.collection = {"records": [rec], "n_images": 1,
                      "feats": {"decoder": np.zeros((1, 4), np.float32),
                                "shapecoord": np.zeros((1, 29), np.float32), "_shapecoord_cols": ["c"] * 29}}
    eng.state.order = [u]; eng.state.meta = {u: InstanceMeta(iuid=u, batch_id="b", row=0, image_id=1000)}
    eng.state.coll_version = 1; eng.store.save_collection(eng.collection); eng.save()

    res = eng.auto_refine_search(u, kind="auto")
    assert res["kind"] == "line" and any(o["name"] == "line_centerline" for o in res["best"]["chain"])
    eng.auto_refine_apply(u)                                 # searches + applies the best chain
    assert eng.state.meta[u].refined is True
    assert eng.state.meta[u].rule_ops == res["best"]["chain"]   # chosen chain logged as a Stage-2 demo
    assert cv2.connectedComponents(eng._mask(u).astype(np.uint8))[0] - 1 == 1   # gap bridged -> one component


def test_engine_auto_refine_class_consensus(tmp_path):
    import cv2
    from chevron import ids
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    H = W = 96
    def _line(off):                                          # a branchy + fragmented line, shifted per instance
        m = np.zeros((H, W), bool)
        m[19 + off:22 + off, 8:81] = True; m[19 + off:22 + off, 40:48] = False
        m[6 + off:21 + off, 29:32] = True
        return m
    uids, recs, feats_d, feats_s = [], [], [], []
    for i in range(2):
        line = _line(i * 10); img = np.zeros((H, W, 3), np.uint8); img[line] = 220
        p = tmp_path / f"im{i}.png"; cv2.imwrite(str(p), img)
        u = ids.new_uid(); ys, xs = np.where(line); uids.append(u)
        recs.append({"iuid": u, "row": i, "inst_id": 0, "image_id": 1000 + i, "H": H, "W": W, "score": 0.9,
                     "rle": _rle(line), "file_name": str(p), "abs_path": str(p), "batch_id": "b",
                     "cx": float(xs.mean() / W), "cy": float(ys.mean() / H), "bw": 0.7, "bh": 0.2,
                     "box_area": 0.14, "mask_area_frac": float(line.mean())})
        feats_d.append(np.zeros(4, np.float32)); feats_s.append(np.zeros(29, np.float32))
    eng.collection = {"records": recs, "n_images": 2,
                      "feats": {"decoder": np.asarray(feats_d), "shapecoord": np.asarray(feats_s),
                                "_shapecoord_cols": ["c"] * 29}}
    eng.state.order = list(uids)
    eng.state.meta = {u: InstanceMeta(iuid=u, batch_id="b", row=i, image_id=1000 + i) for i, u in enumerate(uids)}
    eng.state.coll_version = 1; eng.store.save_collection(eng.collection); eng.save()
    eng.assign(uids, "line1")                                # both instances -> one class

    out = eng.auto_refine_class_consensus("line1")
    assert out["kind"] == "line" and out["reward"] == "geometric"      # category context (no shape prior configured)
    assert any(o["name"] == "line_centerline" for o in out["chain"])   # modal chain is a centerline
    assert out["applied"] == 2 and out.get("saved_rule")
    cid = eng._resolve_cid("line1")
    assert eng.state.class_rules[cid] == out["chain"]                  # saved as the class rule
    assert all(eng.state.meta[u].refined for u in eng.class_rule_members(cid))
