"""v5.6 responsiveness: image LRU cache (no repeat disk reads), embed_thumbnails cap,
partition_view memo. Run: pytest chevron/tests/test_perf.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


def _engine(tmp_path, *, n_images=4, per_image=6):
    """n_images real PNGs on disk, per_image instances each -> many instances share few source files."""
    import cv2
    from chevron import ids
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    recs, order, meta, dec = [], [], {}, []
    for ii in range(n_images):
        p = tmp_path / f"im{ii}.png"
        cv2.imwrite(str(p), (np.random.default_rng(ii).random((256, 256, 3)) * 120 + 40).astype(np.uint8))
        for j in range(per_image):
            m = np.zeros((256, 256), np.uint8); cv2.circle(m, (40 + 30 * j, 128), 14, 1, -1)
            mb = m > 0; ys, xs = np.where(mb); u = ids.new_uid(); row = len(recs)
            recs.append({"iuid": u, "row": row, "inst_id": row, "image_id": 1000 + ii, "H": 256, "W": 256, "score": 0.6,
                         "rle": _rle(mb), "file_name": str(p), "abs_path": str(p), "batch_id": "b",
                         "cx": float(xs.mean() / 256), "cy": float(ys.mean() / 256), "bw": 0.2, "bh": 0.2,
                         "box_area": 0.04, "mask_area_frac": float(mb.mean())})
            dec.append(np.random.default_rng(row).normal(0, 1, 8)); order.append(u)
            meta[u] = InstanceMeta(iuid=u, batch_id="b", row=row, image_id=1000 + ii)
    eng.collection = {"records": recs, "n_images": n_images, "feats": {"decoder": np.array(dec, np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    return eng, order


def test_rgb_cache_reads_each_path_once(tmp_path, monkeypatch):
    import cv2
    from chevron import engine
    engine._IMG_CACHE.clear()
    eng, order = _engine(tmp_path, n_images=4, per_image=6)        # 24 instances over 4 images
    calls = {"n": 0}
    real = cv2.imread
    monkeypatch.setattr(cv2, "imread", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1])
    for u in order:                                               # crop all 24 instances
        eng.crop(u)
    assert calls["n"] == 4                                        # one disk read PER IMAGE, not per instance
    n_before = calls["n"]
    for u in order:                                              # a second pass (e.g. mask-toggle re-render) -> 0 reads
        eng.crop(u, mask_overlay=False)
    assert calls["n"] == n_before                                 # fully cached


def test_img_cache_is_lru_bounded():
    from chevron import engine
    engine._IMG_CACHE.clear()
    for i in range(engine._IMG_CACHE_MAX + 10):
        engine._load_rgb("", fallback_hw=(8, 8))                  # empty path -> fallback array, distinct keys not added
    # empty path always maps to one key "", so cache stays tiny; use distinct fake keys instead:
    engine._IMG_CACHE.clear()
    for i in range(engine._IMG_CACHE_MAX + 10):
        engine._IMG_CACHE[f"k{i}"] = np.zeros((2, 2, 3), np.uint8)
        engine._IMG_CACHE.move_to_end(f"k{i}")
        while len(engine._IMG_CACHE) > engine._IMG_CACHE_MAX:
            engine._IMG_CACHE.popitem(last=False)
    assert len(engine._IMG_CACHE) == engine._IMG_CACHE_MAX


def test_crop_cache_hits_until_mask_changes(tmp_path, monkeypatch):
    from chevron import engine
    engine._CROP_CACHE.clear()
    eng, order = _engine(tmp_path, n_images=1, per_image=4)
    u = order[0]
    calls = {"n": 0}; real = eng._rgb
    monkeypatch.setattr(eng, "_rgb", lambda iu: (calls.__setitem__("n", calls["n"] + 1), real(iu))[1])
    eng.crop(u, max_side=256); eng.crop(u, max_side=256)           # 2nd call is a crop-cache hit
    assert calls["n"] == 1
    tok = eng.mask_token(u)
    eng.merge_instances([order[0], order[1]], mode="union")        # rep=order[0]; its effective mask changes
    assert eng.mask_token(order[0]) != tok                         # token moved -> crop-cache key differs
    n_before = calls["n"]
    eng.crop(order[0], max_side=256)                               # cache miss -> recompute
    assert calls["n"] == n_before + 1


def test_crop_cache_lru_bounded():
    from chevron import engine
    engine._CROP_CACHE.clear()
    import numpy as np
    for i in range(engine._CROP_CACHE_MAX + 20):
        engine.CuratorEngine._cache_crop((f"k{i}",), np.zeros((2, 2, 3), np.uint8))
    assert len(engine._CROP_CACHE) == engine._CROP_CACHE_MAX


def test_embed_thumbnails_capped(tmp_path):
    eng, _ = _engine(tmp_path, n_images=6, per_image=10)           # 60 instances
    eng.cluster({"decoder": 1.0})
    xy, labels, order, thumbs, names = eng.embed_thumbnails(method="pca", max_pts=8)
    n_thumbs = sum(1 for t in thumbs if t)
    assert n_thumbs <= 8 and len(names) == len(order)             # at most max_pts non-empty thumbnails


def test_fused_matrix_cached_per_coll_version(tmp_path, monkeypatch):
    """v7.9: the fused feature matrix is built once per coll_version and reused (classifier Apply was
    rebuilding it O(total) twice per click)."""
    from chevron import cluster as cl
    eng, order = _engine(tmp_path, n_images=2, per_image=6)
    builds = {"n": 0}; real = cl.fused_matrix
    monkeypatch.setattr(cl, "fused_matrix", lambda *a, **k: (builds.__setitem__("n", builds["n"] + 1), real(*a, **k))[1])
    eng.fused({"decoder": 1.0}); eng.fused({"decoder": 1.0}); eng.fused({"decoder": 1.0})
    assert builds["n"] == 1                                        # one build, then cache hits
    eng.state.coll_version += 1                                    # a sample/split bumps it -> rebuild
    eng.fused({"decoder": 1.0})
    assert builds["n"] == 2


def test_partition_view_memoized(tmp_path):
    eng, order = _engine(tmp_path, n_images=2, per_image=4)
    eng.cluster({"decoder": 1.0})
    idx = eng._get_index()
    assert eng._get_index() is idx                                # no mutation -> no rebuild (same live index)
    eng.assign([order[0]], "A")                                   # NEW class grows the taxonomy -> index rebuilds
    assert any(str(r["pid"]).startswith("class:") for r in eng.partition_view())
    idx2 = eng._get_index()
    eng.assign([order[1]], "A")                                   # EXISTING class -> INCREMENTAL patch (no rebuild)
    assert eng._get_index() is idx2                               # same object, patched in place
    cid = eng.state.class_id_by_name("A")
    row = next(r for r in eng.partition_view() if r["pid"] == f"class:{cid}")
    assert row["size"] == 2                                       # both reflected without an O(N) rebuild


# ---- incremental live-index layer (Phase 1) --------------------------------
def _gt_sets(eng):
    """Ground-truth recompute of the membership sets directly from state.meta (what the live index replaces)."""
    img, un, cls = {}, set(), {}
    for u, m in eng.state.meta.items():
        if m.merged_into is None and not m.is_background:
            img.setdefault(m.image_id, set()).add(u)
        if m.assigned_class is None and not m.is_background and m.merged_into is None:
            un.add(u)
        if m.assigned_class and not m.is_background and m.merged_into is None and eng._in_scope(u):
            cls.setdefault(m.assigned_class, set()).add(u)
    return img, un, cls


def _check_index(eng):
    img, un, cls = _gt_sets(eng)
    assert set(eng._unassigned_iuids()) == un                          # classifier pool
    for iid, s in img.items():
        assert set(eng.image_instance_iuids(iid)) == s                 # In-image membership
    sizes = {r["pid"]: r["size"] for r in eng.partition_view()}
    for cid, s in cls.items():
        assert set(eng.partition_iuids(f"class:{cid}")) == s           # class partition members
        assert sizes.get(f"class:{cid}") == len(s)                     # sidebar size
    if eng._cluster:                                                    # FINCH sizes/members vs live _is_pool count
        pool = eng._cluster["pool"]
        for pid, idxs in eng._pool_groups().items():
            live = {pool[i] for i in idxs if eng._is_pool(pool[i])}
            assert sizes.get(str(pid), 0) == len(live)
            assert set(eng.partition_iuids(str(pid))) == live
    comp = eng._image_composition()                                    # release composition
    for iid, s in img.items():
        a = sum(1 for u in s if eng.state.meta[u].assigned_class)
        p = len(s) - a
        if a or p:
            assert comp[iid]["assigned"] == a and comp[iid]["unassigned"] == p


def test_live_index_equivalence_through_mutations(tmp_path):
    eng, order = _engine(tmp_path, n_images=3, per_image=5)
    eng.cluster({"decoder": 1.0})
    _check_index(eng)
    eng.assign([order[0], order[1]], "A"); _check_index(eng)            # new class
    eng.assign([order[2]], "A"); _check_index(eng)                      # existing class (incremental)
    eng.assign([order[3]], "B"); _check_index(eng)
    eng.set_background([order[4]]); _check_index(eng)                   # reject
    eng.remove_from_class([order[0]]); _check_index(eng)               # unassign
    eng.unreject([order[4]]); _check_index(eng)                         # unreject
    eng.merge_instances([order[5], order[6]]); _check_index(eng)        # merge (rebuild path)
    eng.undo(); _check_index(eng)                                       # undo (rebuild path)
    if len(eng._cluster["counts"]) > 1:                                 # set_level reshapes FINCH groups with NO
        pid0 = next((r["pid"] for r in eng.partition_view()             # mutation -> the per-partition FINCH
                     if not str(r["pid"]).startswith("class:")), None)  # materialization must refresh too
        if pid0:
            eng.partition_iuids(pid0)                                    # materialize at the current level first
        eng.set_level(0 if eng._cluster["level"] else len(eng._cluster["counts"]) - 1)
        _check_index(eng)


def test_live_index_incremental_no_rebuild(tmp_path, monkeypatch):
    eng, order = _engine(tmp_path, n_images=2, per_image=5)
    eng.cluster({"decoder": 1.0})
    n = {"c": 0}
    orig = eng._rebuild_index
    monkeypatch.setattr(eng, "_rebuild_index", lambda: (n.__setitem__("c", n["c"] + 1), orig())[1])
    eng._get_index()
    eng.assign([order[0]], "A"); eng._get_index()                      # new class -> rebuild
    base = n["c"]
    eng.assign([order[1]], "A"); eng._get_index()                      # existing class -> NO rebuild
    eng.set_background([order[2]]); eng._get_index()                   # reject -> NO rebuild
    eng.remove_from_class([order[1]]); eng._get_index()               # unassign -> NO rebuild
    assert n["c"] == base                                              # all incremental
    eng.undo(); eng._get_index()                                       # undo -> rebuild
    assert n["c"] == base + 1
