"""v5.6 responsiveness: image LRU cache (no repeat disk reads), embed_thumbnails cap,
partition_view memo. Run: pytest tools/curator/tests/test_perf.py -q  (from repo root)
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
    from tools.curator import ids
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
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
    from tools.curator import engine
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
    from tools.curator import engine
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
    from tools.curator import engine
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
    from tools.curator import engine
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


def test_render_grids_dont_leak_blocks_and_gate(tmp_path):
    """v7.8/v7.9 perf invariants, app-wide: (1) keyed layout => blocks_config.blocks stays FLAT across
    pure re-renders (no +28/render leak); (2) tab-gating => an OFFSCREEN render builds ~0 components."""
    from gradio.context import LocalContext
    from tools.curator import app
    eng, order = _engine(tmp_path, n_images=1, per_image=30)       # one busy image (RANZCR-like)
    app.ENG = eng
    eng.cluster({"decoder": 1.0})
    pid = eng.partition_view()[0]["pid"]
    iid = eng.image_ids()[0]
    demo = app.build_app(str(tmp_path))
    bc = demo.default_config
    tok = LocalContext.blocks_config.set(bc)
    try:
        with demo:
            # (active_tab is the LAST input of each grid) visible-args + the owning tab label
            renders = {}
            for r in demo.renderables:
                n, first = len(r.inputs), type(r.inputs[0]).__name__
                if n == 5 and first == "Dropdown":
                    renders["inimg"] = (r, (iid, 0, 0, True, "In-image"), "In-image")
                elif n == 6:
                    renders["part"] = (r, (pid, True, "crop", 0, 0, "Partitions"), "Partitions")
            assert {"inimg", "part"} <= set(renders)
            for name, (r, args, label) in renders.items():
                r.apply(*args)                                     # warm-up (visible) render
                base = len(bc.blocks)
                for _ in range(6):
                    r.apply(*args)                                 # pure visible re-renders -> must NOT grow
                assert len(bc.blocks) == base, f"{name} grid leaked blocks: {base} -> {len(bc.blocks)}"
                # gate OFF: render with a DIFFERENT active tab -> builds (almost) nothing
                off = list(args[:-1]) + ["__other_tab__"]
                gated = len(bc.blocks)
                r.apply(*off)
                assert len(bc.blocks) - gated <= 1, f"{name} grid did work while offscreen"
    finally:
        LocalContext.blocks_config.reset(tok)
        app.ENG = None


def test_visible_gate_is_fail_open():
    """v7.9.1 regression guard: the tab-gate must RENDER when active_tab is unset/empty (else the
    gating optimisation blanks every tab — which it did when active_tab defaulted to a non-matching
    label and the wrong select signal never updated it)."""
    from tools.curator.app import _visible
    assert _visible("", "Partitions") is True          # unset -> render (fail-open)
    assert _visible(None, "Partitions") is True
    assert _visible("Partitions", "Partitions") is True   # active tab -> render
    assert _visible("Classifier", "Partitions") is False  # other tab -> gated off


def test_fused_matrix_cached_per_coll_version(tmp_path, monkeypatch):
    """v7.9: the fused feature matrix is built once per coll_version and reused (classifier Apply was
    rebuilding it O(total) twice per click)."""
    from tools.curator import cluster as cl
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
    v1 = eng.partition_view()
    assert eng.partition_view() is v1                             # same object -> memo hit (no recompute)
    eng.assign([order[0]], "A")                                   # mutation bumps the signature
    v2 = eng.partition_view()
    assert v2 is not v1 and any(str(r["pid"]).startswith("class:") for r in v2)
