"""New refine ops (within-mask threshold, grabcut, magic_wand, snap_edges) + engine.split_instances.
Run: pytest tools/curator/tests/test_refine_ops.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


# ---- refine.py ops ---------------------------------------------------------
def test_threshold_within_mask_vs_grow():
    import cv2
    from tools.curator import refine
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
    from tools.curator import refine
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
    from tools.curator import refine
    gray = np.full((64, 64), 30, np.uint8)
    cv2.rectangle(gray, (20, 20), (44, 44), 200, -1)
    mask = np.zeros((64, 64), bool); mask[26:38, 26:38] = True
    out = refine.apply_ops(gray, mask, [{"name": "dilate", "kw": {"k": 4, "max_contrast": 0.9}},
                                        {"name": "threshold", "kw": {"val": 128, "within_mask": True}}])
    assert out.dtype == bool and out.shape == (64, 64) and out.sum() > 0


# ---- engine.split_instances ------------------------------------------------
def _png(p, h=64, w=64):
    import cv2
    cv2.imwrite(str(p), (np.random.default_rng(1).random((h, w, 3)) * 100 + 40).astype(np.uint8))


def test_split_instances(tmp_path):
    import cv2
    from tools.curator import ids
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
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
