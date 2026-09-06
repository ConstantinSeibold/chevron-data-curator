"""RAD-DINO dense-correspondence core (match.py) — mechanics on a FAKE extractor (no GPU/model). A support
mask over a colored blob must build a prototype whose query heatmap peaks on the SAME color elsewhere.
Run: pytest tests/test_match.py -q
"""
from __future__ import annotations

import numpy as np


class _FakeExt:
    """grid(rgb) -> [3, g, g] = the image downsampled to a g×g grid of mean-RGB patch features (so a
    patch's 'feature' is just its colour). Mimics ext.grid's [C,gh,gw] torch contract."""
    def __init__(self, g=16):
        self.g = g

    def grid(self, rgb):
        import cv2
        import torch
        small = cv2.resize(rgb.astype(np.float32) / 255.0, (self.g, self.g))   # [g,g,3]
        return torch.from_numpy(small).permute(2, 0, 1).contiguous()           # [3,g,g]


def _img_with_blob(color, cx, cy, r=18, size=128):
    import cv2
    img = np.zeros((size, size, 3), np.uint8)
    cv2.circle(img, (cx, cy), r, color, -1)
    mask = np.zeros((size, size), bool)
    yy, xx = np.ogrid[:size, :size]
    mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = True
    return img, mask


def test_correspondence_localizes_same_color_blob():
    from chevron import match as M
    ext = _FakeExt()
    red = (220, 30, 30)
    sup_img, sup_mask = _img_with_blob(red, 40, 40)            # support: red blob top-left
    proto = M.build_prototype(ext, [(sup_img, sup_mask)])
    assert proto.shape[0] >= 1

    qry_img, qry_mask = _img_with_blob(red, 95, 95)            # query: SAME red, bottom-right
    hm = M.heatmap(ext, qry_img, proto)
    assert hm.shape == qry_img.shape[:2]
    ay, ax = np.unravel_index(int(np.argmax(hm)), hm.shape)
    assert qry_mask[ay, ax]                                    # heatmap argmax lands ON the matching blob
    # the matching blob scores higher than a far background point
    assert hm[95, 95] > hm[5, 5]


def test_heatmap_low_for_absent_color():
    from chevron import match as M
    ext = _FakeExt()
    sup_img, sup_mask = _img_with_blob((220, 30, 30), 40, 40)  # red prototype
    proto = M.build_prototype(ext, [(sup_img, sup_mask)])
    qry_img, qry_mask = _img_with_blob((30, 30, 220), 95, 95)  # query has only a BLUE blob
    hm = M.heatmap(ext, qry_img, proto)
    # red prototype vs a blue query: the blob shouldn't strongly out-score on red-similarity
    assert hm[95, 95] < 0.99


def test_seed_and_peaks():
    from chevron import match as M
    ext = _FakeExt()
    sup_img, sup_mask = _img_with_blob((220, 30, 30), 40, 40)
    proto = M.build_prototype(ext, [(sup_img, sup_mask)])
    qry_img, qry_mask = _img_with_blob((220, 30, 30), 95, 95)
    hm = M.heatmap(ext, qry_img, proto)
    thr = 0.5                                                 # cosine midpoint (blob~1, bg~0 on this synthetic)
    seed = M.seed_mask(hm, thresh=thr)
    assert seed.dtype == bool and seed.any() and not seed.all()
    pk = M.peaks(hm, thresh=thr, max_peaks=5)
    assert pk and qry_mask[pk[0][0], pk[0][1]]                 # top peak is on the blob
