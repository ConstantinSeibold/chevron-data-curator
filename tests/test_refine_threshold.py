"""Unified threshold op: method (otsu/manual/ght) × region (in_mask/in_bb/any) × direction (auto/above/below).
Run: pytest tools/curator/tests/test_refine_threshold.py -q
"""
from __future__ import annotations

import numpy as np


def _scene():
    """bg ~30; bright square [20:44] (the structure); a SECOND bright square [50:60] OUTSIDE the mask bbox;
    mask = a smaller patch [24:40] inside the first square. Mild noise spreads the two intensity modes so the
    auto-threshold split lands strictly between them (a delta-spike histogram makes the split degenerate)."""
    import cv2
    gray = np.full((72, 72), 30.0, np.float32)
    cv2.rectangle(gray, (20, 20), (43, 43), 200, -1)
    cv2.rectangle(gray, (50, 50), (59, 59), 200, -1)        # far bright blob, outside the mask's padded bbox
    gray = np.clip(gray + np.random.default_rng(0).normal(0, 6, gray.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((72, 72), bool); mask[24:40, 24:40] = True
    return gray, mask


def test_ght_value_separates_bimodal():
    from tools.curator.refine import _ght_value
    gray, mask = _scene()
    bb = gray[18:46, 18:46]                                  # local window: dark bg + bright structure
    thr = _ght_value(bb)
    assert 30 < thr < 200                                    # split lands between the two modes
    assert np.isfinite(thr)


def test_methods_all_return_valid_masks():
    from tools.curator import refine
    gray, mask = _scene()
    for method in ("otsu", "manual", "ght"):
        out = refine.threshold_op(gray, mask, method=method, val=128, region="in_bb", direction="above")
        assert out.dtype == bool and out.shape == gray.shape and out.sum() > 0


def test_region_bounds_result():
    from tools.curator import refine
    gray, mask = _scene()
    inm = refine.threshold_op(gray, mask, method="manual", val=128, region="in_mask", direction="above")
    inbb = refine.threshold_op(gray, mask, method="manual", val=128, region="in_bb", direction="above")
    anyr = refine.threshold_op(gray, mask, method="manual", val=128, region="any", direction="above")
    assert not (inm & ~mask).any()                          # in_mask: result ⊆ current mask (never grows)
    assert inbb.sum() > inm.sum()                           # in_bb can grow to fill the structure in the bbox
    assert (inbb & ~mask).any()                             # ...beyond the original mask
    assert anyr.sum() > inbb.sum()                          # any reaches the far bright blob outside the bbox
    assert anyr[50:60, 50:60].any() and not inbb[50:60, 50:60].any()


def test_direction_above_vs_below_complementary():
    from tools.curator import refine
    gray, mask = _scene()
    above = refine.threshold_op(gray, mask, method="manual", val=128, region="any", direction="above")
    below = refine.threshold_op(gray, mask, method="manual", val=128, region="any", direction="below")
    assert not (above & below).any()                        # disjoint sides of the same threshold
    assert above[20:44, 20:44].any()                        # bright structure on the 'above' side
    assert below[0:10, 0:10].any()                          # dark background on the 'below' side


def test_direction_auto_matches_interior():
    from tools.curator import refine
    gray, mask = _scene()                                   # interior is bright (200) -> auto keeps the bright side
    auto = refine.threshold_op(gray, mask, method="otsu", region="in_mask", direction="auto")
    above = refine.threshold_op(gray, mask, method="otsu", region="in_mask", direction="above")
    assert np.array_equal(auto, above)


def test_apply_ops_new_threshold_path_and_legacy_unchanged():
    from tools.curator import refine
    gray, mask = _scene()
    # new path (method/region/direction present) routes through threshold_op
    new = refine.apply_ops(gray, mask, [{"name": "threshold",
                                         "kw": {"method": "otsu", "region": "in_bb", "direction": "above"}}])
    assert new.dtype == bool and (new & ~mask).any()        # grew within the bbox
    # legacy path (only val/within_mask) stays bit-exact with manual_threshold
    legacy = refine.apply_ops(gray, mask, [{"name": "threshold", "kw": {"val": 128, "within_mask": True}}])
    assert np.array_equal(legacy, refine.manual_threshold(gray, mask, 128, within_mask=True))
    # bare otsu op unchanged
    bare = refine.apply_ops(gray, mask, [{"name": "otsu"}])
    assert np.array_equal(bare, refine.otsu_threshold(gray, mask, within_mask=False))
