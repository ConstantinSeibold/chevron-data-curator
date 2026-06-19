"""Curvilinear refinement ops: vessel_extend (line-following completion) + sam_refine (promptable).
Run: pytest tools/curator/tests/test_refine_lines.py -q
"""
from __future__ import annotations

import numpy as np
import pytest


def _line_img(hw=128):
    """Dark background with a bright ~3px diagonal ridge (a synthetic catheter)."""
    import cv2
    g = np.full((hw, hw), 0.2, np.float32)
    cv2.line(g, (12, 12), (hw - 14, hw - 14), 1.0, 3)
    return g


def test_vessel_extend_follows_the_line(tmp_path):
    import cv2
    from tools.curator.refine import vessel_extend
    g = _line_img()
    frag = np.zeros_like(g, np.uint8)
    cv2.line(frag, (12, 12), (55, 55), 1, 3)                # mask covers only the FIRST third of the line
    m = frag > 0
    out = vessel_extend(g, m, max_gap=0)                    # hysteresis grow only (no bridging)
    assert out.sum() > 1.4 * m.sum()                        # grew substantially along the ridge
    assert out[80:110, 80:110].any()                        # reached the far end the fragment didn't cover
    assert not out[:, :5].any()                             # didn't leak into the dark margin


def test_vessel_extend_bridges_a_gap(tmp_path):
    import cv2
    from scipy import ndimage as ndi
    from tools.curator.refine import vessel_extend
    g = _line_img()
    two = np.zeros_like(g, np.uint8)
    cv2.line(two, (12, 12), (45, 45), 1, 3)                 # fragment A
    cv2.line(two, (80, 80), (114, 114), 1, 3)               # fragment B (gap in the middle, along the ridge)
    m = two > 0
    assert ndi.label(m)[1] == 2
    out = vessel_extend(g, m, max_gap=80)
    assert ndi.label(out)[1] == 1                           # the two fragments are now one connected line


def test_vessel_extend_empty_mask_is_noop():
    from tools.curator.refine import vessel_extend
    g = _line_img()
    assert not vessel_extend(g, np.zeros_like(g, bool)).any()


def test_apply_ops_dispatches_vessel_extend():
    import cv2
    from tools.curator.refine import apply_ops
    g = _line_img()
    frag = np.zeros_like(g, np.uint8); cv2.line(frag, (12, 12), (55, 55), 1, 3)
    out = apply_ops(g, frag > 0, [{"name": "vessel_extend", "kw": {"max_gap": 0}}])
    assert out.sum() > (frag > 0).sum()


def test_sam_refine_errors_without_checkpoint(monkeypatch):
    """sam_refine (and the 'sam' op) raise a clear error when no checkpoint is configured."""
    from tools.curator.refine import apply_ops, sam_refine
    monkeypatch.delenv("CURATOR_SAM_CKPT", raising=False)
    g = _line_img(); m = g > 0.5
    with pytest.raises(RuntimeError, match="SAM checkpoint"):
        sam_refine(g, m)
    with pytest.raises(RuntimeError, match="SAM checkpoint"):
        apply_ops(g, m, [{"name": "sam"}])
