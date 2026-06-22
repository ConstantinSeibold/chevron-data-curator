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


def test_sam_refine_errors_without_checkpoint(monkeypatch, tmp_path):
    """sam_refine (and the 'sam' op) raise a clear, catchable error when no checkpoint is available.
    Pin the cache dir to an empty tmp so a checkpoint cached on the dev box can't make this flaky."""
    from tools.curator.refine import apply_ops, sam_refine
    monkeypatch.delenv("CURATOR_SAM_CKPT", raising=False)
    monkeypatch.delenv("CURATOR_SAM_TYPE", raising=False)
    monkeypatch.setenv("CURATOR_SAM_DIR", str(tmp_path / "samcache"))
    g = _line_img(); m = g > 0.5
    with pytest.raises(RuntimeError, match="SAM"):          # message differs by install state; both say "SAM"
        sam_refine(g, m)
    with pytest.raises(RuntimeError, match="SAM"):
        apply_ops(g, m, [{"name": "sam"}])


def test_detect_sam_type():
    from tools.curator.refine import detect_sam_type
    assert detect_sam_type("/x/sam_vit_h_4b8939.pth") == "vit_h"
    assert detect_sam_type("/x/sam_vit_l_0b3195.pth") == "vit_l"
    assert detect_sam_type("/x/medsam_vit_b.pth") == "vit_b"
    assert detect_sam_type("/x/medsam.pth") == "vit_b"            # MedSAM is a vit_b


def test_find_sam_checkpoint_discovers_cached(monkeypatch, tmp_path):
    """find_sam_checkpoint: empty dir -> (None, None); a dropped .pth is discovered with its arch."""
    from tools.curator import refine as r
    d = tmp_path / "samcache"
    monkeypatch.delenv("CURATOR_SAM_CKPT", raising=False)
    monkeypatch.delenv("CURATOR_SAM_TYPE", raising=False)
    monkeypatch.setenv("CURATOR_SAM_DIR", str(d))
    assert r.find_sam_checkpoint() == (None, None)
    d.mkdir(parents=True, exist_ok=True)
    (d / "sam_vit_l_0b3195.pth").write_bytes(b"stub")
    ckpt, mtype = r.find_sam_checkpoint()
    assert ckpt and ckpt.endswith("sam_vit_l_0b3195.pth") and mtype == "vit_l"


def test_sam_prompt_points_sampling():
    """Positives lie ON the mask (skeleton ⊆ mask), negatives lie OUTSIDE it, box encloses the mask+pad."""
    import cv2
    from tools.curator.refine import sam_prompt_points
    g = _line_img()
    m = np.zeros_like(g, np.uint8); cv2.line(m, (12, 12), (114, 114), 1, 5); m = m > 0
    pos, neg, box = sam_prompt_points(m, n_pos=8, n_neg=10, margin=12)
    assert 1 <= len(pos) <= 8 and 1 <= len(neg) <= 10
    assert all(m[y, x] for (x, y) in pos)               # positives sit on the structure (skeleton ⊆ mask)
    assert not any(m[y, x] for (x, y) in neg)           # negatives are outside the mask
    ys, xs = np.where(m)
    assert box[0] <= xs.min() and box[1] <= ys.min() and box[2] >= xs.max() and box[3] >= ys.max()


def test_sam_prompt_points_leave_the_boundary_free():
    """The refinement-critical property: NO prompt sits on the uncertain rim. Positives stay in the
    confident deep interior; negatives stay in clear background beyond a `margin`-px gap; the band around
    the boundary carries no points (so SAM can redraw it instead of reproducing the input)."""
    import cv2
    from scipy import ndimage as ndi
    from tools.curator.refine import sam_prompt_points
    m = np.zeros((200, 200), np.uint8); cv2.circle(m, (100, 100), 34, 1, -1); m = m > 0
    margin = 18
    pos, neg, _ = sam_prompt_points(m, n_pos=6, n_neg=10, margin=margin)
    cyx = ndi.center_of_mass(m)                         # (cy, cx)
    assert abs(pos[0][0] - cyx[1]) <= 4 and abs(pos[0][1] - cyx[0]) <= 4   # pos[0] ≈ centroid
    dt_in = ndi.distance_transform_edt(m)               # each fg pixel's distance to the boundary
    assert dt_in[pos[:, 1], pos[:, 0]].min() >= 5        # positives are deep inside, not on the rim
    dt_out = ndi.distance_transform_edt(~m)             # each bg pixel's distance to the mask
    assert neg.shape[0] >= 1 and dt_out[neg[:, 1], neg[:, 0]].min() >= margin - 1   # negatives beyond the gap


def test_sam_prompt_points_empty():
    from tools.curator.refine import sam_prompt_points
    pos, neg, box = sam_prompt_points(np.zeros((40, 40), bool))
    assert len(pos) == 0 and len(neg) == 0 and box is None


def test_apply_ops_vessel_extend_tunable_max_width():
    """vessel_extend accepts the new max_width kw through apply_ops (tunable per image)."""
    import cv2
    from tools.curator.refine import apply_ops
    g = _line_img()
    frag = np.zeros_like(g, np.uint8); cv2.line(frag, (12, 12), (55, 55), 1, 3)
    out = apply_ops(g, frag > 0, [{"name": "vessel_extend", "kw": {"low": 0.5, "high": 0.8, "max_gap": 10, "max_width": 3}}])
    assert out.dtype == bool and out.shape == g.shape and out.any()
