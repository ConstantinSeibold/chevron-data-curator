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


class _FakeSamPred:
    """Stand-in for SamPredictor so the mask-selection logic is testable without a real checkpoint.
    Records the last predict() kwargs so a MedSAM test can assert box-only / single-mask."""
    def __init__(self, masks, scores): self._m, self._s = masks, scores; self.last = None
    def set_image(self, rgb): self.last_img = rgb
    def predict(self, *, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True):
        self.last = dict(point_coords=point_coords, box=box, mask_input=mask_input, multimask_output=multimask_output)
        return self._m, self._s, None


def _sam_setup(monkeypatch, masks, scores):
    from tools.curator import refine as r
    monkeypatch.setattr(r, "find_sam_checkpoint", lambda ckpt=None, family=None: ("/fake.pth", "vit_b"))
    monkeypatch.setattr(r, "_sam_predictor", lambda c, t, fam="sam": _FakeSamPred(masks, scores))


def test_sam_refine_takes_best_proposal_and_can_shrink(monkeypatch):
    """SAM proposes several masks; the highest-confidence one is taken and (default) REPLACES the input,
    so the boundary can move inward (shrink) — not just echo/grow the original."""
    from tools.curator import refine as r
    H = W = 80
    inp = np.zeros((H, W), bool); inp[20:60, 20:60] = True        # 40x40 (1600 px)
    big = np.zeros((H, W), bool); big[10:70, 10:70] = True
    small = np.zeros((H, W), bool); small[30:50, 30:50] = True    # 20x20 (400 px), ⊂ inp
    masks = np.stack([big, np.zeros((H, W), bool), small])
    scores = np.array([0.20, 0.10, 0.95])                          # 'small' is SAM's best
    g = (np.random.default_rng(0).random((H, W)) * 255).astype("uint8")
    _sam_setup(monkeypatch, masks, scores)
    out = r.sam_refine(g, inp, n_pos=3, n_neg=4)                   # replace (default)
    assert out.sum() == small.sum() and out.sum() < inp.sum()      # picked best AND shrank
    out_u = r.sam_refine(g, inp, n_pos=3, n_neg=4, union=True)     # keep∪: never shrinks
    assert out_u.sum() == inp.sum() and (out_u | inp == out_u).all()


def test_sam_refine_empty_proposal_falls_back_to_input(monkeypatch):
    from tools.curator import refine as r
    H = W = 60
    inp = np.zeros((H, W), bool); inp[20:40, 20:40] = True
    masks = np.stack([np.zeros((H, W), bool)] * 3)                 # best proposal is empty
    _sam_setup(monkeypatch, masks, np.array([0.9, 0.5, 0.3]))
    g = np.zeros((H, W), "uint8")
    out = r.sam_refine(g, inp, n_pos=3, n_neg=4)
    assert (out == inp).all()                                      # never returns an empty mask


def test_medsam_refine_is_box_only_single_mask(monkeypatch):
    """model='medsam' runs the MedSAM recipe: box prompt only (no points, no mask prior), multimask_output
    False — vs SAM's points+box+mask-prior multimask. Verified via the recorded predict() kwargs."""
    from tools.curator import refine as r
    H = W = 80
    inp = np.zeros((H, W), bool); inp[20:60, 20:60] = True
    pred = np.zeros((H, W), bool); pred[18:62, 18:62] = True
    fake = _FakeSamPred(pred[None], np.array([0.9]))             # MedSAM returns a single mask
    monkeypatch.setattr(r, "find_sam_checkpoint", lambda ckpt=None, family=None: ("/x/medsam_vit_b.pth", "vit_b"))
    monkeypatch.setattr(r, "_sam_predictor", lambda c, t, fam="sam": fake)
    g = (np.random.default_rng(0).random((H, W)) * 255).astype("uint8")
    out = r.sam_refine(g, inp, model="medsam")
    assert out.sum() == pred.sum()
    assert fake.last["point_coords"] is None and fake.last["mask_input"] is None     # box-only
    assert fake.last["multimask_output"] is False and fake.last["box"] is not None    # single mask, box prompt


def test_detect_sam_type():
    from tools.curator.refine import detect_sam_type
    assert detect_sam_type("/x/sam_vit_h_4b8939.pth") == "vit_h"
    assert detect_sam_type("/x/sam_vit_l_0b3195.pth") == "vit_l"
    assert detect_sam_type("/x/medsam_vit_b.pth") == "vit_b"
    assert detect_sam_type("/x/medsam.pth") == "vit_b"            # MedSAM is a vit_b


def test_detect_sam_family():
    from tools.curator.refine import detect_sam_family
    assert detect_sam_family("/x/medsam_vit_b.pth") == "medsam"
    assert detect_sam_family("/x/MedSAM.pth") == "medsam"
    assert detect_sam_family("/x/sam_vit_b_01ec64.pth") == "sam"


def test_find_sam_checkpoint_family(monkeypatch, tmp_path):
    """family= prefers a matching cache file; CURATOR_MEDSAM_CKPT wins for family='medsam'."""
    from tools.curator import refine as r
    d = tmp_path / "samcache"; d.mkdir(parents=True)
    for e in ("CURATOR_SAM_CKPT", "CURATOR_SAM_TYPE", "CURATOR_MEDSAM_CKPT"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("CURATOR_SAM_DIR", str(d))
    (d / "sam_vit_b_01ec64.pth").write_bytes(b"s")
    (d / "medsam_vit_b.pth").write_bytes(b"m")
    assert r.find_sam_checkpoint(family="medsam")[0].endswith("medsam_vit_b.pth")
    assert r.find_sam_checkpoint(family="sam")[0].endswith("sam_vit_b_01ec64.pth")
    (d / "custom_medsam.pth").write_bytes(b"x")
    monkeypatch.setenv("CURATOR_MEDSAM_CKPT", str(d / "custom_medsam.pth"))
    assert r.find_sam_checkpoint(family="medsam")[0].endswith("custom_medsam.pth")    # env wins


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


def test_enhance_contrast_expands_range():
    """enhance_contrast boosts contrast of a narrow-band image (returns uint8)."""
    from tools.curator.refine import enhance_contrast
    g = np.random.default_rng(0).integers(110, 140, (64, 64)).astype(np.uint8)   # narrow band
    s = enhance_contrast(g, method="stretch")
    assert s.dtype == np.uint8 and (int(s.max()) - int(s.min())) > (int(g.max()) - int(g.min()))
    c = enhance_contrast(g, method="clahe", clip=4.0)
    assert c.dtype == np.uint8 and c.shape == g.shape and c.std() > g.std()       # CLAHE raises contrast


def test_contrast_op_changes_what_later_ops_see():
    """A `contrast` op enhances the WORKING image (returned via return_image), feeds it to later ops, and
    changes a downstream intensity op's result; `contrast` alone leaves the mask untouched."""
    from tools.curator.refine import apply_ops, enhance_contrast, manual_threshold
    g = np.full((80, 80), 100, np.uint8)
    g[34:46, 34:46] = 150                                      # bright square (the seed)
    g[34:46, 46:52] = 128                                      # faint arm just right of it (below 130)
    m = np.zeros((80, 80), bool); m[34:46, 34:46] = True
    out, work = apply_ops(g, m, [{"name": "contrast", "kw": {"method": "stretch"}}], return_image=True)
    assert work.std() > g.std() and np.array_equal(out, m)     # image enhanced; mask unchanged by contrast alone
    chain = apply_ops(g, m, [{"name": "contrast", "kw": {"method": "stretch"}}, {"name": "threshold", "kw": {"val": 130}}])
    direct = manual_threshold(enhance_contrast(g, method="stretch"), m, 130)
    orig = manual_threshold(g, m, 130)
    assert np.array_equal(chain, direct)                       # the threshold ran on the ENHANCED image (wiring)
    assert not np.array_equal(chain, orig)                     # ...and that changed the result vs the original


def test_apply_ops_vessel_extend_tunable_max_width():
    """vessel_extend accepts the new max_width kw through apply_ops (tunable per image)."""
    import cv2
    from tools.curator.refine import apply_ops
    g = _line_img()
    frag = np.zeros_like(g, np.uint8); cv2.line(frag, (12, 12), (55, 55), 1, 3)
    out = apply_ops(g, frag > 0, [{"name": "vessel_extend", "kw": {"low": 0.5, "high": 0.8, "max_gap": 10, "max_width": 3}}])
    assert out.dtype == bool and out.shape == g.shape and out.any()
