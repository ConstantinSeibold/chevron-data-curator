"""Per-instance mask refinement ops on the image crop. Pure functions (cv2/skimage);
reversibility (keep the immutable base mask) is the engine's job — these just transform
mask + gray -> mask. Ops compose as an ordered stack via `apply_ops`.

Reuses morphology primitives from src/qseg/ssl/refine_anatomy.py; adds Otsu,
contrast-gated dilate/erode, and edge-snap.
"""
from __future__ import annotations

import numpy as np


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    import cv2
    if img_rgb.ndim == 2:
        return img_rgb
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def _bbox(mask: np.ndarray, pad: int = 6):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(W, xs.max() + pad + 1), min(H, ys.max() + pad + 1))


def _threshold_region(mask, x1, y1, x2, y2, *, within_mask: bool, grow: int):
    """The pixels a threshold op is allowed to KEEP. within_mask=True confines it to the CURRENT
    mask (so dilate→threshold carves within the grown region); else a dilated neighborhood."""
    import cv2
    sub = (mask[y1:y2, x1:x2]).astype(np.uint8)
    if within_mask:
        return sub > 0
    return cv2.dilate(sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))) > 0


def otsu_threshold(gray: np.ndarray, mask: np.ndarray, *, pad: int = 6, grow: int = 9,
                   within_mask: bool = False) -> np.ndarray:
    """Otsu within the mask bbox; keep the side whose mean matches the mask interior,
    constrained to the current mask (within_mask) or a dilated neighborhood of it."""
    import cv2
    bb = _bbox(mask, pad)
    if bb is None:
        return mask
    x1, y1, x2, y2 = bb
    sub = gray[y1:y2, x1:x2]
    if sub.size == 0:
        return mask
    _, th = cv2.threshold(sub.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    interior = float(gray[mask].mean()) if mask.any() else 127.0
    hi_mean = sub[th > 0].mean() if (th > 0).any() else 0
    lo_mean = sub[th == 0].mean() if (th == 0).any() else 255
    side = (th > 0) if abs(hi_mean - interior) <= abs(lo_mean - interior) else (th == 0)
    region = _threshold_region(mask, x1, y1, x2, y2, within_mask=within_mask, grow=grow)
    out = mask.copy()
    out[y1:y2, x1:x2] = side & region
    return out


def manual_threshold(gray: np.ndarray, mask: np.ndarray, val: int, *, pad: int = 6, grow: int = 9,
                     within_mask: bool = False) -> np.ndarray:
    import cv2
    bb = _bbox(mask, pad)
    if bb is None:
        return mask
    x1, y1, x2, y2 = bb
    sub = gray[y1:y2, x1:x2]
    interior = float(gray[mask].mean()) if mask.any() else 127.0
    side = (sub >= val) if interior >= val else (sub < val)
    region = _threshold_region(mask, x1, y1, x2, y2, within_mask=within_mask, grow=grow)
    out = mask.copy()
    out[y1:y2, x1:x2] = side & region
    return out


def grabcut(gray: np.ndarray, mask: np.ndarray, *, iters: int = 5, pad: int = 12) -> np.ndarray:
    """Edge/intensity-adaptive ('quick select'): cv2.grabCut seeded from the current mask
    (sure-FG = eroded interior, sure-BG = outside the grown box, rest = probable) → snaps the
    boundary to the underlying gradient. CXR is grey, so grabCut runs on a 3-ch view of gray."""
    import cv2
    if not mask.any():
        return mask
    bb = _bbox(mask, pad)
    if bb is None:
        return mask
    img3 = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    gc = np.full(mask.shape, cv2.GC_PR_BGD, np.uint8)
    gc[mask] = cv2.GC_PR_FGD
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gc[cv2.erode(mask.astype(np.uint8), se) > 0] = cv2.GC_FGD
    grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))) > 0
    gc[~grown] = cv2.GC_BGD
    try:
        cv2.grabCut(img3, gc, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                    int(iters), cv2.GC_INIT_WITH_MASK)
    except Exception:
        return mask
    out = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return out if out.any() else mask


def magic_wand(gray: np.ndarray, mask: np.ndarray, *, tol: float = 0.08, grow: int = 15) -> np.ndarray:
    """Region-grow to the similar-intensity blob containing the mask (Photoshop magic wand): keep
    connected components of |gray - interior| <= tol·255 that touch the mask, within a grown bound.
    Stops at intensity discontinuities (edges); unions with the original seed so it never shrinks."""
    import cv2
    if not mask.any():
        return mask
    interior = float(gray[mask].mean())
    similar = (np.abs(gray.astype(np.float32) - interior) <= tol * 255.0).astype(np.uint8)
    grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))) > 0
    similar = similar & grown
    n, lab = cv2.connectedComponents(similar, connectivity=8)
    keep = set(np.unique(lab[mask & (similar > 0)]).tolist()) - {0}
    grown_sel = np.isin(lab, list(keep)) if keep else np.zeros_like(mask)
    return mask | grown_sel


def active_contour_snap(gray: np.ndarray, mask: np.ndarray, *, iters: int = 20, smoothing: int = 2,
                        balloon: float = 0.0) -> np.ndarray:
    """Edge-snapping boundary ('magnetic lasso'): morphological geodesic active contour evolves the
    mask boundary toward the inverse-gradient edges. Needs skimage; no-op (returns input) if absent."""
    try:
        from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
    except Exception:
        return mask
    if not mask.any():
        return mask
    g = inverse_gaussian_gradient(gray.astype(np.float32) / 255.0)
    try:
        out = morphological_geodesic_active_contour(
            g, num_iter=int(iters), init_level_set=mask.astype(np.uint8),
            smoothing=int(smoothing), balloon=float(balloon))
    except Exception:
        return mask
    out = out.astype(bool)
    return out if out.any() else mask


def contrast_gated_dilate(gray: np.ndarray, mask: np.ndarray, *, k: int = 3, max_contrast: float = 0.15) -> np.ndarray:
    """Grow only into ring pixels whose intensity is within max_contrast (frac of 255)
    of the mask interior mean."""
    import cv2
    if not mask.any():
        return mask
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    ring = (cv2.dilate(mask.astype(np.uint8), se) > 0) & (~mask)
    interior = float(gray[mask].mean())
    similar = np.abs(gray.astype(np.float32) - interior) <= (max_contrast * 255.0)
    return mask | (ring & similar)


def contrast_gated_erode(gray: np.ndarray, mask: np.ndarray, *, k: int = 3, min_contrast: float = 0.15) -> np.ndarray:
    """Strip boundary pixels that differ from the interior mean by >= min_contrast."""
    import cv2
    if not mask.any():
        return mask
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    boundary = mask & (cv2.erode(mask.astype(np.uint8), se) == 0)
    interior = float(gray[mask].mean())
    dissimilar = np.abs(gray.astype(np.float32) - interior) >= (min_contrast * 255.0)
    return mask & ~(boundary & dissimilar)


def edge_snap(gray: np.ndarray, mask: np.ndarray, *, band: int = 3) -> np.ndarray:
    """Clean ragged boundaries by morphological close+open within an edge-aware band."""
    import cv2
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, se)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
    return m > 0


def _ra():
    from qseg.ssl import refine_anatomy as ra
    return ra


# ---- curvilinear (catheter / lead / wire) line-following completion ---------
def _bridge_gaps(mask: np.ndarray, V: np.ndarray, max_gap: int, rad: int) -> np.ndarray:
    """Connect fragments of one tubular instance via minimal-cost paths on (1 - vesselness), but only
    when the path actually runs ALONG a tube (low mean cost) — so true gaps in ONE catheter close while
    unrelated structures are not bridged. The drawn path is dilated to the local tube radius."""
    from scipy import ndimage as ndi
    from skimage.graph import route_through_array
    from skimage.morphology import binary_dilation, disk
    lbl, n = ndi.label(mask)
    if n < 2:
        return mask
    cost = (1.0 - V).astype(np.float64) + 1e-3
    comps = [np.argwhere(lbl == k + 1) for k in range(n)]
    sub = lambda P: P[:: max(1, len(P) // 200)]                       # subsample for the O(|A||B|) nearest-pair
    added = np.zeros_like(mask, bool)
    for a in range(n):
        for b in range(a + 1, n):
            A, B = sub(comps[a]), sub(comps[b])
            d = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
            ia, ib = np.unravel_index(d.argmin(), d.shape)
            if d[ia, ib] > max_gap:
                continue
            try:
                path, total = route_through_array(cost, tuple(A[ia]), tuple(B[ib]), fully_connected=True)
            except Exception:
                continue
            if not path or total / len(path) > 0.6:                   # path mostly OFF a tube -> not a real continuation
                continue
            for (y, x) in path:
                added[y, x] = True
    if added.any():
        added = binary_dilation(added, disk(max(1, rad)))
        return mask | added
    return mask


def vessel_extend(gray: np.ndarray, mask: np.ndarray, *, low: float = 0.4, high: float = 0.7,
                  max_gap: int = 40, sigmas=(1, 2, 3, 4), dark=None, max_width: int = 8) -> np.ndarray:
    """Follow / complete a thin tubular structure (catheter, pacemaker lead, wire) along a Sato
    vesselness ridge map: (1) tubeness with polarity auto-detected from the mask, (2) hysteresis
    region-grow that keeps high-vesselness pixels CONNECTED to the current mask (no spurious blobs),
    (3) bridge fragment gaps via minimal-cost paths that run along the tube, (4) reconstruct width.
    Pure skimage/scipy, CPU. The line-appropriate complement to SAM (which handles compact parts)."""
    from scipy import ndimage as ndi
    from skimage.filters import sato
    from skimage.morphology import binary_dilation, disk, reconstruction, skeletonize
    m = mask > 0
    if not m.any():
        return m
    g = gray.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    if dark is None:                                                  # are the masked pixels darker than the local ring?
        ring = binary_dilation(m, disk(6)) & ~m
        dark = bool(g[m].mean() < (g[ring].mean() if ring.any() else g.mean()))
    V = sato(g, sigmas=sigmas, black_ridges=bool(dark)).astype(np.float32)
    V = (V - V.min()) / (np.ptp(V) + 1e-9)
    ref = float(np.median(V[m]))                                      # in-tube vesselness reference
    seed = m & (V >= high * ref)
    if not seed.any():
        seed = m
    region = (V >= low * ref) | m
    grown = reconstruction(seed.astype(np.uint8), region.astype(np.uint8), method="dilation").astype(bool)
    out = m | grown
    dt = ndi.distance_transform_edt(m)
    sk = skeletonize(m)
    rad = int(max(1, min(max_width, round(float(np.median(dt[sk])) if sk.any() else 1.0))))
    if max_gap:
        out = _bridge_gaps(out, V, int(max_gap), rad)
    return out > 0


# ---- SAM / MedSAM promptable refinement (best for COMPACT parts) ------------
def _sam_predictor(ckpt: str, model_type: str):
    cache = getattr(_sam_predictor, "_cache", None)
    if cache is None or cache[0] != (ckpt, model_type):
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        sam.to("cuda" if torch.cuda.is_available() else "cpu")
        _sam_predictor._cache = ((ckpt, model_type), SamPredictor(sam))
    return _sam_predictor._cache[1]


def sam_refine(gray: np.ndarray, mask: np.ndarray, *, ckpt=None, model_type=None,
               n_pos: int = 10, n_neg: int = 12, margin: int = 10, pad: int = 24, union: bool = True) -> np.ndarray:
    """Promptable SAM/MedSAM refinement: feed the partial mask as a dense (low-res) prompt + its bbox +
    positive points sampled ALONG the skeleton + negative points just outside it. Best for COMPACT
    structures (pacemaker can, catheter hub); thin shafts stay weak — pair with vessel_extend.
    Needs `pip install segment-anything` + a checkpoint via arg or CURATOR_SAM_CKPT (CURATOR_SAM_TYPE
    default vit_b; point at a MedSAM .pth for CXR)."""
    import os

    import cv2
    from skimage.morphology import binary_dilation, disk, skeletonize
    m = mask > 0
    if not m.any():
        return m
    ckpt = ckpt or os.environ.get("CURATOR_SAM_CKPT")
    if not ckpt or not os.path.exists(ckpt):
        raise RuntimeError("SAM checkpoint not found — `pip install segment-anything` and set "
                           "CURATOR_SAM_CKPT to a SAM/MedSAM .pth (CURATOR_SAM_TYPE=vit_b|vit_l|vit_h).")
    predictor = _sam_predictor(ckpt, model_type or os.environ.get("CURATOR_SAM_TYPE", "vit_b"))
    g = gray.astype(np.float32)
    rgb = np.repeat((g if g.max() > 1.5 else g * 255).astype(np.uint8)[..., None], 3, axis=2)
    predictor.set_image(rgb)
    sk = skeletonize(m)
    ys, xs = np.where(sk if sk.any() else m)
    pi = np.linspace(0, len(xs) - 1, min(n_pos, len(xs))).astype(int)
    pos = np.stack([xs[pi], ys[pi]], 1)
    ring = binary_dilation(m, disk(margin)) & ~m
    ry, rx = np.where(ring)
    neg = (np.stack([rx[np.linspace(0, len(rx) - 1, min(n_neg, len(rx))).astype(int)],
                     ry[np.linspace(0, len(ry) - 1, min(n_neg, len(ry))).astype(int)]], 1)
           if len(rx) else np.empty((0, 2), int))
    pts = np.concatenate([pos, neg], 0).astype(float)
    lbls = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)
    ys0, xs0 = np.where(m)
    H, W = m.shape
    box = np.array([max(0, xs0.min() - pad), max(0, ys0.min() - pad),
                    min(W, xs0.max() + pad), min(H, ys0.max() + pad)], float)
    mask_input = (cv2.resize(m.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA) * 16 - 8)[None]
    masks, _scores, _ = predictor.predict(point_coords=pts, point_labels=lbls, box=box,
                                          mask_input=mask_input, multimask_output=False)
    out = masks[0].astype(bool)
    return (out | m) if union else out


def apply_ops(gray: np.ndarray, mask: np.ndarray, ops: list[dict]) -> np.ndarray:
    """Apply an ordered op stack. Each op: {"name": str, "kw": {...}}.
    names: otsu | threshold | dilate | erode | fill | largest_cc | top_k_cc | smooth |
    grabcut | magic_wand | snap_edges | vessel_extend | sam. otsu/threshold take within_mask;
    vessel_extend follows/completes thin tubes (catheters/leads); sam = SAM/MedSAM promptable refine."""
    m = (mask > 0)
    for op in ops:
        name, kw = op.get("name"), op.get("kw", {})
        if name == "otsu":
            m = otsu_threshold(gray, m, within_mask=bool(kw.get("within_mask", False)))
        elif name == "threshold":
            m = manual_threshold(gray, m, int(kw.get("val", 128)), within_mask=bool(kw.get("within_mask", False)))
        elif name == "dilate":
            m = contrast_gated_dilate(gray, m, k=int(kw.get("k", 3)), max_contrast=float(kw.get("max_contrast", 0.15)))
        elif name == "erode":
            m = contrast_gated_erode(gray, m, k=int(kw.get("k", 3)), min_contrast=float(kw.get("min_contrast", 0.15)))
        elif name == "fill":
            m = _ra()._fill(m)
        elif name == "largest_cc":
            m = _ra().largest_cc(m)
        elif name == "top_k_cc":
            m = _ra().top_k_cc(m, int(kw.get("k", 2)))
        elif name == "smooth":
            m = edge_snap(gray, m, band=int(kw.get("band", 3)))
        elif name == "grabcut":
            m = grabcut(gray, m, iters=int(kw.get("iters", 5)))
        elif name == "magic_wand":
            m = magic_wand(gray, m, tol=float(kw.get("tol", 0.08)))
        elif name == "snap_edges":
            m = active_contour_snap(gray, m, iters=int(kw.get("iters", 20)))
        elif name == "vessel_extend":
            m = vessel_extend(gray, m, low=float(kw.get("low", 0.4)), high=float(kw.get("high", 0.7)),
                              max_gap=int(kw.get("max_gap", 40)))
        elif name == "sam":
            m = sam_refine(gray, m, n_pos=int(kw.get("n_pos", 10)), n_neg=int(kw.get("n_neg", 12)))
    return m > 0
