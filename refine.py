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


def apply_ops(gray: np.ndarray, mask: np.ndarray, ops: list[dict]) -> np.ndarray:
    """Apply an ordered op stack. Each op: {"name": str, "kw": {...}}.
    names: otsu | threshold | dilate | erode | fill | largest_cc | top_k_cc | smooth |
    grabcut | magic_wand | snap_edges. otsu/threshold take within_mask (confine to current mask)."""
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
    return m > 0
