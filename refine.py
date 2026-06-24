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


# ---- constrained shortest path (centerline) for line classes ----------------
def _vesselness(g01: np.ndarray, mask: np.ndarray, sigmas=(1, 2, 3, 4)) -> np.ndarray:
    """Sato tubeness, ridge polarity auto-detected from the masked pixels, normalized to [0, 1]."""
    from skimage.filters import sato
    from skimage.morphology import binary_dilation, disk
    ring = binary_dilation(mask, disk(6)) & ~mask
    dark = bool(g01[mask].mean() < (g01[ring].mean() if ring.any() else g01.mean()))
    V = sato(g01, sigmas=sigmas, black_ridges=dark).astype(np.float32)
    return (V - V.min()) / (np.ptp(V) + 1e-9)


def _farthest_in(cost: np.ndarray, seed, region: np.ndarray):
    """The `region` pixel with the largest min-cost (geodesic) distance from `seed` over `cost`.
    One half of the tree-diameter 'double sweep' that locates a line's two tips parameter-free."""
    from skimage.graph import MCP_Geometric
    cum, _ = MCP_Geometric(cost).find_costs([list(seed)])
    d = np.where(region & np.isfinite(cum), cum, -np.inf)
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(d)), d.shape))


def _route_curved(cost: np.ndarray, A, B, *, xi: float, n_theta: int = 60):
    """OPTIONAL curvature-penalized routing (agd Reeds-Shepp, orientation-lifted): the path cannot
    turn sharply, so at a tube crossing it follows the straight continuation instead of hopping onto
    the crosser. Returns [(y, x), ...] or None when agd is missing / the solve fails (caller falls
    back to plain Dijkstra). Requires `pip install agd`; dormant otherwise."""
    try:
        from agd import Eikonal
    except Exception:
        return None
    try:
        H, W = cost.shape
        hi = Eikonal.dictIn({
            "model": "ReedsShepp2",
            "exportValues": 1,
            "cost": np.ascontiguousarray(np.broadcast_to(cost.T[:, :, None], (W, H, int(n_theta))).copy()),
            "xi": float(xi),
            "seeds_Unoriented": [[float(A[1]), float(A[0])]],
            "tips_Unoriented": [[float(B[1]), float(B[0])]],
        })
        hi.SetRect(sides=[[0, W], [0, H]], dimx=W, dimy=H)
        hi["nTheta"] = int(n_theta)
        out = hi.Run()
        geo = np.asarray((out.get("geodesics_Unoriented") or out["geodesics"])[0])
        xy = geo if geo.shape[0] >= geo.shape[1] else geo.T          # -> (k, >=2), rows (x, y, [theta])
        seen = []
        for p in xy:
            y, x = min(max(int(round(p[1])), 0), H - 1), min(max(int(round(p[0])), 0), W - 1)
            if not seen or seen[-1] != (y, x):
                seen.append((y, x))
        return seen if len(seen) >= 2 else None
    except Exception:
        return None


def line_centerline(gray: np.ndarray, mask: np.ndarray, *, alpha: float = 0.7, width: int = 0,
                    curvature: float = 0.0, pad: int = 16, sigmas=(1, 2, 3, 4),
                    max_width: int = 12) -> np.ndarray:
    """Collapse a branchy / fragmented LINE mask to the single constrained shortest path between its
    two extreme tips — a centerline that runs ALONG the mask (medial-axis-biased) and only leaves it,
    preferring image tubeness, to bridge genuine gaps. Unlike `vessel_extend` (which GROWS the mask
    via vesselness region-grow and can flood into ribs / other tubes -> mesh), a path is a single
    simple curve BY CONSTRUCTION: it cannot branch. The route is the global-minimum geodesic
    (Dijkstra), so it is DETERMINISTIC — no run-to-run randomness.

    Tips are found automatically by a two-pass geodesic 'double sweep' over the cost map (the
    tree-diameter trick), so a single `alpha` (mask-trust vs gap-bridging) is the only knob and it
    holds class-wide — no per-instance tuning. `curvature>0` uses the agd Reeds-Shepp backend if
    installed (won't hop onto a crossing tube), else falls back to plain Dijkstra.
    Pure skimage/scipy, CPU. The line-class complement to SAM (compact parts)."""
    import warnings

    from scipy import ndimage as ndi
    from skimage.graph import route_through_array
    from skimage.morphology import binary_dilation, disk, skeletonize

    m_full = mask > 0
    if m_full.sum() < 2:
        return m_full
    H, W = m_full.shape
    ys, xs = np.where(m_full)
    x1, y1 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x2, y2 = min(W, int(xs.max()) + pad + 1), min(H, int(ys.max()) + pad + 1)
    m = m_full[y1:y2, x1:x2]
    g = gray[y1:y2, x1:x2].astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0

    V = _vesselness(g, m, sigmas)
    dt_in = ndi.distance_transform_edt(m).astype(np.float32)
    dtn = dt_in / (dt_in.max() + 1e-9)                               # 1 on the medial axis, 0 at the rim
    beta = float(np.clip(1.0 - alpha, 0.05, 0.5))                   # off-mask tube reward; <= rim cost so inside wins
    lineness = np.where(m, 0.5 + 0.5 * dtn, beta * V)               # inside: medial-biased 0.5..1; outside: 0..beta
    cost = (1.0 - lineness).astype(np.float64) + 1e-3              # cheapest along the medial axis; gaps via tubes

    idx = np.argwhere(m)
    seed0 = tuple(int(v) for v in idx[len(idx) // 2])              # any mask pixel; double sweep is start-invariant
    A = _farthest_in(cost, seed0, m)
    B = _farthest_in(cost, A, m)
    if A == B:
        return m_full

    path = _route_curved(cost, A, B, xi=float(curvature)) if curvature > 0 else None
    if curvature > 0 and path is None:
        warnings.warn("line_centerline: agd unavailable — using plain Dijkstra (pip install agd for curvature).")
    if path is None:
        path, _ = route_through_array(cost, list(A), list(B), fully_connected=True, geometric=True)

    line = np.zeros_like(m)
    for (yy, xx) in path:
        line[yy, xx] = True
    if width and width > 0:
        rad = int(width)
    else:
        sk = skeletonize(m)
        rad = int(max(1, min(max_width, round(float(np.median(dt_in[sk])) if sk.any() else 1.0))))
    line = binary_dilation(line, disk(max(1, rad)))

    out = np.zeros_like(m_full)
    out[y1:y2, x1:x2] = line
    return out


# ---- SAM / MedSAM promptable refinement (best for COMPACT parts) ------------
# Official SAM checkpoints (FAIR). vit_b is the smallest (~375 MB) — the default we auto-fetch.
_SAM_URLS = {
    "vit_b": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "sam_vit_b_01ec64.pth"),
    "vit_l": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "sam_vit_l_0b3195.pth"),
    "vit_h": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "sam_vit_h_4b8939.pth"),
}

# SAM-HQ (HQ-Output token on frozen SAM — crisper masks, esp. on thin/intricate structures). HF mirror of
# the official Google-Drive weights so urlretrieve works; loaded via the `segment_anything_hq` registry.
_HQ_URLS = {
    "vit_b": ("https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth", "sam_hq_vit_b.pth"),
    "vit_l": ("https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth", "sam_hq_vit_l.pth"),
    "vit_h": ("https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth", "sam_hq_vit_h.pth"),
    "vit_tiny": ("https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth", "sam_hq_vit_tiny.pth"),
}


def _sam_dir():
    import os
    from pathlib import Path
    d = os.environ.get("CURATOR_SAM_DIR") or str(Path.home() / ".cache" / "curator" / "sam")
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)


def detect_sam_type(path: str) -> str:
    """Infer the SAM arch from a checkpoint filename (MedSAM is a vit_b). Matches the BASENAME only — a
    parent directory containing 'vit_h'/'tiny'/etc. must not flip the arch."""
    import os
    p = os.path.basename(str(path)).lower()
    if "vit_h" in p or "_h_" in p:
        return "vit_h"
    if "vit_l" in p or "_l_" in p:
        return "vit_l"
    if "vit_tiny" in p or "tiny" in p:
        return "vit_tiny"
    return "vit_b"


def detect_sam_family(path: str) -> str:
    """SAM vs MedSAM vs SAM-HQ from the checkpoint filename. MedSAM loads through the SAME vit_b registry but
    runs a different recipe (box-only, min-max norm, single mask); SAM-HQ loads through the SEPARATE
    `segment_anything_hq` registry (extra HQ token) but is prompted like SAM. The family decides which
    registry weights load AND how `sam_refine` prompts. Matches the BASENAME only (a parent dir with 'hq'
    in its name must not misclassify a vanilla checkpoint)."""
    import os
    p = os.path.basename(str(path)).lower()
    if "medsam" in p:
        return "medsam"
    if "hq" in p:
        return "samhq"
    return "sam"


def find_sam_checkpoint(ckpt=None, family=None):
    """Resolve a SAM/MedSAM checkpoint: explicit arg -> env (CURATOR_MEDSAM_CKPT when family='medsam', else
    CURATOR_SAM_CKPT) -> a .pth/.pt in the SAM cache dir (CURATOR_SAM_DIR or ~/.cache/curator/sam). When
    `family` is given, prefers a cache file of that family (falls back to any). Returns (path, model_type) or
    (None, None)."""
    import os
    env = os.environ.get("CURATOR_MEDSAM_CKPT") if family == "medsam" else None
    ckpt = ckpt or env or os.environ.get("CURATOR_SAM_CKPT")
    if ckpt and os.path.exists(ckpt):
        return ckpt, (os.environ.get("CURATOR_SAM_TYPE") or detect_sam_type(ckpt))
    cands = sorted([*_sam_dir().glob("*.pth"), *_sam_dir().glob("*.pt")])
    if family:
        cands = [c for c in cands if detect_sam_family(c) == family] or cands
    if cands:
        c = str(cands[0])
        return c, (os.environ.get("CURATOR_SAM_TYPE") or detect_sam_type(c))
    return None, None


def sam_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("segment_anything") is not None


def samhq_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("segment_anything_hq") is not None


def ensure_samhq_checkpoint(model_type: str = "vit_b", progress=None) -> str:
    """Make a SAM-HQ checkpoint available locally (HF mirror), downloading if absent. Returns the path.
    Raises RuntimeError if `segment-anything-hq` isn't installed."""
    if not samhq_available():
        raise RuntimeError("SAM-HQ needs the `segment-anything-hq` package — run "
                           "`pip install segment-anything-hq` in the qseg env, then retry.")
    existing, _ = find_sam_checkpoint(family="samhq")
    if existing and detect_sam_family(existing) == "samhq":
        return existing
    import urllib.request
    if model_type not in _HQ_URLS:
        model_type = "vit_b"
    url, fname = _HQ_URLS[model_type]
    dest = _sam_dir() / fname
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _hook(blocks, bs, total):
        if progress and total > 0:
            progress(min(1.0, blocks * bs / total))
    urllib.request.urlretrieve(url, str(tmp), _hook)   # noqa: S310 (HF mirror)
    tmp.replace(dest)
    return str(dest)


def ensure_sam_checkpoint(model_type: str = "vit_b", progress=None) -> str:
    """Make a SAM checkpoint available locally, downloading it to the cache dir if absent. Returns the
    path. Raises RuntimeError with an actionable message if `segment_anything` isn't installed."""
    if not sam_available():
        raise RuntimeError("the `segment-anything` package is not installed — run "
                           "`pip install segment-anything` in the qseg env, then retry.")
    existing, _ = find_sam_checkpoint()
    if existing:
        return existing
    import urllib.request
    if model_type not in _SAM_URLS:
        model_type = "vit_b"
    url, fname = _SAM_URLS[model_type]
    dest = _sam_dir() / fname
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _hook(blocks, bs, total):
        if progress and total > 0:
            progress(min(1.0, blocks * bs / total))
    urllib.request.urlretrieve(url, str(tmp), _hook)   # noqa: S310 (trusted FAIR host)
    tmp.replace(dest)
    return str(dest)


def _sam_predictor(ckpt: str, model_type: str, family: str = "sam"):
    cache = getattr(_sam_predictor, "_cache", None)
    key = (ckpt, model_type, family)
    if cache is None or cache[0] != key:
        import torch
        if family == "samhq":                                  # HQ token -> separate registry/arch
            from segment_anything_hq import SamPredictor, sam_model_registry
        else:
            from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        sam.to("cuda" if torch.cuda.is_available() else "cpu")
        _sam_predictor._cache = (key, SamPredictor(sam))
    return _sam_predictor._cache[1]


def sam_prompt_points(mask: np.ndarray, *, n_pos: int = 10, n_neg: int = 12, margin: int = 24, pad: int = 24,
                      inset: float | None = None):
    """Where SAM's prompts come from, as (pos_xy, neg_xy, box_xyxy) in (x, y) pixel coords. The guiding
    principle for REFINEMENT: never put a prompt on the uncertain BOUNDARY (that just pins the current,
    possibly-wrong outline). Leave the rim unconstrained so SAM can redraw it:
    - POSITIVES only in the CONFIDENT DEEP INTERIOR — the deepest-interior point (medial centre) first,
      then a spread of pixels whose distance-to-boundary >= `inset` (default ½ the max depth, so thin
      structures still get their centerline). None sit on the rim.
    - NEGATIVES only in CLEAR BACKGROUND beyond a `margin`-px GAP — a shell from `margin` to `margin+band`
      out. The `margin`-wide band around the mask carries NO points, so SAM is free to move the boundary
      either way (instead of recreating the input mask).
    - plus the padded bbox.
    Pure numpy/scipy/skimage — no checkpoint needed, so the preview can show them before SAM runs.
    `sam_refine` reuses this so the drawn points are byte-identical to what the model is fed."""
    from scipy import ndimage as ndi
    from skimage.morphology import binary_dilation, disk
    m = mask > 0
    if not m.any():
        return np.empty((0, 2), int), np.empty((0, 2), int), None
    # positives: deepest-interior centre first, then spread over the CONFIDENT core (DT >= inset), so no
    # positive lands on the uncertain rim. inset auto = ½ max depth (thin masks keep their centerline).
    dt = ndi.distance_transform_edt(m)
    dmax = float(dt.max())
    if inset is None:
        inset = 0.5 * dmax
    cy, cx = np.unravel_index(int(np.argmax(dt)), m.shape)
    core = dt >= min(float(inset), dmax)                    # guard: at least the deepest pixel qualifies
    ys, xs = np.where(core)
    pos = [[int(cx), int(cy)]]
    if n_pos > 1 and len(xs):
        pi = np.linspace(0, len(xs) - 1, min(n_pos - 1, len(xs))).astype(int)
        pos += [[int(xs[i]), int(ys[i])] for i in pi]
    pos = np.array(pos[:max(1, n_pos)], int)
    # negatives: a shell from `margin` to `margin+band` out — clear background, beyond the unconstrained gap
    band = max(4, int(margin) // 2)
    inner = binary_dilation(m, disk(int(margin)))           # the margin-px band stays point-free
    outer = binary_dilation(m, disk(int(margin) + band))
    ring = outer & ~inner
    ry, rx = np.where(ring)
    neg = (np.stack([rx[np.linspace(0, len(rx) - 1, min(n_neg, len(rx))).astype(int)],
                     ry[np.linspace(0, len(ry) - 1, min(n_neg, len(ry))).astype(int)]], 1)
           if len(rx) else np.empty((0, 2), int))
    ys0, xs0 = np.where(m)
    H, W = m.shape
    box = np.array([max(0, xs0.min() - pad), max(0, ys0.min() - pad),
                    min(W, xs0.max() + pad), min(H, ys0.max() + pad)], int)
    return pos, neg, box


def sam_refine(gray: np.ndarray, mask: np.ndarray, *, ckpt=None, model_type=None, model: str = "auto",
               n_pos: int = 10, n_neg: int = 12, margin: int = 24, pad: int = 24, union: bool = False,
               use_mask_prompt: bool = True) -> np.ndarray:
    """Promptable SAM/MedSAM refinement for an UNCERTAIN mask. Best for COMPACT structures (pacemaker can,
    catheter hub); thin shafts stay weak — pair with vessel_extend.

    `model` picks the family ("auto" = infer from the checkpoint name, "sam", or "medsam"):
    - SAM: prompt with confident-interior positives + clear-background negatives beyond a gap
      (`sam_prompt_points`) so the boundary stays free to move; bbox + (optional) dense mask prior bound the
      extent; asks for MULTIPLE proposals (`multimask_output=True`) and takes the best — so the result
      follows the image rather than echoing the input.
    - MedSAM: the medical fine-tune was trained with a BOX prompt only, on per-image min-max-normalized
      inputs, single-mask output — so it runs box-only (no points, no mask prior), `multimask_output=False`,
      after stretching the crop to [0,255]. Points/mask-prior knobs are ignored for MedSAM.

    By default the result REPLACES the mask (`union=False`) so the boundary can move BOTH ways; `union=True`
    can never shrink. Empty proposal falls back to the input. Needs `pip install segment-anything` + a
    checkpoint (SAM auto-fetched to ~/.cache/curator/sam; MedSAM via CURATOR_MEDSAM_CKPT or a *medsam*.pth
    dropped in CURATOR_SAM_DIR; CURATOR_SAM_TYPE overrides the arch, CURATOR_SAM_FAMILY the family)."""
    import os

    import cv2
    m = mask > 0
    if not m.any():
        return m
    family = (model if model in ("sam", "medsam", "samhq") else None) or os.environ.get("CURATOR_SAM_FAMILY")
    found, found_type = find_sam_checkpoint(ckpt, family=family)
    if not found:
        if not sam_available():
            raise RuntimeError("SAM refine needs the `segment-anything` package — "
                               "`pip install segment-anything`, then click 'Set up SAM' in Refine.")
        raise RuntimeError("no SAM checkpoint found — click 'Set up SAM' in the Refine tab to download SAM "
                           "(~375 MB), set CURATOR_SAM_CKPT, or for MedSAM drop a *medsam*.pth in "
                           "CURATOR_SAM_DIR / set CURATOR_MEDSAM_CKPT.")
    family = family or detect_sam_family(found)
    if family == "samhq" and detect_sam_family(found) != "samhq":          # a vanilla ckpt won't fit the HQ arch
        raise RuntimeError("no SAM-HQ checkpoint found — click 'Set up SAM-HQ' in Refine to download it "
                           "(the HQ token needs sam_hq_* weights, not vanilla SAM).")
    predictor = _sam_predictor(found, model_type or found_type, family)
    g = gray.astype(np.float32)
    if family == "medsam":                                                     # MedSAM: box-only, min-max norm, 1 mask
        lo, hi = float(g.min()), float(g.max())
        norm = (g - lo) / (hi - lo + 1e-8) * 255.0
        predictor.set_image(np.repeat(norm.astype(np.uint8)[..., None], 3, axis=2))
        ys0, xs0 = np.where(m)
        H, W = m.shape
        box = np.array([max(0, xs0.min() - pad), max(0, ys0.min() - pad),
                        min(W, xs0.max() + pad), min(H, ys0.max() + pad)], float)
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        out = np.asarray(masks)[0].astype(bool)
    else:
        rgb = np.repeat((g if g.max() > 1.5 else g * 255).astype(np.uint8)[..., None], 3, axis=2)
        predictor.set_image(rgb)
        pos, neg, box = sam_prompt_points(m, n_pos=n_pos, n_neg=n_neg, margin=margin, pad=pad)
        pts = np.concatenate([pos, neg], 0).astype(float)
        lbls = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)
        mask_input = ((cv2.resize(m.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA) * 16 - 8)[None]
                      if use_mask_prompt else None)
        masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbls, box=box.astype(float),
                                             mask_input=mask_input, multimask_output=True)
        out = np.asarray(masks)[int(np.argmax(np.asarray(scores)))].astype(bool)   # SAM's best proposal
    if not out.any():                                                          # degenerate -> keep input
        out = m
    return (out | m) if union else out


def enhance_contrast(gray: np.ndarray, *, method: str = "clahe", clip: float = 2.0, grid: int = 8,
                     gamma: float = 1.0) -> np.ndarray:
    """Boost image contrast so the intensity-based ops see sharper edges. Returns uint8.
    method: clahe (local adaptive, default; `clip` = CLAHE clip limit), stretch (2–98 pct linear
    stretch), gamma (`gamma`<1 brightens, >1 darkens)."""
    import cv2
    g = gray
    if g.dtype != np.uint8:                                  # to_gray gives uint8; be robust to float [0,1]
        g = np.clip(g * 255.0 if float(g.max()) <= 1.5 else g, 0, 255).astype(np.uint8)
    if method == "stretch":
        lo, hi = np.percentile(g, [2, 98])
        if hi <= lo:
            return g
        return np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    if method == "gamma":
        lut = (np.linspace(0, 1, 256) ** float(gamma) * 255).astype(np.uint8)
        return lut[g]
    cl = cv2.createCLAHE(clipLimit=max(0.1, float(clip)), tileGridSize=(int(grid), int(grid)))
    return cl.apply(g)


def apply_ops(gray: np.ndarray, mask: np.ndarray, ops: list[dict], *, return_image: bool = False):
    """Apply an ordered op stack. Each op: {"name": str, "kw": {...}}.
    names: contrast | otsu | threshold | dilate | erode | fill | largest_cc | top_k_cc | smooth |
    grabcut | magic_wand | snap_edges | vessel_extend | line_centerline | sam. `contrast` enhances the
    WORKING image that every later op sees (add it FIRST); otsu/threshold take within_mask; vessel_extend
    GROWS a thin tube along vesselness (catheters/leads); line_centerline REDUCES a line mask to the single
    shortest path between its tips (deterministic, cannot branch); sam = SAM/MedSAM promptable refine.
    With return_image=True returns (mask, working_image) so the preview can show the enhanced image."""
    g = gray                                                 # working image; a `contrast` op replaces it
    m = (mask > 0)
    for op in ops:
        name, kw = op.get("name"), op.get("kw", {})
        if name == "contrast":
            g = enhance_contrast(g, method=str(kw.get("method", "clahe")), clip=float(kw.get("clip", 2.0)),
                                 gamma=float(kw.get("gamma", 1.0)))
        elif name == "otsu":
            m = otsu_threshold(g, m, within_mask=bool(kw.get("within_mask", False)))
        elif name == "threshold":
            m = manual_threshold(g, m, int(kw.get("val", 128)), within_mask=bool(kw.get("within_mask", False)))
        elif name == "dilate":
            m = contrast_gated_dilate(g, m, k=int(kw.get("k", 3)), max_contrast=float(kw.get("max_contrast", 0.15)))
        elif name == "erode":
            m = contrast_gated_erode(g, m, k=int(kw.get("k", 3)), min_contrast=float(kw.get("min_contrast", 0.15)))
        elif name == "fill":
            m = _ra()._fill(m)
        elif name == "largest_cc":
            m = _ra().largest_cc(m)
        elif name == "top_k_cc":
            m = _ra().top_k_cc(m, int(kw.get("k", 2)))
        elif name == "smooth":
            m = edge_snap(g, m, band=int(kw.get("band", 3)))
        elif name == "grabcut":
            m = grabcut(g, m, iters=int(kw.get("iters", 5)))
        elif name == "magic_wand":
            m = magic_wand(g, m, tol=float(kw.get("tol", 0.08)))
        elif name == "snap_edges":
            m = active_contour_snap(g, m, iters=int(kw.get("iters", 20)))
        elif name == "vessel_extend":
            m = vessel_extend(g, m, low=float(kw.get("low", 0.4)), high=float(kw.get("high", 0.7)),
                              max_gap=int(kw.get("max_gap", 40)), max_width=int(kw.get("max_width", 8)))
        elif name == "line_centerline":
            m = line_centerline(g, m, alpha=float(kw.get("alpha", 0.7)), width=int(kw.get("width", 0)),
                                curvature=float(kw.get("curvature", 0.0)), max_width=int(kw.get("max_width", 12)))
        elif name == "sam":
            m = sam_refine(g, m, model=str(kw.get("model", "auto")), n_pos=int(kw.get("n_pos", 10)),
                           n_neg=int(kw.get("n_neg", 12)), margin=int(kw.get("margin", 24)),
                           use_mask_prompt=bool(kw.get("mask_prior", 1)), union=bool(kw.get("keep", 0)))
    return ((m > 0), g) if return_image else (m > 0)
