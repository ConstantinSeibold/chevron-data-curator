"""RAD-DINO dense feature correspondence for one/few-shot mask discovery (the Task-2 core).

Given a few SUPPORT (image, refined-mask) examples, build a per-patch FOREGROUND prototype bank, then for a
QUERY image produce a cosine-similarity heatmap + coarse seed mask. The caller turns seeds into masks:
compact -> SAM-HQ (`refine.sam_refine`), tubular -> `refine.apply_ops` vessel_extend.

Engine-independent on purpose: it takes numpy images/masks + a grid extractor exposing
`ext.grid(rgb_uint8_HxWx3) -> torch.Tensor[C, gh, gw]` (the curator's `RadDinoExtractor`). This is the
Matcher mechanic (DINO dense correspondence) on the domain-matched RAD-DINO features.
"""
from __future__ import annotations

import numpy as np


def _grid_feats(ext, rgb: np.ndarray):
    """L2-normalized patch features [P, C] + grid (gh, gw) for one image."""
    g = ext.grid(rgb)                                   # [C, gh, gw] torch tensor
    C, gh, gw = g.shape
    f = g.reshape(C, -1).T                              # [P, C]
    f = f / (f.norm(dim=1, keepdim=True) + 1e-8)
    return f, gh, gw


def _fg_patch_feats(ext, rgb: np.ndarray, mask: np.ndarray, *, gate: float = 0.1):
    """The normalized patch features whose grid cell overlaps the mask (>= `gate` soft coverage). Falls back
    to the single patch at the mask centroid when the mask is sub-patch."""
    import torch
    import torch.nn.functional as F
    f, gh, gw = _grid_feats(ext, rgb)                   # [P, C]
    m = torch.from_numpy(np.ascontiguousarray(mask).astype(np.float32))[None, None]
    soft = F.interpolate(m, size=(gh, gw), mode="bilinear", align_corners=False).reshape(-1).to(f.device)
    sel = soft > float(gate)
    if not bool(sel.any()):
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return f[:0]
        gy = min(gh - 1, int(ys.mean() / mask.shape[0] * gh))
        gx = min(gw - 1, int(xs.mean() / mask.shape[1] * gw))
        return f[gy * gw + gx][None]
    return f[sel]


def build_prototype(ext, supports, *, max_vecs: int = 512):
    """supports: iterable of (rgb uint8 HxWx3, mask bool HxW). Returns a normalized prototype bank
    [N, C] of foreground patch features (subsampled to `max_vecs`), on the extractor's device."""
    import torch
    banks = []
    for rgb, mask in supports:
        b = _fg_patch_feats(ext, rgb, mask)
        if b.numel():
            banks.append(b)
    if not banks:
        raise ValueError("no foreground patches in the support set")
    proto = torch.cat(banks, 0)
    if proto.shape[0] > max_vecs:                        # even, deterministic subsample
        idx = torch.linspace(0, proto.shape[0] - 1, max_vecs).round().long()
        proto = proto[idx]
    return proto                                         # [N, C] normalized


def heatmap(ext, rgb_query: np.ndarray, proto) -> np.ndarray:
    """Per-pixel MAX cosine similarity of the query patches to the prototype bank, upsampled to the query
    image size. Returns [H, W] float in [-1, 1] (1 = identical to the most similar support patch)."""
    import torch.nn.functional as F
    f, gh, gw = _grid_feats(ext, rgb_query)             # [P, C]
    sim = (f @ proto.T).max(dim=1).values                # [P] max over prototypes
    H, W = rgb_query.shape[:2]
    hm = F.interpolate(sim.reshape(1, 1, gh, gw), size=(H, W), mode="bilinear", align_corners=False)
    return hm[0, 0].detach().cpu().numpy()


def seed_mask(hm: np.ndarray, *, thresh: float) -> np.ndarray:
    """Coarse foreground from the heatmap (cosine >= thresh)."""
    return hm >= float(thresh)


def peaks(hm: np.ndarray, *, thresh: float, min_dist: int = 8, max_peaks: int = 32):
    """Local maxima of the heatmap above `thresh`, as [(y, x, score)] sorted by score (point prompts)."""
    import cv2
    H, W = hm.shape
    above = hm >= float(thresh)
    if not above.any():
        return []
    k = max(1, int(min_dist) | 1)
    dil = cv2.dilate(hm.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ismax = (hm >= dil - 1e-6) & above
    ys, xs = np.where(ismax)
    pts = sorted(((int(y), int(x), float(hm[y, x])) for y, x in zip(ys, xs)), key=lambda p: -p[2])
    return pts[:int(max_peaks)]
