"""Stage-1 auto-refine: pick the refinement chain a mask NEEDS by label-free search.

For one instance we enumerate a small set of candidate chains (the curator's own ops), apply
each via `refine.apply_ops`, score the result with a GT-free reward, and keep the argmax. No
training data — the chosen (features, chain) pairs are exactly the supervision a Stage-2 learned
policy would later imitate.

The reward is shape-kind-aware:
- `line`  : single component + simple path (2 skeleton endpoints, no mesh) + sits on an image
            ridge (Sato) + fidelity (spans the structure, doesn't spill outside its hull).
- `blob`  : single component + solidity + boundary smoothness + IoU-to-original (edge cleanup
            shouldn't move the mask much).
`kind="auto"` routes by elongation of the original mask. A near-empty mask scores 0 so the search
can prefer the no-op chain (leave a clean mask alone) — encoded by a tiny per-op simplicity penalty.

Pure cv2/skimage/scipy, CPU. The reward weights are module constants — tune in one place.
"""
from __future__ import annotations

import numpy as np

# Candidate chains. Each is a list of {"name", "kw"} the same shape `refine.apply_ops` consumes.
# Keep lists short — search cost is linear in candidates × ops. SAM/grabcut excluded (ckpt / slow).
LINE_CANDIDATES: list[list[dict]] = [
    [],
    [{"name": "largest_cc", "kw": {}}],
    [{"name": "line_centerline", "kw": {"alpha": 0.5}}],
    [{"name": "line_centerline", "kw": {"alpha": 0.7}}],
    [{"name": "line_centerline", "kw": {"alpha": 0.9}}],
    [{"name": "smooth", "kw": {}}, {"name": "line_centerline", "kw": {"alpha": 0.7}}],
    [{"name": "vessel_extend", "kw": {}}],
]
BLOB_CANDIDATES: list[list[dict]] = [
    [],
    [{"name": "largest_cc", "kw": {}}],
    [{"name": "fill", "kw": {}}],
    [{"name": "fill", "kw": {}}, {"name": "largest_cc", "kw": {}}],
    [{"name": "smooth", "kw": {}}],
    [{"name": "fill", "kw": {}}, {"name": "smooth", "kw": {}}],
    [{"name": "fill", "kw": {}}, {"name": "largest_cc", "kw": {}}, {"name": "smooth", "kw": {}}],
]

_W_LINE = {"connected": 0.30, "simple": 0.25, "on_tube": 0.20, "fidelity": 0.25}
_W_BLOB = {"connected": 0.25, "solidity": 0.30, "smooth": 0.20, "iou": 0.25}
_SIMPLICITY_PENALTY = 1e-3            # per op, to break ties toward the shorter chain (and toward no-op)
_THINNESS_LINE = 0.12                # median tube width / bbox diagonal below this -> treat as a line


# ---- geometry helpers ------------------------------------------------------
def _ncc(mask: np.ndarray) -> int:
    import cv2
    return int(cv2.connectedComponents(mask.astype(np.uint8))[0]) - 1


def _skeleton(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    return skeletonize(mask > 0)


def _skel_endpoints(mask: np.ndarray) -> int:
    from scipy import ndimage as ndi
    sk = _skeleton(mask).astype(np.uint8)
    nb = ndi.convolve(sk, np.ones((3, 3), np.uint8), mode="constant") - sk
    return int(((nb == 1) & (sk > 0)).sum())


def _bbox_diag(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    return float(np.hypot(ys.max() - ys.min(), xs.max() - xs.min()))


def _convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    import cv2
    pts = np.argwhere(mask)
    if len(pts) < 3:
        return mask.copy()
    hull = cv2.convexHull(pts[:, ::-1].astype(np.int32))     # (x, y) for cv2
    out = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out > 0


def _perimeter(mask: np.ndarray) -> float:
    import cv2
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(c, True) for c in cnts))


def _solidity(mask: np.ndarray) -> float:
    a = float(mask.sum())
    h = float(_convex_hull_mask(mask).sum())
    return a / h if h > 0 else 0.0


def _thinness(mask: np.ndarray) -> float | None:
    """Median tube width / bbox diagonal — small for lines, large for blobs. None if too small to judge.
    Width-based (not elongation), so it is robust to holes: an annulus' long skeleton would fool a
    skeleton-length ratio, but its width is large."""
    from scipy import ndimage as ndi
    m = mask > 0
    if m.sum() < 8:
        return None
    sk = _skeleton(m)
    if not sk.any():
        return None
    width = 2.0 * float(np.median(ndi.distance_transform_edt(m)[sk]))
    return width / (_bbox_diag(m) + 1e-9)


def classify_shape(mask: np.ndarray) -> str:
    """`line` if the single mask is THIN, else `blob`. Per-instance — a weak proxy for the category;
    prefer `partition_kind` over a class's members when the category is known."""
    t = _thinness(mask)
    return "blob" if t is None else ("line" if t < _THINNESS_LINE else "blob")


def partition_kind(masks: list[np.ndarray]) -> str:
    """Category kind from a partition/class IN CONTEXT: the MEDIAN thinness across members. Robust where
    a single instance lies (a short fragment looks blobby alone, but the class's members are mostly thin),
    which is exactly why this beats per-instance `classify_shape` when the category is known."""
    ts = [t for t in (_thinness(m) for m in masks) if t is not None]
    if not ts:
        return "blob"
    return "line" if float(np.median(ts)) < _THINNESS_LINE else "blob"


# ---- label-free rewards ----------------------------------------------------
def reward_line(orig: np.ndarray, cand: np.ndarray, V: np.ndarray | None) -> tuple[float, dict]:
    a = int(cand.sum())
    if a < 2:
        return 0.0, {"empty": True}
    connected = 1.0 / max(1, _ncc(cand))
    simple = 2.0 / max(2, _skel_endpoints(cand))                       # 2 endpoints -> 1; mesh -> <1
    on_tube = float(np.clip(V[cand].mean(), 0, 1)) if V is not None else 0.5
    span = min(1.0, _bbox_diag(cand) / (_bbox_diag(orig) + 1e-9))      # still reaches the structure's extent
    margin = int(np.clip(0.08 * _bbox_diag(orig), 3, 30))
    hull = _convex_hull_mask(orig)
    if margin:
        import cv2
        hull = cv2.dilate(hull.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))) > 0
    spill = float((cand & ~hull).sum()) / a                            # leaked outside the structure's hull
    fidelity = span * (1.0 - spill)
    br = {"connected": connected, "simple": simple, "on_tube": on_tube,
          "span": span, "spill": spill, "fidelity": fidelity}
    return float(sum(_W_LINE[k] * br[k] for k in _W_LINE)), br


def reward_blob(orig: np.ndarray, cand: np.ndarray) -> tuple[float, dict]:
    a = int(cand.sum())
    if a < 2:
        return 0.0, {"empty": True}
    connected = 1.0 / max(1, _ncc(cand))
    solidity = _solidity(cand)
    smooth = min(1.0, _perimeter(_convex_hull_mask(cand)) / (_perimeter(cand) + 1e-9))
    union = int((orig | cand).sum())
    iou = int((orig & cand).sum()) / union if union else 0.0
    br = {"connected": connected, "solidity": solidity, "smooth": smooth, "iou": iou}
    return float(sum(_W_BLOB[k] * br[k] for k in _W_BLOB)), br


def shape_prior_reward(model, *, device: str = "cpu", weights=(0.5, 0.25, 0.25)):
    """CATEGORY-conditioned reward: how PLAUSIBLE the candidate is under class C's trained shape-prior DAE
    (label-free — the same per-class manifold the OOD evaluator uses), guarded by connectivity + IoU so a
    'plausible' shape can't be a hallucination drifted off the original. Returns a reward_fn(orig, cand).
    This is the reward whose very CRITERION is category-specific (a lung manifold ≠ a rib manifold)."""
    from .core.shape_prior import canonicalize, implausibility_detail
    wp, wc, wi = weights

    def _reward(orig: np.ndarray, cand: np.ndarray) -> tuple[float, dict]:
        if int(cand.sum()) < 2:
            return 0.0, {"empty": True}
        m01 = canonicalize(cand.astype(np.uint8))
        if m01 is None or m01.sum() == 0:
            return 0.0, {"empty": True}
        glob, band, _ = implausibility_detail(model, m01, device)
        plaus = float(max(0.0, 1.0 - 0.5 * (glob + band)))            # mean implausibility -> plausibility
        connected = 1.0 / max(1, _ncc(cand))
        union = int((orig | cand).sum())
        iou = int((orig & cand).sum()) / union if union else 0.0
        br = {"plausibility": plaus, "connected": connected, "iou": iou}
        return float(wp * plaus + wc * connected + wi * iou), br

    return _reward


# ---- search ----------------------------------------------------------------
def search(gray: np.ndarray, mask: np.ndarray, *, kind: str = "auto",
           candidates: list[list[dict]] | None = None, reward_fn=None,
           reward_name: str = "geometric") -> dict:
    """Score every candidate chain on this mask, return the ranked list + the argmax `best`.

    `kind` selects the candidate set (pass the CATEGORY's kind from `partition_kind` for context, not the
    lone mask's). `reward_fn(orig, cand) -> (score, breakdown)` overrides the default geometric reward with
    a category-conditioned one (e.g. `shape_prior_reward`). Returns {kind, reward, best, candidates}.
    `best.chain` is `[]` (no-op) when the mask is already clean."""
    from . import refine

    m = mask > 0
    if m.sum() < 4:
        return {"kind": "blob", "reward": reward_name,
                "best": {"chain": [], "score": 0.0, "breakdown": {"empty": True}, "ncc": 0, "area": int(m.sum())},
                "candidates": []}
    kind = classify_shape(m) if kind == "auto" else kind
    cands = candidates if candidates is not None else (LINE_CANDIDATES if kind == "line" else BLOB_CANDIDATES)

    V = None
    if reward_fn is None and kind == "line":
        g01 = gray.astype(np.float32)
        g01 = g01 / 255.0 if g01.max() > 1.5 else g01
        V = refine._vesselness(g01, m)

    results = []
    for chain in cands:
        try:
            out = refine.apply_ops(gray, m, chain) > 0
        except Exception:
            continue
        if reward_fn is not None:
            score, br = reward_fn(m, out)
        else:
            score, br = reward_line(m, out, V) if kind == "line" else reward_blob(m, out)
        score -= _SIMPLICITY_PENALTY * len(chain)
        results.append({"chain": chain, "score": float(score), "breakdown": br,
                        "ncc": _ncc(out), "area": int(out.sum())})
    results.sort(key=lambda r: -r["score"])
    best = results[0] if results else {"chain": [], "score": 0.0, "breakdown": {}, "ncc": _ncc(m), "area": int(m.sum())}
    return {"kind": kind, "reward": reward_name, "best": best, "candidates": results}
