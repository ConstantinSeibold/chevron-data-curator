"""Anatomy-aware deterministic refinement of (pseudo-)label masks.

OPTIONAL postprocessing — gated by config (``train.ssl.refine.enable``) and selected
by PROFILE (``train.ssl.refine.profile``). Different tasks/datasets pick different
profiles (or ``none``); this is NOT applied to every SSL run.

The rule BODIES are lifted from the original MIMIC-anatomy pseudo-label pipeline
(``anatomy18_mimic.py``). That pipeline addressed a fixed ~184-channel tensor by
hardcoded position; here we instead dispatch by anatomical TYPE inferred from the
qseg category NAMES, so the same rules apply to our (different) contiguous label
space without reconstructing the original channel order.

Each class maps to ONE of five recipes — and the mapping matters: a blanket
largest-CC is WRONG for multi-piece structures (costal cartilage = one piece per
rib; vessels/trachea are tubular). The original is careful about this (median-only,
no CC, for cartilage / esophagus / trachea / vessels / lung zones), and so are we:

- ``compact``        median -> largest_cc -> fill     rigid single blobs
                     (vertebrae, sternum, single scapula/clavicle, organs, heart,
                      mediastinum parts, single lung field)
- ``bilateral``      fill -> closing -> top_k_cc(2) -> median   both-sides unions
                     (clavicles, scapulas, breast, diaphragm, lung union)
- ``rib_single``     largest_cc -> median(3)           single-side rib (one thin curved
                     bone): enforce the single CC, gentle non-eroding smooth (k=3 not 5)
- ``rib_union``      top_k_cc(2) -> median(3)          L/R rib union: keep both sides
- ``tubular``        median ONLY (no CC, no fill)      WIDE smooth tubular
                     (aorta, IVC/vena cava, esophagus, trachea, trunc)
- ``tubular_fine``   NOTHING (passthrough)             THIN / branching tubular
                     (pulmonary arteries, vessels, veins, costal cartilage): branches
                     + tips are ~1-3 px, so ANY median/morphology erodes them. Never
                     smooth, CC, or fill — this is the class (pulmonary artery) we
                     diagnosed as a thin-branch RECALL problem; smoothing makes it worse.
- ``lung_subregion`` median -> intersect lung field    zones / lobes (never CC-reduce)
- ``passthrough``    unchanged

Pure numpy/skimage/scipy — no detectron2, no mmdet, no cxas (the original's
``cxas.label_mapper`` import was dead code).

Vendored from qseg `src/qseg/ssl/refine_anatomy.py` — pure numpy/cv2/scipy morphology
(largest_cc / top_k_cc / fill / median / closing + the anatomy recipe dispatch). Chevron uses the
primitives; the paxray-specific profile tables come along unchanged and are simply unused by
non-anatomy projects.
"""
from __future__ import annotations

from typing import Callable, Iterable

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


# --- primitives (lifted from anatomy18_mimic; OpenCV-backed for speed) -------
# cv2.connectedComponentsWithStats / medianBlur / morphologyEx are optimized C
# and give the same result as the original skimage rank-filter / label calls.

def _cc(mask: np.ndarray):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    return n, lab, stats


def largest_cc(mask: np.ndarray) -> np.ndarray:
    n, lab, stats = _cc(mask)
    if n <= 2:
        return mask > 0
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def top_k_cc(mask: np.ndarray, k: int = 2) -> np.ndarray:
    n, lab, stats = _cc(mask)
    if n <= 1:
        return mask > 0
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:k] + 1
    out = np.zeros(mask.shape, dtype=bool)
    for u in order:
        out |= lab == u
    return out


def get_left_right(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, lab, stats = _cc(mask)
    if n <= 2:
        return mask > 0, np.zeros_like(mask, dtype=bool)
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:2] + 1
    a, b = int(order[0]), int(order[1])
    if np.nonzero(lab == a)[1].mean() > np.nonzero(lab == b)[1].mean():
        return lab == a, lab == b
    return lab == b, lab == a


def _median(mask: np.ndarray, k: int = 5) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if min(m.shape) < k:
        return m > 0
    return cv2.medianBlur(m * 255, k) > 127


def _fill(mask: np.ndarray) -> np.ndarray:
    return binary_fill_holes(mask > 0)


def _closing(mask: np.ndarray, k: int = 5) -> np.ndarray:
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, kern) > 0


def n_components(mask: np.ndarray) -> int:
    n, _, _ = _cc(mask)
    return max(0, n - 1)


# --- recipes ----------------------------------------------------------------

def _recipe_compact(m: np.ndarray, lung_field=None) -> np.ndarray:
    return _fill(_median(largest_cc(m), 5))


def _recipe_bilateral(m: np.ndarray, lung_field=None) -> np.ndarray:
    m = _fill(m)
    m = _closing(m, 5)
    return _median(top_k_cc(m, 2), 5)


def _recipe_tubular(m: np.ndarray, lung_field=None) -> np.ndarray:
    # WIDE smooth tubular (aorta/trachea/...): smooth ragged edges, never drop/fill.
    # Safe here because the structure is >>3 px wide so a 3x3 median barely touches it.
    return _median(m, 3)


def _recipe_tubular_fine(m: np.ndarray, lung_field=None) -> np.ndarray:
    # THIN / branching tubular (pulmonary arteries, vessels, cartilage): branches+tips
    # are ~1-3 px, so a median erodes them. Do NOTHING — pass the teacher mask through.
    return m > 0


def _recipe_rib_single(m: np.ndarray, lung_field=None) -> np.ndarray:
    # single-side rib = one connected curved bone. Enforce largest CC (drop spurious
    # teacher blobs / occlusion noise), then a SMALL median (k=3) to clean jagged edges
    # without eroding the ~3-6 px body (k=5 ate the thin ribs).
    return _median(largest_cc(m), 3)


def _recipe_rib_union(m: np.ndarray, lung_field=None) -> np.ndarray:
    # L/R rib union: keep top-2 CC (never collapse to a single side), then small median.
    return _median(top_k_cc(m, 2), 3)


def _recipe_lung_subregion(m: np.ndarray, lung_field=None) -> np.ndarray:
    m = _median(m, 5)
    if lung_field is not None and np.any(lung_field):
        m = m & (lung_field > 0)
    return m


def _recipe_passthrough(m: np.ndarray, lung_field=None) -> np.ndarray:
    return m > 0


RECIPES: dict[str, Callable] = {
    "compact": _recipe_compact,
    "bilateral": _recipe_bilateral,
    "tubular": _recipe_tubular,
    "tubular_fine": _recipe_tubular_fine,
    "rib_single": _recipe_rib_single,
    "rib_union": _recipe_rib_union,
    "lung_subregion": _recipe_lung_subregion,
    "passthrough": _recipe_passthrough,
}


# --- profile: paxray anatomy -------------------------------------------------

# thin / branching tubular -> NO smoothing (passthrough); wide smooth tubular -> light median
_TUBULAR_FINE_KEYS = ("cartilage", "artery", "pulmonary", "vessel", "vein")
_TUBULAR_SMOOTH_KEYS = ("aorta", "aortic", "trunc", "vena cava", "esophagus", "trachea")
_BILATERAL_IRREGULAR = {"clavicles", "scapulas", "breast", "breasts", "diaphragm"}
_LUNG_FIELD_NAMES = {
    "lung", "left lung", "right lung",
    "less obstructed lung", "less obstructed left lung", "less obstructed right lung",
}


def _has_side(name: str) -> bool:
    toks = name.lower().replace("_", " ").split()
    return ("left" in toks) or ("right" in toks)


def _sided_siblings(name: str, present: set[str]) -> list[str]:
    """The left/right sibling names of a sideless generic class that are PRESENT, e.g.
    "breast" -> ["breast left","breast right"], "humerus" -> ["left humerus","right humerus"]."""
    for lr in ((f"{name} left", f"{name} right"), (f"left {name}", f"right {name}")):
        sib = [s for s in lr if s in present]
        if sib:
            return sib
    return []


def _is_bilateral(name: str, all_names: set[str]) -> bool:
    if name in _BILATERAL_IRREGULAR:
        return True
    if f"{name} left" in all_names and f"{name} right" in all_names:
        return True
    if f"left {name}" in all_names and f"right {name}" in all_names:
        return True
    # (ribs no longer reach here — routed to tubular_fine/passthrough upstream, since
    #  median/cc would erode/fragment thin ribs.)
    n = name.lower()
    # lung-field union: covers both lungs, no side (e.g. "lung", "less obstructed lung")
    if "lung" in n and not _has_side(name) and not any(z in n for z in ("zone", "lobe", "base")):
        return True
    return False


def profile_paxray_anatomy(name: str, all_names: set[str]) -> str:
    """Map a qseg category name to a refinement recipe key."""
    n = name.strip().lower()
    # 1. lung sub-regions: smooth + containment, NEVER cc-reduce (keeps both sides)
    if "lung" in n and any(z in n for z in ("zone", "lobe", "base")):
        return "lung_subregion"
    # 1b. ribs: thin curved bones -> enforce connectivity + a SMALL non-eroding median.
    #     single-side = one CC -> largest_cc; L/R union (no side) -> top-2 CC. median k=3
    #     (NOT 5, which ate the ~3 px ribs). rib_cartilage is fine/branching -> tubular_fine.
    if "rib" in n and "cartilage" not in n:
        return "rib_single" if _has_side(name) else "rib_union"
    # 1d. humerus: long curved bones, like ribs -> connectivity + a SMALL non-eroding median,
    #     NOT compact's fill/closing (which over-grows a bone). single side = largest CC;
    #     merged sideless "humerus" = top-2 CC (both arms). Must precede the bilateral check
    #     (sideless "humerus" would otherwise route to the fill+closing "bilateral" recipe).
    if "humerus" in n:
        return "rib_single" if _has_side(name) else "rib_union"
    # 2a. thin/branching tubular: NEVER smooth (median erodes pulmonary-artery branches)
    if any(k in n for k in _TUBULAR_FINE_KEYS):
        return "tubular_fine"
    # 2b. wide smooth tubular: light edge median only (no cc, no fill)
    if any(k in n for k in _TUBULAR_SMOOTH_KEYS):
        return "tubular"
    # 3. bilateral unions: keep both sides
    if _is_bilateral(name, all_names):
        return "bilateral"
    # 4. rigid single blobs
    return "compact"


def profile_none(name: str, all_names: set[str]) -> str:
    return "passthrough"


PROFILES: dict[str, Callable[[str, set], str]] = {
    "paxray_anatomy": profile_paxray_anatomy,
    "none": profile_none,
}


def get_profile(name: str) -> Callable[[str, set], str]:
    if name not in PROFILES:
        raise KeyError(
            f"unknown refine profile {name!r}; available: {sorted(PROFILES)}"
        )
    return PROFILES[name]


# --- public API -------------------------------------------------------------

def recipe_for(name: str, all_names: Iterable[str], profile: str = "paxray_anatomy") -> str:
    return get_profile(profile)(name, set(all_names))


def refine_mask(
    mask: np.ndarray,
    name: str,
    all_names: Iterable[str],
    profile: str = "paxray_anatomy",
    lung_field: np.ndarray | None = None,
) -> np.ndarray:
    m = mask > 0
    if not m.any():
        return m
    recipe = get_profile(profile)(name, set(all_names))
    if recipe == "passthrough":
        return m
    # Crop to the instance bbox (+pad for the footprint) so morphology runs on the
    # structure's box, not the full frame — the dominant speedup for small classes.
    ys, xs = np.nonzero(m)
    pad = 8
    y0, y1 = max(0, int(ys.min()) - pad), min(m.shape[0], int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(m.shape[1], int(xs.max()) + 1 + pad)
    sub = m[y0:y1, x0:x1]
    lf_sub = lung_field[y0:y1, x0:x1] if (recipe == "lung_subregion" and lung_field is not None) else None
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = RECIPES[recipe](sub, lung_field=lf_sub)
    return out


def refine_instances(
    instances: list[dict],
    cat_id_to_name: dict[int, str],
    all_names: Iterable[str],
    decode_fn,
    profile: str = "paxray_anatomy",
) -> list[dict]:
    """Singleton-collapse per class (argmax score) then refine each mask by recipe.

    ``instances`` are COCO-style dicts with ``category_id`` and ``segmentation``;
    ``decode_fn(seg) -> HxW uint8`` decodes the mask. Returns new dicts with a
    refined boolean ``mask`` field (RLE re-encoding left to the caller).
    """
    if profile == "none":
        return [{**ins, "mask": decode_fn(ins["segmentation"]) > 0} for ins in instances]

    all_names = set(all_names)
    best: dict[int, dict] = {}
    for ins in instances:
        cid = ins["category_id"]
        if cid not in best or ins.get("score", 1.0) > best[cid].get("score", 1.0):
            best[cid] = ins
    decoded = {cid: decode_fn(ins["segmentation"]) for cid, ins in best.items()}

    lung_field = None
    for cid, m in decoded.items():
        if cat_id_to_name.get(cid) in _LUNG_FIELD_NAMES:
            lf = refine_mask(m, cat_id_to_name[cid], all_names, profile)
            lung_field = lf if lung_field is None else (lung_field | lf)

    out = []
    for cid, ins in best.items():
        name = cat_id_to_name.get(cid, str(cid))
        refined = refine_mask(decoded[cid], name, all_names, profile, lung_field=lung_field)
        if refined.sum() == 0:
            continue
        out.append({**ins, "mask": refined})

    # Consistency pass: a sideless "generic" class (breast=174, merged humerus=180) is by
    # definition the UNION of its left/right siblings, but the decoder predicts all three
    # independently -> they can contradict. When the siblings are present, set the generic
    # mask = OR(left, right) so it exactly matches its parts (the same definition used to
    # build the GT). No-op when no sibling is present (keeps the raw generic prediction), so
    # non-paired classes are untouched. Dispatches on NAMES -> cannot affect other classes.
    by_name = {cat_id_to_name.get(o["category_id"], str(o["category_id"])): o for o in out}
    for o in out:
        nm = cat_id_to_name.get(o["category_id"], str(o["category_id"]))
        if _has_side(nm):
            continue
        sibs = _sided_siblings(nm, set(by_name) - {nm})
        if not sibs:
            continue
        u = np.zeros_like(o["mask"])
        for s in sibs:
            u = u | by_name[s]["mask"]
        o["mask"] = u
    return out
