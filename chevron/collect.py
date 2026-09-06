"""Instance collection over a generic image folder + handcrafted shape-coordinate
features + the additive append protocol.

Wraps the qseg backend's `collect_instances` (model inference + decoder/maskpool/roialign/
backbone features + shape descriptors + keypoints) by registering an images-only
detectron2 split. Adds shape-COORDINATE features (PCA axes, radial signature, contour
Fourier) that describe the shape geometry beyond the centroid.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from . import ids

# --------------------------------------------------------------------------- #
# Shape-coordinate features (pure numpy/cv2 — unit-testable without a model)
# --------------------------------------------------------------------------- #
def _largest_contour(mask: np.ndarray):
    import cv2
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


def pca_axes(mask: np.ndarray) -> np.ndarray:
    """(5,): [major_len/sqrt(area), minor_len/sqrt(area), minor/major ratio, sin(2θ), cos(2θ)].
    Describes the principal-axis geometry; 2θ handles the axis 180° ambiguity."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return np.zeros(5, np.float32)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    pts -= pts.mean(0, keepdims=True)
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0, None)
    minor_len, major_len = 2.0 * np.sqrt(evals[0]), 2.0 * np.sqrt(evals[1])
    v = evecs[:, 1]                                   # major axis eigenvector
    ang = np.arctan2(v[1], v[0])
    s = float(np.sqrt(len(xs))) + 1e-6
    return np.array([major_len / s, minor_len / s,
                     (minor_len / (major_len + 1e-6)),
                     np.sin(2 * ang), np.cos(2 * ang)], np.float32)


def radial_signature(mask: np.ndarray, n: int = 16) -> np.ndarray:
    """(n,): max centroid->boundary distance in n angular bins, normalized by the mean.
    A rotation-anchored outline signature."""
    cont = _largest_contour(mask)
    if cont is None or len(cont) < 3:
        return np.zeros(n, np.float32)
    c = cont.mean(0)
    d = cont - c
    r = np.hypot(d[:, 0], d[:, 1])
    th = (np.arctan2(d[:, 1], d[:, 0]) + 2 * np.pi) % (2 * np.pi)
    bins = np.minimum((th / (2 * np.pi) * n).astype(int), n - 1)
    sig = np.zeros(n, np.float64)
    for b in range(n):
        sel = r[bins == b]
        sig[b] = sel.max() if len(sel) else 0.0
    m = sig[sig > 0].mean() if np.any(sig > 0) else 1.0
    return (sig / (m + 1e-6)).astype(np.float32)


def contour_fourier(mask: np.ndarray, n: int = 8, resample: int = 128) -> np.ndarray:
    """(n,): magnitude of the first n FFT harmonics of the centroid-distance-vs-arclength
    signal, normalized by the DC term. Rotation/start-point invariant, scale invariant."""
    cont = _largest_contour(mask)
    if cont is None or len(cont) < 4:
        return np.zeros(n, np.float32)
    c = cont.mean(0)
    r = np.hypot(cont[:, 0] - c[0], cont[:, 1] - c[1])
    idx = np.linspace(0, len(r) - 1, resample)
    rr = np.interp(idx, np.arange(len(r)), r)
    f = np.abs(np.fft.rfft(rr))
    dc = f[0] if f[0] > 1e-6 else 1.0
    out = f[1:n + 1] / dc
    if len(out) < n:
        out = np.concatenate([out, np.zeros(n - len(out))])
    return out.astype(np.float32)


_SHAPECOORD_COLS = (
    ["pca_major", "pca_minor", "pca_ratio", "pca_sin2a", "pca_cos2a"]
    + [f"radial{i}" for i in range(16)]
    + [f"fourier{i}" for i in range(8)]
)


def shapecoord_vector(mask: np.ndarray) -> np.ndarray:
    return np.concatenate([pca_axes(mask), radial_signature(mask, 16), contour_fourier(mask, 8)]).astype(np.float32)


def attach_shapecoord(collection: dict) -> dict:
    """Add per-instance f_shapecoord (+ feats['shapecoord'] + _shapecoord_cols)."""
    from pycocotools import mask as mask_util
    recs = collection["records"]
    if not recs:
        return collection
    for r in recs:
        if "f_shapecoord" not in r:
            m = mask_util.decode(r["rle"]).astype(bool)
            r["f_shapecoord"] = shapecoord_vector(m)
    collection["feats"]["shapecoord"] = np.stack([r["f_shapecoord"] for r in recs]).astype(np.float32)
    collection["feats"]["_shapecoord_cols"] = list(_SHAPECOORD_COLS)
    return collection


# --------------------------------------------------------------------------- #
# images-only detectron2 split + collection over a generic folder
# --------------------------------------------------------------------------- #
def path_image_id(path: str) -> int:
    """Stable per-file id (cross-batch grouping + de-dup)."""
    return int(hashlib.blake2b(str(path).encode(), digest_size=7).hexdigest(), 16)


def register_images_split(name: str, file_list: list[str], num_classes: int) -> str:
    """Register an images-only split (dicts carry file_name/image_id/height/width)."""
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from PIL import Image

    def _loader():
        dicts = []
        for p in file_list:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            dicts.append({"file_name": str(p), "image_id": path_image_id(p),
                          "height": int(h), "width": int(w), "annotations": []})
        return dicts

    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        try:
            MetadataCatalog.remove(name)
        except Exception:
            pass
    DatasetCatalog.register(name, _loader)
    MetadataCatalog.get(name).set(thing_classes=[str(i) for i in range(num_classes)],
                                  evaluator_type="coco")
    return name


def collect_batch(model, cfg, d2_cfg, file_list, *, score_thresh: float, feature_cfg: dict,
                  batch=None) -> dict:
    """Run the model over file_list -> collection, tagging each record with a fresh iuid +
    batch_id + abspath. feature_cfg: {with_features, with_shape, with_backbone, backbone_level,
    shapecoord:bool, raddino:bool}."""
    from ._bootstrap import get_P
    P = get_P()
    split = "curator"
    register_images_split(f"{cfg.data.name}_{split}", list(file_list), int(cfg.data.num_classes))
    col = P.collect_instances(
        model, cfg, d2_cfg, split=split, score_thresh=float(score_thresh),
        with_features=bool(feature_cfg.get("with_features", True)),
        with_shape=bool(feature_cfg.get("with_shape", True)),
        with_backbone=bool(feature_cfg.get("with_backbone", True)),
        backbone_level=feature_cfg.get("backbone_level", "p16"), verbose=False,
    )
    if feature_cfg.get("shapecoord", True):
        attach_shapecoord(col)
    if feature_cfg.get("raddino", False):
        _raddino_by_path(col, P)
    if float(feature_cfg.get("nms_iou", 0.0)) > 0:
        mask_nms(col, float(feature_cfg["nms_iou"]))
    bid = batch or ids.batch_id()
    for r in col["records"]:
        r["iuid"] = ids.new_uid()
        r["batch_id"] = bid
        r["abs_path"] = r.get("file_name", "")
    return col


def _bbox_grid_cells(rec, g):
    """Grid-cell slice (gy1,gy2,gx1,gx2) for an instance's bbox, mapped linearly into the gxg patch grid
    (same stretch assumption as the mask interpolate). box_xyxy if present, else cx/cy/bw/bh."""
    H, W = float(rec["H"]), float(rec["W"])
    bx = rec.get("box_xyxy")
    if bx is None:
        cx, cy, bw, bh = rec["cx"], rec["cy"], rec["bw"], rec["bh"]
        bx = [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]
    gx1 = max(0, int(bx[0] / W * g)); gx2 = min(g, int(np.ceil(bx[2] / W * g)))
    gy1 = max(0, int(bx[1] / H * g)); gy2 = min(g, int(np.ceil(bx[3] / H * g)))
    return gy1, max(gy1 + 1, gy2), gx1, max(gx1 + 1, gx2)


def pool_by_path(col, ext, key: str, progress=None, pool="mask") -> dict:
    """Pool ANY extractor's patch grids into per-instance features under `col["feats"][key]`.

    Generalised from the RAD-DINO path: the encoder differs, the pooling does not. `ext` only has to
    provide `grid_batch(images) -> (B, C, g, g)`.

    `progress(done, total)` is called per image (for a UI progress bar).
    `pool`: 'mask' (soft mask-pool — precise, decodes each RLE) | 'bbox' (max-pool the patches inside the
    instance's bbox — SKIPS the per-instance RLE decode, the ~1.7 ms/inst cost; for scale).

    Pooling with an all-ones mask instead of the instance mask is what gives a whole-image SAMPLE
    embedding from this same code — see chevron.extractors.base.
    """
    import torch
    import torch.nn.functional as F
    import cv2
    from .core.collection import decode_mask
    recs = col["records"]
    from collections import defaultdict
    by_img = defaultdict(list)
    for i, r in enumerate(recs):
        by_img[r["file_name"]].append(i)
    cdim = None
    items = list(by_img.items())
    n_img = len(items)
    B = int(os.environ.get("CURATOR_EXTRACT_BATCH", os.environ.get("CURATOR_RADDINO_BATCH", "8")))   # images per forward
    done = 0
    # parallel image DECODE within each chunk (cv2.imread releases the GIL, so threads overlap disk+decode).
    # Bounded to B images in flight -> no RAM blowup (vs pre-loading the whole list). ~zero gain when the page
    # cache is warm, but a real win on a COLD first run over many images. (Measured: I/O << compute when warm,
    # so a full prefetch pipeline isn't worth it; this is the cheap, RAM-safe half.)
    from concurrent.futures import ThreadPoolExecutor
    tpool = ThreadPoolExecutor(max_workers=min(int(os.environ.get("CURATOR_IMG_WORKERS", "8")), max(1, B)))
    _read = lambda pi: cv2.cvtColor(cv2.imread(pi[0]), cv2.COLOR_BGR2RGB)
    try:
        for s in range(0, n_img, B):
            chunk = items[s:s + B]
            imgs = list(tpool.map(_read, chunk))                 # decode this chunk in parallel
            grids = ext.grid_batch(imgs)                         # (b, C, g, g) in ONE forward
            for k, (path, idxs) in enumerate(chunk):
                grid = grids[k]
                C, g, _ = grid.shape; cdim = C
                if pool == "bbox":                               # max-pool patches in the bbox, NO RLE decode
                    for i in idxs:
                        gy1, gy2, gx1, gx2 = _bbox_grid_cells(recs[i], g)
                        recs[i][f"f_{key}"] = grid[:, gy1:gy2, gx1:gx2].reshape(C, -1).amax(1) \
                            .detach().cpu().numpy().astype(np.float32)
                    continue
                gf = grid.reshape(C, -1)
                masks = torch.stack([torch.from_numpy(decode_mask(recs[i])).float() for i in idxs])
                soft = F.interpolate(masks.unsqueeze(1), size=(g, g), mode="bilinear", align_corners=False).squeeze(1)
                sf = soft.reshape(len(idxs), -1).to(grid.device)
                pooled = (sf @ gf.t()) / sf.sum(1, keepdim=True).clamp_min(1e-6)
                for j, i in enumerate(idxs):
                    recs[i][f"f_{key}"] = pooled[j].detach().cpu().numpy().astype(np.float32)
            done += len(chunk)
            if progress:
                progress(done, n_img)
    finally:
        tpool.shutdown(wait=True)
    if cdim:
        # a shared-space extractor (CLIP/SigLIP) maps the pooled region into the image-text space,
        # so a text query and an instance are comparable — see ProjectedVisionExtractor
        proj = getattr(ext, "project_pooled", None)
        if proj is not None:
            stacked = torch.from_numpy(np.stack([r[f"f_{key}"] for r in recs]))
            for i, v in enumerate(proj(stacked).detach().cpu().numpy().astype(np.float32)):
                recs[i][f"f_{key}"] = v
        col["feats"][key] = np.stack([r[f"f_{key}"] for r in recs]).astype(np.float32)
    return col


def _raddino_by_path(col, P, progress=None, pool="mask") -> dict:
    """Back-compat shim: the original RAD-DINO-only entry point."""
    import torch
    return pool_by_path(col, P.RadDinoExtractor("cuda" if torch.cuda.is_available() else "cpu"),
                        "raddino", progress=progress, pool=pool)


# --------------------------------------------------------------------------- #
# Additive concat (preserves the feats row == records order invariant)
# --------------------------------------------------------------------------- #
def mask_nms(collection: dict, iou_thresh: float = 0.8) -> dict:
    """Per-image mask-IoU NMS: keep highest-score instances, drop those overlapping a kept
    one by >= iou_thresh. Filters records AND every feats row in lockstep (batch-local, so
    dropping rows is safe pre-concat). Returns the same dict, filtered in place."""
    from collections import defaultdict
    from pycocotools import mask as mu
    recs = collection["records"]
    if not recs or iou_thresh <= 0:
        return collection
    by_img = defaultdict(list)
    for i, r in enumerate(recs):
        by_img[r["image_id"]].append(i)
    keep = np.ones(len(recs), bool)
    for idxs in by_img.values():
        order = sorted(idxs, key=lambda i: -float(recs[i]["score"]))
        kept_rles = []
        for i in order:
            rle = recs[i]["rle"]
            if kept_rles:
                ious = mu.iou([rle], kept_rles, [0] * len(kept_rles))  # [1, K]
                if float(np.max(ious)) >= iou_thresh:
                    keep[i] = False
                    continue
            kept_rles.append(rle)
    if keep.all():
        return collection
    sel = np.where(keep)[0]
    collection["records"] = [recs[i] for i in sel]
    for k in list(collection["feats"].keys()):
        if not k.startswith("_"):
            collection["feats"][k] = collection["feats"][k][sel]
    _reindex(collection)
    return collection


def concat_collections(master: dict | None, batch: dict) -> dict:
    """Append batch into master, vstacking each feats method. Refuses on method/column
    mismatch (feature config is fixed per project). Rewrites rec['row']/['inst_id']."""
    if master is None or not master.get("records"):
        out = {"records": list(batch["records"]), "feats": dict(batch["feats"]),
               "n_images": batch.get("n_images", 0)}
        _reindex(out)
        return out
    if not batch.get("records"):
        return master                                 # a 0-detection chunk carries no feats methods; the
                                                      # method-set check below would spuriously reject it
    mf, bf = master["feats"], batch["feats"]
    m_methods = {k for k in mf if not k.startswith("_")}
    b_methods = {k for k in bf if not k.startswith("_")}
    if m_methods != b_methods:
        raise ValueError(f"feature-method mismatch on append: master={sorted(m_methods)} batch={sorted(b_methods)}")
    out_feats = {}
    for k in m_methods:
        if mf[k].shape[1] != bf[k].shape[1]:
            raise ValueError(f"feature-dim mismatch for '{k}': {mf[k].shape[1]} vs {bf[k].shape[1]}")
        out_feats[k] = np.vstack([mf[k], bf[k]])
    for k in mf:                                      # carry column-name lists (_shape_cols etc.)
        if k.startswith("_"):
            out_feats[k] = mf[k]
    out = {"records": list(master["records"]) + list(batch["records"]), "feats": out_feats,
           "n_images": master.get("n_images", 0) + batch.get("n_images", 0)}
    _reindex(out)
    return out


def _reindex(col: dict) -> None:
    for i, r in enumerate(col["records"]):
        r["row"] = i
        r["inst_id"] = i


def subset_collection(col: dict, idx) -> dict:
    """A collection holding only the records at row indices `idx` (each feats matrix row-sliced to
    match, `_`-prefixed column-name lists carried verbatim). Used to drop already-present rows when
    folding recovered ingest shards back in (de-dup by iuid)."""
    idx = list(idx)
    recs = [col["records"][i] for i in idx]
    feats = {k: (v if k.startswith("_") else v[idx]) for k, v in col["feats"].items()}
    out = {"records": recs, "feats": feats, "n_images": col.get("n_images", 0)}
    _reindex(out)
    return out
