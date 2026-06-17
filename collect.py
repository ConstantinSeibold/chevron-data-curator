"""Instance collection over a generic image folder + handcrafted shape-coordinate
features + the additive append protocol.

Wraps `qseg_playground.collect_instances` (model inference + decoder/maskpool/roialign/
backbone features + shape descriptors + keypoints) by registering an images-only
detectron2 split. Adds shape-COORDINATE features (PCA axes, radial signature, contour
Fourier) that describe the shape geometry beyond the centroid.
"""
from __future__ import annotations

import hashlib

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
    import qseg_playground as P  # noqa: lazy (notebooks/ must be on sys.path)
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
    bid = batch or ids.batch_id()
    for r in col["records"]:
        r["iuid"] = ids.new_uid()
        r["batch_id"] = bid
        r["abs_path"] = r.get("file_name", "")
    return col


def _raddino_by_path(col, P) -> dict:
    """RAD-DINO features for a generic folder: P.add_raddino_features loads images by
    record file_name; here file_name is already an abspath, so load by that directly."""
    import torch
    import torch.nn.functional as F
    import cv2
    ext = P.RadDinoExtractor("cuda" if torch.cuda.is_available() else "cpu")
    recs = col["records"]
    from collections import defaultdict
    by_img = defaultdict(list)
    for i, r in enumerate(recs):
        by_img[r["file_name"]].append(i)
    cdim = None
    for path, idxs in by_img.items():
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        grid = ext.grid(img)
        C, g, _ = grid.shape; cdim = C
        gf = grid.reshape(C, -1)
        masks = torch.stack([torch.from_numpy(P.decode_mask(recs[i])).float() for i in idxs])
        soft = F.interpolate(masks.unsqueeze(1), size=(g, g), mode="bilinear", align_corners=False).squeeze(1)
        sf = soft.reshape(len(idxs), -1).to(grid.device)
        denom = sf.sum(1, keepdim=True)
        pooled = (sf @ gf.t()) / denom.clamp_min(1e-6)
        for j, i in enumerate(idxs):
            recs[i]["f_raddino"] = pooled[j].detach().cpu().numpy().astype(np.float32)
    if cdim:
        col["feats"]["raddino"] = np.stack([r["f_raddino"] for r in recs]).astype(np.float32)
    return col


# --------------------------------------------------------------------------- #
# Additive concat (preserves the feats row == records order invariant)
# --------------------------------------------------------------------------- #
def concat_collections(master: dict | None, batch: dict) -> dict:
    """Append batch into master, vstacking each feats method. Refuses on method/column
    mismatch (feature config is fixed per project). Rewrites rec['row']/['inst_id']."""
    if master is None or not master.get("records"):
        out = {"records": list(batch["records"]), "feats": dict(batch["feats"]),
               "n_images": batch.get("n_images", 0)}
        _reindex(out)
        return out
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
