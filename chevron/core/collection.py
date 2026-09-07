"""Generic instance-collection machinery — vendored from qseg's `notebooks/qseg_playground.py`.

These functions are model- and dataset-agnostic (numpy / cv2 / sklearn only): shape descriptors,
feature stacking + fusion, clustering (incl. FINCH), 2D embedding, pair features for the merge
recommender, in-image merge grouping, overlays, and collection (de)serialisation.

They were extracted VERBATIM so behaviour is bit-identical to the qseg curator; the qseg-coupled
parts (model loading, `collect_instances`, synth generation, GT matching) deliberately stayed
behind and live in `chevron.backends.qseg`.

A `collection` is `{"records": [...], "feats": {method: (N, D) matrix}, "n_images": int}` with the
row-alignment invariant: `records[i]` corresponds to row `i` of every `feats` matrix.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

def shape_descriptors(mask: np.ndarray) -> dict[str, float]:
    """Geometry/topology descriptors for a single binary mask (HxW, bool/0-1).
    Pure cv2+numpy (+ optional skimage for skeleton length)."""
    import cv2
    m = (mask > 0).astype(np.uint8)
    out = dict(area=0.0, perimeter=0.0, solidity=0.0, extent=0.0, aspect=0.0,
               elongation=0.0, orientation=0.0, eccentricity=0.0, n_cc=0.0,
               fill=0.0, skel_len=0.0, tortuosity=0.0)
    out.update({f"hu{i}": 0.0 for i in range(7)})
    area = float(m.sum())
    if area < 1:
        return out
    n_cc, _ = cv2.connectedComponents(m)
    out["n_cc"] = float(max(n_cc - 1, 1))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:                                           # area>=1 but no external contour -> keep zeros
        return out
    peri = float(sum(cv2.arcLength(c, True) for c in cnts))
    out["area"], out["perimeter"] = area, peri
    big = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(big)
    ha = float(cv2.contourArea(hull))
    out["solidity"] = area / ha if ha > 0 else 0.0
    x, y, w, h = cv2.boundingRect(big)
    out["extent"] = area / float(w * h) if w * h > 0 else 0.0
    out["aspect"] = float(max(w, h)) / float(max(min(w, h), 1))
    out["fill"] = 2.0 * area / max(peri, 1.0)              # ~mean width of a ribbon
    mom = cv2.moments(m, binaryImage=True)
    hu = cv2.HuMoments(mom).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-30)        # log-scaled Hu moments
    for i in range(7):
        out[f"hu{i}"] = float(hu[i])
    if len(big) >= 5:
        try:
            (_, _), (MA, ma), ang = cv2.fitEllipse(big)    # degenerate/collinear points -> NaN axes (or raises)
        except Exception:
            MA = ma = ang = 0.0
        if np.isfinite(MA) and np.isfinite(ma) and np.isfinite(ang) and min(MA, ma) > 0:
            out["orientation"] = float(ang)
            out["elongation"] = float(max(MA, ma)) / float(max(min(MA, ma), 1e-6))
            a, b = max(MA, ma) / 2.0, min(MA, ma) / 2.0
            out["eccentricity"] = float(np.sqrt(max(1.0 - (b * b) / (a * a + 1e-9), 0.0)))
    try:
        from skimage.morphology import skeletonize
        sk = skeletonize(m > 0)
        sl = float(sk.sum())
        out["skel_len"] = sl
        # tortuosity = skeleton length / endpoint chord (per-instance shape complexity)
        ys, xs = np.where(sk)
        if len(xs) > 1:
            chord = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
            out["tortuosity"] = sl / max(chord, 1.0)
    except Exception:
        pass
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}   # never store NaN/inf


def _coord_feats(box_xyxy, mask, H, W):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx, cy = (x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H
    bw, bh = (x2 - x1) / W, (y2 - y1) / H
    return dict(cx=cx, cy=cy, bw=bw, bh=bh, box_area=bw * bh,
                mask_area_frac=float((mask > 0).sum()) / float(H * W))


# --------------------------------------------------------------------------- #
# Collection: run inference over a split, build per-instance records
# --------------------------------------------------------------------------- #

def _stack_feats(records: list[dict]) -> dict[str, np.ndarray]:
    out = {}
    if not records:
        return out
    for key in ("f_decoder", "f_maskpool", "f_roialign", "f_backbone"):
        if key in records[0]:
            out[key.replace("f_", "")] = np.stack([r[key] for r in records]).astype(np.float32)
    if "shape" in records[0]:
        cols = list(records[0]["shape"].keys())
        S = np.array([[r["shape"][c] for c in cols] for r in records], dtype=np.float32)
        out["shape"] = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)   # never store NaN/inf shape feats
        out["_shape_cols"] = cols  # names alongside the matrix
    out["coords"] = np.array([[r["cx"], r["cy"], r["bw"], r["bh"], r["box_area"], r["mask_area_frac"]]
                              for r in records], dtype=np.float32)
    return out


# --------------------------------------------------------------------------- #
# Clustering + coordinate biasing
# --------------------------------------------------------------------------- #
def build_matrix(collection: dict, method: str = "decoder", *, standardize: bool = True,
                 coord_bias: float = 0.0, l2norm: bool = False) -> np.ndarray:
    """Assemble a clustering matrix from a feature method, optionally appending
    coordinate features scaled by `coord_bias` (0 = appearance only; large =
    spatial dominates). Standardize per-column by default."""
    feats = collection["feats"]
    if method not in feats:
        raise KeyError(f"method '{method}' not in {[k for k in feats if not k.startswith('_')]}")
    X = feats[method].copy()
    if l2norm:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    if standardize:
        X = _zscore(X)
    if coord_bias > 0:
        C = _zscore(feats["coords"].copy()) * float(coord_bias)
        X = np.concatenate([X, C], axis=1)
    return X.astype(np.float32)


def _zscore(X):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    return (X - mu) / (sd + 1e-9)


def cluster(X: np.ndarray, algo: str = "kmeans", **kw):
    """Cluster rows of X. algo in {kmeans, agglomerative, hdbscan, dbscan}.
    Returns integer labels (-1 = noise for density algos)."""
    if algo == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=int(kw.get("k", 8)), n_init=10, random_state=0).fit_predict(X)
    if algo == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=int(kw.get("k", 8)),
                                       linkage=kw.get("linkage", "ward")).fit_predict(X)
    if algo == "dbscan":
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=float(kw.get("eps", 3.0)), min_samples=int(kw.get("min_samples", 4))).fit_predict(X)
    if algo == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN
        except Exception:
            import hdbscan
            return hdbscan.HDBSCAN(min_cluster_size=int(kw.get("min_cluster_size", 5))).fit_predict(X)
        return HDBSCAN(min_cluster_size=int(kw.get("min_cluster_size", 5))).fit_predict(X)
    if algo == "finch":
        # FINCH (Sarfraz et al., CVPR'19): parameter-free hierarchical clustering — no k.
        # Pass req_clust=<n> to force a count; else pick a hierarchy level via
        # partition=<idx> or the level whose count is nearest to k; default finest (col 0).
        from finch import FINCH
        req = kw.get("req_clust", None)
        Xc, dist = np.ascontiguousarray(X), kw.get("distance", "cosine")
        try:
            c, num_clust, req_c = FINCH(Xc, req_clust=req, distance=dist, verbose=False)
        except UnboundLocalError:
            # finch-clust <=0.2.2 bug: when req_clust is ALREADY one of the hierarchy's own counts,
            # the `if req_clust not in num_clust` branch never assigns `requested_c` and the function
            # dies on its own return statement. That case needs no refinement at all — the level we
            # want is a column of `c` — so re-run without req_clust and pick it out below.
            c, num_clust, req_c = FINCH(Xc, distance=dist, verbose=False)
        if req is not None and req_c is not None:
            return np.asarray(req_c)
        if c.ndim == 1:
            return c
        if req is not None and req in list(num_clust):
            # the exact count is a level of the hierarchy (the path the bug above lands on)
            return c[:, list(num_clust).index(req)]
        p = kw.get("partition", None)
        if p is None:
            p = int(np.argmin([abs(n - kw["k"]) for n in num_clust])) if kw.get("k") else 0
        return c[:, int(p)]
    raise ValueError(f"unknown algo {algo}")


def finch_hierarchy(X: np.ndarray, distance: str = "cosine"):
    """Parameter-free FINCH. Returns (partitions [N, P], cluster_counts [P]) — one
    column per hierarchy level, coarser to the right. Pick a level for `cluster`'s
    `partition=` arg, or call `cluster(X, 'finch', req_clust=n)` for an exact count."""
    from finch import FINCH
    c, num_clust, _ = FINCH(np.ascontiguousarray(X), distance=distance, verbose=False)
    if c.ndim == 1:
        c = c[:, None]
    return c, list(num_clust)


def cluster_per_image(collection: dict, method: str = "decoder", *, algo: str = "agglomerative",
                      coord_bias: float = 1.0, **kw) -> np.ndarray:
    """Cluster instances WITHIN each image independently (the natural setting for
    'which predicted fragments belong to the same line'). Returns a global label
    array where labels are namespaced per-image (image_id*1000 + local label)."""
    records = collection["records"]
    img_ids = np.array([r["image_id"] for r in records])
    labels = np.full(len(records), -1, dtype=np.int64)
    for iid in np.unique(img_ids):
        sel = np.where(img_ids == iid)[0]
        if len(sel) == 1:
            labels[sel] = iid * 1000
            continue
        sub = dict(feats={k: (v[sel] if isinstance(v, np.ndarray) and v.ndim == 2 else v)
                           for k, v in collection["feats"].items()},
                   records=[records[i] for i in sel])
        X = build_matrix(sub, method=method, coord_bias=coord_bias)
        k = int(kw.pop("k", min(max(2, len(sel) // 2), len(sel))))
        try:
            lab = cluster(X, algo=algo, k=k, **kw)
        except Exception:
            lab = np.zeros(len(sel), dtype=np.int64)
        labels[sel] = iid * 1000 + lab
    return labels


def embed2d(X: np.ndarray, method: str = "umap", **kw):
    """2D embedding for scatter plots. method in {umap, tsne, pca}."""
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=float(kw.get("perplexity", 30)),
                    init="pca", random_state=0).fit_transform(X)
    try:
        import umap
        return umap.UMAP(n_neighbors=int(kw.get("n_neighbors", 15)),
                         min_dist=float(kw.get("min_dist", 0.1)), random_state=0).fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA
        print("[embed2d] umap unavailable -> PCA fallback")
        return PCA(n_components=2).fit_transform(X)


# --------------------------------------------------------------------------- #
# Feature fusion + pairwise (merge / same-type) features
# --------------------------------------------------------------------------- #
def fuse_features(collection: dict, spec, *, standardize: bool = True,
                  l2_per_block: bool = True) -> tuple[np.ndarray, list]:
    """Concatenate several feature methods into ONE matrix for clustering / classification.
    `spec` = list of method names (equal weight) or dict {method: weight}; methods are any
    of feats keys (decoder/maskpool/backbone/roialign/raddino/shape/coords). Each block is
    L2-normalized per row then z-scored then scaled by its weight. Returns (X, blocks) where
    blocks = [(method, slice, dim), ...] so you can inspect each block's contribution.
    This generalizes build_matrix (one method + coords) to arbitrary fusion."""
    feats = collection["feats"]
    if isinstance(spec, (list, tuple)):
        spec = {m: 1.0 for m in spec}
    mats, blocks, off = [], [], 0
    for m, w in spec.items():
        if m not in feats:
            print(f"[fuse] skip '{m}' (absent; have {[k for k in feats if not k.startswith('_')]})")
            continue
        B = feats[m].astype(np.float32).copy()
        if l2_per_block:
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        if standardize:
            B = _zscore(B)
        B = B * float(w)
        mats.append(B); blocks.append((m, slice(off, off + B.shape[1]), B.shape[1])); off += B.shape[1]
    X = np.concatenate(mats, axis=1).astype(np.float32)
    return X, blocks


def cluster_per_image_X(collection: dict, X: np.ndarray, *, algo: str = "agglomerative", **kw) -> np.ndarray:
    """Cluster a PRECOMPUTED (N,D) matrix (e.g. from fuse_features) within each image —
    so fused features can drive the fragmentation/merge view. Labels namespaced per image."""
    recs = collection["records"]
    img = np.array([r["image_id"] for r in recs])
    labels = np.full(len(recs), -1, dtype=np.int64)
    for iid in np.unique(img):
        sel = np.where(img == iid)[0]
        if len(sel) == 1:
            labels[sel] = iid * 1000; continue
        k = int(kw.get("k", min(max(2, len(sel) // 2), len(sel))))
        try:
            lab = cluster(X[sel], algo=algo, **{**kw, "k": k})
        except Exception:
            lab = np.zeros(len(sel), dtype=np.int64)
        labels[sel] = iid * 1000 + lab
    return labels



def candidate_pairs(collection: dict, *, within_image: bool = True,
                    max_pairs: int | None = 200000) -> np.ndarray:
    """Candidate instance-index pairs. within_image=True -> only same-image pairs (the
    merge setting); False -> all pairs (capped at max_pairs, subsampled if exceeded)."""
    import itertools
    recs = collection["records"]
    if within_image:
        from collections import defaultdict
        by = defaultdict(list)
        for i, r in enumerate(recs):
            by[r["image_id"]].append(i)
        pairs = [p for idxs in by.values() for p in itertools.combinations(idxs, 2)]
    else:
        pairs = list(itertools.combinations(range(len(recs)), 2))
    pairs = np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), np.int64)
    if max_pairs and len(pairs) > max_pairs:
        rng = np.random.default_rng(0)
        pairs = pairs[rng.choice(len(pairs), max_pairs, replace=False)]
    return pairs


def pair_features(collection: dict, pairs: np.ndarray, *, methods=("decoder",)) -> tuple[np.ndarray, list]:
    """Vectorized per-PAIR features for a merge / same-type classifier. For each chosen
    encoder: cosine similarity + L2 distance of the two instances. Plus geometry:
    centroid distance, bbox IoU, bbox gap, class agreement, score min/mean, orientation
    diff, elongation & area ratios. Returns (Xp [P,D], feature_names)."""
    feats = collection["feats"]; recs = collection["records"]
    if len(pairs) == 0:
        return np.zeros((0, 0), np.float32), []
    I, J = pairs[:, 0], pairs[:, 1]
    cols, names = [], []
    for m in methods:
        if m not in feats:
            continue
        R = feats[m].astype(np.float32)
        Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
        cols.append((Rn[I] * Rn[J]).sum(1)); names.append(f"{m}_cos")
        cols.append(np.linalg.norm(R[I] - R[J], axis=1)); names.append(f"{m}_l2")
    # geometry
    cx = np.array([r["cx"] for r in recs]); cy = np.array([r["cy"] for r in recs])
    bx = np.stack([r["box_xyxy"] for r in recs]).astype(np.float32)
    sc = np.array([r["score"] for r in recs]); pc = np.array([r["pred_class"] for r in recs])
    HW = np.array([(r["H"] + r["W"]) / 2.0 for r in recs])
    a, b = bx[I], bx[J]
    x1 = np.maximum(a[:, 0], b[:, 0]); y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2]); y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / (aa + ab - inter + 1e-6)
    gapx = np.clip(np.maximum(b[:, 0] - a[:, 2], a[:, 0] - b[:, 2]), 0, None)
    gapy = np.clip(np.maximum(b[:, 1] - a[:, 3], a[:, 1] - b[:, 3]), 0, None)
    gap = np.hypot(gapx, gapy) / HW[I]
    cols += [np.hypot(cx[I] - cx[J], cy[I] - cy[J]), iou, gap,
             (pc[I] == pc[J]).astype(np.float32), np.minimum(sc[I], sc[J]), (sc[I] + sc[J]) / 2.0]
    names += ["centroid_dist", "bbox_iou", "bbox_gap", "class_agree", "score_min", "score_mean"]
    if "shape" in feats and "_shape_cols" in feats:
        scols = feats["_shape_cols"]; S = feats["shape"]
        gi = {c: k for k, c in enumerate(scols)}
        if "orientation" in gi:
            d = np.abs(S[I, gi["orientation"]] - S[J, gi["orientation"]]) % 180.0
            cols.append(np.minimum(d, 180.0 - d)); names.append("orient_diff")
        for c in ("elongation", "area"):
            if c in gi:
                v = S[:, gi[c]]
                # CLIP the ratio: a tiny/degenerate mask gives min~0 -> ratio ~1e6, which dominates the scaler.
                cols.append(np.clip(np.maximum(v[I], v[J]) / (np.minimum(v[I], v[J]) + 1e-6), 1.0, 50.0))
                names.append(f"{c}_ratio")
        if "orientation" in gi and "elongation" in gi:
            # COLLINEARITY (mask-free, the decisive LINE-fragment signal): do both fragments' major axes align
            # with the line joining their centroids? |cos| (axes are undirected). Gated by elongation so round
            # blobs (whose ellipse angle is noise) read ~0 -> only genuinely elongated fragments use it.
            o = np.deg2rad(S[:, gi["orientation"]].astype(np.float32))
            thc = np.arctan2(cy[J] - cy[I], cx[J] - cx[I])
            align = np.abs(np.cos(o[I] - thc)) * np.abs(np.cos(o[J] - thc))
            el = S[:, gi["elongation"]]
            elong = np.clip((np.minimum(el[I], el[J]) - 1.5) / 1.5, 0.0, 1.0)
            cols.append((align * elong).astype(np.float32)); names.append("collinearity")
    Xp = np.stack(cols, axis=1).astype(np.float32)
    return Xp, names


def pair_labels(collection: dict, pairs: np.ndarray, *, mode: str = "merge",
                gt_id: np.ndarray | None = None) -> np.ndarray:
    """Binary pair labels. mode='merge': 1 iff the two preds map to the SAME GT instance
    (needs gt_id from match_pred_to_gt) — the within-image fragmentation target.
    mode='sameclass': 1 iff equal predicted class — meaningful across images (type matching)."""
    if len(pairs) == 0:
        return np.zeros(0, np.int64)
    I, J = pairs[:, 0], pairs[:, 1]
    if mode == "merge":
        if gt_id is None:
            raise ValueError("mode='merge' needs gt_id (from match_pred_to_gt)")
        return ((gt_id[I] == gt_id[J]) & (gt_id[I] >= 0)).astype(np.int64)
    if mode == "sameclass":
        pc = np.array([r["pred_class"] for r in collection["records"]])
        return (pc[I] == pc[J]).astype(np.int64)
    raise ValueError(f"unknown mode {mode}")


# --------------------------------------------------------------------------- #
# Within-image instance MERGING (distance / cluster based) + visualization
# --------------------------------------------------------------------------- #
def _img_indices(collection, image_id):
    return [i for i, r in enumerate(collection["records"]) if r["image_id"] == image_id]


def _pairwise_dist(collection, idxs, kind, method, combo_weights):
    """Within-image pairwise distance matrix (n,n). kind: feature (1-cosine on `method`),
    centroid (normalized), mask_gap (min mask-to-mask px gap / image diag), combo (fw*feature+gw*mask_gap)."""
    import cv2
    recs = collection["records"]; n = len(idxs)

    def feat():
        F = collection["feats"][method][idxs].astype(np.float32)
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        return (1.0 - Fn @ Fn.T).astype(np.float32)           # cosine distance in [0,2]

    def centroid():
        c = np.array([[recs[i]["cx"], recs[i]["cy"]] for i in idxs], np.float32)
        return np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)

    def mask_gap():
        masks = [decode_mask(recs[i]) for i in idxs]
        H, W = masks[0].shape; diag = float(np.hypot(H, W))
        dt = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3) for m in masks]
        D = np.zeros((n, n), np.float32)
        for a in range(n):
            for b in range(a + 1, n):
                ga = dt[b][masks[a]].min() if masks[a].any() else diag
                gb = dt[a][masks[b]].min() if masks[b].any() else diag
                D[a, b] = D[b, a] = min(ga, gb) / diag
        return D

    if kind == "feature":  return feat()
    if kind == "centroid": return centroid()
    if kind == "mask_gap": return mask_gap()
    if kind == "combo":
        fw, gw = combo_weights
        return fw * feat() + gw * mask_gap()
    raise ValueError(f"unknown dist_kind {kind}")


def merge_groups_in_image(collection, image_id, *, dist_kind: str = "mask_gap", method: str = "decoder",
                          thresh: float = 0.05, max_group_size: int | None = None,
                          cluster_labels: np.ndarray | None = None,
                          restrict_same_cluster: bool = False, combo_weights=(1.0, 1.0)):
    """Group an image's instances into merge sets. Returns (groups, D) where groups is a
    list of lists of GLOBAL instance indices (singletons included).

    - dist_kind='cluster': merge instances sharing a per-image cluster label (needs cluster_labels).
    - else: connect pairs with distance < thresh, then greedy union-find capped at max_group_size
      (2 -> only pairs, 3 -> triples, None -> connected components / "any"). With cluster_labels +
      restrict_same_cluster=True, only same-cluster pairs may link. thresh scale depends on dist_kind
      (feature ~0.3-0.7 cosine-dist; centroid ~0.1-0.3; mask_gap ~0.02-0.1 of the image diagonal)."""
    from collections import defaultdict
    idxs = _img_indices(collection, image_id)
    n = len(idxs)
    if n == 0:
        return [], None
    if dist_kind == "cluster":
        if cluster_labels is None:
            raise ValueError("dist_kind='cluster' needs cluster_labels")
        g = defaultdict(list)
        for i in idxs:
            g[int(cluster_labels[i])].append(i)
        return list(g.values()), None

    D = _pairwise_dist(collection, idxs, dist_kind, method, combo_weights)
    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            if D[a, b] < thresh:
                if restrict_same_cluster and cluster_labels is not None \
                        and cluster_labels[idxs[a]] != cluster_labels[idxs[b]]:
                    continue
                edges.append((float(D[a, b]), a, b))
    edges.sort()
    parent = list(range(n)); size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for _, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if max_group_size and size[ra] + size[rb] > max_group_size:
            continue
        parent[ra] = rb; size[rb] += size[ra]
    g = defaultdict(list)
    for k in range(n):
        g[find(k)].append(idxs[k])
    return list(g.values()), D


def union_group_mask(collection, group) -> np.ndarray:
    m = None
    for i in group:
        mm = decode_mask(collection["records"][i])
        m = mm if m is None else (m | mm)
    return m


def overlay_groups(img: np.ndarray, collection, groups, *, alpha: float = 0.5, outline: bool = True) -> np.ndarray:
    """Color each merge group distinctly (union mask + outline) on the image."""
    import cv2
    import colorsys
    out = img.copy().astype(np.float32)
    for gi, grp in enumerate(groups):
        m = union_group_mask(collection, grp)
        r, g, b = colorsys.hsv_to_rgb((gi * 0.61803) % 1.0, 0.85, 1.0)
        c = np.array([r, g, b]) * 255
        out[m] = (1 - alpha) * out[m] + alpha * c
        if outline:
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, tuple(int(v) for v in c), 2)
    return out.astype(np.uint8)


def busiest_image(collection, rank: int = 0) -> int:
    from collections import Counter
    return Counter(r["image_id"] for r in collection["records"]).most_common(rank + 1)[rank][0]


# --------------------------------------------------------------------------- #
# Keypoints (catheter centerlines): draw + parse GT
# --------------------------------------------------------------------------- #
def overlay_keypoints(img: np.ndarray, kpts_list, vis_list=None, *, vis_thresh: float = 0.3,
                      radius: int = 3, line_w: int = 2, connect: bool = True) -> np.ndarray:
    """Draw per-instance keypoints (dots) and connect the visible ones IN ORDER as a
    centerline polyline — for catheters the kpts are the ordered bezier centerline points.
    kpts_list: list of (K,2) image-coord arrays. vis_list: list of (K,) visibility (pred
    score in [0,1] or COCO v in {0,1,2}); a point is drawn where vis > vis_thresh. Connects
    consecutive visible points (so an invisible middle point bridges its neighbours)."""
    import cv2
    import colorsys
    out = img.copy()
    for i, kp in enumerate(kpts_list):
        kp = np.asarray(kp, float)
        if kp.size == 0:
            continue
        v = np.asarray(vis_list[i], float) if vis_list is not None else np.ones(len(kp))
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.9, 1.0)
        c = (int(r * 255), int(g * 255), int(b * 255))
        pts = [(int(round(x)), int(round(y))) for (x, y), vv in zip(kp, v) if vv > vis_thresh]
        if connect:
            for a_, b_ in zip(pts[:-1], pts[1:]):
                cv2.line(out, a_, b_, c, line_w, cv2.LINE_AA)
        for q in pts:
            cv2.circle(out, q, radius, c, -1, cv2.LINE_AA)
    return out



def save_collection(collection: dict, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(collection, f, protocol=4)
    print(f"[save] {len(collection['records'])} instances -> {path}")


def load_collection(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def decode_mask(rec: dict) -> np.ndarray:
    from pycocotools import mask as mask_util
    return mask_util.decode(rec["rle"]).astype(bool)


def load_image(rec_or_fname, image_root: str | Path) -> np.ndarray:
    # `image_root` is explicit here (the qseg original defaulted to a hardcoded RANZCR path).
    import cv2
    fn = rec_or_fname["file_name"] if isinstance(rec_or_fname, dict) else rec_or_fname
    p = Path(image_root) / Path(fn).name
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
