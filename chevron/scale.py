"""Scalable pseudo-labeling: train-on-sample, propagate-at-scale, sharded export.

The interactive curator holds the whole collection (records + feature matrices + masks) in RAM, which tops out
around ~1M instances on a 100GB box. For pseudo-labeling 64k images / millions of instances we instead:
  1. curate a manageable SAMPLE normally -> a trained classifier + reference bank + taxonomy;
  2. run the seg model over the rest in SHARDS, propagate labels per shard with that trained model, write one
     COCO per shard, and DROP the shard collection (RAM bounded to one shard);
  3. merge the shard COCOs.
This module is the propagation + per-shard COCO + merge logic (pure-ish, no model); the engine orchestrates
the sharded model runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def assign_by_classifier(batch: dict, clf, spec: dict, thresh: float):
    """Per-record class id (or None) from a trained classifier. Builds the fused matrix on the BATCH (it must
    carry the classifier's feature methods) and argmaxes proba above `thresh`. Returns (class_ids, scores)."""
    from ._bootstrap import get_P
    n = len(batch["records"])
    if n == 0 or clf is None:
        return [None] * n, [0.0] * n
    X, _ = get_P().fuse_features(batch, dict(spec))
    proba = clf.proba(np.asarray(X, np.float32))
    classes = list(clf.classes)
    out_c, out_s = [], []
    for p in proba:
        p = np.asarray(p, float)
        j = int(np.argmax(p)) if len(p) else -1
        s = float(p[j]) if j >= 0 else 0.0
        out_c.append(classes[j] if (j >= 0 and s >= float(thresh)) else None)
        out_s.append(s)
    return out_c, out_s


def assign_by_reference(batch_emb: np.ndarray, bank, thresh: float):
    """Per-record reference class name (or None) by CSLS top-1 against the bank, above `thresh`."""
    from . import reference_bank as _rb
    if batch_emb is None or len(batch_emb) == 0 or bank is None or bank.n == 0:
        return [None] * (0 if batch_emb is None else len(batch_emb)), []
    ranked = _rb.suggest(np.asarray(batch_emb, np.float32), bank.emb, bank.labels, topk=1, knn=8)
    names, scores = [], []
    for r in ranked:
        if r and r[0][1] >= float(thresh):
            names.append(bank.class_names.get(r[0][0], str(r[0][0]))); scores.append(float(r[0][1]))
        else:
            names.append(None); scores.append(float(r[0][1]) if r else 0.0)
    return names, scores


def batch_to_coco(batch: dict, class_of: list, *, class_agnostic: bool = False,
                  drop_unlabeled: bool = True, scores=None) -> dict:
    """COCO for one shard: each record whose class_of[i] is not None becomes an annotation (RLE seg + xywh
    bbox). `class_of` are class NAMES (strings). class_agnostic collapses all to one 'object' category."""
    from pycocotools import mask as mu
    recs = batch["records"]
    names = ["object"] if class_agnostic else sorted({c for c in class_of if c})
    cat_id = {n: i + 1 for i, n in enumerate(names)}
    cats = [{"id": i, "name": n, "supercategory": "device"} for n, i in cat_id.items()]
    images, anns, seen, aid = [], [], {}, 1
    for i, rec in enumerate(recs):
        cls = class_of[i]
        if cls is None and drop_unlabeled:
            continue
        rle = rec["rle"]
        if list(rle.get("size", [])) != [int(rec["H"]), int(rec["W"])]:
            continue                                       # stale/legacy mask not at image size
        iid = int(rec["image_id"])
        if iid not in seen:
            seen[iid] = True
            images.append({"id": iid, "file_name": rec.get("file_name", ""),
                           "height": int(rec["H"]), "width": int(rec["W"])})
        a = {"id": aid, "image_id": iid, "category_id": cat_id["object" if class_agnostic else cls],
             "bbox": [float(v) for v in mu.toBbox(rle)], "area": float(mu.area(rle)), "iscrowd": 0,
             "score": float(rec.get("score", 1.0)),
             "segmentation": {"size": rle["size"], "counts": rle["counts"]}}
        if scores is not None and scores[i] is not None:
            a["assign_score"] = round(float(scores[i]), 4)
        anns.append(a); aid += 1
    return {"images": images, "annotations": anns, "categories": cats}


def merge_cocos(paths: list, out_path: str | Path) -> dict:
    """Merge shard COCO files into one, unifying categories BY NAME and reindexing image/annotation ids."""
    out = {"images": [], "annotations": [], "categories": []}
    cat_id, img_id, aid = {}, {}, 1
    for p in paths:
        d = json.loads(Path(p).read_text())
        loc_cat = {}
        for c in d.get("categories", []):
            if c["name"] not in cat_id:
                cat_id[c["name"]] = len(cat_id) + 1
                out["categories"].append({"id": cat_id[c["name"]], "name": c["name"],
                                          "supercategory": c.get("supercategory", "device")})
            loc_cat[c["id"]] = cat_id[c["name"]]
        for im in d.get("images", []):
            key = (str(p), im["id"])
            if key not in img_id:
                img_id[key] = len(img_id) + 1
                out["images"].append({**im, "id": img_id[key]})
        for a in d.get("annotations", []):
            out["annotations"].append({**a, "id": aid, "image_id": img_id[(str(p), a["image_id"])],
                                       "category_id": loc_cat[a["category_id"]]})
            aid += 1
    Path(out_path).write_text(json.dumps(out))
    return {"images": len(out["images"]), "annotations": len(out["annotations"]),
            "categories": len(out["categories"]), "path": str(out_path)}
