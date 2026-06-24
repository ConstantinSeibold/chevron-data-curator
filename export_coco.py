"""Curated COCO export + round-trip import.

assemble_curated_coco mirrors assemble_pseudo_coco's annotation construction (RLE seg,
xywh bbox/area via pycocotools, keypoints) but SKIPS its per-class argmax + trust gate —
curation legitimately has multiple instances of one class per image. Each annotation also
carries a non-standard `iuid` for exact re-import.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .state import CuratorState


def _kpt_flat(kpts, vis, vis_thresh=0.3):
    """(K,2)+(K,) -> COCO flat [x,y,v]*K (v: 2 visible / 0 absent); returns (flat, num)."""
    flat, num = [], 0
    for (x, y), v in zip(np.asarray(kpts), np.asarray(vis)):
        vv = 2 if float(v) > vis_thresh else 0
        flat += [float(x), float(y), vv]
        num += int(vv > 0)
    return flat, num


def _prov(m) -> dict:
    """Per-mask provenance for the released COCO: how the label was produced (assign_source), the classifier
    confidence if it came from the classifier (assign_score), and the detector's original score + checkpoint
    that first proposed the instance (from meta.provenance). Carries the datasheet/QC trail (paper §5) into
    the annotation. Omits None values so labeled-by-hand masks stay clean."""
    p = m.provenance or {}
    fields = {"assign_source": m.assign_source, "assign_score": m.assign_score,
              "src_score": p.get("src_score"), "detector_ckpt": p.get("ckpt")}
    return {k: v for k, v in fields.items() if v is not None}


def _rle_fits(rle, rec) -> bool:
    """True iff the (effective) mask is encoded at the instance's image size. A mismatch means a stale /
    legacy mask (e.g. a pre-fix cross-image merge left a union mask from a DIFFERENT image on the rep) —
    such an annotation is invalid COCO and crashes the trainer's augmentation, so the export drops it."""
    return isinstance(rle, dict) and list(rle.get("size", [])) == [int(rec["H"]), int(rec["W"])]


def _rle_to_poly(rle):
    import cv2
    from pycocotools import mask as mu
    m = mu.decode(rle).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [c.reshape(-1).astype(float).tolist() for c in cnts if len(c) >= 3]
    return polys or None


def assemble_curated_coco(collection: dict, state: CuratorState, *, classes=None, iuids=None,
                          with_keypoints: bool = True, include_unassigned: bool = False,
                          polygon: bool = False, rle_override: dict | None = None,
                          partial_labels: bool = False, class_agnostic: bool = False) -> dict:
    if partial_labels:
        return _assemble_partial(collection, state, with_keypoints=with_keypoints, polygon=polygon,
                                 rle_override=rle_override, class_agnostic=class_agnostic,
                                 iuids=iuids, classes=classes)
    from pycocotools import mask as mu
    rle_override = rle_override or {}
    recs = collection["records"]

    # which instances to export
    sel = []
    for u, m in state.meta.items():
        if m.is_background or m.merged_into is not None:
            continue
        if m.assigned_class is None and not include_unassigned:
            continue
        tc = state.taxonomy.get(m.assigned_class) if m.assigned_class else None
        if tc is not None and tc.temp:                       # temp/scratch class -> excluded from the release
            continue
        if classes is not None and m.assigned_class not in classes:
            continue
        if iuids is not None and u not in iuids:
            continue
        sel.append(u)

    # categories (stable order; pinned coco_cat_id else 1..K). Unassigned -> reserved id 0.
    used_classes = []
    for u in sel:
        cid = state.meta[u].assigned_class
        if cid and cid not in used_classes:
            used_classes.append(cid)
    cat_id_map, cats = {}, []
    if class_agnostic:                                       # collapse every class -> one "object"
        cats.append({"id": 1, "name": "object", "supercategory": "device"})
        cat_id_map = {cid: 1 for cid in used_classes}
        used_classes = []                                    # skip the per-class loop below
    next_id = 1
    for cid in used_classes:
        tc = state.taxonomy.get(cid)
        coco_id = tc.coco_cat_id if (tc and tc.coco_cat_id) else next_id
        next_id = max(next_id, coco_id + 1)
        cat_id_map[cid] = coco_id
        sup = state.superclasses.get(tc.supercategory) if (tc and tc.supercategory) else None   # real supercategory
        con = state.concepts.get(tc.concept) if (tc and tc.concept) else None
        cat = {"id": coco_id, "name": state.class_name(cid), "supercategory": sup.name if sup else "device"}
        if con:                                              # carry the concept + mimic crosswalk for roll-up eval
            cat["concept"] = con.name
            if con.mimic_family:
                cat["mimic_family"] = con.mimic_family
        cats.append(cat)
    if include_unassigned:
        cat_id_map[None] = 0
        cats.append({"id": 0, "name": "__unassigned__", "supercategory": "device"})

    # group by image_id
    images, anns, seen_img, skipped = [], [], {}, 0
    aid = 1
    for u in sel:
        m = state.meta[u]
        rec = recs[m.row]
        rle = rle_override.get(u, rec["rle"])
        if not _rle_fits(rle, rec):                       # stale/legacy mask not at the image size -> drop
            skipped += 1
            continue
        iid = int(rec["image_id"])
        if iid not in seen_img:
            seen_img[iid] = True
            images.append({"id": iid, "file_name": rec.get("file_name", ""),
                           "height": int(rec["H"]), "width": int(rec["W"])})
        bbox = [float(v) for v in mu.toBbox(rle)]
        area = float(mu.area(rle))
        a = {"id": aid, "image_id": iid, "category_id": cat_id_map[m.assigned_class],
             "bbox": bbox, "area": area, "iscrowd": 0,
             "score": float(rec["score"]), "iuid": u, **_prov(m),
             "segmentation": (_rle_to_poly(rle) if polygon else
                              {"size": rle["size"], "counts": rle["counts"]})}
        if with_keypoints and "keypoints" in rec:
            flat, num = _kpt_flat(rec["keypoints"], rec.get("keypoint_vis", np.ones(len(rec["keypoints"]))))
            a["keypoints"] = flat; a["num_keypoints"] = num
        anns.append(a); aid += 1

    return {"images": images, "annotations": anns, "categories": cats,
            "info": {"description": "qseg curator export", "version": "1.0", "n_skipped_bad_mask": skipped}}


def _assemble_partial(collection: dict, state: CuratorState, *, with_keypoints: bool, polygon: bool,
                      rle_override: dict | None, class_agnostic: bool, iuids=None, classes=None) -> dict:
    """PARTIAL-LABEL export for self-training where images are only partially curated. Emits:
    - POSITIVES (assigned, reviewed) as normal GT annotations (iscrowd=0, their class — or one 'object'
      class if class_agnostic);
    - UNREVIEWED (unassigned predictions) as iscrowd=1 '__ignore__' annotations, so the trainer must NOT
      supervise those regions as background (the false-negative trap);
    - REJECTED (background) are OMITTED -> treated as true background (the human confirmed not-object).
    Each image carries `reviewed_exhaustive` (true = every prediction on it was reviewed, so absence is a
    true negative) + n_positive/n_ignore/n_negative counts; each annotation carries iuid + curator_status."""
    from collections import Counter

    from pycocotools import mask as mu
    rle_override = rle_override or {}
    recs = collection["records"]
    IGNORE_ID = 0

    pos, ign, neg = [], [], []
    for u, m in state.meta.items():
        if m.merged_into is not None:
            continue
        if iuids is not None and u not in iuids:
            continue
        if m.is_background:
            neg.append(u)
        elif m.assigned_class is not None:
            if classes is None or m.assigned_class in classes:
                pos.append(u)
        else:
            ign.append(u)

    cat_id_map, cats = {}, []
    if class_agnostic:
        cats.append({"id": 1, "name": "object", "supercategory": "device"})
        cat_id_map = {state.meta[u].assigned_class: 1 for u in pos}
    else:
        nid = 1
        for u in pos:
            cid = state.meta[u].assigned_class
            if cid in cat_id_map:
                continue
            tc = state.taxonomy.get(cid)
            coco_id = tc.coco_cat_id if (tc and tc.coco_cat_id) else nid
            nid = max(nid, coco_id + 1)
            cat_id_map[cid] = coco_id
            cats.append({"id": coco_id, "name": state.class_name(cid), "supercategory": "device"})
    cats.append({"id": IGNORE_ID, "name": "__ignore__", "supercategory": "device"})

    def iid_of(u):
        return int(recs[state.meta[u].row]["image_id"])
    pc, ic, nc = Counter(map(iid_of, pos)), Counter(map(iid_of, ign)), Counter(map(iid_of, neg))

    images, seen = [], set()
    for u in pos + ign + neg:
        iid = iid_of(u)
        if iid in seen:
            continue
        seen.add(iid)
        rec = recs[state.meta[u].row]
        exhaustive = ic.get(iid, 0) == 0 and (pc.get(iid, 0) + nc.get(iid, 0)) > 0
        images.append({"id": iid, "file_name": rec.get("file_name", ""),
                       "height": int(rec["H"]), "width": int(rec["W"]),
                       "reviewed_exhaustive": bool(exhaustive),
                       "n_positive": int(pc.get(iid, 0)), "n_ignore": int(ic.get(iid, 0)),
                       "n_negative": int(nc.get(iid, 0))})

    anns, aid = [], 1
    skipped = [0]
    def _emit(u, cat, crowd, status):
        nonlocal aid
        rec = recs[state.meta[u].row]
        rle = rle_override.get(u, rec["rle"])
        if not _rle_fits(rle, rec):                       # stale/legacy mask not at the image size -> drop
            skipped[0] += 1
            return
        a = {"id": aid, "image_id": iid_of(u), "category_id": cat,
             "bbox": [float(v) for v in mu.toBbox(rle)], "area": float(mu.area(rle)),
             "iscrowd": crowd, "score": float(rec["score"]), "iuid": u, "curator_status": status,
             **_prov(state.meta[u]),
             "segmentation": (_rle_to_poly(rle) if polygon else {"size": rle["size"], "counts": rle["counts"]})}
        if with_keypoints and "keypoints" in rec:
            flat, num = _kpt_flat(rec["keypoints"], rec.get("keypoint_vis", np.ones(len(rec["keypoints"]))))
            a["keypoints"] = flat; a["num_keypoints"] = num
        anns.append(a); aid += 1

    for u in pos:
        _emit(u, cat_id_map[state.meta[u].assigned_class], 0, "positive")
    for u in ign:
        _emit(u, IGNORE_ID, 1, "ignore")

    return {"images": images, "annotations": anns, "categories": cats,
            "info": {"description": "qseg curator partial-label export", "version": "1.0",
                     "partial_labels": True, "class_agnostic": bool(class_agnostic),
                     "n_images": len(images), "n_positive": len(pos), "n_ignore": len(ign), "n_negative": len(neg),
                     "n_skipped_bad_mask": skipped[0],
                     "semantics": ("positive=reviewed GT; iscrowd/__ignore__=unreviewed (do NOT supervise as "
                                   "background); rejected omitted (true background); per-image "
                                   "reviewed_exhaustive=true means absence is a true negative.")}}


def merge_coco_sources(curated, extra, *, class_agnostic: bool = True, extra_exhaustive: bool = True,
                       extra_image_root: str | None = None) -> dict:
    """Merge a curated partial-label COCO with an EXTRA fully-labeled COCO (e.g. synthfb, complete masks)
    into one training json. Image + annotation ids are reindexed to avoid collisions; categories align by
    NAME (union) — or, with class_agnostic, both collapse to a single 'object' (id 1) + '__ignore__' (id 0).
    Extra images are marked `reviewed_exhaustive` (their masks are complete) unless extra_exhaustive=False;
    curated images keep their own per-image flags. The two sources can thus be supervised correctly:
    complete synth = real negatives, partial real = ignore (the PU consumer reads `reviewed_exhaustive`).
    The curated export uses ABSOLUTE file_names; the extra source's RELATIVE file_names are absolutized
    against `extra_image_root` (default: <extra_json_dir>/images) so both resolve under one image_root."""
    import os
    def _load(x):
        return x if isinstance(x, dict) else json.loads(Path(x).read_text())
    cur, ext = _load(curated), _load(extra)
    if extra_image_root is None and not isinstance(extra, dict):
        extra_image_root = str(Path(extra).resolve().parent / "images")

    # ---- aligned category map (name -> new id), keeping __ignore__ pinned at 0 ----
    names = []
    for coco in (cur, ext):
        for c in coco.get("categories", []):
            if c["name"] != "__ignore__" and c["name"] not in names:
                names.append(c["name"])
    if class_agnostic:
        name2id = {n: 1 for n in names}
        cats = [{"id": 1, "name": "object", "supercategory": "device"}]
    else:
        name2id = {n: i + 1 for i, n in enumerate(names)}
        cats = [{"id": i + 1, "name": n, "supercategory": "device"} for i, n in enumerate(names)]
    name2id["__ignore__"] = 0
    cats.append({"id": 0, "name": "__ignore__", "supercategory": "device"})

    images, anns = [], []
    iid_next, aid_next = 1, 1
    for coco, is_extra in ((cur, False), (ext, True)):
        catid2name = {c["id"]: c["name"] for c in coco.get("categories", [])}
        iid_map = {}
        for im in coco.get("images", []):
            new = dict(im); new["id"] = iid_next; iid_map[im["id"]] = iid_next; iid_next += 1
            if is_extra:
                fn = str(new.get("file_name", ""))                    # absolutize relative synth paths so
                if extra_image_root and not os.path.isabs(fn):        # both sources resolve under one root
                    new["file_name"] = os.path.join(extra_image_root, fn)
                new["reviewed_exhaustive"] = bool(extra_exhaustive)   # synth masks are complete
                new["source"] = "extra"
            else:
                new["source"] = "curated"
            images.append(new)
        for a in coco.get("annotations", []):
            nm = catid2name.get(a.get("category_id"))
            if nm is None or nm not in name2id:
                continue
            na = dict(a); na["id"] = aid_next; aid_next += 1
            na["image_id"] = iid_map[a["image_id"]]; na["category_id"] = name2id[nm]
            anns.append(na)

    return {"images": images, "annotations": anns, "categories": cats,
            "info": {"description": "qseg curator merged (curated + extra)", "version": "1.0",
                     "class_agnostic": bool(class_agnostic), "n_curated_images": len(cur.get("images", [])),
                     "n_extra_images": len(ext.get("images", []))}}


def to_class_agnostic(coco, *, image_root: str | None = None) -> dict:
    """Collapse a COCO to a single 'object' class (id 1) and absolutize relative file_names (against
    image_root, default <json_dir>/images) — for using a multi-class set (e.g. synthfb) as the VAL/TEST
    target of a class-agnostic model (the eval cat space must match the model's one 'object' class)."""
    import os
    c = coco if isinstance(coco, dict) else json.loads(Path(coco).read_text())
    if image_root is None and not isinstance(coco, dict):
        image_root = str(Path(coco).resolve().parent / "images")
    images = []
    for im in c.get("images", []):
        ni = dict(im); fn = str(ni.get("file_name", ""))
        if image_root and not os.path.isabs(fn):
            ni["file_name"] = os.path.join(image_root, fn)
        images.append(ni)
    anns = [{**a, "category_id": 1} for a in c.get("annotations", [])]
    return {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "object", "supercategory": "device"}],
            "info": {"description": "class-agnostic", "class_agnostic": True}}


def export(collection: dict, state: CuratorState, out_path: str | Path, **kw) -> Path:
    coco = assemble_curated_coco(collection, state, **kw)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(coco))
    os.replace(tmp, out_path)
    return out_path


def import_coco(path: str | Path, collection: dict, state: CuratorState, *, iou_floor: float = 0.3) -> dict:
    """Match annotations back to instances by `iuid` (exact), else by image_id + mask IoU,
    and set assigned_class (creating taxonomy classes). Returns a report."""
    from pycocotools import mask as mu
    coco = json.loads(Path(path).read_text())
    catid2name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    recs = collection["records"]
    by_img = {}
    for u, m in state.meta.items():
        by_img.setdefault(m.image_id, []).append(u)

    matched, by_iuid, by_iou, unmatched = 0, 0, 0, 0
    touched = []
    for a in coco.get("annotations", []):
        name = catid2name.get(a.get("category_id"))
        if not name or name == "__unassigned__":
            continue
        cid = state.add_class(name)
        target = None
        if a.get("iuid") in state.meta:
            target = a["iuid"]; by_iuid += 1
        else:
            seg = a.get("segmentation")
            am = mu.decode(seg) if isinstance(seg, dict) else None
            best, best_iou = None, iou_floor
            for u in by_img.get(int(a.get("image_id", -1)), []):
                if am is None:
                    break
                pm = mu.decode(recs[state.meta[u].row]["rle"])
                if pm.shape != am.shape:
                    continue
                inter = float(np.logical_and(pm, am).sum()); union = float(np.logical_or(pm, am).sum())
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best, best_iou = u, iou
            if best is not None:
                target = best; by_iou += 1
        if target is not None:
            state.meta[target].assigned_class = cid
            state.meta[target].assign_source = "import"
            touched.append(target); matched += 1
        else:
            unmatched += 1
    return {"matched": matched, "by_iuid": by_iuid, "by_iou": by_iou, "unmatched": unmatched, "touched": touched}
