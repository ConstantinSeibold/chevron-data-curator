"""Custom (non-Gradio) web frontend for the curator — a thin FastAPI layer over CuratorEngine.

Gradio's per-interaction round-trip + full DOM reconciliation does not scale to 25k+ instances (every
click re-serialises/re-renders the whole component tree). This serves WINDOWED JSON + LAZY per-instance
crop images, so the browser only ever fetches what is on screen, mutations are targeted, and the
partition/instance count stops driving responsiveness. All curation logic is reused from CuratorEngine
unchanged — this is purely a transport + UI swap.

Run:  python -m tools.curator.server --project DIR [--port 7870]
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

from .engine import CuratorEngine

WEB = Path(__file__).parent / "web"


def _png_bytes(arr) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def _png_data_uri(arr) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(arr)).decode("ascii")


def _clf_report(eng, rep: dict) -> dict:
    """JSON-safe summary of a classifier-train report (drops the heavy numpy PR curves)."""
    if rep.get("error"):
        return {"ok": False, "error": rep["error"], "skipped": rep.get("skipped_names", [])}
    clf = getattr(eng, "_clf", None)
    classes = [eng.state.class_name(c) for c in getattr(clf, "classes", [])]
    youden = {}
    for cid, cv in ((rep.get("pr") or {}).get("curves") or {}).items():
        if cv.get("youden") is not None:
            youden[eng.state.class_name(cid)] = round(float(cv["youden"]), 2)
    return {"ok": True, "algo": rep.get("algo"), "n_classes": rep.get("n_classes", len(classes)),
            "classes": classes, "skipped": rep.get("skipped_names", []), "youden": youden}


def _partition_rows(eng: CuratorEngine, query: str = ""):
    """Formatted partition rows (full list; the API windows with offset/limit)."""
    rows = [{"pid": str(r["pid"]), "size": int(r["size"]),
             "score": r["mean_score"], "cls": r["majority_class"] or ""}
            for r in eng.partition_view()]
    q = (query or "").strip().lower()
    if q:
        rows = [r for r in rows if q in r["pid"].lower() or q in r["cls"].lower()]
    return rows


def create_app(project: str | None = None, *, engine: CuratorEngine | None = None):
    from fastapi import Body, FastAPI, HTTPException, Response
    from fastapi.responses import HTMLResponse

    eng = engine if engine is not None else CuratorEngine(project)
    app = FastAPI(title="qseg curator")
    app.state.eng = eng

    _NOCACHE = {"Cache-Control": "no-store, must-revalidate"}   # always serve fresh page/JS (no stale UI)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((WEB / "index.html").read_text(), headers=_NOCACHE)

    @app.get("/app.js")
    def appjs():
        return Response((WEB / "app.js").read_text(), media_type="application/javascript", headers=_NOCACHE)

    @app.get("/api/state")
    def state():
        cl = eng._cluster
        return {
            "stats": eng.stats(),
            "clustered": cl is not None,
            "level": cl["level"] if cl else None,
            "levels": [{"i": i, "n": int(c)} for i, c in enumerate(cl["counts"])] if cl else [],
            "classes": eng.state.class_names(),
            "features": eng.available_features(),
            "model_config": eng.state.config.get("model", {}).get("config_name"),
            "model_ckpt": eng.state.config.get("model", {}).get("ckpt"),
        }

    @app.post("/api/cluster")
    def cluster(body: dict = Body(default={})):
        feats = body.get("features") or ["decoder"]
        try:
            info = eng.cluster({m: 1.0 for m in feats})
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, **info}

    @app.post("/api/level")
    def set_level(body: dict = Body(...)):
        eng.set_level(int(body["level"]))
        return {"ok": True}

    @app.get("/api/statistics")
    def statistics():
        return eng.statistics()

    @app.get("/api/partitions")
    def partitions(offset: int = 0, limit: int = 100, query: str = ""):
        rows = _partition_rows(eng, query)
        return {"total": len(rows), "rows": rows[offset:offset + limit]}

    def _items(iuids):
        # image_id is a 56-bit hash (> 2^53) -> emit as a STRING so JS doesn't round it (a rounded id
        # round-trips to a non-existent image -> "no instances on this image"). JS passes it back verbatim.
        return [{"iuid": u, "caption": eng._caption(u), "image_id": str(int(eng.state.meta[u].image_id))} for u in iuids]

    @app.get("/api/instances")
    def instances(pid: str, offset: int = 0, limit: int = 60):
        iu = eng.partition_iuids(pid)
        return {"total": len(iu), "items": _items(iu[offset:offset + limit])}

    @app.get("/api/crop")
    def crop(iuid: str, mask: int = 1, max_side: int = 256, context: int = 0):
        if iuid not in eng.state.meta:
            raise HTTPException(404, "unknown iuid")
        arr = eng.crop(iuid, mask_overlay=bool(mask), max_side=int(max_side), context=bool(context))
        return Response(_png_bytes(arr), media_type="image/png",
                        headers={"Cache-Control": "max-age=31536000"})   # content-stable per (iuid,mask,context)

    @app.get("/api/images")
    def images(query: str = "", limit: int = 100):
        """Windowed image-id list (most-populated first) + per-image instance count — so the file
        picker never ships thousands of options (the Gradio dropdown-freeze trap)."""
        from collections import Counter
        c = Counter(int(m.image_id) for m in eng.state.meta.values())
        items = c.most_common()
        q = query.strip()
        if q:
            items = [(i, n) for i, n in items if q in str(i)]
        return {"total": len(items), "items": [{"image_id": str(i), "n": n} for i, n in items[:limit]]}

    @app.post("/api/assign")
    def assign(body: dict = Body(...)):
        iuids, cls = body.get("iuids") or [], (body.get("cls") or "").strip()
        if iuids and cls:
            eng.assign(iuids, cls)
        return {"ok": True, "stats": eng.stats(), "classes": eng.state.class_names()}

    @app.post("/api/reject")
    def reject(body: dict = Body(...)):
        if body.get("iuids"):
            eng.set_background(body["iuids"])
        return {"ok": True, "stats": eng.stats()}

    @app.post("/api/unassign")
    def unassign(body: dict = Body(...)):
        if body.get("iuids"):
            eng.remove_from_class(body["iuids"])
        return {"ok": True, "stats": eng.stats()}

    @app.post("/api/merge")
    def merge(body: dict = Body(...)):
        iuids = body.get("iuids") or []
        n = eng.merge_instances(iuids, mode=body.get("mode", "union")) if len(iuids) >= 2 else 0
        return {"ok": True, "n_groups": int(n), "stats": eng.stats()}

    @app.post("/api/export")
    def export(body: dict = Body(default={})):
        path = eng.export_coco(partial_labels=bool(body.get("partial", False)),
                               class_agnostic=bool(body.get("class_agnostic", False)))
        return {"ok": True, "path": str(path), "partial": bool(body.get("partial", False)),
                "class_agnostic": bool(body.get("class_agnostic", False)), "stats": eng.stats()}

    # ---- Phase 2: undo/redo, in-image, classifier, refine, rejected, sampling ----
    @app.post("/api/undo")
    def undo(body: dict = Body(default={})):
        eng.undo()
        return {"ok": True, "stats": eng.stats(), "classes": eng.state.class_names()}

    @app.post("/api/redo")
    def redo(body: dict = Body(default={})):
        eng.redo()
        return {"ok": True, "stats": eng.stats(), "classes": eng.state.class_names()}

    @app.get("/api/image_instances")
    def image_instances(image_id: int, offset: int = 0, limit: int = 120):
        iu = eng.image_instance_iuids(int(image_id))
        return {"total": len(iu), "items": _items(iu[offset:offset + limit])}

    @app.get("/api/image_overlay")
    def image_overlay(image_id: int, color_by: str = "partition", masks: int = 1, max_side: int = 900):
        arr = eng.image_overlay(int(image_id), color_by=color_by, show_masks=bool(masks), max_side=int(max_side))
        return Response(_png_bytes(arr), media_type="image/png")

    @app.post("/api/match_image")
    def match_image(body: dict = Body(...)):
        """Find the partition whose instance is closest to an uploaded reference image, in the chosen
        feature space (default roialign). Image sent as a base64 data-URI (no multipart dep)."""
        import cv2
        import numpy as np
        b64 = (body.get("image") or "").split(",")[-1]
        if not b64:
            raise HTTPException(400, "no image")
        arr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise HTTPException(400, "could not decode image")
        res = eng.match_image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB),
                              feature=body.get("feature", "roialign"), k=int(body.get("k", 12)))
        if "error" not in res:
            for key in ("matches", "matches_class", "matches_pool"):
                for m in res.get(key, []):
                    m["crop"] = _png_data_uri(eng.crop(m["iuid"], max_side=160))
        return res

    @app.get("/api/progress")
    def progress():
        """Live progress of the running inference/RAD-DINO job (polled by the UI; served from another worker
        thread while the blocking inference POST runs — torch releases the GIL so this stays responsive)."""
        return eng.progress()

    @app.get("/api/features")
    def features():
        """Feature methods present in the collection (the selectors' source of truth) + whether RAD-DINO
        (mask-pooled, the best fine-device feature) has been extracted."""
        avail = eng.available_features()
        return {"available": avail, "has_raddino": "raddino" in avail}

    @app.post("/api/compute_raddino")
    def compute_raddino(body: dict = Body(default={})):
        """Extract mask-pooled RAD-DINO embeddings for every instance (no re-detection) -> 'raddino' becomes
        a selectable feature for clustering / classifier / substructure / merge-rec / reference suggest."""
        rep = eng.compute_raddino(force=bool(body.get("force", False)))
        if rep.get("error"):
            raise HTTPException(400, rep["error"])
        return rep

    # ---- reference exemplar bank (suggest a fine class for unassigned instances) ----
    @app.post("/api/reference/load")
    def reference_load(body: dict = Body(...)):
        """Build/load the RAD-DINO reference bank from a labeled COCO + bootstrap the taxonomy."""
        path = (body.get("coco_path") or "").strip()
        if not path or not Path(path).exists():
            raise HTTPException(400, f"reference coco not found: {path}")
        rep = eng.load_reference_bank(path, rebuild=bool(body.get("rebuild", False)))
        if rep.get("error"):
            raise HTTPException(400, rep["error"])
        return {"ok": True, "class_names": eng.state.class_names(), "n_classes": rep["classes"],
                "exemplars": rep["exemplars"], "added_classes": rep["added_classes"]}

    @app.get("/api/reference/classes")
    def reference_classes():
        return {"loaded": getattr(eng, "_ref_bank", None) is not None, "rows": eng.reference_classes()}

    @app.get("/api/reference/exemplars")
    def reference_exemplars(cls: str = "", limit: int = 8):
        bank = getattr(eng, "_ref_bank", None)
        if bank is None:
            return {"items": []}
        return {"items": [{"file_name": e["file_name"], "bbox": e.get("bbox")}
                          for e in bank.exemplars_for(cls, int(limit))]}

    @app.get("/api/reference/exemplar")
    def reference_exemplar(file_name: str = "", x: int = -1, y: int = -1, w: int = -1, h: int = -1):
        bbox = [x, y, w, h] if w > 0 else None
        arr = eng.reference_exemplar(file_name, bbox)
        if arr is None:
            raise HTTPException(404, "exemplar not found")
        return Response(_png_bytes(arr), media_type="image/png")

    @app.post("/api/reference/suggest")
    def reference_suggest(body: dict = Body(...)):
        """Top-k reference classes per instance (for a partition or an explicit iuid list)."""
        ius = body.get("iuids")
        if not ius and body.get("pid"):
            ius = eng.partition_iuids(str(body["pid"]))
        rep = eng.reference_suggest(list(ius or [])[:int(body.get("cap", 120))],
                                    topk=int(body.get("topk", 5)), use_csls=bool(body.get("csls", True)))
        if rep.get("error"):
            raise HTTPException(400, rep["error"])
        return rep

    @app.post("/api/reference/add")
    def reference_add(body: dict = Body(...)):
        """Self-improving: add confirmed in-domain instances to the bank."""
        return eng.add_to_reference_bank(body.get("iuids") or [])

    @app.post("/api/merge_preview")
    def merge_preview(body: dict = Body(...)):
        iuids = body.get("iuids") or []
        if len(iuids) < 2:
            return {"img": None}
        arr = eng.merge_result_preview(iuids, body.get("mode", "union"), max_side=320)
        return {"img": _png_data_uri(arr)}

    # ---- learned merge recommender (train on past merges -> suggest new ones) ----
    @app.post("/api/train_merge_recommender")
    def train_merge_recommender(body: dict = Body(...)):
        feats = body.get("features") or ["decoder"]
        rep = eng.train_merge_recommender({m: 1.0 for m in feats}, algo=body.get("algo", "logreg"))
        if rep.get("error"):
            return {"ok": False, "error": rep["error"]}
        return {"ok": True, "n_pos": int(rep.get("n_pos", 0)), "n_neg": int(rep.get("n_neg", 0)),
                "n_merge_events": int(rep.get("n_merge_events", 0)), "youden": round(float(rep.get("youden", 0.5)), 3)}

    @app.get("/api/recommend_merges")
    def recommend_merges(thresh: float = 0.5, image_id: str = "", max_groups: int = 50):
        """Candidate merge groups (connected components of pair P(merge) >= thresh) — globally, or scoped
        to one image when image_id is given (the In-image-tab in-context suggestions)."""
        if image_id:
            cands = eng.recommend_merges_for_image(int(image_id), float(thresh), max_groups=int(max_groups))
        else:
            cands = eng.recommend_merges(float(thresh), max_groups=int(max_groups))
        return {"trained": getattr(eng, "_merge_clf", None) is not None,
                "groups": [{"iuids": list(c["iuids"]), "image_id": str(int(c["image_id"])),
                            "prob": round(float(c["prob"]), 3), "n": len(c["iuids"])} for c in cands]}

    @app.get("/api/merge_result")
    def merge_result(iuids: str = "", mode: str = "union", max_side: int = 220):
        """PNG of the would-be merged mask (per mode) for a candidate group — lazy <img> for the cards."""
        ius = [u for u in iuids.split(",") if u and u in eng.state.meta]
        if len(ius) < 2:
            raise HTTPException(400, "need >=2 valid iuids")
        arr = eng.merge_result_preview(ius, mode, max_side=int(max_side))
        return Response(_png_bytes(arr), media_type="image/png")

    @app.post("/api/accept_merge")
    def accept_merge(body: dict = Body(...)):
        iuids = body.get("iuids") or []
        if len(iuids) >= 2:
            eng.accept_merge(iuids, mode=body.get("mode", "union"))      # logs a positive merge event
        return {"ok": True, "stats": eng.stats(), "classes": eng.state.class_names()}

    @app.post("/api/reject_merge")
    def reject_merge(body: dict = Body(...)):
        iuids = body.get("iuids") or []
        if len(iuids) >= 2:
            eng.reject_merge(iuids)                                      # logs a negative (no state change)
        return {"ok": True}

    @app.post("/api/train_classifier")
    def train_classifier(body: dict = Body(...)):
        feats = body.get("features") or ["decoder"]
        rep = eng.train_classifier({m: 1.0 for m in feats}, algo=body.get("algo", "logreg"),
                                   use_unassigned_negatives=bool(body.get("openset", True)),
                                   knn_k=int(body.get("knn_k", 5)), knn_metric=body.get("knn_metric", "cosine"),
                                   knn_weights=body.get("knn_weights", "distance"))
        return _clf_report(eng, rep)

    @app.get("/api/predict")
    def predict(thresh: float = 0.5, only_class: str = "", offset: int = 0, limit: int = 60):
        cid = eng.state.class_id_by_name(only_class) if only_class else None
        preds = sorted(eng.predict_and_threshold(float(thresh), only_class=cid), key=lambda t: -t[2])
        page = preds[offset:offset + limit]
        items = [{"iuid": u, "cls": eng.state.class_name(c), "conf": round(float(conf), 3),
                  "image_id": str(int(eng.state.meta[u].image_id))} for u, c, conf in page]
        return {"total": len(preds), "items": items}

    @app.post("/api/apply_predictions")
    def apply_predictions(body: dict = Body(...)):
        cid = eng.state.class_id_by_name(body["only_class"]) if body.get("only_class") else None
        n, _ = eng.apply_predictions(float(body.get("thresh", 0.5)), only_class=cid,
                                     exclude=set(body.get("exclude") or []))
        return {"ok": True, "n": n, "stats": eng.stats(), "classes": eng.state.class_names()}

    @app.get("/api/recommend_rejections")
    def recommend_rejections(max_conf: float = 0.3, offset: int = 0, limit: int = 60):
        """Unassigned instances the classifier matches to NO class (max prob < max_conf) — reject candidates."""
        recs = eng.recommend_rejections(float(max_conf))
        page = recs[offset:offset + limit]
        items = [{"iuid": u, "cls": eng.state.class_name(c) if c is not None else "?",
                  "conf": round(float(conf), 3),
                  "image_id": str(int(eng.state.meta[u].image_id))} for u, c, conf in page]
        return {"total": len(recs), "items": items}

    @app.post("/api/refine_preview")
    def refine_preview(body: dict = Body(...)):
        try:
            before, after = eng.refine_preview(body["iuid"], body.get("ops", []),
                                               mask_overlay=bool(body.get("mask", 1)))
        except RuntimeError as e:                       # e.g. SAM not set up — show it, don't 500
            raise HTTPException(400, str(e))
        return {"before": _png_data_uri(before), "after": _png_data_uri(after)}

    @app.post("/api/apply_refine")
    def apply_refine(body: dict = Body(...)):
        try:
            eng.apply_refine(body["iuid"], body.get("ops", []))
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "stats": eng.stats()}

    @app.post("/api/apply_refine_partition")
    def apply_refine_partition(body: dict = Body(...)):
        """Apply a refine chain to EVERY instance in a partition (finch cluster or class: pseudo-partition)."""
        try:
            n = eng.apply_refine_partition(str(body["pid"]), body.get("ops", []))
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "n": n, "stats": eng.stats()}

    @app.post("/api/apply_class_rule")
    def apply_class_rule(body: dict = Body(...)):
        """Save a refine chain as a class's rule and apply it to all the class's instances."""
        try:
            n = eng.apply_class_rule((body.get("cls") or "").strip(), body.get("ops"))
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "n": n, "stats": eng.stats(), "classes": eng.state.class_names()}

    @app.get("/api/class_rules")
    def class_rules():
        return {"rules": eng.class_rules_summary()}

    @app.get("/api/class_rule")
    def class_rule(cls: str = ""):
        """The FULL saved refine chain (ops + kw) for a class, so Refine can reload it into the live chain."""
        return {"cls": cls, "ops": eng.class_rule_for(cls)}

    @app.get("/api/recommend_interesting")
    def recommend_interesting(n: int = 60, metric: str = "entropy", offset: int = 0, limit: int = 60):
        """Active-learning acquisition: unassigned instances most informative to label next (classifier most
        uncertain; falls back to lowest-detection-score when nothing is trained)."""
        recs = eng.recommend_interesting(int(n), metric=metric)
        page = recs[offset:offset + limit]
        items = [{"iuid": u, "cls": (eng.state.class_name(c) if c is not None else "?"),
                  "score": round(float(s), 3),
                  "image_id": str(int(eng.state.meta[u].image_id))} for u, c, s in page]
        return {"total": len(recs), "trained": getattr(eng, "_clf", None) is not None, "items": items}

    # ---- training-loop orchestration ----
    @app.post("/api/train/launch")
    def train_launch(body: dict = Body(default={})):
        return eng.launch_training(mode=body.get("mode", "finetune"), epochs=body.get("epochs") or None,
                                   config_name=(body.get("config_name") or None),
                                   image_root=(body.get("image_root") or None),
                                   json_val=(body.get("json_val") or None), json_test=(body.get("json_test") or None),
                                   partial=bool(body.get("partial", True)),
                                   class_agnostic=bool(body.get("class_agnostic", False)),
                                   extra_train_json=(body.get("extra_train_json") or None),
                                   extra_image_root=(body.get("extra_image_root") or None))

    @app.get("/api/train/status")
    def train_status():
        return eng.training_status()

    @app.post("/api/train/stop")
    def train_stop(body: dict = Body(default={})):
        return {"ok": eng.stop_training()}

    @app.post("/api/train/adopt")
    def train_adopt(body: dict = Body(default={})):
        return eng.adopt_checkpoint(body.get("ckpt") or "", force=bool(body.get("force", False)))

    @app.post("/api/train/overfit_check")
    def train_overfit_check(body: dict = Body(default={})):
        """Gate 2: train on a few human-verified instances + eval on the SAME images (train==val==test). High
        segm/AP => the pipeline can learn; low => the loss/LR/label/inference path is broken, not the data."""
        return eng.launch_overfit_check(n=int(body.get("n", 12)), epochs=int(body.get("epochs", 60)))

    @app.get("/api/classes")
    def classes():
        return {"classes": eng.classes_summary()}

    @app.post("/api/merge_classes")
    def merge_classes(body: dict = Body(...)):
        res = eng.merge_classes(body.get("sources") or [], body.get("into") or "")
        return {**res, "stats": eng.stats(), "classes": eng.state.class_names()}

    # ---- within-class substructure (contrastive + FINCH) ----
    @app.post("/api/subcluster")
    def subcluster(body: dict = Body(...)):
        feats = body.get("features") or ["decoder"]
        return eng.subcluster(body.get("target") or "", spec={m: 1.0 for m in feats},
                              dim=int(body.get("dim", 64)), epochs=int(body.get("epochs", 150)),
                              temperature=float(body.get("temperature", 0.2)),
                              device=body.get("device", "cpu"))

    @app.post("/api/subcluster_level")
    def subcluster_level(body: dict = Body(...)):
        eng.subcluster_set_level(int(body["level"]))
        return {"ok": True}

    @app.get("/api/subclusters")
    def subclusters():
        sc = eng._subcluster
        if not sc:
            return {"active": False, "rows": []}
        return {"active": True, "target": sc["target"], "level": sc["level"],
                "levels": [{"i": i, "n": int(c)} for i, c in enumerate(sc["counts"])],
                "rows": eng.subcluster_view()}

    @app.get("/api/subcluster_instances")
    def subcluster_instances(subpid: str, offset: int = 0, limit: int = 60):
        iu = eng.subcluster_iuids(subpid)
        return {"total": len(iu), "items": _items(iu[offset:offset + limit])}

    @app.post("/api/sam_prompt_preview")
    def sam_prompt_preview(body: dict = Body(...)):
        """Show WHERE SAM's positive/negative prompt points (and box) are sampled, for the given iuid
        and op chain — pure geometry, so it works even before a checkpoint is downloaded."""
        img, npos, nneg = eng.sam_prompt_preview(body["iuid"], body.get("ops", []),
                                                 n_pos=int(body.get("n_pos", 10)),
                                                 n_neg=int(body.get("n_neg", 12)),
                                                 margin=int(body.get("margin", 24)))
        return {"img": _png_data_uri(img), "n_pos": npos, "n_neg": nneg}

    @app.get("/api/find_instances")
    def find_instances(query: str = "", limit: int = 60):
        """Search instances for the Refine picker — match iuid prefix, source filename, class name, or
        image id. Empty query returns the first `limit` instances so the picker is never blank."""
        q = (query or "").strip().lower()
        out = []
        for u, m in eng.state.meta.items():
            if m.merged_into is not None:
                continue
            if q:
                cls = (eng.state.class_name(m.assigned_class) or "").lower() if m.assigned_class else ""
                if not (u.lower().startswith(q) or q in eng._src_name(u).lower()
                        or q in cls or q == str(int(m.image_id))):
                    continue
            out.append(u)
            if len(out) >= limit:
                break
        return {"total": len(out), "items": _items(out)}

    @app.get("/api/sam_status")
    def sam_status(family: str = ""):
        from . import refine as _rf
        ckpt, mtype = _rf.find_sam_checkpoint(family=family or None)
        d = _rf._sam_dir()
        avail = sorted({_rf.detect_sam_family(c) for c in [*d.glob("*.pth"), *d.glob("*.pt")]})
        return {"installed": _rf.sam_available(), "ckpt": ckpt, "model_type": mtype,
                "family": (_rf.detect_sam_family(ckpt) if ckpt else None), "families": avail}

    @app.post("/api/sam_setup")
    def sam_setup(body: dict = Body(default={})):
        """Download a SAM checkpoint (default vit_b, ~375 MB) into the cache dir so `sam` refine works.
        Reports a clear instruction if the `segment-anything` package isn't installed."""
        from . import refine as _rf
        try:
            path = _rf.ensure_sam_checkpoint(body.get("model_type", "vit_b"))
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "ckpt": path, "installed": _rf.sam_available()}

    @app.post("/api/split")
    def split(body: dict = Body(...)):
        n = eng.split_instances(body.get("iuids") or [])
        return {"ok": True, "n": n, "stats": eng.stats()}

    @app.get("/api/rejected")
    def rejected(offset: int = 0, limit: int = 60):
        iu = eng.background_iuids()
        return {"total": len(iu), "items": _items(iu[offset:offset + limit])}

    @app.post("/api/unreject")
    def unreject(body: dict = Body(...)):
        if body.get("iuids"):
            eng.unreject(body["iuids"])
        return {"ok": True, "stats": eng.stats()}

    def _thr(body):                                  # optional per-run detection knobs (None = config default)
        st = body.get("score_thresh"); nms = body.get("nms_iou")
        return (float(st) if st not in (None, "") else None), (float(nms) if nms not in (None, "") else None)

    @app.post("/api/sample")
    def sample(body: dict = Body(default={})):
        st, nms = _thr(body)
        info = eng.sample_more(int(body.get("n", 10)), smart=bool(body.get("smart", False)),
                               score_thresh=st, nms_iou=nms)
        return {"ok": True, "stats": eng.stats(),
                "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str))},
                "features": eng.available_features()}

    @app.post("/api/infer_dir")
    def infer_dir(body: dict = Body(...)):
        d = (body.get("dir") or "").strip()
        if not d or not Path(d).exists():
            raise HTTPException(400, f"folder not found: {d}")
        st, nms = _thr(body)
        info = eng.infer_dir(d, limit=int(body.get("limit", 50)), mode=body.get("mode", "new"),
                             score_thresh=st, nms_iou=nms)
        return {"ok": "error" not in info, **info, "features": eng.available_features()}

    @app.post("/api/preview_infer")
    def preview_infer(body: dict = Body(default={})):
        """Non-destructive preview: render the current model's predictions on a random sample of N
        processed images at the given score/nms thresholds (does NOT modify the collection)."""
        st, nms = _thr(body)
        res = eng.preview_processed(int(body.get("n", 6)), score_thresh=st, nms_iou=nms)
        return {"n_inst": res["n_inst"], "n_before": res.get("n_before", 0), "sampled": res.get("sampled", 0),
                "items": [{"before": _png_data_uri(it["before"]), "after": _png_data_uri(it["after"]),
                           "caption": it["caption"]} for it in res["items"]]}

    @app.post("/api/reinfer")
    def reinfer(body: dict = Body(default={})):
        """Re-run the (adopted) model on ALREADY-processed images at the given score/nms thresholds.
        mode: append (add alongside old) | replace (hide old un-curated first, keep curation)."""
        st, nms = _thr(body)
        info = eng.reinfer_processed(mode=body.get("mode", "replace"),
                                     limit=(int(body["limit"]) if body.get("limit") else None),
                                     score_thresh=st, nms_iou=nms)
        return {"ok": "error" not in info, **info, "features": eng.available_features()}

    @app.post("/api/infer_upload")
    def infer_upload(body: dict = Body(...)):
        """Run inference on browser-uploaded images (base64 data-URIs); they're saved under
        <project>/uploads/ so the crops remain readable afterwards."""
        import hashlib
        import cv2
        import numpy as np
        updir = Path(eng.store.dir) / "uploads"
        updir.mkdir(parents=True, exist_ok=True)
        paths = []
        for d in (body.get("images") or []):
            raw = base64.b64decode((d or "").split(",")[-1])
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                continue
            p = updir / f"{hashlib.blake2b(raw, digest_size=8).hexdigest()}.png"
            cv2.imwrite(str(p), arr)
            paths.append(str(p))
        if not paths:
            raise HTTPException(400, "no decodable images")
        info = eng.ingest_paths(paths)
        return {"ok": True, **info, "features": eng.available_features()}

    return app


def main():
    ap = argparse.ArgumentParser(description="qseg curator — custom web frontend")
    ap.add_argument("--project", required=True, help="project dir (created by the curator)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7870)
    args = ap.parse_args()
    import uvicorn
    print(f"qseg curator (custom frontend) → http://{args.host}:{args.port}  project={args.project}")
    uvicorn.run(create_app(args.project), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
