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
        return {"ok": False, "error": rep["error"], "skipped": rep.get("skipped_names", []),
                "dropped_nan": rep.get("dropped_nan_features", [])}
    clf = getattr(eng, "_clf", None)
    classes = [eng.state.class_name(c) for c in getattr(clf, "classes", [])]
    youden = {}
    for cid, cv in ((rep.get("pr") or {}).get("curves") or {}).items():
        if cv.get("youden") is not None:
            youden[eng.state.class_name(cid)] = round(float(cv["youden"]), 2)
    return {"ok": True, "algo": rep.get("algo"), "n_classes": rep.get("n_classes", len(classes)),
            "classes": classes, "skipped": rep.get("skipped_names", []), "youden": youden,
            "dropped_nan": rep.get("dropped_nan_features", [])}


def _partition_rows(eng: CuratorEngine, query: str = "", kind: str = "all"):
    """Formatted partition rows (full list; the API windows with offset/limit). `kind` scopes the list so
    FINCH partitions aren't crowded out of the first page by many class pseudo-partitions (classes sort
    first): 'all' | 'part' (FINCH clusters only) | 'class' (assigned-class pseudo-partitions only)."""
    rows = [{"pid": str(r["pid"]), "size": int(r["size"]),
             "score": r["mean_score"], "cls": r["majority_class"] or ""}
            for r in eng.partition_view()]
    if kind == "part":
        rows = [r for r in rows if not r["pid"].startswith("class:")]
    elif kind == "class":
        rows = [r for r in rows if r["pid"].startswith("class:")]
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
            "feature_nan": sorted(eng.feature_nan_methods()),   # NaN/inf features -> classifier marks them unusable
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

    @app.get("/api/activity")
    def activity(bins: int = 48, session_gap_s: float = 1800.0):
        """Read-only curation-activity timeline from the append-only logs — for the Activity tab."""
        return eng.activity_summary(bins=int(bins), session_gap_s=float(session_gap_s))

    @app.get("/api/partitions")
    def partitions(offset: int = 0, limit: int = 100, query: str = "", kind: str = "all"):
        rows = _partition_rows(eng, query, kind)
        return {"total": len(rows), "rows": rows[offset:offset + limit]}

    def _items(iuids):
        # image_id is a 56-bit hash (> 2^53) -> emit as a STRING so JS doesn't round it (a rounded id
        # round-trips to a non-existent image -> "no instances on this image"). JS passes it back verbatim.
        return [{"iuid": u, "caption": eng._caption(u), "image_id": str(int(eng.state.meta[u].image_id))} for u in iuids]

    @app.get("/api/instances")
    def instances(pid: str, offset: int = 0, limit: int = 60):
        iu = eng.partition_iuids(pid)
        return {"total": len(iu), "items": _items(iu[offset:offset + limit])}

    @app.get("/api/partition_suggestion")
    def partition_suggestion(pid: str, gate_mult: float = 1.0, thr: float | None = None):
        """1-NN 'most likely class' (+ reject likelihood, + 'no likely class' gate) for the selected
        partition. Read-only; n/a and error cases come back as JSON (not HTTP errors)."""
        return eng.partition_class_suggestion(pid, gate_mult=float(gate_mult),
                                              thr=(float(thr) if thr is not None else None))

    @app.get("/api/instance_peers")
    def instance_peers(iuid: str, limit: int = 120):
        """Partition peers of an instance — the same-partition samples of the Refine tab's currently
        previewed instance (NOT a global search). Falls back to the instance alone when it has no
        partition (rejected / merged-away / not clustered)."""
        if iuid not in eng.state.meta:
            raise HTTPException(404, "unknown iuid")
        pid = eng.partition_of(iuid)
        iu = eng.partition_iuids(pid) if pid is not None else [iuid]
        return {"pid": (str(pid) if pid is not None else None), "total": len(iu),
                "items": _items(iu[:limit])}

    @app.get("/api/crop")
    def crop(iuid: str, mask: int = 1, max_side: int = 256, context: int = 0):
        if iuid not in eng.state.meta:
            raise HTTPException(404, "unknown iuid")
        arr = eng.crop(iuid, mask_overlay=bool(mask), max_side=int(max_side), context=bool(context))
        return Response(_png_bytes(arr), media_type="image/png",
                        headers={"Cache-Control": "max-age=31536000"})   # content-stable per (iuid,mask,context)

    @app.post("/api/crops")
    def crops(body: dict = Body(...)):
        """Batch crop thumbnails: a whole grid page of crops as data-URIs in ONE request (vs one GET per
        cell). Per-crop work is the same server-cached eng.crop(); this collapses the round-trips + the
        browser's 6-connections-per-host cap so a set of instances renders together at scale."""
        mask, ctx, ms = bool(body.get("mask", 1)), bool(body.get("context", 0)), int(body.get("max_side", 256))
        out = {}
        for u in (body.get("iuids") or [])[:200]:
            if u in eng.state.meta:
                out[u] = _png_data_uri(eng.crop(u, mask_overlay=mask, max_side=ms, context=ctx))
        return {"crops": out}

    @app.get("/api/images")
    def images(query: str = "", limit: int = 100):
        """Windowed image-id list (most-populated first) + per-image instance count, respecting the
        active ingest scope — so the file picker never ships thousands of options."""
        return eng.image_counts(query=query, limit=limit)

    @app.get("/api/ingests")
    def ingests():
        """List recorded (re)inference runs + the active scope, for the 'Scope: latest ingest' selector."""
        return {"ingests": eng.list_ingests(), "scope": eng._scope_id}

    @app.post("/api/scope")
    def scope(body: dict = Body(default={})):
        """Restrict the cluster pool + image picker to one ingest's instances (ingest_id=None/'all' = clear)."""
        res = eng.set_scope(body.get("ingest_id"))
        return {**res, "stats": eng.stats()}

    @app.post("/api/reset")
    def reset(body: dict = Body(default={})):
        """Drop EVERYTHING (full reset) — every instance + all curation + classes + ingest/merge/history
        logs; keeps only the project config. IRREVERSIBLE: requires {"confirm": true}."""
        if not body.get("confirm"):
            raise HTTPException(400, "reset requires confirm=true")
        return {"ok": True, "stats": eng.reset(keep_config=True)}

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

    # ---- image-level release gate ----
    @app.get("/api/release_images")
    def release_images(filter: str = "all", offset: int = 0, limit: int = 24):
        """Windowed list of FINAL images (fully categorized, >1 instance) + gate stats, for the Release tab."""
        return eng.release_view(filter=filter, offset=int(offset), limit=int(limit))

    @app.post("/api/release_set")
    def release_set(body: dict = Body(...)):
        """Accept/reject whole images FOR RELEASE (image-level gate; not instance reject). status in
        {accepted, rejected, pending} (pending clears)."""
        ids = body.get("image_ids") or ([body["image_id"]] if body.get("image_id") is not None else [])
        n = eng.set_release([int(i) for i in ids], str(body.get("status", "")))
        return {"ok": True, "n": n, "stats": eng.release_stats()}

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
                              feature=body.get("feature", "raddino"), k=int(body.get("k", 12)))
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
        return {"available": avail, "has_raddino": "raddino" in avail,
                "nan": sorted(eng.feature_nan_methods())}      # features with NaN/inf -> not classifier-selectable

    @app.post("/api/scaled_pseudolabel")
    def scaled_pseudolabel(body: dict = Body(default={})):
        """Train-on-sample, propagate-at-scale: shard the seg model over a folder (or the processed pool),
        propagate labels (classifier | reference | raw), write per-shard COCO + a merged COCO. RAM-bounded."""
        st, nms = _thr(body)
        rep = eng.scaled_pseudolabel(
            directory=(body.get("dir") or "").strip() or None, out_dir=(body.get("out_dir") or "").strip() or None,
            shard_size=int(body.get("shard_size", 2000)), method=body.get("method", "classifier"),
            thresh=float(body.get("thresh", 0.5)), score_thresh=st, nms_iou=nms,
            pool=body.get("pool", "bbox"), class_agnostic=bool(body.get("class_agnostic", False)),
            limit=(int(body["limit"]) if body.get("limit") else None))
        if rep.get("error"):
            raise HTTPException(400, rep["error"])
        return rep

    @app.post("/api/compute_raddino")
    def compute_raddino(body: dict = Body(default={})):
        """Extract mask-pooled RAD-DINO embeddings for every instance (no re-detection) -> 'raddino' becomes
        a selectable feature for clustering / classifier / substructure / merge-rec / reference suggest."""
        rep = eng.compute_raddino(force=bool(body.get("force", False)), pool=body.get("pool", "mask"))
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
        rep = eng.load_reference_bank(path, rebuild=bool(body.get("rebuild", False)),
                                      image_root=(body.get("image_root") or "").strip() or None)
        if rep.get("error"):
            raise HTTPException(400, rep["error"])
        return {"ok": True, "class_names": eng.state.class_names(), "n_classes": rep["classes"],
                "exemplars": rep["exemplars"], "added_classes": rep["added_classes"],
                "image_root": rep.get("image_root"), "exemplars_ok": rep.get("exemplars_ok")}

    @app.get("/api/reference/classes")
    def reference_classes():
        return {"loaded": getattr(eng, "_ref_bank", None) is not None, "rows": eng.reference_classes(),
                "last_coco_path": eng.state.config.get("last_reference_coco", "")}

    @app.get("/api/fs/suggest")
    def fs_suggest(path: str = "", limit: int = 40):
        """Filesystem path completions for a path input (Tab-complete + datalist). Lists the children of
        `path` when it ends in '/', else siblings in its directory whose name prefix-matches. Dirs get a
        trailing '/'; only dirs + .json files are surfaced (the reference bank wants a coco.json)."""
        from pathlib import Path as _P
        p = (path or "").strip()
        base = _P(p) if (p.endswith("/") or p == "") else _P(p).parent
        stem = "" if (p.endswith("/") or p == "") else _P(p).name
        base = base if str(base) else _P(".")
        try:
            entries = sorted(base.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except (OSError, PermissionError):
            return {"items": []}
        out = []
        for e in entries:
            if stem and not e.name.startswith(stem):
                continue
            if e.is_dir():
                out.append(str(e) + "/")
            elif e.suffix.lower() == ".json":
                out.append(str(e))
            if len(out) >= int(limit):
                break
        return {"items": out}

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
                                    topk=int(body.get("topk", 5)), use_csls=bool(body.get("csls", False)))
        if rep.get("error"):
            raise HTTPException(400, rep["error"])
        return rep

    @app.post("/api/reference/find")
    def reference_find(body: dict = Body(...)):
        """Reverse retrieval: rank the collection's instances/partitions by similarity to a reference CLASS,
        across ALL present instances — no partition preselect needed."""
        cls = str(body.get("cls", "")).strip()
        if not cls:
            raise HTTPException(400, "cls required")
        rep = eng.reference_find_instances(cls, k=int(body.get("k", 24)), knn=int(body.get("knn", 8)),
                                           use_csls=bool(body.get("csls", False)),
                                           dedup_partition=bool(body.get("dedup_partition", True)))
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
                                               mask_overlay=bool(body.get("mask", 1)),
                                               context=bool(body.get("context", 0)))
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

    @app.post("/api/propagate_refinement")
    def propagate_refinement(body: dict = Body(...)):
        """Within-partition propagation: replay a refine chain across a reference instance's partition, so
        its peers get the SAME segmentation treatment. ops default to the reference's recorded chain; pid
        defaults to its partition; match_thresh (0-1) gates to RAD-DINO-similar members (GPU/HF)."""
        mt = body.get("match_thresh")
        try:
            res = eng.propagate_refinement(str(body["ref_iuid"]), pid=body.get("pid"),
                                           ops=body.get("ops"),
                                           match_thresh=(float(mt) if mt is not None else None))
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        if res.get("error"):
            raise HTTPException(400, res["error"])
        return {"ok": True, **res, "stats": eng.stats()}

    def _chain_label(res: dict) -> dict:
        best = res.get("best", {})
        names = [o.get("name") for o in best.get("chain", [])] or ["(leave as-is)"]
        return {"kind": res.get("kind"), "reward": res.get("reward", "geometric"),
                "chain": names, "ops": best.get("chain", []),
                "score": round(float(best.get("score", 0.0)), 3), "breakdown": best.get("breakdown", {}),
                "candidates": [{"chain": [o.get("name") for o in c["chain"]] or ["(leave as-is)"],
                                "score": round(float(c["score"]), 3)} for c in res.get("candidates", [])]}

    @app.post("/api/auto_refine_preview")
    def auto_refine_preview(body: dict = Body(...)):
        """Stage-1: search the candidate chains for the one THIS mask needs, return its before/after + pick."""
        try:
            before, after, res = eng.auto_refine_preview(body["iuid"], kind=str(body.get("kind", "auto")))
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"before": _png_data_uri(before), "after": _png_data_uri(after), "pick": _chain_label(res)}

    @app.post("/api/auto_refine_apply")
    def auto_refine_apply(body: dict = Body(...)):
        try:
            res = eng.auto_refine_apply(body["iuid"], kind=str(body.get("kind", "auto")))
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "pick": _chain_label(res), "stats": eng.stats()}

    @app.post("/api/auto_refine_many")
    def auto_refine_many(body: dict = Body(...)):
        """Auto-refine a whole partition or class — each instance gets its OWN best chain."""
        kind = str(body.get("kind", "auto"))
        try:
            if body.get("cls"):
                out = eng.auto_refine_class(str(body["cls"]), kind=kind)
            else:
                out = eng.auto_refine_partition(str(body["pid"]), kind=kind)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "n": out["n"], "kind": out.get("kind"), "reward": out.get("reward"),
                "summary": out["summary"], "stats": eng.stats()}

    @app.post("/api/auto_refine_consensus")
    def auto_refine_consensus(body: dict = Body(...)):
        """Category consensus: the class's MODAL best chain. apply=false previews it (no mutation);
        apply=true applies it uniformly to all the class's instances and saves it as the class rule."""
        cls = (body.get("cls") or "").strip()
        if not cls:
            raise HTTPException(400, "consensus needs a class name")
        kind = str(body.get("kind", "auto"))
        try:
            if body.get("apply"):
                out = eng.auto_refine_class_consensus(cls, kind=kind)
            else:
                cid = eng._resolve_cid(cls)
                out = eng.auto_refine_consensus(eng.class_rule_members(cid), kind=kind) if cid \
                    else {"n": 0, "chain": [], "summary": []}
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "n": out["n"], "kind": out.get("kind"), "reward": out.get("reward"),
                "votes": out.get("votes", 0), "applied": out.get("applied", 0),
                "chain": [o.get("name") for o in out.get("chain", [])] or ["(leave as-is)"],
                "summary": out.get("summary", []), "stats": eng.stats(), "classes": eng.state.class_names()}

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

    @app.get("/api/class_samples")
    def class_samples(class_id: str = "", limit: int = 24):
        """Representative iuids for a class (Classes-tab sample preview); render each via /api/crop."""
        return {"iuids": eng.class_samples(class_id, int(limit))}

    @app.post("/api/merge_classes")
    def merge_classes(body: dict = Body(...)):
        res = eng.merge_classes(body.get("sources") or [], body.get("into") or "")
        return {**res, "stats": eng.stats(), "classes": eng.state.class_names()}

    # ---- nested taxonomy (superclass -> concept -> part leaves) ----
    @app.get("/api/taxonomy")
    def taxonomy():
        return eng.taxonomy_tree()

    @app.post("/api/taxonomy/seed")
    def taxonomy_seed(body: dict = Body(default={})):
        """Load the nested taxonomy seed (or a given path) into state."""
        rep = eng.seed_taxonomy(body.get("path") or None, replace=bool(body.get("replace", False)))
        return {"ok": True, **rep}

    @app.post("/api/taxonomy/temp")
    def taxonomy_temp(body: dict = Body(...)):
        """Flag class_ids temp/scratch (excluded from export) or un-flag (temp=false)."""
        return eng.set_class_temp(body.get("class_ids") or [], bool(body.get("temp", True)))

    @app.post("/api/taxonomy/assign_leaf")
    def taxonomy_assign_leaf(body: dict = Body(...)):
        """Place a leaf class under a concept (promote a temp/scratch class into the taxonomy)."""
        return eng.assign_leaf(body["class_id"], body.get("concept"))

    @app.get("/api/taxonomy/concepts")
    def taxonomy_concepts():
        """Flat concept list (for the 'promote to concept' picker), grouped by superclass."""
        return {"concepts": [{"id": c.concept_id, "name": c.name, "superclass": c.superclass}
                             for c in eng.state.concepts.values()]}

    @app.get("/api/taxonomy/release_qc")
    def taxonomy_release_qc():
        """Per-image part-rule completeness gate — images that fail are held back from the release."""
        return eng.release_qc()

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
        pkg = _rf.samhq_available() if family == "samhq" else _rf.sam_available()   # the family's package
        return {"installed": pkg, "samhq_installed": _rf.samhq_available(), "ckpt": ckpt, "model_type": mtype,
                "family": (_rf.detect_sam_family(ckpt) if ckpt else None), "families": avail}

    @app.post("/api/sam_setup")
    def sam_setup(body: dict = Body(default={})):
        """Download a SAM checkpoint into the cache dir so `sam` refine works. family='samhq' fetches the
        SAM-HQ weights (HF mirror) instead of vanilla SAM. Reports a clear instruction if the package
        isn't installed."""
        from . import refine as _rf
        fam, mt = body.get("family", "sam"), body.get("model_type", "vit_b")
        try:
            path = _rf.ensure_samhq_checkpoint(mt) if fam == "samhq" else _rf.ensure_sam_checkpoint(mt)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "ckpt": path, "family": fam,
                "installed": _rf.samhq_available() if fam == "samhq" else _rf.sam_available()}

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

    def _rad(body):                                  # opt-in: chain RAD-DINO feature extraction after the run
        return {"with_raddino": bool(body.get("with_raddino", False)),
                "raddino_pool": str(body.get("raddino_pool", "mask"))}

    @app.post("/api/sample")
    def sample(body: dict = Body(default={})):
        st, nms = _thr(body)
        info = eng.sample_more(int(body.get("n", 10)), smart=bool(body.get("smart", False)),
                               score_thresh=st, nms_iou=nms, **_rad(body))
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
                             score_thresh=st, nms_iou=nms, **_rad(body))
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
                                     score_thresh=st, nms_iou=nms, **_rad(body))
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
        info = eng.ingest_paths(paths, **_rad(body))
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
