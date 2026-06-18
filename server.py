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

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB / "index.html").read_text()

    @app.get("/app.js")
    def appjs():
        return Response((WEB / "app.js").read_text(), media_type="application/javascript")

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

    @app.get("/api/partitions")
    def partitions(offset: int = 0, limit: int = 100, query: str = ""):
        rows = _partition_rows(eng, query)
        return {"total": len(rows), "rows": rows[offset:offset + limit]}

    def _items(iuids):
        return [{"iuid": u, "caption": eng._caption(u), "image_id": int(eng.state.meta[u].image_id)} for u in iuids]

    @app.get("/api/instances")
    def instances(pid: str, offset: int = 0, limit: int = 60):
        iu = eng.partition_iuids(pid)
        return {"total": len(iu), "items": _items(iu[offset:offset + limit])}

    @app.get("/api/crop")
    def crop(iuid: str, mask: int = 1, max_side: int = 256):
        if iuid not in eng.state.meta:
            raise HTTPException(404, "unknown iuid")
        arr = eng.crop(iuid, mask_overlay=bool(mask), max_side=int(max_side))
        return Response(_png_bytes(arr), media_type="image/png",
                        headers={"Cache-Control": "max-age=31536000"})   # content-stable per (iuid,mask)

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
        if len(iuids) >= 2:
            eng.merge_instances(iuids, mode=body.get("mode", "union"))
        return {"ok": True, "stats": eng.stats()}

    @app.post("/api/export")
    def export(body: dict = Body(default={})):
        path = eng.export_coco()
        return {"ok": True, "path": str(path)}

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
                  "image_id": int(eng.state.meta[u].image_id)} for u, c, conf in page]
        return {"total": len(preds), "items": items}

    @app.post("/api/apply_predictions")
    def apply_predictions(body: dict = Body(...)):
        cid = eng.state.class_id_by_name(body["only_class"]) if body.get("only_class") else None
        n, _ = eng.apply_predictions(float(body.get("thresh", 0.5)), only_class=cid,
                                     exclude=set(body.get("exclude") or []))
        return {"ok": True, "n": n, "stats": eng.stats(), "classes": eng.state.class_names()}

    @app.post("/api/refine_preview")
    def refine_preview(body: dict = Body(...)):
        before, after = eng.refine_preview(body["iuid"], body.get("ops", []), mask_overlay=True)
        return {"before": _png_data_uri(before), "after": _png_data_uri(after)}

    @app.post("/api/apply_refine")
    def apply_refine(body: dict = Body(...)):
        eng.apply_refine(body["iuid"], body.get("ops", []))
        return {"ok": True, "stats": eng.stats()}

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

    @app.post("/api/sample")
    def sample(body: dict = Body(default={})):
        info = eng.sample_more(int(body.get("n", 10)), smart=bool(body.get("smart", False)))
        return {"ok": True, "stats": eng.stats(),
                "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str))},
                "features": eng.available_features()}

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
