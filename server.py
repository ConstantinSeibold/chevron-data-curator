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
from pathlib import Path
from typing import Optional

from .engine import CuratorEngine

WEB = Path(__file__).parent / "web"


def _png_bytes(arr) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


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

    @app.get("/api/instances")
    def instances(pid: str, offset: int = 0, limit: int = 60):
        iu = eng.partition_iuids(pid)
        page = iu[offset:offset + limit]
        return {"total": len(iu), "items": [{"iuid": u, "caption": eng._caption(u)} for u in page]}

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
