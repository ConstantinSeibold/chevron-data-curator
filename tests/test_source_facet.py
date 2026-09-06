"""Proposal-source facet: tag ingests with the model that proposed them, then filter by source across every
tab via the shared view predicate (composes with ingest scope). Read-only + additive (facet None = no change).
Run: pytest chevron/tests/test_source_facet.py -q
"""
from __future__ import annotations

import numpy as np


def _engine(tmp_path, n=40):
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x", "config_name": "modelA"},
                      "features": {"model_features": ["decoder"]}})
    rng = np.random.default_rng(0)
    feats = rng.normal(0, 1, (n, 8)).astype(np.float32)
    order, meta = [], {}
    for i in range(n):
        bid = "bA" if i < n // 2 else "bB"            # two proposal batches
        order.append(f"u{i}"); meta[f"u{i}"] = InstanceMeta(f"u{i}", bid, i, 1000 + i)
    eng.collection = {"records": [{"iuid": f"u{i}", "row": i, "score": 0.6} for i in range(n)],
                      "n_images": n, "feats": {"decoder": feats}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    # register two ingests tagging the sources
    eng.store.append_ingest_event({"ingest_id": "ing_000", "batch_ids": ["bA"], "source": "modelA", "n_instances": n // 2})
    eng.store.append_ingest_event({"ingest_id": "ing_001", "batch_ids": ["bB"], "source": "modelB", "n_instances": n // 2})
    eng._bsrc_cache = None
    return eng, order


def test_sources_list_and_source_of(tmp_path):
    eng, order = _engine(tmp_path)
    s = eng.sources()
    assert {x["source"] for x in s["sources"]} == {"modelA", "modelB"} and s["active"] is None
    assert eng._source_of("u0") == "modelA" and eng._source_of("u39") == "modelB"


def test_source_filter_threads_through_scope_and_pool(tmp_path):
    eng, order = _engine(tmp_path)
    assert len(eng._pool_iuids()) == 40                      # no facet -> all
    eng.set_source_filter(["modelA"])
    assert eng._in_scope("u0") and not eng._in_scope("u39")  # folded into the shared predicate
    assert len(eng._pool_iuids()) == 20                      # pool (and thus every tab) filters by source
    eng.set_source_filter(None)
    assert len(eng._pool_iuids()) == 40                      # cleared


def test_source_filter_composes_with_ingest_scope(tmp_path):
    eng, order = _engine(tmp_path)
    eng.set_scope("ing_000")                                 # scope to modelA's ingest
    eng.set_source_filter(["modelB"])                        # ...but facet to modelB -> empty intersection
    assert len(eng._pool_iuids()) == 0
    eng.set_source_filter(["modelA"])
    assert len(eng._pool_iuids()) == 20


def test_projection_carries_source_and_filters(tmp_path):
    eng, order = _engine(tmp_path)
    r = eng.projection_points(method="pca")
    assert {p["source"] for p in r["points"]} == {"modelA", "modelB"}
    eng.set_source_filter(["modelA"])
    r2 = eng.projection_points(method="pca")
    assert r2["n"] == 20 and {p["source"] for p in r2["points"]} == {"modelA"}


def test_ingest_paths_tags_source(tmp_path, monkeypatch):
    # _record_ingest should stamp the source onto the ingest event (default = model config_name)
    eng, order = _engine(tmp_path)
    ev = eng._record_ingest([{"batch_id": "bC", "image_id": 5000}], context={"source": "modelC", "mode": "append"})
    assert ev["source"] == "modelC"
    eng._bsrc_cache = None
    assert "modelC" in {x["source"] for x in eng.sources()["sources"]} or eng._batch_source().get("bC") == "modelC"


def test_endpoints_sources_and_filter(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, order = _engine(tmp_path)
    c = TestClient(create_app(engine=eng))
    s = c.get("/api/sources").json()
    assert {x["source"] for x in s["sources"]} == {"modelA", "modelB"}
    r = c.post("/api/source_filter", json={"sources": ["modelB"]}).json()
    assert r["active"] == ["modelB"]
    assert len(eng._pool_iuids()) == 20
    c.post("/api/source_filter", json={"sources": None})
    assert len(eng._pool_iuids()) == 40


# ---- tagged-COCO import (slice 2): ingest an external model's proposals as a source ----------------
def _disk_engine(tmp_path):
    """Native collection of circle masks on 2 real PNGs, feats = decoder(8) + shapecoord(29)."""
    import cv2
    from pycocotools import mask as mu
    from chevron import collect as _co
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x", "config_name": "native"},
                      "features": {"model_features": ["decoder"]}})
    for f in ("im0.png", "im1.png"):
        cv2.imwrite(str(tmp_path / f), (np.random.default_rng(0).random((64, 64, 3)) * 200).astype(np.uint8))

    def _rle(m):
        r = mu.encode(np.asfortranarray(m.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii"); return r

    def _circ(cx):
        mm = np.zeros((64, 64), np.uint8); cv2.circle(mm, (cx, 30), 8, 1, -1); return mm > 0
    recs, order, meta, dec, shp = [], [], {}, [], []
    for i in range(4):
        f = "im0.png" if i % 2 == 0 else "im1.png"; m = _circ(20 + 8 * i); u = f"n{i}"
        recs.append({"iuid": u, "row": i, "inst_id": i, "image_id": 1000 + (i % 2), "H": 64, "W": 64, "score": 0.9,
                     "rle": _rle(m), "file_name": str(tmp_path / f), "abs_path": str(tmp_path / f), "batch_id": "native",
                     "cx": 0.3, "cy": 0.5, "bw": 0.2, "bh": 0.2, "box_area": 0.04, "mask_area_frac": float(m.mean())})
        order.append(u); meta[u] = InstanceMeta(u, "native", i, 1000 + (i % 2))
        dec.append(np.zeros(8, np.float32)); shp.append(_co.shapecoord_vector(m))
    eng.collection = {"records": recs, "n_images": 2, "feats": {"decoder": np.array(dec, np.float32),
                      "shapecoord": np.array(shp, np.float32), "_shapecoord_cols": ["c"] * 29}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    return eng


def _write_coco(tmp_path, name, n_by_img):
    """A COCO of rectangle proposals on im0/im1 (matched by basename). n_by_img=(n0,n1)."""
    import cv2, json
    from pycocotools import mask as mu
    def _rect():
        mm = np.zeros((64, 64), np.uint8); cv2.rectangle(mm, (35, 10), (55, 40), 1, -1); return mm > 0
    def _rle(m):
        r = mu.encode(np.asfortranarray(m.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii"); return r
    coco = {"images": [{"id": 7, "file_name": "im0.png"}, {"id": 8, "file_name": "im1.png"}], "annotations": []}
    for _ in range(n_by_img[0]):
        coco["annotations"].append({"image_id": 7, "score": 0.8, "segmentation": _rle(_rect())})
    for _ in range(n_by_img[1]):
        coco["annotations"].append({"image_id": 8, "score": 0.7, "segmentation": _rle(_rect())})
    p = tmp_path / name; json.dump(coco, open(p, "w")); return str(p)


def test_import_proposals_tags_source_shared_space_no_poison(tmp_path):
    eng = _disk_engine(tmp_path)
    cp = _write_coco(tmp_path, "sam.coco.json", (2, 1))
    r = eng.import_proposals_coco(cp, source="sam")
    assert r["ok"] and r["n_imported"] == 3 and r["source"] == "sam" and r["n_images"] == 2
    assert len(eng.state.meta) == 7                                        # 4 native + 3 imported
    assert {s["source"]: s["n"] for s in eng.sources()["sources"]} == {"native": 4, "sam": 3}
    eng.state.assert_aligned(eng.collection["feats"]["decoder"].shape[0])  # feats stay row-aligned
    row = eng.state.meta[[u for u in eng.state.order if eng._source_of(u) == "sam"][0]].row
    assert np.all(eng.collection["feats"]["decoder"][row] == 0)            # detector feature 0-filled...
    assert np.any(eng.collection["feats"]["shapecoord"][row] != 0)        # ...shapecoord computed (shared space)
    assert "decoder" not in eng.feature_nan_methods()                     # NOT NaN-poisoned -> native clustering intact


def test_import_then_facet_filters_by_source(tmp_path):
    eng = _disk_engine(tmp_path)
    eng.import_proposals_coco(_write_coco(tmp_path, "sam.coco.json", (2, 1)), source="sam")
    eng.set_source_filter(["sam"]);    assert len(eng._pool_iuids()) == 3
    eng.set_source_filter(["native"]); assert len(eng._pool_iuids()) == 4
    eng.set_source_filter(None);       assert len(eng._pool_iuids()) == 7


def test_import_unmatched_images_errors(tmp_path):
    import json
    eng = _disk_engine(tmp_path)
    p = tmp_path / "other.coco.json"
    json.dump({"images": [{"id": 1, "file_name": "NOT_in_project.png"}],
               "annotations": [{"image_id": 1, "bbox": [1, 1, 10, 10]}]}, open(p, "w"))
    r = eng.import_proposals_coco(str(p), source="x")
    assert r.get("error") and "matched" in r["error"]


def test_import_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng = _disk_engine(tmp_path)
    cp = _write_coco(tmp_path, "medsam.coco.json", (1, 1))
    c = TestClient(create_app(engine=eng))
    r = c.post("/api/import_proposals", json={"path": cp, "source": "medsam"}).json()
    assert r["ok"] and r["n_imported"] == 2 and "medsam" in {s["source"] for s in r["sources"]["sources"]}
    assert c.post("/api/import_proposals", json={"path": cp}).status_code == 400   # missing source
