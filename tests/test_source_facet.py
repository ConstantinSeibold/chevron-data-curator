"""Proposal-source facet: tag ingests with the model that proposed them, then filter by source across every
tab via the shared view predicate (composes with ingest scope). Read-only + additive (facet None = no change).
Run: pytest tools/curator/tests/test_source_facet.py -q
"""
from __future__ import annotations

import numpy as np


def _engine(tmp_path, n=40):
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
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
    from tools.curator.server import create_app
    eng, order = _engine(tmp_path)
    c = TestClient(create_app(engine=eng))
    s = c.get("/api/sources").json()
    assert {x["source"] for x in s["sources"]} == {"modelA", "modelB"}
    r = c.post("/api/source_filter", json={"sources": ["modelB"]}).json()
    assert r["active"] == ["modelB"]
    assert len(eng._pool_iuids()) == 20
    c.post("/api/source_filter", json={"sources": None})
    assert len(eng._pool_iuids()) == 40
