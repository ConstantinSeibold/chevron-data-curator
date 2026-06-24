"""Full project reset: drop every instance + all curation + classes + ingest/merge/history logs, keep only
the config. Plus the /api/reset confirm guard. Model-free.
Run: pytest tools/curator/tests/test_reset.py -q
"""
from __future__ import annotations

import tempfile

import numpy as np


def _rle(h=32, w=32):
    import cv2
    from pycocotools import mask as mu
    m = np.zeros((h, w), np.uint8); cv2.circle(m, (16, 16), 6, 1, -1)
    r = mu.encode(np.asfortranarray(m)); r["counts"] = r["counts"].decode("ascii")
    return r


def _fb(files, bid="b0", dim=8):
    from tools.curator import ids
    recs = [{"iuid": ids.new_uid(), "batch_id": bid, "abs_path": f, "file_name": f,
             "image_id": abs(hash(f)) % 1000000, "score": 0.9, "rle": _rle(), "H": 32, "W": 32}
            for f in files]
    return {"records": recs, "feats": {"decoder": np.zeros((len(files), dim), np.float32)}, "n_images": len(files)}


def _eng(tmp_path):
    from tools.curator.engine import CuratorEngine
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    eng.collection = {"records": [], "n_images": 0, "feats": {"decoder": np.zeros((0, 8), np.float32)}}
    return eng


def test_reset_drops_everything_keeps_config(tmp_path, monkeypatch):
    from tools.curator import collect as _co
    eng = _eng(tmp_path)
    monkeypatch.setattr(eng, "_ensure_model", lambda: (None, None, None))
    monkeypatch.setattr(_co, "collect_batch",
                        lambda m, c, d, fs, *, score_thresh, feature_cfg: _fb(fs))
    eng.ingest_paths([f"/x/{i}.png" for i in range(6)])
    eng.assign(list(eng.state.meta)[:2], "foo")
    eng.set_scope("ing_000")
    assert len(eng.state.order) == 6 and eng.store.has_collection() and len(eng.state.taxonomy) == 1
    assert eng.store.ingests_path.exists() and eng._scope_id == "ing_000"
    cfg = dict(eng.state.config)

    stats = eng.reset(keep_config=True)

    assert stats["n_instances"] == 0
    assert eng.collection is None and not eng.store.has_collection()
    assert eng.state.order == [] and dict(eng.state.meta) == {} and len(eng.state.taxonomy) == 0
    assert eng.list_ingests() == [] and not eng.store.ingests_path.exists()
    man = eng.store.load_manifest()
    assert man["processed_paths"] == [] and man["n_instances"] == 0 and man["coll_version"] == 0
    assert eng._scope_id is None and eng._scope_bids is None and eng._cluster is None
    assert dict(eng.state.config) == cfg                          # config preserved


def _client(tmp_path):
    from fastapi.testclient import TestClient
    from tools.curator.server import create_app
    return TestClient(create_app(str(tmp_path)))


def test_reset_route_requires_confirm(tmp_path):
    c = _client(tmp_path)
    assert c.post("/api/reset", json={}).status_code == 400          # no confirm -> rejected
    assert c.post("/api/reset", json={"confirm": False}).status_code == 400
    r = c.post("/api/reset", json={"confirm": True})
    assert r.status_code == 200 and r.json()["ok"] is True
