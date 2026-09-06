"""Ingest registry + view scope: each (re)inference run is recorded with its batch_ids, and a view can
SCOPE to one ingest so the cluster pool + image picker show only THAT run's instances ("see only the
newly predicted"). Model/collect_batch stubbed (no GPU).
Run: pytest tests/test_ingest_scope.py -q
"""
from __future__ import annotations

import numpy as np


def _rle(h=32, w=32):
    import cv2
    from pycocotools import mask as mu
    m = np.zeros((h, w), np.uint8); cv2.circle(m, (16, 16), 6, 1, -1)
    r = mu.encode(np.asfortranarray(m)); r["counts"] = r["counts"].decode("ascii")
    return r


def _fb(files, bid, dim=8):
    """Mimic collect_batch: one instance per image, all sharing batch_id `bid` (a single run's chunk)."""
    from chevron import ids
    recs = [{"iuid": ids.new_uid(), "batch_id": bid, "abs_path": f, "file_name": f,
             "image_id": abs(hash(f)) % 1000000, "score": 0.9, "rle": _rle(), "H": 32, "W": 32}
            for f in files]
    return {"records": recs, "feats": {"decoder": np.zeros((len(files), dim), np.float32)}, "n_images": len(files)}


def _eng(tmp_path):
    from chevron.engine import CuratorEngine
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    eng.collection = {"records": [], "n_images": 0, "feats": {"decoder": np.zeros((0, 8), np.float32)}}
    return eng


def _two_runs(tmp_path, monkeypatch):
    from chevron import collect as _co
    eng = _eng(tmp_path)
    monkeypatch.setattr(eng, "_ensure_model", lambda: (None, None, None))
    monkeypatch.setattr(_co, "collect_batch",
                        lambda m, c, d, fs, *, score_thresh, feature_cfg: _fb(fs, "bA"))
    eng.ingest_paths([f"/A/{i}.png" for i in range(10)])
    monkeypatch.setattr(_co, "collect_batch",
                        lambda m, c, d, fs, *, score_thresh, feature_cfg: _fb(fs, "bB"))
    eng.ingest_paths([f"/B/{i}.png" for i in range(6)])
    return eng


def test_ingest_paths_records_a_registry_event(tmp_path, monkeypatch):
    eng = _two_runs(tmp_path, monkeypatch)
    ings = eng.list_ingests()
    assert [g["ingest_id"] for g in ings] == ["ing_001", "ing_000"]     # newest first
    by_id = {g["ingest_id"]: g for g in ings}
    assert by_id["ing_000"]["n_instances"] == 10 and by_id["ing_001"]["n_instances"] == 6
    assert by_id["ing_000"]["n_live"] == 10 and by_id["ing_001"]["n_live"] == 6   # nothing curated yet
    raw = {e["ingest_id"]: set(e["batch_ids"]) for e in eng.store.read_ingests()}
    assert raw["ing_000"] == {"bA"} and raw["ing_001"] == {"bB"}


def test_scope_restricts_pool_images_and_partitions(tmp_path, monkeypatch):
    eng = _two_runs(tmp_path, monkeypatch)
    assert len(eng._pool_iuids()) == 16 and eng.image_counts()["total"] == 16   # no scope = everything

    res = eng.set_scope("ing_000")                                              # scope to run A
    assert res["scope"] == "ing_000" and res["n_pool"] == 10
    assert all(eng.state.meta[u].batch_id == "bA" for u in eng._pool_iuids())
    assert eng.image_counts()["total"] == 10
    assert eng._cluster is None                                                 # cluster cleared on scope change

    eng.set_scope("ing_001")
    assert len(eng._pool_iuids()) == 6
    assert all(eng.state.meta[u].batch_id == "bB" for u in eng._pool_iuids())

    eng.set_scope("all")                                                        # clear
    assert eng._scope_id is None and len(eng._pool_iuids()) == 16
    assert "error" in eng.set_scope("does_not_exist")                           # unknown ingest -> error, no raise


def test_scope_busts_view_signature(tmp_path, monkeypatch):
    eng = _two_runs(tmp_path, monkeypatch)
    sig0 = eng._view_sig()
    eng.set_scope("ing_000")
    assert eng._view_sig() != sig0                                              # partition_view cache must bust


def test_scoped_class_partition_membership(tmp_path, monkeypatch):
    eng = _two_runs(tmp_path, monkeypatch)
    a_iuids = [u for u in eng.state.order if eng.state.meta[u].batch_id == "bA"]
    eng.assign(a_iuids[:3], "lineX")                                            # 3 run-A instances assigned
    cid = eng.state.class_id_by_name("lineX")
    assert len(eng.partition_iuids(f"class:{cid}")) == 3                        # unscoped: all 3
    eng.set_scope("ing_001")                                                    # run B has none of them
    assert eng.partition_iuids(f"class:{cid}") == []
    eng.set_scope("ing_000")
    assert len(eng.partition_iuids(f"class:{cid}")) == 3
