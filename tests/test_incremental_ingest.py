"""Incremental, crash-safe, RAM-bounded ingest (Scope 1): ingest_paths writes one append-only shard per
chunk + advances processed_paths per chunk; the shards fold into the collection at the end, OR are
recovered on the next project open if the run was interrupted. Model/collect_batch stubbed (no GPU).
Run: pytest tools/curator/tests/test_incremental_ingest.py -q
"""
from __future__ import annotations

import numpy as np
import pytest


def _rle(h=32, w=32):
    import cv2
    from pycocotools import mask as mu
    m = np.zeros((h, w), np.uint8); cv2.circle(m, (16, 16), 6, 1, -1)
    r = mu.encode(np.asfortranarray(m)); r["counts"] = r["counts"].decode("ascii")
    return r


def _fake_batch(files, dim=8):
    """Mimic collect_batch: one instance per image, fresh iuid, aligned feats row."""
    from tools.curator import ids
    recs = [{"iuid": ids.new_uid(), "batch_id": "b", "abs_path": f, "file_name": f,
             "image_id": abs(hash(f)) % 100000, "score": 0.9, "rle": _rle(), "H": 32, "W": 32}
            for f in files]
    feats = {"decoder": np.zeros((len(files), dim), np.float32)}
    return {"records": recs, "feats": feats, "n_images": len(files)}


def _eng(tmp_path):
    from tools.curator.engine import CuratorEngine
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    eng.collection = {"records": [], "n_images": 0, "feats": {"decoder": np.zeros((0, 8), np.float32)}}
    return eng


def test_collection_shards_roundtrip(tmp_path):
    from tools.curator.store import Store
    st = Store(tmp_path); st.ensure()
    assert st.list_collection_shards() == [] and st.load_collection_shards() is None
    st.append_collection_shard(_fake_batch(["/a.png", "/b.png"]))
    st.append_collection_shard(_fake_batch(["/c.png"]))
    assert len(st.list_collection_shards()) == 2
    merged = st.load_collection_shards()
    assert len(merged["records"]) == 3 and merged["feats"]["decoder"].shape == (3, 8)   # concat in write order
    st.clear_collection_shards()
    assert st.list_collection_shards() == [] and st.load_collection_shards() is None


def test_ingest_writes_a_shard_per_chunk_then_folds(tmp_path, monkeypatch):
    from tools.curator import collect as _co
    eng = _eng(tmp_path)
    monkeypatch.setattr(eng, "_ensure_model", lambda: (None, None, None))
    chunks = {"n": 0}
    def fake_collect(model, cfg, d2, fs, *, score_thresh, feature_cfg):
        chunks["n"] += 1
        return _fake_batch(fs)
    monkeypatch.setattr(_co, "collect_batch", fake_collect)
    files = [f"/x/im{i}.png" for i in range(20)]                     # CHUNK=8 -> 3 chunks (8,8,4)
    rep = eng.ingest_paths(files)
    assert chunks["n"] == 3 and rep["n_new_instances"] == 20
    assert len(eng.collection["records"]) == 20                     # folded into the live collection
    assert eng.store.list_collection_shards() == []                 # ...and the shards cleared
    assert eng.store.has_collection()                               # collection.pkl written once at the end
    assert len(eng.store.load_manifest()["processed_paths"]) == 20  # processed advanced (per chunk)
    eng.state.assert_aligned(eng.collection["feats"]["decoder"].shape[0])


def test_ingest_interrupt_is_recovered_on_reopen(tmp_path, monkeypatch):
    from tools.curator import collect as _co
    from tools.curator.engine import CuratorEngine
    # run 1: a clean ingest of 20 instances (folded + saved)
    eng = _eng(tmp_path)
    monkeypatch.setattr(eng, "_ensure_model", lambda: (None, None, None))
    monkeypatch.setattr(_co, "collect_batch",
                        lambda model, cfg, d2, fs, *, score_thresh, feature_cfg: _fake_batch(fs))
    eng.ingest_paths([f"/x/im{i}.png" for i in range(20)])
    assert len(eng.collection["records"]) == 20

    # run 2: crash on the 2nd chunk — chunk 1's shard must persist, processed advances, collection NOT yet grown
    eng2 = CuratorEngine(tmp_path)
    assert len(eng2.collection["records"]) == 20 and eng2.store.list_collection_shards() == []
    monkeypatch.setattr(eng2, "_ensure_model", lambda: (None, None, None))
    calls = {"n": 0}
    def crashy(model, cfg, d2, fs, *, score_thresh, feature_cfg):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return _fake_batch(fs)
    monkeypatch.setattr(_co, "collect_batch", crashy)
    files2 = [f"/y/im{i}.png" for i in range(20)]
    with pytest.raises(RuntimeError):
        eng2.ingest_paths(files2)
    assert len(eng2.store.list_collection_shards()) == 1            # chunk-1 shard kept on disk
    proc = set(eng2.store.load_manifest()["processed_paths"])
    assert len([f for f in files2 if f in proc]) == 8              # processed advanced for chunk 1 only
    assert len(eng2.collection["records"]) == 20                   # not folded yet (merge never reached)

    # reopen -> recovery folds the orphaned shard (idempotent, by iuid)
    eng3 = CuratorEngine(tmp_path)
    assert eng3.store.list_collection_shards() == []               # recovered + cleared
    assert len(eng3.collection["records"]) == 28                   # 20 + chunk-1's 8
    eng3.state.assert_aligned(eng3.collection["feats"]["decoder"].shape[0])
    # opening yet again is a no-op (nothing pending)
    eng4 = CuratorEngine(tmp_path)
    assert len(eng4.collection["records"]) == 28
