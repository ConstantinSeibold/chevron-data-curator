"""v3 tests: unreject (bg -> unassigned, reversible) + reset (drop everything, keep config).
Run: pytest tools/curator/tests/test_v3.py -q  (repo root)
"""
from __future__ import annotations

import numpy as np

from tools.curator import ids
from tools.curator.engine import CuratorEngine
from tools.curator.state import InstanceMeta


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


def _engine(tmp_path, n=5):
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x", "score_thresh": 0.3},
                      "features": {"model_features": ["decoder"]}})
    recs, order, meta = [], [], {}
    m = np.zeros((32, 32), np.uint8); m[8:20, 8:20] = 1
    for i in range(n):
        u = ids.new_uid()
        recs.append({"iuid": u, "row": i, "inst_id": i, "image_id": 1, "H": 32, "W": 32, "score": 0.6,
                     "rle": _rle(m > 0), "file_name": str(tmp_path / "x.png")})
        order.append(u); meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=1)
    eng.collection = {"records": recs, "n_images": 1, "feats": {"decoder": np.zeros((n, 4), np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    return eng, order


def test_unreject(tmp_path):
    eng, order = _engine(tmp_path)
    eng.set_background([order[0], order[1]])
    assert eng.stats()["n_background"] == 2
    assert set(eng.background_iuids()) == {order[0], order[1]}
    n = eng.unreject([order[0]])
    assert n == 1
    assert eng.state.meta[order[0]].is_background is False
    assert eng.state.meta[order[0]].assigned_class is None       # back to unassigned
    assert eng.stats()["n_background"] == 1
    eng.undo()                                                   # reversible
    assert eng.state.meta[order[0]].is_background is True
    # unreject all
    assert eng.unreject(eng.background_iuids()) == 2
    assert eng.stats()["n_background"] == 0


def test_reset_keeps_config(tmp_path):
    eng, order = _engine(tmp_path)
    eng.state.add_class("foo"); eng.state.meta[order[0]].assigned_class = list(eng.state.taxonomy)[0]
    eng.apply_refine(order[0], [{"name": "fill"}])               # creates a refine overlay file
    assert eng.store.collection_path.exists()
    cfg = dict(eng.state.config)
    eng.reset(keep_config=True)
    assert eng.collection is None
    assert len(eng.state.order) == 0 and len(eng.state.meta) == 0 and len(eng.state.taxonomy) == 0
    assert eng.state.coll_version == 0
    assert eng.state.config == cfg                               # config preserved
    assert not eng.store.collection_path.exists()                # collection dropped
    assert list(eng.store.refine_dir.glob("*.pkl")) == []        # overlays dropped
    assert eng.store.load_manifest()["processed_paths"] == []
    # resume from disk -> still empty (reset persisted)
    eng2 = CuratorEngine(tmp_path)
    assert eng2.collection is None and len(eng2.state.order) == 0
    assert eng2.state.config.get("model", {}).get("ckpt") == "x"
