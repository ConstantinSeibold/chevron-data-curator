"""Multi-session safety: the shared in-process engine is reachable by CONCURRENT requests (threadpool), so
state mutations are serialized by self._mutate_lock (@_mutating). Plus the /api/version stamp that drives
the live-refresh banner. Run: pytest chevron/tests/test_concurrency.py -q
"""
from __future__ import annotations

import threading

import numpy as np


def _engine(tmp_path, n=200):
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    eng.collection = {"records": [{"iuid": f"u{i}", "row": i, "score": 0.6} for i in range(n)],
                      "n_images": n, "feats": {"decoder": np.zeros((n, 8), np.float32)}}
    eng.state.order = [f"u{i}" for i in range(n)]
    eng.state.meta = {f"u{i}": InstanceMeta(f"u{i}", "b", i, 1000 + i) for i in range(n)}
    eng.state.coll_version = 1
    return eng


def test_concurrent_mutations_no_corruption(tmp_path):
    eng = _engine(tmp_path, 200)

    def worker(t):
        for i in range(t * 25, (t + 1) * 25):
            eng.assign([f"u{i}"], f"c{t}")
    ths = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for x in ths:
        x.start()
    for x in ths:
        x.join()
    assigned = sum(1 for m in eng.state.meta.values() if m.assigned_class)
    assert assigned == 200                                       # every assign landed; none lost to a race
    assert len(eng.state.taxonomy) == 8                          # 8 distinct classes created cleanly
    eng.flush(); eng.state.assert_aligned(eng.collection["feats"]["decoder"].shape[0])   # feats stay aligned


def test_mutate_lock_is_reentrant(tmp_path):
    eng = _engine(tmp_path, 4)
    with eng._mutate_lock:                                       # already held -> a nested mutation must not deadlock
        eng.assign(["u0"], "A")
    assert eng.state.meta["u0"].assigned_class is not None


def test_version_stamp_advances(tmp_path):
    eng = _engine(tmp_path, 4)
    s0 = eng._mutation_serial
    eng.assign(["u0"], "A")
    assert eng._mutation_serial > s0 and eng.stats()["serial"] == eng._mutation_serial


def test_version_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng = _engine(tmp_path, 4)
    c = TestClient(create_app(engine=eng))
    v0 = c.get("/api/version").json()
    assert {"serial", "coll_version", "scope_token"} <= set(v0)
    c.post("/api/assign", json={"iuids": ["u0"], "cls": "A"})
    assert c.get("/api/version").json()["serial"] > v0["serial"]   # another session would see the bump
