"""Latent-space Map: 2D/3D projection of in-scope instances + per-point color fields, and the endpoint.
Projection is label-INDEPENDENT (cached on coll_version/scope/spec) so labeling recolors instantly.
Run: pytest tools/curator/tests/test_projection_map.py -q
"""
from __future__ import annotations

import numpy as np


def _engine(tmp_path, n=120):
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    rng = np.random.default_rng(0)
    half = n // 2
    feats = np.concatenate([rng.normal(0, 1, (half, 8)), rng.normal(6, 1, (n - half, 8))]).astype(np.float32)
    eng.collection = {"records": [{"iuid": f"u{i}", "row": i, "score": round(0.4 + 0.005 * i, 3)} for i in range(n)],
                      "n_images": n, "feats": {"decoder": feats}}
    eng.state.order = [f"u{i}" for i in range(n)]
    eng.state.meta = {f"u{i}": InstanceMeta(f"u{i}", "b", i, 1000 + i) for i in range(n)}
    eng.state.coll_version = 1
    return eng


def test_projection_points_shape_and_fields(tmp_path):
    eng = _engine(tmp_path)
    eng.assign([f"u{i}" for i in range(10)], "A")
    eng.set_background([f"u{i}" for i in range(10, 15)])
    r = eng.projection_points(method="hnne")                        # hnne not installed -> falls back to umap/pca
    assert r["n"] == 120 and r["dims"] == 2 and r["method"] in ("hnne", "umap", "pca")
    p = r["points"][0]
    assert {"iuid", "x", "y", "state", "cls", "pid", "image_id", "score"} <= set(p)
    xs = [q["x"] for q in r["points"]]; ys = [q["y"] for q in r["points"]]
    assert 0.0 <= min(xs) and max(xs) <= 1.0 and 0.0 <= min(ys) and max(ys) <= 1.0   # normalized to [0,1]
    st = {q["state"] for q in r["points"]}
    assert {"class", "reject", "pool"} <= st                        # labeled/rejected/unassigned all present
    assert any(q["cls"] == "A" for q in r["points"])


def test_projection_is_label_independent_cache(tmp_path):
    eng = _engine(tmp_path)
    eng.projection_points(method="hnne")
    cached = eng._proj_cache                                          # coords computed once (label-independent)
    eng.assign([f"u{i}" for i in range(5)], "A")                     # a LABEL mutation (bumps _mutation_serial, not coll_version)
    eng.projection_points(method="hnne")
    assert eng._proj_cache is cached                                  # same coords reused -> labeling only recolors
    # but the recolored payload reflects the new labels
    r = eng.projection_points(method="hnne")
    assert sum(1 for q in r["points"] if q["state"] == "class") == 5


def test_projection_3d_and_pca(tmp_path):
    eng = _engine(tmp_path, n=40)
    r = eng.projection_points(method="pca", dims=3)
    assert r["method"] == "pca" and r["dims"] == 3 and all("z" in q for q in r["points"])


def test_projection_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from tools.curator.server import create_app
    eng = _engine(tmp_path)
    c = TestClient(create_app(engine=eng))
    r = c.get("/api/projection_points?method=hnne").json()
    assert r["n"] == 120 and r["points"] and {"x", "y", "state"} <= set(r["points"][0])


def test_projection_no_features_400(tmp_path):
    from fastapi.testclient import TestClient
    from tools.curator.server import create_app
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    # feature is all-NaN -> _present_spec_nanfree drops it -> error
    eng.collection = {"records": [{"iuid": "u0", "row": 0, "score": 0.5}], "n_images": 1,
                      "feats": {"decoder": np.full((1, 8), np.nan, np.float32)}}
    eng.state.order = ["u0"]; eng.state.meta = {"u0": InstanceMeta("u0", "b", 0, 1000)}; eng.state.coll_version = 1
    r = c = TestClient(create_app(engine=eng)).get("/api/projection_points")
    assert r.status_code == 400
