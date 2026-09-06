"""Persisted dimensionality reduction and placing a NEW point on the map.

The property under test is not "a number comes back" but "the number is in the RIGHT PLACE". A
query is only meaningful if it is transformed exactly the way the fit was:

  * through the same reducer,
  * through the fit-time coordinate frame (not a fresh min/max of the current points),
  * through the fit-time per-block z-score statistics (fuse_features standardises against the
    collection, so fresh statistics silently move the point).

The round-trip test below is the sharp one: projecting an instance that was IN the fit must land on
that instance's own map position.

Run: pytest tests/test_projection_query.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chevron import ids
from chevron.engine import CuratorEngine
from chevron.projection import FittedProjection, block_stats
from chevron.server import create_app
from chevron.state import InstanceMeta


def _engine(tmp_path, n=60, dim=12, clusters=4):
    """Clustered feature data, so a projection has real structure to preserve."""
    rng = np.random.default_rng(0)
    cent = rng.normal(0, 3, (clusters, dim))
    X = np.stack([cent[i % clusters] + rng.normal(0, 0.3, dim) for i in range(n)]).astype(np.float32)
    eng = CuratorEngine(tmp_path); eng.init_project({})
    recs, order, meta = [], [], {}
    for i in range(n):
        u = ids.new_uid()
        recs.append({"iuid": u, "row": i, "inst_id": i, "image_id": 1000 + i % 5, "H": 8, "W": 8,
                     "score": 0.8, "file_name": "x.png", "abs_path": "x.png", "batch_id": "b",
                     "rle": {"size": [8, 8], "counts": "0"}})
        order.append(u); meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=1000 + i % 5)
    eng.collection = {"records": recs, "n_images": 5, "feats": {"decoder": X}}
    eng.state.order, eng.state.meta, eng.state.coll_version = order, meta, 1
    return eng


# --------------------------------------------------------------------------- the fitted artefact
def test_projection_keeps_its_reducer_and_frame(tmp_path):
    eng = _engine(tmp_path)
    p = eng.project({"decoder": 1.0}, method="pca", dims=2)
    fit = eng._fit
    assert fit.method == "pca" and fit.queryable
    assert fit.coord_min.shape == (2,) and fit.coord_max.shape == (2,)
    assert np.allclose(fit.coord_min, p["coords"].min(0))       # the frame is the FIT's, not a later one
    assert set(fit.block_stats) == {"decoder"}
    assert fit.block_stats["decoder"]["dim"] == 12
    eng.close()


def test_fitted_projection_round_trips_to_disk(tmp_path):
    eng = _engine(tmp_path)
    eng.project({"decoder": 1.0}, method="pca", dims=2)
    f = tmp_path / "dr" / "pca_2d.joblib"
    assert f.is_file(), "the fit should persist so a query survives a restart"
    back = FittedProjection.load(f)
    assert back is not None and back.method == "pca" and back.queryable
    assert np.allclose(back.coord_min, eng._fit.coord_min)
    assert np.allclose(back.block_stats["decoder"]["mean"], eng._fit.block_stats["decoder"]["mean"])
    eng.close()


def test_a_corrupt_artefact_means_refit_not_crash(tmp_path):
    (tmp_path / "x.joblib").write_bytes(b"not a joblib file")
    assert FittedProjection.load(tmp_path / "x.joblib") is None
    assert FittedProjection.load(tmp_path / "missing.joblib") is None


# --------------------------------------------------------------------------- placing a point
def test_querying_an_instance_lands_on_that_instance(tmp_path):
    """THE correctness test. An instance that was in the fit must project onto its own position —
    if any of reducer, coordinate frame or z-score statistics is recomputed, this drifts."""
    eng = _engine(tmp_path)
    p = eng.project({"decoder": 1.0}, method="pca", dims=2)
    fit = eng._fit
    target = p["iuids"][7]
    truth = fit.normalise(p["coords"])[7]

    q = eng.project_query({"decoder": 1.0}, iuid=target, method="pca", dims=2)
    assert q.get("ok"), q
    got = np.array([q["point"]["x"], q["point"]["y"]])
    assert np.allclose(got, truth, atol=1e-4), f"query landed at {got}, the instance is at {truth}"
    assert q["neighbors"][0]["iuid"] == target and q["neighbors"][0]["dist"] < 1e-4
    eng.close()


def test_fresh_statistics_would_move_the_point(tmp_path):
    """Guards the reason the fusion statistics are stored: standardising a query against anything
    other than the fit-time mean/std puts it somewhere else."""
    eng = _engine(tmp_path)
    eng.project({"decoder": 1.0}, method="pca", dims=2)
    fit = eng._fit
    v = eng.collection["feats"]["decoder"][3]

    right = fit.fuse_query({"decoder": v})
    wrong_stats = dict(fit.block_stats)
    wrong_stats["decoder"] = {**wrong_stats["decoder"], "mean": np.zeros_like(wrong_stats["decoder"]["mean"])}
    wrong = FittedProjection(fit.reducer, fit.method, fit.dims, fit.spec, wrong_stats,
                             fit.coord_min, fit.coord_max, fit.n_fit, fit.truncated).fuse_query({"decoder": v})
    assert not np.allclose(right, wrong), "the stored statistics are not actually being used"
    eng.close()


def test_neighbours_are_ranked_in_map_space(tmp_path):
    eng = _engine(tmp_path)
    eng.project({"decoder": 1.0}, method="pca", dims=2)
    q = eng.project_query({"decoder": 1.0}, iuid=eng.state.order[0], method="pca", dims=2, k=6)
    d = [n["dist"] for n in q["neighbors"]]
    assert len(d) == 6 and d == sorted(d)
    eng.close()


# --------------------------------------------------------------------------- honest refusals
def test_a_reducer_that_cannot_transform_says_so(tmp_path):
    """h-NNE / t-SNE style embeddings may have no `.transform`. Refuse clearly rather than silently
    refitting (which would move every existing point)."""
    eng = _engine(tmp_path)
    eng.project({"decoder": 1.0}, method="pca", dims=2)
    eng._fit.reducer = object()                       # no .transform
    assert eng._fit.queryable is False
    r = eng.project_query({"decoder": 1.0}, iuid=eng.state.order[0], method="pca", dims=2)
    assert "cannot place new points" in r["error"]
    eng.close()


def test_text_query_refuses_a_fused_map(tmp_path):
    """A phrase yields ONE feature block; it cannot be placed on a map fused over several."""
    eng = _engine(tmp_path)
    eng.collection["feats"]["shapecoord"] = np.ones((len(eng.state.order), 4), np.float32)
    eng.state.coll_version += 1
    eng.project({"decoder": 1.0, "shapecoord": 1.0}, method="pca", dims=2)
    r = eng.project_query({"decoder": 1.0, "shapecoord": 1.0}, text="a catheter", method="pca", dims=2)
    assert "single-feature map" in r["error"]
    eng.close()


def test_wrong_dimensionality_is_caught(tmp_path):
    eng = _engine(tmp_path)
    eng.project({"decoder": 1.0}, method="pca", dims=2)
    with pytest.raises(ValueError, match="was fitted on"):
        eng._fit.fuse_query({"decoder": np.ones(999, np.float32)})
    eng.close()


# --------------------------------------------------------------------------- API
def test_api_places_a_point(tmp_path):
    eng = _engine(tmp_path)
    c = TestClient(create_app(engine=eng))
    c.get("/api/projection_points?method=pca&dims=2")
    r = c.post("/api/project_query", json={"spec": {"decoder": 1.0}, "iuid": eng.state.order[2], "method": "pca", "dims": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"x", "y"} <= set(body["point"]) and body["neighbors"]
    assert c.post("/api/project_query", json={"spec": {"decoder": 1.0}, "iuid": "nope", "method": "pca"}).status_code == 400
    eng.close()
