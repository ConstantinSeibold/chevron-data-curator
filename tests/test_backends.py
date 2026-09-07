"""Proposal backends — where instances come from when you have no trained model.

The one that matters most here is `coco`: it BOOTSTRAPS an empty project with no ML stack at all,
which is what makes a fresh clone usable. The ML-backed proposers (SAM, torchvision, HF) are covered
through the shared assembly path with a stub backend, so their bookkeeping is tested without
downloading half a gigabyte of weights.

Run: pytest tests/test_backends.py -q
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pycocotools import mask as mu

from chevron.backends import base as B
from chevron.backends import list_backends
from chevron.engine import CuratorEngine
from chevron.server import create_app


def _images(root, n=3, size=64):
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = root / f"im{i}.png"
        cv2.imwrite(str(p), (np.random.default_rng(i).random((size, size, 3)) * 200).astype(np.uint8))
        paths.append(str(p))
    return paths


def _coco(paths, out, size=64, per_image=2):
    """A COCO with polygon segmentations — the format most tools actually emit."""
    images, anns, aid = [], [], 1
    for i, p in enumerate(paths):
        images.append({"id": i + 1, "file_name": p, "width": size, "height": size})
        for k in range(per_image):
            x, y, w, h = 6 + 20 * k, 8, 14, 18
            anns.append({"id": aid, "image_id": i + 1, "category_id": 1, "score": 0.9 - 0.1 * k,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                         "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]]})
            aid += 1
    out.write_text(json.dumps({"images": images, "annotations": anns,
                               "categories": [{"id": 1, "name": "thing"}]}))
    return out


# --------------------------------------------------------------------------- registry
def test_every_backend_reports_availability_without_raising():
    """The picker must be able to list proposers on a machine with no ML stack."""
    got = {b["name"]: b for b in list_backends()}
    assert {"coco", "sam_auto", "torchvision_maskrcnn", "hf_seg"} <= set(got)
    for b in got.values():
        assert isinstance(b["available"], bool) and b["label"] and "detail" in b


def test_coco_backend_needs_nothing():
    got = {b["name"]: b for b in list_backends()}
    assert got["coco"]["available"] is True
    assert got["coco"]["requires"] == "nothing"


def test_unknown_backend_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown proposal backend"):
        B.get("no_such_thing")


# --------------------------------------------------------------------------- bootstrap
def test_coco_bootstraps_an_empty_project(tmp_path):
    """THE cold-start path: a fresh project, a COCO of masks, no model, no torch."""
    paths = _images(tmp_path / "img")
    cj = _coco(paths, tmp_path / "p.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    assert eng.collection is None                       # genuinely empty to begin with

    rep = eng.propose_instances("coco", coco_path=str(cj))
    assert rep.get("ok"), rep
    assert rep["n_instances"] == 6 and rep["n_images"] == 3
    assert eng.collection is not None
    assert len(eng.state.order) == 6 == len(eng.state.meta)
    eng.state.assert_aligned(eng.collection["feats"]["shapecoord"].shape[0])
    assert {"shapecoord", "coords"} <= set(eng.available_features())
    # every proposal arrives unassigned and class-agnostic — the human supplies the taxonomy
    assert all(m.assigned_class is None and not m.is_background for m in eng.state.meta.values())
    eng.close()


def test_coco_resolves_a_flat_image_root(tmp_path):
    """COCO file_names are often bare basenames against a separate image root."""
    paths = _images(tmp_path / "img")
    cj = tmp_path / "p.json"
    _coco([p.split("/")[-1] for p in paths], cj)        # basenames only
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    rep = eng.propose_instances("coco", coco_path=str(cj), image_root=str(tmp_path / "img"))
    assert rep.get("ok") and rep["n_instances"] == 6
    eng.close()


def test_unresolvable_images_report_rather_than_crash(tmp_path):
    cj = tmp_path / "p.json"
    _coco(["/nowhere/a.png"], cj)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    rep = eng.propose_instances("coco", coco_path=str(cj))
    assert "error" in rep and "image root" in rep["error"]
    eng.close()


def test_a_second_backend_run_appends(tmp_path):
    """Two proposal runs on one project: additive, and the row-alignment invariant holds."""
    paths = _images(tmp_path / "img")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(_coco(paths, tmp_path / "a.json")))
    eng.propose_instances("coco", coco_path=str(_coco(paths, tmp_path / "b.json", per_image=1)),
                          source="second")
    assert len(eng.state.order) == 9
    eng.state.assert_aligned(eng.collection["feats"]["shapecoord"].shape[0])
    assert {"coco", "second"} <= {s["source"] for s in eng.sources()["sources"]}
    eng.close()


def test_a_proposal_run_survives_a_reopen(tmp_path):
    """A proposer builds the whole collection in RAM — it must reach collection.pkl, not just state.json.

    When it did not, reopening the project left 6 instances in `state` with no records behind them, and
    the NEXT run concat'd into an empty master: `order` reset to the new batch while `meta` kept both
    halves ("row-alignment broken: order=6 meta=12"), which then failed every later call.
    """
    paths = _images(tmp_path / "img")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(_coco(paths, tmp_path / "a.json")))
    eng.close()
    assert (tmp_path / "proj" / "collection.pkl").exists()

    eng = CuratorEngine(tmp_path / "proj")              # constructing on a project dir reopens it
    assert eng.collection is not None and len(eng.collection["records"]) == 6
    eng.propose_instances("coco", coco_path=str(_coco(paths, tmp_path / "b.json", per_image=1)),
                          source="second")
    assert len(eng.state.order) == 9 == len(eng.state.meta)
    eng.state.assert_aligned(eng.collection["feats"]["shapecoord"].shape[0])
    eng.close()


def test_ingesting_over_a_lost_collection_refuses_instead_of_orphaning(tmp_path):
    """collection.pkl gone but state.json intact: refuse BEFORE mutating, and say what to do."""
    paths = _images(tmp_path / "img")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(_coco(paths, tmp_path / "a.json")))
    eng.close()
    (tmp_path / "proj" / "collection.pkl").unlink()

    eng = CuratorEngine(tmp_path / "proj")
    assert eng.collection is None and len(eng.state.order) == 6
    with pytest.raises(RuntimeError, match="no collection loaded"):
        eng.propose_instances("coco", coco_path=str(_coco(paths, tmp_path / "b.json")))
    assert len(eng.state.order) == 6 == len(eng.state.meta)     # state untouched by the refusal
    eng.close()


# --------------------------------------------------------------------------- shared assembly
class _StubBackend:
    """Stands in for SAM/torchvision/HF: the framework's bookkeeping is what is under test."""
    name, label, requires = "stub", "Stub", "nothing"
    def available(self): return True, "ok"
    def propose(self, image_rgb, path=None, **cfg):
        H, W = image_rgb.shape[:2]
        a = np.zeros((H, W), bool); a[5:20, 5:20] = True
        b = np.zeros((H, W), bool); b[30:50, 30:50] = True
        empty = np.zeros((H, W), bool)                  # must be dropped, not stored
        return [B.Proposal(a, 0.9), B.Proposal(b, 0.4), B.Proposal(empty, 0.99)]


def test_assembly_builds_records_features_and_drops_empties(tmp_path):
    paths = _images(tmp_path / "img", n=2)
    col = B.build_collection(_StubBackend(), paths)
    assert len(col["records"]) == 4 and col["n_images"] == 2        # the empty mask is dropped
    assert col["feats"]["shapecoord"].shape[0] == 4
    assert col["feats"]["coords"].shape[0] == 4
    assert np.isfinite(col["feats"]["shapecoord"]).all(), "NaN features would disable the method globally"
    r = col["records"][0]
    assert {"iuid", "rle", "score", "H", "W", "abs_path", "cx", "cy", "bw", "bh"} <= set(r)
    assert [x["row"] for x in col["records"]] == [0, 1, 2, 3]       # the row invariant
    assert mu.decode(r["rle"]).any()


def test_assembly_honours_the_score_threshold(tmp_path):
    paths = _images(tmp_path / "img", n=1)
    assert len(B.build_collection(_StubBackend(), paths, score_thresh=0.5)["records"]) == 1
    assert len(B.build_collection(_StubBackend(), paths, score_thresh=0.0)["records"]) == 2


def test_geometry_only_batch_appends_to_a_richer_collection():
    """A model-free proposer produces geometry only; a qseg-seeded project also has `decoder`.
    concat_collections refuses a method mismatch, so the batch must be zero-filled to match."""
    master = {"records": [{"iuid": "a", "row": 0}],
              "feats": {"decoder": np.ones((1, 8), np.float32),
                        "shapecoord": np.ones((1, 29), np.float32)}}
    batch = {"records": [{"iuid": "b", "row": 0}, {"iuid": "c", "row": 1}],
             "feats": {"shapecoord": np.ones((2, 29), np.float32),
                       "coords": np.ones((2, 6), np.float32)}}
    out = CuratorEngine._align_batch_feats(batch, master)
    assert set(out["feats"]) == {"decoder", "shapecoord"}, "master-only methods filled, batch-only dropped"
    assert out["feats"]["decoder"].shape == (2, 8) and not out["feats"]["decoder"].any()
    assert np.isfinite(out["feats"]["decoder"]).all()


# --------------------------------------------------------------------------- API
def test_api_lists_backends_and_bootstraps(tmp_path):
    paths = _images(tmp_path / "img")
    cj = _coco(paths, tmp_path / "p.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    c = TestClient(create_app(engine=eng))

    names = {b["name"] for b in c.get("/api/backends").json()["backends"]}
    assert "coco" in names and "sam_auto" in names

    r = c.post("/api/propose", json={"backend": "coco", "coco_path": str(cj)})
    assert r.status_code == 200, r.text
    assert r.json()["n_instances"] == 6
    assert c.get("/api/state").json()["stats"]["n_instances"] == 6

    assert c.post("/api/propose", json={}).status_code == 400            # backend required
    assert c.post("/api/propose", json={"backend": "nope"}).status_code == 400
    eng.close()


def test_api_reports_a_missing_backend_as_400(tmp_path):
    """An uninstalled proposer is a configuration problem, not a crash."""
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    c = TestClient(create_app(engine=eng))

    class _Unavailable:
        name, label, requires = "nope2", "Needs a thing", "pip install a-thing"
        def available(self): return False, "the thing is not installed"
        def propose(self, *a, **k): return []
    B.register("nope2", _Unavailable)
    try:
        r = c.post("/api/propose", json={"backend": "nope2", "image_root": str(tmp_path)})
        assert r.status_code == 400
        assert "pip install a-thing" in r.json()["detail"]
    finally:
        B._REGISTRY.pop("nope2", None)
    eng.close()


# --------------------------------------------------------------------------- weights before images
class _SlowStartBackend(_StubBackend):
    """A proposer that must fetch weights first — SAM's shape, without SAM's half a gigabyte.

    Records the progress it was handed so a test can assert the download is REPORTED, not merely
    survived: the bug this guards against is a 400 MB fetch counted as "image 0 of 80", a counter
    that cannot move and that the UI is right to call stalled.
    """
    name, label = "slowstart", "Slow start"

    def __init__(self):
        self.ticks, self.stages, self.ready = [], [], False

    def prepare(self, *, progress=None, stage=None, **cfg):
        if stage:
            stage("downloading the checkpoint", 45)
        for done in (50, 100):
            if progress:
                progress(done, 100)
        if stage:
            stage("loading the model", 600)
        self.ready = True

    def propose(self, image_rgb, path=None, **cfg):
        assert self.ready, "the weights must be fetched before the image loop, not inside it"
        return super().propose(image_rgb, path=path, **cfg)


def test_a_proposer_fetches_its_weights_before_the_image_loop(tmp_path):
    """The download gets a phase of its own, with bytes — not a frozen image counter."""
    be = _SlowStartBackend()
    seen = []
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    _images(tmp_path / "img", n=2)
    orig = eng._set_progress

    def _spy(phase, done, total, **kw):
        seen.append({"phase": phase, "done": done, "total": total, **kw})
        orig(phase, done, total, **kw)

    eng._set_progress = _spy
    B.register("slowstart", lambda: be)
    try:
        rep = eng.propose_instances("slowstart", image_root=str(tmp_path / "img"))
    finally:
        B._REGISTRY.pop("slowstart", None)
    assert rep.get("ok"), rep

    dl = [s for s in seen if s["unit"] == "bytes"]
    got = [(s["done"], s["total"]) for s in dl if "downloading" in s["phase"]]
    assert (100, 100) in got, f"the fetch reported no bytes: {got}"
    load = [s for s in dl if "loading the model" in s["phase"]]
    # the load is not a download: a generous budget, so a silent minute is not called a stall
    assert load and load[0]["stall_after"] >= 300

    img = [s for s in seen if s["unit"] == "images"]
    assert img and all(s["phase"] == "proposing (slowstart)" for s in img), \
        "the filename belongs in `detail` — a phase that changes per image resets rate, ETA and stall"
    assert [s["detail"] for s in img if s.get("detail")] == ["im0.png", "im1.png"]
    assert all(s["stall_after"] > 0 for s in img), "one image can outlast the 20s default"
    eng.close()


def test_a_backend_without_prepare_still_runs(tmp_path):
    """`prepare` is optional — the coco/whole-image proposers have nothing to fetch."""
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    _images(tmp_path / "img", n=1)
    B.register("noprep", _StubBackend)
    try:
        rep = eng.propose_instances("noprep", image_root=str(tmp_path / "img"))
    finally:
        B._REGISTRY.pop("noprep", None)
    assert rep.get("ok") and rep["n_instances"] == 2
    eng.close()


# --------------------------------------------------------------------------- MPS dtype
def test_the_sam_point_grid_is_handed_over_in_float32():
    """SAM's generator builds its grid in numpy (float64) and hands it straight to torch, and Metal
    supports no float64 at all — so the whole run died on "Cannot convert a MPS Tensor to float64"."""
    from chevron.backends.sam_auto import _coords_in_float32

    class _Transform:
        def apply_coords(self, coords, original_size):
            return np.asarray(coords, np.float64) * 2      # what the real transform returns

    class _Gen:
        def __init__(self): self.predictor = type("P", (), {"transform": _Transform()})()

    gen = _coords_in_float32(_Gen())
    tr = gen.predictor.transform
    out = tr.apply_coords(np.array([[1.0, 2.0]]), (10, 10))
    assert out.dtype == np.float32 and out.tolist() == [[2.0, 4.0]], "the transform must still transform"
    once = tr.apply_coords
    assert _coords_in_float32(gen).predictor.transform.apply_coords is once, "patched twice = a cast per call"
    assert _coords_in_float32(object()) is not None                  # a generator with no predictor
