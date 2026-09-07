"""Sample mode: labelling whole images with the SAME machinery that labels mask instances.

This is the test of the claim the whole port rests on — *a sample is an instance with a trivial
mask*. If it holds, sample mode needs no second data model, grid, selection or feature pipeline, and
clustering / projection / classifier work untouched because they only ever read `feats[name]`.

So these tests deliberately exercise the SHARED paths rather than a sample-specific one.

Run: pytest tests/test_sample_mode.py -q
"""
from __future__ import annotations

import csv
import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from chevron.engine import CuratorEngine
from chevron.server import create_app


def _images(root, n=6, size=48):
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = root / f"img{i:02d}.png"
        # two visually distinct groups, so clustering has something real to find
        base = 40 if i % 2 == 0 else 200
        a = np.full((size, size, 3), base, np.uint8)
        a[:, :, i % 3] = (base + 40) % 255
        cv2.imwrite(str(p), a)
        out.append(str(p))
    return out


def _sample_project(tmp_path, n=6):
    paths = _images(tmp_path / "img", n)
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({"mode": "sample", "modality": "image"})
    rep = eng.propose_instances("whole_image", image_root=str(tmp_path / "img"))
    return eng, paths, rep


# --------------------------------------------------------------------------- the claim
def test_whole_images_become_items_through_the_normal_ingest(tmp_path):
    eng, paths, rep = _sample_project(tmp_path)
    assert rep.get("ok"), rep
    assert rep["n_instances"] == len(paths) == rep["n_images"], "exactly one item per image"
    eng.state.assert_aligned(eng.collection["feats"]["shapecoord"].shape[0])
    eng.close()


def test_items_are_stamped_as_samples(tmp_path):
    """The project's mode decides granularity; nothing special-cases the backend."""
    eng, _, _ = _sample_project(tmp_path)
    assert eng.state.is_sample_mode()
    assert {m.granularity for m in eng.state.meta.values()} == {"sample"}
    assert {m.modality for m in eng.state.meta.values()} == {"image"}
    eng.close()


def test_a_sample_mask_covers_the_whole_image(tmp_path):
    """The 'trivial mask' is not a metaphor — it is an all-ones mask, which is why pooling,
    cropping and rendering all work with no branch."""
    eng, _, _ = _sample_project(tmp_path, n=2)
    u = eng.state.order[0]
    m = eng._mask(u)
    assert m.all(), "a sample's mask must cover the image"
    crop = eng.crop(u, max_side=64)
    assert crop is not None and crop.ndim == 3
    eng.close()


def test_a_whole_image_mask_is_not_tinted(tmp_path):
    """A mask that covers everything distinguishes nothing, so the crop is the plain picture —
    an instance colour over the whole frame only stains what the user is trying to look at."""
    import numpy as np

    eng, _, _ = _sample_project(tmp_path, n=2)
    u = eng.state.order[0]
    plain = eng.crop(u, mask_overlay=False, max_side=64)
    tinted = eng.crop(u, mask_overlay=True, max_side=64)
    assert np.array_equal(plain, tinted)
    ov = eng.image_overlay(eng.state.meta[u].image_id, max_side=64)
    assert np.array_equal(ov, eng.image_overlay(eng.state.meta[u].image_id, max_side=64, show_masks=False))
    eng.close()


def test_clustering_and_projection_work_untouched(tmp_path):
    """The point of the design: the shared machinery needs no sample-specific code path."""
    eng, _, _ = _sample_project(tmp_path, n=8)
    info = eng.cluster({"shapecoord": 1.0, "coords": 1.0})
    assert info.get("n_levels", 0) >= 1
    p = eng.project({"coords": 1.0}, method="pca", dims=2)
    assert p["coords"].shape[0] == len(eng.state.order)
    assert eng.projection_points({"coords": 1.0}, method="pca", dims=2)["points"]
    eng.close()


def test_assign_and_reject_are_the_same_verbs(tmp_path):
    eng, _, _ = _sample_project(tmp_path, n=6)
    order = list(eng.state.order)
    eng.assign(order[:3], "healthy")
    eng.reject(order[3:4]) if hasattr(eng, "reject") else eng.set_background(order[3:4])
    assert sum(m.assigned_class is not None for m in eng.state.meta.values()) == 3
    eng.close()


# --------------------------------------------------------------------------- capabilities
def test_mask_only_tools_are_reported_unavailable(tmp_path):
    eng, _, _ = _sample_project(tmp_path, n=2)
    caps = eng.state.capabilities()
    assert caps["masks"] is False and caps["refine"] is False
    assert caps["merge"] is False and caps["substructure"] is False
    assert caps["coco_export"] is False
    c = TestClient(create_app(engine=eng))
    assert c.get("/api/state").json()["capabilities"]["refine"] is False
    eng.close()


# --------------------------------------------------------------------------- export
def test_manifest_export_lists_labelled_items(tmp_path):
    eng, paths, _ = _sample_project(tmp_path, n=6)
    order = list(eng.state.order)
    eng.assign(order[:2], "cat")
    eng.assign(order[2:4], "dog")
    rep = eng.export_manifest()
    assert rep["ok"] and rep["n_items"] == 4 and rep["n_classes"] == 2

    d = json.loads(open(rep["path"]).read())
    assert d["info"]["mode"] == "sample" and sorted(d["classes"]) == ["cat", "dog"]
    assert all({"file", "class"} <= set(i) for i in d["items"])
    assert all(i["granularity"] == "sample" for i in d["items"])

    rows = list(csv.DictReader(open(rep["csv"])))
    assert len(rows) == 4 and {r["class"] for r in rows} == {"cat", "dog"}
    eng.close()


def test_unreviewed_items_are_not_labels(tmp_path):
    """Only decided items belong in a manifest; an unreviewed image is not a negative."""
    eng, _, _ = _sample_project(tmp_path, n=5)
    eng.assign(list(eng.state.order)[:1], "cat")
    assert eng.export_manifest()["n_items"] == 1
    eng.close()


def test_rejected_items_are_opt_in(tmp_path):
    eng, _, _ = _sample_project(tmp_path, n=4)
    order = list(eng.state.order)
    eng.assign(order[:1], "cat")
    eng.set_background(order[1:3])
    assert eng.export_manifest()["n_items"] == 1
    rep = eng.export_manifest(include_rejected=True)
    assert rep["n_items"] == 3
    d = json.loads(open(rep["path"]).read())
    assert "__rejected__" in d["classes"]
    eng.close()


def test_api_export_returns_a_manifest_in_sample_mode(tmp_path):
    eng, _, _ = _sample_project(tmp_path, n=4)
    eng.assign(list(eng.state.order)[:2], "thing")
    c = TestClient(create_app(engine=eng))
    r = c.post("/api/export", json={})
    assert r.status_code == 200 and r.json()["kind"] == "manifest"
    assert r.json()["n_items"] == 2
    eng.close()


def test_instance_mode_still_exports_coco(tmp_path):
    """The branch must not disturb the normal path."""
    paths = _images(tmp_path / "img", 2)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})      # default = instance mode
    eng.propose_instances("whole_image", image_root=str(tmp_path / "img"))
    assert eng.state.capabilities()["coco_export"] is True
    c = TestClient(create_app(engine=eng))
    assert c.post("/api/export", json={}).json().get("kind") != "manifest"
    eng.close()
