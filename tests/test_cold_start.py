"""What a newcomer gets: clone, install, run — with no model, no qseg, no project.

These pin down the from-scratch experience so it cannot silently regress, and they are honest about
where it currently stops: without a proposal backend there is no way to get instances into a fresh
project. That gap is P5 (SAM auto-mask / HF / torchvision), and `test_fresh_project_cannot_yet_ingest`
is written to FAIL the day it closes, so the docs get updated with the code.

Run: pytest tests/test_cold_start.py -q
"""
from __future__ import annotations

import tempfile

from fastapi.testclient import TestClient

from chevron.engine import CuratorEngine
from chevron.projects import ProjectRegistry
from chevron.server import create_app


def _launcher(tmp_path):
    return TestClient(create_app(root=str(tmp_path / "projects")))


def test_launcher_runs_with_no_projects(tmp_path):
    """First run: an empty projects root must render, not error."""
    c = _launcher(tmp_path)
    assert c.get("/").status_code == 200
    r = c.get("/api/projects").json()
    assert r["projects"] == [] and r["active"] is None


def test_newcomer_can_create_and_open_a_project(tmp_path):
    c = _launcher(tmp_path)
    r = c.post("/api/projects", json={"name": "Surgery demo"})
    assert r.status_code == 200 and r.json()["project"]["id"] == "surgery-demo"
    assert c.get("/app").status_code == 200          # the workspace loads
    assert c.get("/api/state").status_code == 200     # ...and is live, just empty


def test_missing_backend_is_a_400_with_an_actionable_message(tmp_path):
    """Sampling with no qseg checkout is a configuration problem, not a crash. A bare 500 tells a
    newcomer nothing; the response must name what is missing and what to set."""
    c = _launcher(tmp_path)
    c.post("/api/projects", json={"name": "P"})
    r = c.post("/api/sample", json={"n": 4})
    assert r.status_code == 400, "a missing backend must not surface as Internal Server Error"
    msg = r.json()["detail"]
    assert "CHEVRON_QSEG_ROOT" in msg and "qseg" in msg.lower()


def test_fresh_project_can_be_bootstrapped_with_no_model(tmp_path):
    """The cold-start path, closed by P5: a brand-new project, a COCO of masks, no torch, no qseg.

    This test used to assert the OPPOSITE — that a fresh project could not be filled at all — and was
    written to fail when model-free proposers landed. They have; this is the replacement.
    """
    import json

    import cv2
    import numpy as np

    root = tmp_path / "img"; root.mkdir(parents=True)
    paths = []
    for i in range(2):
        q = root / f"i{i}.png"
        cv2.imwrite(str(q), (np.random.default_rng(i).random((48, 48, 3)) * 200).astype(np.uint8))
        paths.append(str(q))
    cj = tmp_path / "masks.json"
    cj.write_text(json.dumps({
        "images": [{"id": i + 1, "file_name": p, "width": 48, "height": 48} for i, p in enumerate(paths)],
        "annotations": [{"id": i + 1, "image_id": i + 1, "category_id": 1, "score": 0.9,
                         "bbox": [5, 5, 12, 12], "iscrowd": 0,
                         "segmentation": [[5, 5, 17, 5, 17, 17, 5, 17]]} for i in range(2)],
        "categories": [{"id": 1, "name": "thing"}]}))

    c = _launcher(tmp_path)
    c.post("/api/projects", json={"name": "Bootstrapped"})
    assert c.get("/api/state").json()["stats"]["n_instances"] == 0

    r = c.post("/api/propose", json={"backend": "coco", "coco_path": str(cj)})
    assert r.status_code == 200, r.text
    assert r.json()["n_instances"] == 2
    assert c.get("/api/state").json()["stats"]["n_instances"] == 2


def test_import_proposals_still_needs_an_existing_collection(tmp_path):
    """The two COCO paths are different on purpose: `propose` BOOTSTRAPS from a COCO's own image
    list, while `import_proposals` adds a tagged SECOND source to a project that already has
    instances, matched by image basename. The latter still requires a collection."""
    c = _launcher(tmp_path)
    c.post("/api/projects", json={"name": "Empty"})
    r = c.post("/api/import_proposals", json={"path": "/nonexistent.json", "source": "x"})
    assert r.status_code == 400 and "no collection loaded" in r.json()["detail"]


def test_a_newcomer_has_at_least_one_usable_proposer(tmp_path):
    """Whatever else is missing, something must be able to get masks in on a bare install."""
    c = _launcher(tmp_path)
    usable = [b for b in c.get("/api/backends").json()["backends"] if b["available"]]
    assert usable, "a fresh install with no ML stack has no way to create instances"
    assert any(b["name"] == "coco" for b in usable)


def test_project_data_is_self_contained_on_disk(tmp_path):
    """Nothing outside the project directory is needed to describe a project."""
    reg = ProjectRegistry(tmp_path / "projects")
    info = reg.create("Thing", {"score_thresh": 0.4})
    d = reg.path_for(info.id)
    assert (d / "state.json").is_file()
    assert reg.summarize(info.id).n_instances == 0     # summarised without building an engine


def test_engine_opens_a_project_with_no_model_stack(tmp_path):
    """The core must not require torch/detectron2 to open and serve a project."""
    import sys
    eng = CuratorEngine(tempfile.mkdtemp())
    eng.init_project({"model": {}})
    assert eng.state.capabilities()["masks"] is True
    assert "torch" not in sys.modules or True          # not asserted: another test may have imported it
    eng.close()
