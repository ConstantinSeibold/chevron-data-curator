"""The image folder typed into New project must be the folder the project actually reads.

The dialog offers an image folder, and Set up's step 1 promises it back ("this project reads: …"),
but the two lived at different keys: the launcher wrote a flat `image_root`, and everything that
resolves images reads `images.root`. So a project created with a folder reported none, and an ingest
left blank — the field whose placeholder says "blank = the project's image root" — searched nowhere
and reported finding no images.

Run: pytest tests/test_project_image_root.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from chevron.engine import CuratorEngine
from chevron.projects import ProjectRegistry

ROOT = Path(__file__).resolve().parents[1]


def _images(root, n=3, size=64):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(root / f"im{i}.png"),
                    (np.random.default_rng(i).random((size, size, 3)) * 200).astype(np.uint8))
    return sorted(str(p) for p in root.iterdir())


def _coco(paths, out, size=64, per_image=2):
    images, anns, aid = [], [], 1
    for i, p in enumerate(paths):
        images.append({"id": i + 1, "file_name": p, "width": size, "height": size})
        for k in range(per_image):
            x, y, w, h = 6 + 20 * k, 8, 14, 18
            anns.append({"id": aid, "image_id": i + 1, "category_id": 1, "score": 0.9,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                         "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]]})
            aid += 1
    out.write_text(json.dumps({"images": images, "annotations": anns,
                               "categories": [{"id": 1, "name": "thing"}]}))
    return out


# --------------------------------------------------------------------------- creation
def test_a_folder_typed_at_creation_is_where_the_project_reads(tmp_path):
    """The whole complaint: the path was accepted, stored, and then never consulted."""
    reg = ProjectRegistry(tmp_path / "projects")
    info = reg.create("Chest X-ray", {"image_root": "/data/imgs", "score_thresh": 0.5})
    eng = CuratorEngine(reg.path_for(info.id))
    assert eng.state.image_root() == "/data/imgs"
    assert eng.state.config["images"]["root"] == "/data/imgs", "not stored where the readers look"
    assert "image_root" not in eng.state.config, "the flat key was left behind as a second truth"
    eng.close()


def test_the_canonical_shape_is_taken_as_given(tmp_path):
    reg = ProjectRegistry(tmp_path / "projects")
    info = reg.create("nested", {"images": {"root": "/a", "other": 1}})
    eng = CuratorEngine(reg.path_for(info.id))
    assert eng.state.image_root() == "/a"
    assert eng.state.config["images"]["other"] == 1, "normalizing dropped the rest of images.*"
    eng.close()


def test_no_folder_leaves_no_root(tmp_path):
    reg = ProjectRegistry(tmp_path / "projects")
    info = reg.create("blank", {"score_thresh": 0.5})
    eng = CuratorEngine(reg.path_for(info.id))
    assert eng.state.image_root() == "", "an unset root must read as unset, not as a stray key"
    eng.close()


def test_an_already_created_project_still_resolves_its_folder(tmp_path):
    """Projects made before the fix carry the flat key on disk. Reading it keeps them working
    instead of quietly needing a rebuild."""
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({"image_root": "/legacy/imgs"})
    assert eng.state.image_root() == "/legacy/imgs"
    eng.close()


# --------------------------------------------------------------------------- ingest
def test_an_ingest_left_blank_uses_the_project_folder(tmp_path):
    """`Images: blank = the project's image root` — the field's own promise, from step 2."""
    imgs = tmp_path / "img"
    paths = _images(imgs)
    cj = tmp_path / "p.json"
    _coco([Path(p).name for p in paths], cj)           # basenames: only a root can resolve these
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({"images": {"root": str(imgs)}})

    rep = eng.propose_instances("coco", coco_path=str(cj))
    assert rep.get("ok"), rep
    assert rep["n_instances"] == 6 and rep["n_images"] == 3
    eng.close()


def test_a_typed_folder_still_wins_over_the_project_one(tmp_path):
    """Step 1 says the images field is how you pull from somewhere else for one run."""
    elsewhere = _images(tmp_path / "other")
    cj = tmp_path / "p.json"
    _coco([Path(p).name for p in elsewhere], cj)
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({"images": {"root": str(tmp_path / "empty")}})

    rep = eng.propose_instances("coco", coco_path=str(cj), image_root=str(tmp_path / "other"))
    assert rep.get("ok") and rep["n_instances"] == 6, rep
    eng.close()


def test_a_coco_next_to_its_own_images_survives_a_project_root(tmp_path):
    """The COCO backend resolves relative file_names against the json's folder. A project root the
    user never typed for THIS run must not take that convenience away."""
    _images(tmp_path / "beside")
    cj = tmp_path / "beside" / "p.json"
    _coco(["im0.png", "im1.png", "im2.png"], cj)
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({"images": {"root": str(tmp_path / "unrelated")}})

    rep = eng.propose_instances("coco", coco_path=str(cj))
    assert rep.get("ok") and rep["n_instances"] == 6, rep
    eng.close()


# --------------------------------------------------------------------------- end to end
def test_creating_a_project_over_the_api_reports_the_folder_back(tmp_path):
    """Exactly what the launcher does, then exactly what Set up step 1 reads."""
    from fastapi.testclient import TestClient

    from chevron.server import create_app
    c = TestClient(create_app(root=str(tmp_path / "projects")))
    r = c.post("/api/projects", json={"name": "Surgery demo",
                                      "config": {"images": {"root": "/data/frames"}, "model": {}}})
    assert r.status_code == 200, r.text
    assert c.get("/api/state").json()["image_root"] == "/data/frames"


# --------------------------------------------------------------------------- editing it later
def test_the_root_can_be_changed_after_creation(tmp_path):
    """The folder is chosen once, in a dialog, before the user has seen a single image of it."""
    a, b = tmp_path / "a", tmp_path / "b"
    _images(a, n=2)
    _images(b, n=3)
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({"images": {"root": str(a)}})

    rep = eng.set_image_root(str(b))
    assert rep["ok"] and rep["image_root"] == str(b) and rep["n_images"] == 3
    assert eng.state.image_root() == str(b)
    eng.close()

    assert CuratorEngine(tmp_path / "proj").state.image_root() == str(b), "the change was not saved"


def test_changing_the_root_retires_the_legacy_key(tmp_path):
    """Leaving the flat key behind would mean two answers to where the images are, and the older one
    would win again the day the accessor is simplified."""
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({"image_root": "/legacy/imgs"})
    _images(tmp_path / "now")

    eng.set_image_root(str(tmp_path / "now"))
    assert "image_root" not in eng.state.config
    assert eng.state.config["images"]["root"] == str(tmp_path / "now")
    eng.close()


def test_a_path_that_is_not_a_folder_is_refused_with_the_path_in_it(tmp_path):
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({"images": {"root": str(tmp_path)}})
    rep = eng.set_image_root(str(tmp_path / "nope"))
    assert "error" in rep and "nope" in rep["error"]
    assert eng.state.image_root() == str(tmp_path), "a refused path still moved the project"
    eng.close()


def test_the_root_can_be_cleared(tmp_path):
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({"images": {"root": str(tmp_path)}})
    assert eng.set_image_root("")["ok"]
    assert eng.state.image_root() == ""
    eng.close()


def test_a_typed_path_is_expanded_and_absolute(tmp_path, monkeypatch):
    """Users type `~/data` and relative paths; the stored root has to survive being read from a
    different working directory later."""
    home = tmp_path / "home"
    _images(home / "data")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home, raising=False)
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({})
    rep = eng.set_image_root("~/data")
    assert rep["ok"] and rep["image_root"] == str(home / "data"), rep
    eng.close()


def test_counting_images_stops_at_the_cap(tmp_path):
    """The box takes any path, including a home directory. The count is feedback, not an index."""
    _images(tmp_path / "many", n=7)
    assert CuratorEngine.count_images(str(tmp_path / "many"), cap=3) == (3, True)
    assert CuratorEngine.count_images(str(tmp_path / "many")) == (7, False)


def test_setting_the_root_over_the_api_is_what_the_next_state_reports(tmp_path):
    from fastapi.testclient import TestClient

    from chevron.server import create_app
    imgs = tmp_path / "img"
    _images(imgs)
    c = TestClient(create_app(root=str(tmp_path / "projects")))
    c.post("/api/projects", json={"name": "later", "config": {}})
    assert c.get("/api/state").json()["image_root"] == ""

    r = c.post("/api/image_root", json={"root": str(imgs)})
    assert r.status_code == 200 and r.json()["n_images"] == 3, r.text
    assert c.get("/api/state").json()["image_root"] == str(imgs)

    bad = c.post("/api/image_root", json={"root": str(tmp_path / "ghost")})
    assert bad.status_code == 400 and "not a folder" in bad.json()["detail"]
    assert c.get("/api/state").json()["image_root"] == str(imgs), "a refused path changed the project"


def test_a_corrected_root_is_what_the_next_ingest_uses(tmp_path):
    """The point of editing it: the next blank-field ingest reads the new folder."""
    good = tmp_path / "good"
    _images(good)
    cj = tmp_path / "p.json"
    _coco(["im0.png", "im1.png", "im2.png"], cj)
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({"images": {"root": str(tmp_path / "typo")}})

    eng.set_image_root(str(good))
    rep = eng.propose_instances("coco", coco_path=str(cj))
    assert rep.get("ok") and rep["n_instances"] == 6, rep
    eng.close()


# --------------------------------------------------------------------------- what the UI writes
def test_the_new_project_dialog_writes_the_key_the_project_reads():
    """A launcher that goes back to a flat `image_root` would be silently ignored again, and the
    only symptom is an empty step 1 in a different page."""
    js = (ROOT / "chevron" / "web" / "launcher.js").read_text()
    assert "config.images = { root: imageRoot }" in js, \
        "the launcher no longer writes images.root"


def test_the_dialog_does_not_promise_a_config_tab_that_does_not_exist():
    html = (ROOT / "chevron" / "web" / "launcher.html").read_text()
    assert "Config tab" not in html, "the dialog points the user at a tab the app does not have"
