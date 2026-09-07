"""Multi-project support: the registry, cheap summaries, and the server's active-project swap.

Run: pytest tests/test_projects.py -q
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chevron import ids
from chevron.engine import CuratorEngine
from chevron.projects import ProjectRegistry, slugify
from chevron.server import create_app
from chevron.state import InstanceMeta


def _populate(path, *, n=6, assigned=2, rejected=1):
    """A minimal but REAL project on disk (no collection.pkl needed for summaries)."""
    eng = CuratorEngine(path)
    eng.init_project({"model": {}, "score_thresh": 0.5})
    order, meta = [], {}
    for j in range(n):
        u = ids.new_uid()
        order.append(u)
        meta[u] = InstanceMeta(iuid=u, batch_id="b", row=j, image_id=1000 + (j % 3))
    cid = eng.state.add_class("device")
    for u in order[:assigned]:
        meta[u].assigned_class = cid
    for u in order[assigned:assigned + rejected]:
        meta[u].is_background = True
    eng.state.order, eng.state.meta = order, meta
    eng.save()
    man = eng.store.load_manifest()
    man["n_instances"] = n
    eng.store.save_manifest(man)
    eng.close()
    return order


# --------------------------------------------------------------------------- registry
def test_slugify():
    assert slugify("Chest X-ray  foreign bodies!") == "chest-x-ray-foreign-bodies"
    assert slugify("   ") == "project"          # never yields an empty directory name
    assert slugify("../etc") == "etc"           # traversal characters cannot survive


def test_create_list_and_summarize(tmp_path):
    reg = ProjectRegistry(tmp_path / "root")
    assert reg.list() == []

    info = reg.create("My Dataset", {"score_thresh": 0.4})
    assert info.id == "my-dataset" and info.name == "My Dataset"
    assert reg.exists("my-dataset")

    _populate(reg.path_for("my-dataset"), n=6, assigned=2, rejected=1)
    s = reg.summarize("my-dataset")
    assert s.n_instances == 6
    assert (s.n_assigned, s.n_rejected, s.n_unassigned) == (2, 1, 3)
    assert s.n_classes == 1 and s.n_images == 3
    assert s.pct_curated == pytest.approx(50.0)          # 3 of 6 decided

    assert [p.id for p in reg.list()] == ["my-dataset"]


def test_name_collision_gets_a_new_dir(tmp_path):
    reg = ProjectRegistry(tmp_path / "root")
    a = reg.create("Ribs")
    b = reg.create("Ribs")
    assert (a.id, b.id) == ("ribs", "ribs-2")            # never adopts an existing directory


def test_summary_is_cached_on_state_mtime(tmp_path):
    """The launcher must not re-parse a large state.json on every page load."""
    reg = ProjectRegistry(tmp_path / "root")
    reg.create("P")
    _populate(reg.path_for("p"), n=4, assigned=1, rejected=0)
    first = reg.summarize("p")
    assert first.n_assigned == 1

    calls = {"n": 0}
    import chevron.projects as mod
    real = mod._summarize_state

    def counting(path):
        calls["n"] += 1
        return real(path)

    mod._summarize_state = counting
    try:
        reg.summarize("p")
        assert calls["n"] == 0                            # served from the cache
        # touching state.json invalidates the stamp
        sp = reg.path_for("p") / "state.json"
        d = json.loads(sp.read_text())
        sp.write_text(json.dumps(d) + " ")                # changes size -> new stamp
        reg.summarize("p")
        assert calls["n"] == 1
    finally:
        mod._summarize_state = real


def test_rename_delete_and_bad_id(tmp_path):
    reg = ProjectRegistry(tmp_path / "root")
    reg.create("Alpha")
    assert reg.rename("alpha", "Renamed").name == "Renamed"
    assert reg.summarize("alpha").name == "Renamed"      # persisted in the registry file

    with pytest.raises(ValueError):
        reg.path_for("../escape")                        # no traversal outside the root

    reg.delete("alpha")
    assert not reg.exists("alpha") and reg.list() == []
    with pytest.raises(KeyError):
        reg.delete("alpha")


def test_unreadable_project_does_not_break_the_list(tmp_path):
    """One corrupt project must not take the whole launcher down."""
    reg = ProjectRegistry(tmp_path / "root")
    reg.create("Good")
    reg.create("Bad")
    (reg.path_for("bad") / "state.json").write_text("{ not json")
    infos = {p.id: p for p in reg.list()}
    assert infos["bad"].error and infos["good"].error is None


# --------------------------------------------------------------------------- server
def test_single_project_mode_is_unchanged(tmp_path):
    """Back-compat: --project serves the curator UI at / and has no project API."""
    _populate(tmp_path / "solo")
    c = TestClient(create_app(str(tmp_path / "solo")))
    assert "<title>" in c.get("/").text                   # the curator page, not the launcher
    assert c.get("/api/state").status_code == 200
    assert c.get("/api/projects").status_code == 400      # registry-only endpoint


def test_multi_project_flow(tmp_path):
    root = tmp_path / "root"
    reg = ProjectRegistry(root)
    reg.create("One")
    _populate(reg.path_for("one"), n=5, assigned=1, rejected=1)
    reg.create("Two")
    _populate(reg.path_for("two"), n=3, assigned=3, rejected=0)

    c = TestClient(create_app(root=str(root)))

    # / is the launcher; the curator page moves to /app
    assert "Chevron" in c.get("/").text and "launcher.js" in c.get("/").text
    assert c.get("/launcher.js").status_code == 200
    assert c.get("/app").status_code == 200

    # nothing open yet -> engine-backed endpoints refuse cleanly rather than 500
    assert c.get("/api/state").status_code == 409
    assert c.get("/api/session").json() == {"multi_project": True, "active": None, "project": None}

    listed = c.get("/api/projects").json()
    assert {p["id"] for p in listed["projects"]} == {"one", "two"} and listed["active"] is None

    assert c.post("/api/projects/one/open").status_code == 200
    assert c.get("/api/state").json()["stats"]["n_instances"] == 5
    assert c.get("/api/session").json()["active"] == "one"

    # switching swaps the engine behind the same captured `eng` reference
    assert c.post("/api/projects/two/open").status_code == 200
    assert c.get("/api/state").json()["stats"]["n_instances"] == 3

    assert c.post("/api/projects/close").status_code == 200
    assert c.get("/api/state").status_code == 409
    assert c.post("/api/projects/nope/open").status_code == 404


def test_switching_projects_clears_shared_image_caches(tmp_path):
    """The crop/image LRUs are process-wide; a project switch must not leave the previous
    project's working set resident to evict the incoming one."""
    import chevron.engine as E
    root = tmp_path / "root"
    reg = ProjectRegistry(root)
    reg.create("A"); _populate(reg.path_for("a"))
    reg.create("B"); _populate(reg.path_for("b"))
    c = TestClient(create_app(root=str(root)))

    c.post("/api/projects/a/open")
    E._IMG_CACHE["/some/a.png"] = np.zeros((2, 2, 3), np.uint8)
    E._CROP_CACHE[("iuid-a", 1, 2)] = np.zeros((2, 2, 3), np.uint8)

    c.post("/api/projects/b/open")
    assert len(E._IMG_CACHE) == 0 and len(E._CROP_CACHE) == 0


def test_create_via_api_opens_the_project(tmp_path):
    root = tmp_path / "root"
    c = TestClient(create_app(root=str(root)))
    r = c.post("/api/projects", json={"name": "Fresh Set", "config": {"score_thresh": 0.3}})
    assert r.status_code == 200
    assert r.json()["project"]["id"] == "fresh-set" and r.json()["active"] == "fresh-set"
    assert c.get("/api/state").status_code == 200        # engine is live immediately

    cfg = json.loads((root / "fresh-set" / "state.json").read_text())["config"]
    assert cfg["score_thresh"] == 0.3

    assert c.post("/api/projects", json={"name": "  "}).status_code == 400


def test_delete_active_project_releases_it(tmp_path):
    root = tmp_path / "root"
    reg = ProjectRegistry(root)
    reg.create("Doomed"); _populate(reg.path_for("doomed"))
    c = TestClient(create_app(root=str(root)))
    c.post("/api/projects/doomed/open")
    assert c.delete("/api/projects/doomed").json()["active"] is None
    assert not (root / "doomed").exists()
    assert c.get("/api/state").status_code == 409


# --------------------------------------------------------------- linked projects
def test_link_adopts_a_project_outside_the_root(tmp_path):
    """A project folder anywhere on disk lists and summarises like a local one — in place."""
    elsewhere = tmp_path / "elsewhere" / "old-run"
    _populate(elsewhere, n=4, assigned=2, rejected=1)

    reg = ProjectRegistry(tmp_path / "root")
    info = reg.link(elsewhere)
    assert info.id == "old-run" and info.linked and info.path == str(elsewhere)
    assert info.n_instances == 4 and (info.n_assigned, info.n_rejected) == (2, 1)

    assert [p.id for p in reg.list()] == ["old-run"]
    assert reg.exists("old-run") and reg.path_for("old-run") == elsewhere
    assert not (reg.root / "old-run").exists()               # nothing was copied or moved


def test_link_is_idempotent_and_names_are_deduped(tmp_path):
    reg = ProjectRegistry(tmp_path / "root")
    a = tmp_path / "a" / "runs"
    b = tmp_path / "b" / "runs"
    _populate(a); _populate(b)

    first = reg.link(a, "Runs")
    assert reg.link(a).id == first.id                        # re-adding the same path is a no-op
    assert len(reg.ids()) == 1

    second = reg.link(b)
    assert second.id == "runs-2" and second.path == str(b)   # same basename, distinct id

    # a linked id also reserves the name against a freshly created project
    assert reg.create("Runs").id == "runs-3"


def test_link_rejects_non_projects(tmp_path):
    reg = ProjectRegistry(tmp_path / "root")
    (tmp_path / "plain").mkdir()
    with pytest.raises(ValueError):
        reg.link(tmp_path / "plain")
    with pytest.raises(FileNotFoundError):
        reg.link(tmp_path / "nope")


def test_link_of_a_directory_inside_the_root_just_names_it(tmp_path):
    """Already discoverable — adopting it must not create a second, linked entry for the same dir."""
    reg = ProjectRegistry(tmp_path / "root")
    reg.create("Local")
    info = reg.link(reg.root / "local", "Renamed")
    assert info.id == "local" and not info.linked and info.name == "Renamed"
    assert reg.ids() == ["local"]


def test_unlink_forgets_without_deleting(tmp_path):
    src = tmp_path / "keepme"
    _populate(src)
    reg = ProjectRegistry(tmp_path / "root")
    pid = reg.link(src).id

    with pytest.raises(ValueError):
        reg.delete(pid)                                      # never rmtree outside the root
    reg.unlink(pid)
    assert reg.list() == [] and (src / "state.json").exists()
    with pytest.raises(KeyError):
        reg.unlink(pid)


def test_a_linked_path_that_vanished_still_lists_with_an_error(tmp_path):
    """So the stale entry stays visible and removable rather than silently disappearing."""
    src = tmp_path / "moved"
    _populate(src)
    reg = ProjectRegistry(tmp_path / "root")
    pid = reg.link(src).id
    shutil.rmtree(src)

    info = reg.summarize(pid)
    assert info.linked and info.error and "missing" in info.error
    assert [p.id for p in reg.list()] == [pid]
    reg.unlink(pid)
    assert reg.list() == []


def test_discover_handles_both_a_project_and_its_parent(tmp_path):
    holder = tmp_path / "holder"
    _populate(holder / "one")
    _populate(holder / "two")
    (holder / "not-a-project").mkdir()

    reg = ProjectRegistry(tmp_path / "root")
    assert [f["name"] for f in reg.discover(holder)] == ["one", "two"]
    assert [f["path"] for f in reg.discover(holder / "one")] == [str(holder / "one")]

    reg.link(holder / "one")
    found = {f["name"]: f for f in reg.discover(holder)}
    assert found["one"]["known_as"] == "one" and found["two"]["known_as"] is None

    with pytest.raises(FileNotFoundError):
        reg.discover(tmp_path / "nowhere")


# --------------------------------------------------------------------------- server
def test_scan_and_link_over_the_api(tmp_path):
    holder = tmp_path / "holder"
    _populate(holder / "external", n=7, assigned=3, rejected=0)
    root = tmp_path / "root"
    c = TestClient(create_app(root=str(root)))

    r = c.post("/api/projects/scan", json={"path": str(holder)})
    assert r.status_code == 200
    assert [f["name"] for f in r.json()["found"]] == ["external"]
    assert c.post("/api/projects/scan", json={"path": str(tmp_path / "nope")}).status_code == 400
    assert c.post("/api/projects/scan", json={"path": " "}).status_code == 400

    r = c.post("/api/projects/link", json={"path": str(holder / "external"), "open": True})
    assert r.status_code == 200 and r.json()["project"]["linked"] is True
    assert r.json()["active"] == "external"
    assert c.get("/api/state").json()["stats"]["n_instances"] == 7   # opens like any other project

    assert c.post("/api/projects/link", json={"path": str(holder)}).status_code == 400  # not a project


def test_api_refuses_to_delete_a_linked_project_but_unlinks_it(tmp_path):
    src = tmp_path / "outside"
    _populate(src)
    root = tmp_path / "root"
    c = TestClient(create_app(root=str(root)))
    c.post("/api/projects/link", json={"path": str(src), "open": True})

    assert c.delete("/api/projects/outside").status_code == 400
    assert (src / "state.json").exists()

    assert c.post("/api/projects/outside/unlink").json()["active"] is None
    assert (src / "state.json").exists()                     # the folder survives the unlink
    assert c.get("/api/projects").json()["projects"] == []
    assert c.post("/api/projects/outside/unlink").status_code == 404
