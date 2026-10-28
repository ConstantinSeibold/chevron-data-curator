"""Hosted mode end to end: sign-in, two people at once, their own views, shared data, safe paths.
Run: pytest tests/test_hosted.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chevron.auth import UserStore
from chevron.projects import ProjectRegistry
from chevron.server import create_app

PW = "correct horse"


def _fill(path, n=60):
    """A project with a real collection, so it can be clustered and labelled through the API."""
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    from pycocotools import mask as mu
    eng = CuratorEngine(path)
    m = np.zeros((8, 8), np.uint8); m[2:6, 2:6] = 1
    rle = mu.encode(np.asfortranarray(m)); rle["counts"] = rle["counts"].decode()
    rng = np.random.default_rng(0)
    cents = rng.normal(0, 4, (6, 8))
    feats = np.array([cents[i % 6] + rng.normal(0, 0.3, 8) for i in range(n)], np.float32)
    eng.collection = {"records": [{"iuid": f"u{i}", "row": i, "score": 0.6, "batch_id": "b0",
                                   "image_id": 1000 + i // 4, "file_name": f"img{i // 4}.png", "H": 8, "W": 8,
                                   "rle": rle, "bbox": [2, 2, 4, 4]}
                                  for i in range(n)], "n_images": n // 4, "feats": {"decoder": feats}}
    eng.state.order = [f"u{i}" for i in range(n)]
    eng.state.meta = {f"u{i}": InstanceMeta(f"u{i}", "b0", i, 1000 + i // 4) for i in range(n)}
    eng.state.coll_version = 1
    eng.store.save_collection(eng.collection)
    eng.save()
    eng.close()


@pytest.fixture
def site(tmp_path):
    root, data = tmp_path / "projects", tmp_path / "data"
    data.mkdir()
    reg = ProjectRegistry(root)
    for name in ("Alpha", "Beta"):
        info = reg.create(name, {"images": {"root": str(data)}})
        _fill(reg.path_for(info.id))
    us = UserStore(root)
    us.add("ana", PW, role="admin")
    us.add("bo", PW)
    app = create_app(root=str(root), data_root=str(data))
    return app, root, data


def _login(app, name):
    c = TestClient(app)
    r = c.post("/api/auth/login", json={"username": name, "password": PW})
    assert r.status_code == 200, r.text
    return c


def test_signed_out_requests_are_refused(site):
    app, _, _ = site
    c = TestClient(app)
    assert c.get("/api/state").status_code == 401
    r = c.get("/app?x=1", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith("/login?next=%2Fapp%3Fx%3D1")
    assert c.get("/login").status_code == 200
    assert c.get("/api/auth/me").json() == {"hosted": True, "user": None, "role": None}


def test_login_logout(site):
    app, _, _ = site
    c = TestClient(app)
    assert c.post("/api/auth/login", json={"username": "ana", "password": "nope"}).status_code == 401
    c = _login(app, "ana")
    assert c.get("/api/auth/me").json()["user"] == "ana"
    assert c.get("/api/projects").status_code == 200
    c.post("/api/auth/logout")
    assert c.get("/api/projects").status_code == 401


def test_cookie_is_httponly_and_secure_behind_https(site):
    app, _, _ = site
    r = TestClient(app).post("/api/auth/login", json={"username": "ana", "password": PW},
                             headers={"X-Forwarded-Proto": "https"})
    ck = r.headers["set-cookie"]
    assert "HttpOnly" in ck and "SameSite=Lax" in ck and "Secure" in ck


def test_guessing_is_throttled(site):
    app, _, _ = site
    c = TestClient(app)
    codes = [c.post("/api/auth/login", json={"username": "bo", "password": f"wrong{i}"}).status_code
             for i in range(6)]
    assert codes[:5] == [401] * 5 and codes[5] == 429
    assert c.post("/api/auth/login", json={"username": "bo", "password": PW}).status_code == 429


def test_two_people_two_projects_at_once(site):
    app, _, _ = site
    a, b = _login(app, "ana"), _login(app, "bo")
    a.post("/api/projects/alpha/open")
    b.post("/api/projects/beta/open")
    assert a.get("/api/session").json()["active"] == "alpha"
    assert b.get("/api/session").json()["active"] == "beta"
    a.post("/api/assign", json={"iuids": ["u0"], "cls": "cat"})
    assert a.get("/api/state").json()["classes"] == ["cat"]
    assert b.get("/api/state").json()["classes"] == []          # beta is untouched


def test_same_project_shared_labels_separate_views(site):
    app, _, _ = site
    a, b = _login(app, "ana"), _login(app, "bo")
    a.post("/api/projects/alpha/open"); b.post("/api/projects/alpha/open")
    a.post("/api/assign", json={"iuids": ["u0", "u1"], "cls": "cat"})
    assert b.get("/api/state").json()["stats"]["n_assigned"] == 2
    a.post("/api/kind_filter", json={"granularity": "sample"})
    assert a.get("/api/version").json()["scope_token"] != b.get("/api/version").json()["scope_token"]
    assert b.get("/api/version").json()["scope_token"] == 0          # bo's view is not narrowed
    a.post("/api/kind_filter", json={})
    r = a.post("/api/cluster", json={"features": ["decoder"]})
    assert r.status_code == 200
    a.post("/api/level", json={"level": 0})
    b.post("/api/level", json={"level": 1})
    assert a.get("/api/state").json()["level"] == 0 and b.get("/api/state").json()["level"] == 1


def test_undo_is_per_person_over_http(site):
    app, _, _ = site
    a, b = _login(app, "ana"), _login(app, "bo")
    a.post("/api/projects/alpha/open"); b.post("/api/projects/alpha/open")
    a.post("/api/assign", json={"iuids": ["u0"], "cls": "cat"})
    b.post("/api/assign", json={"iuids": ["u0"], "cls": "dog"})
    r = a.post("/api/undo").json()
    assert r["ok"] is False and "bo" in r["error"]
    assert b.post("/api/undo").json()["ok"] is True


def test_stale_partition_is_409(site):
    app, _, _ = site
    a, b = _login(app, "ana"), _login(app, "bo")
    a.post("/api/projects/alpha/open"); b.post("/api/projects/alpha/open")
    rev = a.post("/api/cluster", json={"features": ["decoder"]}).json()["rev"]
    b.post("/api/cluster", json={"features": ["decoder"]})
    r = a.post("/api/reject_partition", json={"pid": "0"}, headers={"X-Chevron-Cluster": str(rev)})
    assert r.status_code == 409 and r.json()["code"] == "stale_cluster"
    assert a.get("/api/state").json()["stats"]["n_background"] == 0


def test_paths_outside_the_data_folder_are_refused(site):
    app, root, data = site
    a = _login(app, "ana")
    a.post("/api/projects/alpha/open")
    assert a.post("/api/image_root", json={"root": "/etc"}).status_code == 403
    assert a.post("/api/image_root", json={"root": str(data)}).status_code == 200
    assert a.get("/api/fs/suggest", params={"path": "/etc/"}).json()["items"] == []
    r = a.post("/api/projects", json={"name": "x", "config": {"images": {"root": "/home"}}})
    assert r.status_code == 403
    assert a.post("/api/projects/link", json={"path": "/tmp"}).status_code == 403


def test_cross_site_post_is_refused(site):
    app, _, _ = site
    a = _login(app, "ana")
    r = a.post("/api/projects/alpha/open", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = a.post("/api/projects/alpha/open", headers={"Origin": "http://testserver"})
    assert r.status_code == 200


def test_only_admins_manage_users_and_delete_projects(site):
    app, _, _ = site
    a, b = _login(app, "ana"), _login(app, "bo")
    assert b.get("/api/admin/users").status_code == 403
    assert b.delete("/api/projects/beta").status_code == 403
    assert a.post("/api/admin/users", json={"name": "cy", "password": PW}).status_code == 200
    assert [u["name"] for u in a.get("/api/admin/users").json()["users"]] == ["ana", "bo", "cy"]
    assert a.post("/api/admin/users/bo", json={"role": "admin"}).status_code == 200
    assert b.get("/api/admin/users").status_code == 200        # promoted: takes effect at once


def test_removing_a_user_ends_their_session(site):
    app, _, _ = site
    a, b = _login(app, "ana"), _login(app, "bo")
    assert a.delete("/api/admin/users/bo").status_code == 200
    assert b.get("/api/projects").status_code == 401


def test_deleting_an_open_project_sends_everyone_back(site):
    app, _, _ = site
    a, b = _login(app, "ana"), _login(app, "bo")
    b.post("/api/projects/beta/open")
    assert a.delete("/api/projects/beta").status_code == 200
    assert b.get("/api/session").json()["active"] is None
    assert b.get("/api/state").status_code == 409


def test_presence(site):
    app, _, _ = site
    a, b = _login(app, "ana"), _login(app, "bo")
    a.post("/api/projects/alpha/open"); b.post("/api/projects/alpha/open")
    b.get("/api/state")                                      # present = working in it, not just opened
    assert [u["user"] for u in a.get("/api/presence").json()["users"]] == ["bo"]
    b.post("/api/projects/beta/open")
    assert a.get("/api/presence").json()["users"] == []


def test_change_own_password(site):
    app, _, _ = site
    b = _login(app, "bo")
    assert b.post("/api/auth/password", json={"current": "nope", "new": "brand new pw"}).status_code == 400
    assert b.post("/api/auth/password", json={"current": PW, "new": "brand new pw"}).status_code == 200
    c = TestClient(app)
    assert c.post("/api/auth/login", json={"username": "bo", "password": "brand new pw"}).status_code == 200


def test_local_mode_has_no_sign_in(tmp_path):
    reg = ProjectRegistry(tmp_path / "p")
    reg.create("Alpha")
    c = TestClient(create_app(root=str(tmp_path / "p")))
    assert c.get("/api/projects").status_code == 200
    assert c.get("/api/auth/me").json() == {"hosted": False, "user": None, "role": None}
    assert c.post("/api/image_root", json={"root": "/etc"}).status_code != 403


def test_export_can_be_downloaded(site):
    app, _, _ = site
    a = _login(app, "ana")
    a.post("/api/projects/alpha/open")
    a.post("/api/assign", json={"iuids": ["u0"], "cls": "cat"})
    r = a.post("/api/export", json={})
    assert r.status_code == 200
    name = r.json()["path"].rsplit("/", 1)[-1]
    d = a.get(f"/api/exports/{name}")
    assert d.status_code == 200 and d.json()["categories"][0]["name"] == "cat"
    assert a.get("/api/exports/..%2F..%2Fstate.json").status_code == 404
    assert a.get("/api/exports/state.json").status_code == 404


def test_folder_completion_starts_at_the_data_folder(site):
    app, root, data = site
    (data / "set1").mkdir()
    a = _login(app, "ana")
    roots = a.get("/api/fs/suggest", params={"path": ""}).json()["items"]
    assert str(data) + "/" in roots
    assert a.get("/api/fs/suggest", params={"path": "/et"}).json()["items"] == []
    inside = a.get("/api/fs/suggest", params={"path": str(data) + "/"}).json()["items"]
    assert str(data / "set1") + "/" in inside


@pytest.mark.parametrize("ctype", ["Application/JSON", "application/x+json", "text/plain", "multipart/form-data",
                                   None])
def test_guard_checks_bodies_whatever_their_content_type(site, ctype):
    import json as _json
    app, _, _ = site
    a = _login(app, "ana")
    headers = {"Content-Type": ctype} if ctype else {}
    r = a.post("/api/projects/scan", content=_json.dumps({"path": "/etc"}), headers=headers)
    assert r.status_code == 403


def test_a_second_content_type_header_does_not_skip_the_guard(site):
    import json as _json
    app, _, _ = site
    a = _login(app, "ana")
    r = a.post("/api/projects/scan", content=_json.dumps({"path": "/etc"}),
               headers=[("Content-Type", "application/json"), ("Content-Type", "multipart/form-data")])
    assert r.status_code == 403


def test_only_admins_reset_a_project(site):
    app, _, _ = site
    a, b = _login(app, "ana"), _login(app, "bo")
    a.post("/api/projects/alpha/open"); b.post("/api/projects/alpha/open")
    assert b.post("/api/reset", json={"confirm": True}).status_code == 403
    assert a.get("/api/state").status_code == 200
    assert a.post("/api/reset", json={"confirm": True}).status_code == 200


def test_spoofed_forwarding_headers_do_not_reset_the_throttle(site):
    app, _, _ = site
    c = TestClient(app)
    codes = [c.post("/api/auth/login", json={"username": "bo", "password": "nope"},
                    headers={"CF-Connecting-IP": f"10.0.0.{i}", "X-Forwarded-For": f"10.1.0.{i}"}).status_code
             for i in range(6)]
    assert codes[-1] == 429


def test_changing_your_password_signs_out_your_other_sessions(site):
    app, _, _ = site
    b1, b2 = _login(app, "bo"), _login(app, "bo")
    assert b1.post("/api/auth/password", json={"current": PW, "new": "brand new pw"}).status_code == 200
    assert b1.get("/api/projects").status_code == 200
    assert b2.get("/api/projects").status_code == 401


def test_completion_never_lists_above_the_data_folder(site):
    app, root, data = site
    a = _login(app, "ana")
    items = a.get("/api/fs/suggest", params={"path": str(data) + "/../"}).json()["items"]
    assert not any(i.startswith(str(data.parent) + "/") and not i.startswith(str(data)) and
                   not i.startswith(str(root)) for i in items)


def test_two_tabs_of_one_browser_work_in_two_projects(site):
    app, _, _ = site
    a = _login(app, "ana")
    a.post("/api/projects/alpha/open")
    a.post("/api/projects/beta/open")                       # the second tab opened beta last
    tab1, tab2 = {"X-Chevron-Project": "alpha"}, {"X-Chevron-Project": "beta"}
    a.post("/api/assign", json={"iuids": ["u0"], "cls": "cat"}, headers=tab1)
    assert a.get("/api/state", headers=tab1).json()["classes"] == ["cat"]
    assert a.get("/api/state", headers=tab2).json()["classes"] == []
    # an <img> carries no header: the page it was requested from says which project
    r = a.get("/api/state", headers={"Referer": "http://testserver/app?p=alpha"})
    assert r.json()["classes"] == ["cat"]


def test_a_tab_on_a_deleted_project_is_sent_back(site):
    app, _, _ = site
    a, b = _login(app, "ana"), _login(app, "bo")
    b.post("/api/projects/beta/open")
    a.delete("/api/projects/beta")
    r = b.get("/api/state", headers={"X-Chevron-Project": "beta"})
    assert r.status_code == 409 and r.json()["code"] == "no_project"


def test_bad_project_names_from_a_tab_are_ignored(site):
    app, _, _ = site
    a = _login(app, "ana")
    a.post("/api/projects/alpha/open")
    r = a.get("/api/state", headers={"X-Chevron-Project": "../../etc"})
    assert r.status_code == 200                             # falls back to the session's project


def test_completion_hides_the_servers_dot_files(site):
    app, root, _ = site
    a = _login(app, "ana")
    items = a.get("/api/fs/suggest", params={"path": str(root) + "/"}).json()["items"]
    assert items and not any("/." in i for i in items)


def test_opening_a_second_project_keeps_the_first_tabs_project_resident(site):
    app, _, _ = site
    a = _login(app, "ana")
    a.post("/api/projects/alpha/open")
    tab1 = {"X-Chevron-Project": "alpha"}
    rev = a.post("/api/cluster", json={"features": ["decoder"]}, headers=tab1).json()["rev"]
    a.post("/api/projects/beta/open")                      # the second tab
    r = a.post("/api/reject_partition", json={"pid": "0"}, headers={**tab1, "X-Chevron-Cluster": rev})
    assert r.status_code == 200 and r.json()["n"] > 0      # alpha's clustering survived
