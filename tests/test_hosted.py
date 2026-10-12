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
    eng = CuratorEngine(path)
    rng = np.random.default_rng(0)
    cents = rng.normal(0, 4, (6, 8))
    feats = np.array([cents[i % 6] + rng.normal(0, 0.3, 8) for i in range(n)], np.float32)
    eng.collection = {"records": [{"iuid": f"u{i}", "row": i, "score": 0.6, "batch_id": "b0",
                                   "image_id": 1000 + i // 4, "file_name": f"img{i // 4}.png", "H": 8, "W": 8}
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
    assert a.get("/api/fs/suggest", params={"path": "/etc/"}).status_code == 403
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
