"""Reference-bank exemplar path resolution (tolerate a moved/renamed dataset) + the filesystem-suggest
endpoint behind the path-input Tab autocomplete. Model-free.
Run: pytest tools/curator/tests/test_reference_paths.py -q
"""
from __future__ import annotations

import tempfile


def test_resolve_ref_root_finds_moved_dataset(tmp_path):
    from tools.curator.engine import CuratorEngine
    rel = "reference_db/raw/dev/x.jpeg"
    moved = tmp_path / "fb-coco-reference 2"                      # dataset moved DOWN into a sibling dir
    (moved / "reference_db/raw/dev").mkdir(parents=True)
    (moved / rel).write_bytes(b"x")
    # the stale root (no reference_db directly under it) must be re-resolved to the moved dir
    assert CuratorEngine._resolve_ref_root(tmp_path, rel) == moved
    # already-correct root is returned unchanged
    assert CuratorEngine._resolve_ref_root(moved, rel) == moved
    # dataset moved UP a level
    up_rel = "imgs/y.png"; (tmp_path / "imgs").mkdir(); (tmp_path / up_rel).write_bytes(b"y")
    assert CuratorEngine._resolve_ref_root(tmp_path / "imgs", up_rel) == tmp_path
    # unresolvable -> returns start (no crash)
    assert CuratorEngine._resolve_ref_root(tmp_path, "nope/nope.jpg") == tmp_path


def _client():
    from fastapi.testclient import TestClient
    from tools.curator.server import create_app
    return TestClient(create_app(tempfile.mkdtemp()))


def test_fs_suggest_lists_children_and_prefix(tmp_path):
    (tmp_path / "alpha").mkdir(); (tmp_path / "beta").mkdir()
    (tmp_path / "coco.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("x")                     # non-json file is filtered out
    c = _client()
    # trailing slash -> list children (dirs get '/', only dirs + .json surfaced)
    items = c.get("/api/fs/suggest", params={"path": str(tmp_path) + "/"}).json()["items"]
    assert f"{tmp_path}/alpha/" in items and f"{tmp_path}/beta/" in items
    assert f"{tmp_path}/coco.json" in items
    assert not any(i.endswith("notes.txt") for i in items)
    # partial -> prefix-match siblings in the parent dir
    pref = c.get("/api/fs/suggest", params={"path": str(tmp_path / "al")}).json()["items"]
    assert pref == [f"{tmp_path}/alpha/"]
    # nonexistent dir -> empty, no error
    assert c.get("/api/fs/suggest", params={"path": "/no/such/dir/x"}).json()["items"] == []
