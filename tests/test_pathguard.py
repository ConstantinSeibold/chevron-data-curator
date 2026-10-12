"""Hosted mode: no client-supplied path may leave the data and project roots.
Run: pytest tests/test_pathguard.py -q
"""
from __future__ import annotations

import os

from chevron.pathguard import PathGuard


def _guard(tmp_path):
    data, projects = tmp_path / "data", tmp_path / "projects"
    data.mkdir(); projects.mkdir()
    return PathGuard([data, projects]), data, projects


def test_inside_and_outside(tmp_path):
    g, data, projects = _guard(tmp_path)
    assert g.allowed(str(data / "set1" / "img.png"))
    assert g.allowed(str(projects / "p1"))
    assert not g.allowed("/etc/passwd")
    assert not g.allowed(str(data / ".." / "elsewhere"))
    assert not g.allowed(str(tmp_path / "data-other"))           # a sibling sharing the prefix


def test_symlink_out_of_the_root_is_caught(tmp_path):
    g, data, _ = _guard(tmp_path)
    os.symlink("/etc", data / "link")
    assert not g.allowed(str(data / "link" / "passwd"))


def test_home_tilde_is_expanded(tmp_path):
    g, _, _ = _guard(tmp_path)
    assert not g.allowed("~/.ssh/id_rsa")


def test_violations_walk_nested_bodies(tmp_path):
    g, data, _ = _guard(tmp_path)
    ok = {"name": "x", "config": {"images": {"root": str(data)}}, "paths": [str(data / "a.png")]}
    assert g.violations(ok) == []
    bad = {"name": "x", "config": {"images": {"root": "/etc"}}, "paths": [str(data / "a.png"), "/root/x"],
           "cfg": {"ckpt": "/home/someone/model.pth"}, "label": "/not/a/path/key"}
    assert sorted(g.violations(bad)) == ["/etc", "/home/someone/model.pth", "/root/x"]


def test_empty_values_are_fine(tmp_path):
    g, _, _ = _guard(tmp_path)
    assert g.violations({"path": "", "image_root": None, "paths": []}) == []
