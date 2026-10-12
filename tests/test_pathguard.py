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


def test_non_string_values_under_a_path_key_are_refused(tmp_path):
    g, data, _ = _guard(tmp_path)
    assert g.violations({"paths": {"/etc/x.png": 1}})                  # iterating a dict yields its keys
    assert g.violations({"paths": [["/etc/x.png"]]})
    assert g.violations({"path": 5})


def test_model_id_is_a_hub_id_or_a_checked_local_folder(tmp_path):
    g, data, _ = _guard(tmp_path)
    assert g.violations({"cfg": {"model_id": "facebook/mask2former-swin-small-coco-instance"}}) == []
    assert g.violations({"cfg": {"model_id": "/etc"}})
    assert g.violations({"cfg": {"model_id": str(data)}}) == []


def test_free_text_into_configs_is_refused(tmp_path):
    g, _, _ = _guard(tmp_path)
    assert g.violations({"cfg": {"overrides": ["MODEL.WEIGHTS=/root/x.pth"]}})
    assert g.violations({"config_name": "../../elsewhere"})
    assert g.violations({"config_name": "", "overrides": []}) == []


def test_prior_dir_in_a_project_config(tmp_path):
    g, _, _ = _guard(tmp_path)
    assert g.violations({"config": {"shape_prior": {"prior_dir": "/root"}}})


def test_the_servers_own_dot_files_are_off_limits(tmp_path):
    g, data, projects = _guard(tmp_path)
    assert not g.allowed(str(projects / ".chevron-users.json"))
    assert not g.allowed(str(projects / ".chevron-sessions.json"))
    assert not g.allowed(str(data / ".hidden" / "x.png"))
    assert g.allowed(str(projects / "p1" / "state.json"))
