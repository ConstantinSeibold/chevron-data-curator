"""`chevron` started from the environment alone, as the container does.
Run: pytest tests/test_server_cli.py -q
"""
from __future__ import annotations

import sys

import uvicorn

from chevron import server
from chevron.auth import UserStore


def _start(monkeypatch, argv):
    got = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: got.update(app=app, **kw))
    monkeypatch.setattr(sys, "argv", ["chevron", *argv])
    server.main()
    return got


def test_environment_turns_on_hosted_mode_and_trusts_the_proxy(tmp_path, monkeypatch):
    root, data = tmp_path / "projects", tmp_path / "data"
    data.mkdir()
    UserStore(root).add("ana", "correct horse", role="admin")
    monkeypatch.setenv("CHEVRON_DATA_ROOT", str(data))
    monkeypatch.setenv("CHEVRON_HOST", "0.0.0.0")
    monkeypatch.setenv("CHEVRON_PORT", "7999")
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    got = _start(monkeypatch, ["--root", str(root)])
    assert (got["host"], got["port"], got["forwarded_allow_ips"]) == ("0.0.0.0", 7999, "*")
    assert got["app"].state.hosting.hosted


def test_defaults_stay_local(tmp_path, monkeypatch):
    for k in ("CHEVRON_DATA_ROOT", "CHEVRON_HOST", "CHEVRON_PORT", "FORWARDED_ALLOW_IPS"):
        monkeypatch.delenv(k, raising=False)
    got = _start(monkeypatch, ["--root", str(tmp_path)])
    assert (got["host"], got["port"], got["forwarded_allow_ips"]) == ("127.0.0.1", 7870, "127.0.0.1")
    assert not got["app"].state.hosting.hosted
