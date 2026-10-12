"""`chevron user ...` (chevron.usercli). Run: pytest tests/test_usercli.py -q"""
from __future__ import annotations

import io

import pytest

from chevron.auth import UserStore
from chevron.usercli import main


@pytest.fixture
def stdin(monkeypatch):
    def feed(text):
        monkeypatch.setattr("sys.stdin", io.StringIO(text))
    return feed


@pytest.mark.parametrize("before", [True, False])
def test_root_before_or_after_the_subcommand(tmp_path, stdin, before):
    stdin("correct horse\n")
    root = ["--root", str(tmp_path)]
    argv = root + ["add", "ana", "--admin"] if before else ["add", "ana", "--admin"] + root
    assert main(argv) == 0
    assert UserStore(tmp_path).verify("ana", "correct horse").role == "admin"


def test_passwd_role_remove_list(tmp_path, stdin, capsys):
    r = ["--root", str(tmp_path)]
    stdin("correct horse\n"); main(["add", "ana", "--admin"] + r)
    stdin("correct horse\n"); main(["add", "bo"] + r)
    stdin("new password\n"); assert main(["passwd", "bo"] + r) == 0
    assert UserStore(tmp_path).verify("bo", "new password")
    assert main(["role", "bo", "admin"] + r) == 0
    assert main(["remove", "ana"] + r) == 0
    capsys.readouterr()
    main(["list"] + r)
    assert capsys.readouterr().out.split() == ["bo", "admin"]


def test_errors_are_reported_not_raised(tmp_path, stdin, capsys):
    r = ["--root", str(tmp_path)]
    stdin("short\n")
    assert main(["add", "ana"] + r) == 1
    assert "at least" in capsys.readouterr().err
    assert main(["remove", "ghost"] + r) == 1
