"""Accounts for hosted mode: the user store, sessions, and the login throttle.
Run: pytest tests/test_auth.py -q
"""
from __future__ import annotations

import json

import pytest

from chevron.auth import SessionStore, Throttle, UserStore


def test_add_and_verify(tmp_path):
    us = UserStore(tmp_path)
    us.add("ana", "correct horse", role="admin")
    assert us.verify("ana", "correct horse").name == "ana"
    assert us.verify("ana", "wrong password") is None
    assert us.verify("nobody", "correct horse") is None
    assert us.has_admin()


def test_password_is_never_stored_in_clear(tmp_path):
    UserStore(tmp_path).add("ana", "correct horse")
    raw = (tmp_path / ".chevron-users.json").read_text()
    assert "correct horse" not in raw
    assert json.loads(raw)["users"]["ana"]["hash"].startswith("scrypt$")


def test_store_survives_reload(tmp_path):
    UserStore(tmp_path).add("ana", "correct horse", role="admin")
    us = UserStore(tmp_path)
    assert us.verify("ana", "correct horse").role == "admin"


@pytest.mark.parametrize("name", ["", "Ana Bell", "a/b", "x" * 33, "../etc"])
def test_rejects_bad_names(tmp_path, name):
    with pytest.raises(ValueError):
        UserStore(tmp_path).add(name, "correct horse")


def test_rejects_short_password_duplicate_and_bad_role(tmp_path):
    us = UserStore(tmp_path)
    with pytest.raises(ValueError):
        us.add("ana", "short")
    us.add("ana", "correct horse")
    with pytest.raises(ValueError):
        us.add("ana", "another one")
    with pytest.raises(ValueError):
        us.add("bo", "correct horse", role="root")


def test_names_are_case_insensitive(tmp_path):
    us = UserStore(tmp_path)
    us.add("Ana", "correct horse")
    assert us.verify("ANA", "correct horse").name == "ana"


def test_passwd_role_remove(tmp_path):
    us = UserStore(tmp_path)
    us.add("ana", "correct horse", role="admin")
    us.add("bo", "battery staple")
    us.set_password("bo", "new password")
    assert us.verify("bo", "battery staple") is None
    assert us.verify("bo", "new password")
    us.set_role("bo", "admin")
    assert us.get("bo").role == "admin"
    us.remove("bo")
    assert us.get("bo") is None
    assert [u["name"] for u in us.list()] == ["ana"]


def test_last_admin_cannot_be_removed_or_demoted(tmp_path):
    us = UserStore(tmp_path)
    us.add("ana", "correct horse", role="admin")
    with pytest.raises(ValueError):
        us.remove("ana")
    with pytest.raises(ValueError):
        us.set_role("ana", "member")


def test_sessions_create_get_revoke(tmp_path):
    ss = SessionStore(tmp_path)
    tok = ss.create("ana")
    s = ss.get(tok)
    assert s.user == "ana" and s.pid is None
    assert ss.get("not-a-token") is None
    ss.revoke(tok)
    assert ss.get(tok) is None


def test_session_token_is_stored_hashed_and_survives_restart(tmp_path):
    ss = SessionStore(tmp_path)
    tok = ss.create("ana")
    ss.get(tok).pid = "proj"
    ss.save()
    assert tok not in (tmp_path / ".chevron-sessions.json").read_text()
    s = SessionStore(tmp_path).get(tok)
    assert s.user == "ana" and s.pid == "proj"


def test_session_expires(tmp_path):
    now = [1000.0]
    ss = SessionStore(tmp_path, ttl=60, clock=lambda: now[0])
    tok = ss.create("ana")
    now[0] += 59
    assert ss.get(tok)                  # touching it slides the expiry
    now[0] += 59
    assert ss.get(tok)
    now[0] += 61
    assert ss.get(tok) is None


def test_revoke_user_ends_all_their_sessions(tmp_path):
    ss = SessionStore(tmp_path)
    a1, a2, b = ss.create("ana"), ss.create("ana"), ss.create("bo")
    ss.revoke_user("ana")
    assert ss.get(a1) is None and ss.get(a2) is None and ss.get(b)


def test_throttle_blocks_after_failures_then_recovers():
    now = [0.0]
    th = Throttle(max_failures=3, window=600, clock=lambda: now[0])
    for _ in range(3):
        assert th.allowed("ana|1.2.3.4")
        th.fail("ana|1.2.3.4")
    assert not th.allowed("ana|1.2.3.4")
    assert th.allowed("ana|5.6.7.8")
    now[0] += 601
    assert th.allowed("ana|1.2.3.4")


def test_throttle_reset_on_success():
    th = Throttle(max_failures=2, window=600)
    th.fail("k"); th.fail("k")
    th.reset("k")
    assert th.allowed("k")


def test_a_revocation_written_by_another_process_takes_effect(tmp_path):
    """`chevron user passwd` rewrites the file while the server holds sessions in memory."""
    server = SessionStore(tmp_path)
    tok = server.create("bo")
    assert server.get(tok)
    SessionStore(tmp_path).revoke_user("bo")          # the CLI, a separate store on the same file
    assert server.get(tok) is None
    server.save()
    assert SessionStore(tmp_path).get(tok) is None    # and the server's next save does not bring it back


def test_revoke_user_can_keep_the_current_session(tmp_path):
    ss = SessionStore(tmp_path)
    a, b = ss.create("ana"), ss.create("ana")
    ss.revoke_user("ana", keep=ss.get(a).id)
    assert ss.get(a) and ss.get(b) is None


def test_a_session_created_since_the_last_read_survives_a_sync(tmp_path):
    """Another process rewrote the file without knowing a session we created meanwhile: that session
    was never revoked, it is just newer than the file."""
    server = SessionStore(tmp_path)
    old = server.create("ana")
    other = SessionStore(tmp_path)                    # e.g. the CLI, loaded before the next login
    fresh = server.create("bo")                       # written to disk...
    other.revoke_user("ana")                          # rewrites from its stale view: no `bo` in it
    assert server.get(old) is None                    # the revocation reaches the server
    assert server.get(fresh) is not None              # the login made meanwhile does not vanish
