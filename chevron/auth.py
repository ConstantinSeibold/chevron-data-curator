"""Accounts for hosted mode: users, sessions and a login throttle.

Local use has no accounts at all. Hosted mode (`--data-root`) needs them, and nothing more than a
small team needs: a user store an admin manages, a session cookie, and a brake on password guessing.
Both stores are JSON files next to the projects (`<root>/.chevron-users.json`,
`<root>/.chevron-sessions.json`), written tmp -> os.replace like every other small file.

Passwords are hashed with scrypt from the standard library, so this adds no dependency. Session
tokens are only ever stored as their sha256, so a leaked sessions file cannot be replayed.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

USERS_FILE = ".chevron-users.json"
SESSIONS_FILE = ".chevron-sessions.json"
ROLES = ("admin", "member")
MIN_PASSWORD = 8
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    b64 = lambda b: base64.b64encode(b).decode()
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${b64(salt)}${b64(dk)}"


def check_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, want = stored.split("$")
        if algo != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), dklen=32,
                            n=int(n), r=int(r), p=int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk, base64.b64decode(want))


# verifying an unknown name costs the same as a known one, so timing does not reveal who exists
_DUMMY_HASH = hash_password(secrets.token_hex(8))


@dataclass
class User:
    name: str
    role: str
    created: float

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class UserStore:
    def __init__(self, root: str | Path):
        self.path = Path(root).expanduser() / USERS_FILE
        self._lock = threading.Lock()
        self._cache: tuple | None = None              # (mtime_ns, size, users): read once per change

    def _load(self) -> dict:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return {}
        key = (st.st_mtime_ns, st.st_size)
        if self._cache is None or self._cache[:2] != key:
            self._cache = (*key, json.loads(self.path.read_text()).get("users", {}))
        return json.loads(json.dumps(self._cache[2]))  # a copy: callers edit what they get

    def _save(self, users: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(self.path, {"users": users})
        self._cache = None

    @staticmethod
    def _norm(name: str) -> str:
        name = str(name or "").strip().lower()
        if not _NAME_RE.match(name):
            raise ValueError("user names are 1-32 characters: letters, digits, '.', '_' or '-'")
        return name

    @staticmethod
    def _check_password(password: str) -> None:
        if len(password or "") < MIN_PASSWORD:
            raise ValueError(f"passwords need at least {MIN_PASSWORD} characters")

    @staticmethod
    def _check_role(role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"role must be one of {', '.join(ROLES)}")

    def get(self, name: str) -> User | None:
        try:
            name = self._norm(name)
        except ValueError:
            return None
        d = self._load().get(name)
        return User(name, d["role"], d.get("created", 0.0)) if d else None

    def list(self) -> list[dict]:
        return [{"name": n, "role": d["role"], "created": d.get("created", 0.0)}
                for n, d in sorted(self._load().items())]

    def has_admin(self) -> bool:
        return any(d["role"] == "admin" for d in self._load().values())

    def add(self, name: str, password: str, role: str = "member") -> User:
        name = self._norm(name)
        self._check_password(password)
        self._check_role(role)
        with self._lock:
            users = self._load()
            if name in users:
                raise ValueError(f"user {name!r} already exists")
            users[name] = {"hash": hash_password(password), "role": role, "created": time.time()}
            self._save(users)
        return User(name, role, users[name]["created"])

    def verify(self, name: str, password: str) -> User | None:
        try:
            key = self._norm(name)
        except ValueError:
            key = None
        d = self._load().get(key) if key else None
        ok = check_password(password or "", d["hash"] if d else _DUMMY_HASH)
        return User(key, d["role"], d.get("created", 0.0)) if (d and ok) else None

    def _edit(self, name: str, fn) -> None:
        name = self._norm(name)
        with self._lock:
            users = self._load()
            if name not in users:
                raise KeyError(name)
            fn(users, name)
            self._save(users)

    @staticmethod
    def _keeps_an_admin(users: dict, name: str) -> None:
        if not any(d["role"] == "admin" for n, d in users.items() if n != name):
            raise ValueError("the last admin cannot be removed or demoted")

    def set_password(self, name: str, password: str) -> None:
        self._check_password(password)
        self._edit(name, lambda users, n: users[n].update(hash=hash_password(password)))

    def set_role(self, name: str, role: str) -> None:
        self._check_role(role)

        def fn(users, n):
            if role != "admin" and users[n]["role"] == "admin":
                self._keeps_an_admin(users, n)
            users[n]["role"] = role
        self._edit(name, fn)

    def remove(self, name: str) -> None:
        def fn(users, n):
            if users[n]["role"] == "admin":
                self._keeps_an_admin(users, n)
            del users[n]
        self._edit(name, fn)


@dataclass
class Session:
    id: str                       # sha256 of the cookie token
    user: str
    created: float
    seen: float
    pid: str | None = None        # the project this browser has open
    csrf: str = field(default_factory=lambda: secrets.token_urlsafe(16))


class SessionStore:
    """Sessions live in memory and are written through on create/revoke, and at most once a minute for
    `seen`/`pid` changes, so a restart keeps everyone signed in without a write per request."""

    def __init__(self, root: str | Path, ttl: float = 14 * 86400, clock=time.time):
        self.path = Path(root).expanduser() / SESSIONS_FILE
        self.ttl, self.clock = ttl, clock
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._saved_at = 0.0
        self._file_sig = None                        # (mtime_ns, size) of the file as we last saw it
        self._known_ids: set = set()                 # ids that file held when we last read or wrote it
        self._revoked: set = set()                   # ids THIS store revoked (a merge must not revive them)
        for d in self._read():
            try:
                self._sessions[d["id"]] = Session(**d)
            except TypeError:
                pass

    def _read(self) -> list[dict]:
        try:
            st = self.path.stat()
            self._file_sig = (st.st_mtime_ns, st.st_size)
            rows = json.loads(self.path.read_text()).get("sessions", [])
            self._known_ids = {d.get("id") for d in rows}
            return rows
        except (OSError, ValueError, AttributeError):
            return []

    def _sync(self) -> None:
        """Someone else rewrote the file (`chevron user passwd/remove` while the server runs): sessions it
        dropped are revoked here too. Only ones the file HAD: a session created since our last read is
        not in it yet, and must not be taken for revoked."""
        with self._write_lock:
            try:
                st = self.path.stat()
            except OSError:
                return
            if (st.st_mtime_ns, st.st_size) == self._file_sig:
                return
            before = set(self._known_ids)
            alive = {d.get("id") for d in self._read()}
            gone = before - alive
            if gone:
                with self._lock:
                    self._sessions = {k: v for k, v in self._sessions.items() if k not in gone}

    @staticmethod
    def _id(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def save(self) -> None:
        with self._write_lock:                       # read, reconcile and write as one step, in order
            known = set(self._known_ids)
            rows = self._read()
            on_disk = {d.get("id") for d in rows}
            with self._lock:
                for sid in known - on_disk:           # another process revoked these since we last looked
                    self._sessions.pop(sid, None)
                for d in rows:                        # ...and added these (we may be the stale writer)
                    if d.get("id") not in self._sessions and d.get("id") not in self._revoked:
                        try:
                            self._sessions[d["id"]] = Session(**d)
                        except TypeError:
                            pass
            with self._lock:
                now = self.clock()
                live = [asdict(s) for s in self._sessions.values() if now - s.seen <= self.ttl]
                self._saved_at = now
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(self.path, {"sessions": live})
            st = self.path.stat()
            self._file_sig = (st.st_mtime_ns, st.st_size)
            self._known_ids = {d["id"] for d in live}

    def create(self, user: str) -> str:
        token = secrets.token_urlsafe(32)
        now = self.clock()
        with self._lock:
            self._sessions[self._id(token)] = Session(self._id(token), user, now, now)
        self.save()
        return token

    def get(self, token: str | None) -> Session | None:
        if not token:
            return None
        sid = self._id(token)
        self._sync()
        now = self.clock()
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return None
            if now - s.seen > self.ttl:
                del self._sessions[sid]
                return None
            s.seen = now
            stale = now - self._saved_at > 60
        if stale:
            self.save()
        return s

    def all(self) -> list[Session]:
        now = self.clock()
        with self._lock:
            return [s for s in self._sessions.values() if now - s.seen <= self.ttl]

    def _forget_revoked(self) -> None:
        if len(self._revoked) > 10000:                # only needs to outlive the file's copy of them
            self._revoked = set(list(self._revoked)[-5000:])

    def revoke(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(self._id(token), None)
            self._revoked.add(self._id(token))
            self._forget_revoked()
        self.save()

    def revoke_user(self, user: str, *, keep: str | None = None) -> None:
        """End every session of `user`, except the one with id `keep` (the one changing its password)."""
        with self._lock:
            self._revoked |= {k for k, s in self._sessions.items() if s.user == user and k != keep}
            self._sessions = {k: s for k, s in self._sessions.items() if s.user != user or k == keep}
        self.save()


class Throttle:
    """At most `max_failures` failed logins per key (user + client address) within `window` seconds."""

    def __init__(self, max_failures: int = 5, window: float = 600, clock=time.monotonic):
        self.max_failures, self.window, self.clock = max_failures, window, clock
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str) -> list[float]:
        cutoff = self.clock() - self.window
        recent = [t for t in self._fails.get(key, []) if t > cutoff]
        if recent:
            self._fails[key] = recent
        else:
            self._fails.pop(key, None)
        return recent

    def allowed(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key)) < self.max_failures

    def fail(self, key: str) -> None:
        with self._lock:
            self._recent(key)
            self._fails.setdefault(key, []).append(self.clock())
            if len(self._fails) > 10000:              # many one-off keys: forget the expired ones
                for k in list(self._fails):
                    self._recent(k)

    def reset(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)
