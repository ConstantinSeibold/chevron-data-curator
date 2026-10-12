"""What turns the local curator into a server a team can share.

`server.create_app` wires these in. Local mode (no `--data-root`) uses only the parts that keep a
single person's behaviour exactly as it was: one implicit session, one resident project, the
engine's own default view. Hosted mode adds sign-in, a session per browser, a view per session,
several resident projects and the path guard.

The request path, outermost first:

    SessionMiddleware      who is asking (cookie -> session -> user), Origin check, request context
    PathGuardMiddleware    hosted: every client path must stay inside the allowed roots
    DetachMiddleware       slow POSTs become pollable jobs (chevron.jobs)
    UsingMiddleware        the session's project stays resident while its request runs
    FastAPI app
"""
from __future__ import annotations

import json
import time
import weakref
from http.cookies import SimpleCookie
from urllib.parse import parse_qsl, quote, urlsplit

from . import context
from .auth import Session

COOKIE = "chevron_session"
CLUSTER_HEADER = b"x-chevron-cluster"
PUBLIC_PATHS = frozenset({"/login", "/api/auth/login", "/api/auth/me", "/favicon.ico"})
UNSAFE = frozenset({"POST", "PUT", "PATCH", "DELETE"})
FRESH_PATHS = frozenset({"/api/partitions", "/api/state", "/api/cluster", "/api/level", "/api/version"})
PRESENCE_WINDOW = 60.0


async def _send_json(send, status: int, obj: dict, headers: list | None = None):
    body = json.dumps(obj).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())] + (headers or [])})
    await send({"type": "http.response.body", "body": body})


async def _redirect(send, location: str):
    await send({"type": "http.response.start", "status": 302,
                "headers": [(b"location", location.encode()), (b"content-length", b"0")]})
    await send({"type": "http.response.body", "body": b""})


def _headers(scope) -> dict[bytes, bytes]:
    return {k.lower(): v for k, v in scope.get("headers") or []}


def is_https(scope) -> bool:
    h = _headers(scope)
    return scope.get("scheme") == "https" or h.get(b"x-forwarded-proto", b"").split(b",")[0].strip() == b"https"


def client_ip(scope) -> str:
    h = _headers(scope)
    for k in (b"cf-connecting-ip", b"x-real-ip", b"x-forwarded-for"):
        if h.get(k):
            return h[k].decode("latin-1").split(",")[0].strip()
    return (scope.get("client") or ("?",))[0]


def session_cookie(token: str, *, secure: bool, max_age: int) -> str:
    return (f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}"
            + ("; Secure" if secure else ""))


class Hosting:
    """Shared state of one app: sessions, users, the local session, the pool, per-session views."""

    def __init__(self, *, pool, registry, users=None, sessions=None, guard=None):
        self.pool, self.registry = pool, registry
        self.users, self.sessions, self.guard = users, sessions, guard
        self.hosted = users is not None
        self.local = Session(id="local", user="", created=0.0, seen=0.0)
        self._views: dict[str, weakref.WeakKeyDictionary] = {}
        self.jobs: dict = {}                          # {"mw": DetachMiddleware} once the stack is built

    def views_for(self, sess: Session):
        if not self.hosted:
            return None                               # single user: the engine's own default view
        return self._views.setdefault(sess.id, weakref.WeakKeyDictionary())

    def drop_views(self, sess_id: str) -> None:
        self._views.pop(sess_id, None)

    def all_sessions(self) -> list[Session]:
        return self.sessions.all() if self.hosted else [self.local]

    def presence(self, pid: str | None, me: Session) -> list[dict]:
        if not self.hosted or pid is None:
            return []
        now, seen = time.time(), {}
        for s in self.sessions.all():
            if s.pid == pid and s.user != me.user and now - s.seen <= PRESENCE_WINDOW:
                seen[s.user] = max(seen.get(s.user, 0), s.seen)
        return [{"user": u, "seen": t} for u, t in sorted(seen.items())]


class SessionProjects:
    """The `active` project as seen by the CURRENT request's browser session.

    Same surface the server's endpoints always used (`pid`, `engine`, `open`, `close`, `adopt`), so
    they did not change; underneath, each session points at its own project in the pool.
    """

    def __init__(self, hosting: Hosting):
        self.h = hosting

    @property
    def session(self) -> Session:
        c = context.current()
        return c.session if c is not None and c.session is not None else self.h.local

    @property
    def pid(self):
        return self.session.pid

    @property
    def engine(self):
        pool, reg = self.h.pool, self.h.registry
        if reg is None:
            return pool.peek(None)
        pid = self.session.pid
        if pid is None:
            return None
        eng = pool.peek(pid)
        if eng is None:
            if not reg.exists(pid):                   # deleted or unlinked by someone else meanwhile
                self.session.pid = None
                return None
            eng = pool.get(pid)
        return eng

    def open(self, pid: str, path=None):
        prev, self.session.pid = self.session.pid, pid
        eng = self.h.pool.get(pid)
        if prev is not None and prev != pid and not self._wanted(prev):
            self.h.pool.release(prev)                 # nobody else has it open: let it go now
        if self.h.hosted:
            self.h.sessions.save()                    # a restart lands everyone back in their project
        return eng

    def _wanted(self, pid) -> bool:
        me = self.session
        return any(s.pid == pid and s.id != me.id for s in self.h.all_sessions())

    def close(self) -> None:
        pid, self.session.pid = self.session.pid, None
        if pid is not None and not self._wanted(pid):
            self.h.pool.close(pid)

    def close_for_everyone(self, pid) -> None:
        """Before a project directory goes away: release it and send every session back to the launcher."""
        for s in self.h.all_sessions():
            if s.pid == pid:
                s.pid = None
        self.h.pool.close(pid)

    def adopt(self, engine, pid=None) -> None:
        self.h.pool.adopt(engine, pid)


class SessionMiddleware:
    def __init__(self, app, hosting: Hosting):
        self.app, self.h = app, hosting

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        h, path = _headers(scope), scope["path"]
        if scope["method"] in UNSAFE and not self._same_origin(h):
            return await _send_json(send, 403, {"detail": "cross-site request refused"})
        rev = h.get(CLUSTER_HEADER, b"").decode("latin-1").strip() or None
        if path in FRESH_PATHS:
            rev = None                                # these (re)load partitions: nothing to be stale against
        if not self.h.hosted:
            ctx = context.Ctx(actor=None, views=None, cluster_rev=rev, session=self.h.local)
            with context.use(ctx):
                return await self.app(scope, receive, send)
        sess, user = self._who(h)
        if user is None and path not in PUBLIC_PATHS:
            if path.startswith("/api/"):
                return await _send_json(send, 401, {"detail": "sign in first", "code": "signed_out"})
            nxt = path + (("?" + scope["query_string"].decode("latin-1")) if scope.get("query_string") else "")
            return await _redirect(send, "/login?next=" + quote(nxt, safe=""))
        ctx = context.Ctx(actor=user.name if user else None,
                          views=self.h.views_for(sess) if sess else None,
                          cluster_rev=rev, session=sess, user=user)
        with context.use(ctx):
            return await self.app(scope, receive, send)

    def _who(self, h):
        cookie = SimpleCookie()
        try:
            cookie.load(h.get(b"cookie", b"").decode("latin-1"))
        except Exception:
            return None, None
        tok = cookie.get(COOKIE)
        sess = self.h.sessions.get(tok.value) if tok else None
        user = self.h.users.get(sess.user) if sess else None
        if sess is not None and user is None:         # the account was removed: the session goes too
            self.h.sessions.revoke(tok.value)
            return None, None
        return sess, user

    @staticmethod
    def _same_origin(h) -> bool:
        """A browser always sends Origin on a cross-site POST; it must name the host it was sent to.
        No Origin at all (curl, scripts, same-origin GET-style forms in old browsers) is let through."""
        origin = h.get(b"origin")
        if origin is None:
            ref = h.get(b"referer")
            if ref is None:
                return True
            origin = ref
        try:
            return urlsplit(origin.decode("latin-1")).netloc.lower() == h.get(b"host", b"").decode("latin-1").lower()
        except Exception:
            return False


class PathGuardMiddleware:
    def __init__(self, app, hosting: Hosting):
        self.app, self.h = app, hosting

    async def __call__(self, scope, receive, send):
        guard = self.h.guard
        if guard is None or scope["type"] != "http" or not scope["path"].startswith("/api/"):
            return await self.app(scope, receive, send)
        query = dict(parse_qsl(scope.get("query_string", b"").decode("latin-1")))
        bad = guard.violations(query)
        ctype = _headers(scope).get(b"content-type", b"")
        replay = receive
        if not bad and scope["method"] in UNSAFE and ctype.startswith(b"application/json"):
            chunks = []
            while True:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    return
                chunks.append(msg.get("body", b""))
                if not msg.get("more_body"):
                    break
            body, done = b"".join(chunks), False
            try:
                bad = guard.violations(json.loads(body or b"null"))
            except ValueError:
                bad = []

            async def replay():
                nonlocal done
                if not done:
                    done = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()
        if bad:
            return await _send_json(send, 403, {"detail": f"outside the data folder: {bad[0]}",
                                                "code": "path_outside_roots"})
        return await self.app(scope, replay, send)


class UsingMiddleware:
    def __init__(self, app, hosting: Hosting):
        self.app, self.h = app, hosting

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self.h.registry is None:
            return await self.app(scope, receive, send)
        c = context.current()
        sess = c.session if c is not None and c.session is not None else self.h.local
        self.h.pool.sweep()                           # over budget, or untouched for a while (hosted)
        pid = sess.pid
        # project management never needs the session's CURRENT engine resident, and holding it would
        # keep the project being switched away from alive past the switch
        if pid is None or scope["path"].startswith("/api/projects"):
            return await self.app(scope, receive, send)
        with self.h.pool.using(pid):
            return await self.app(scope, receive, send)
