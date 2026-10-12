"""Long requests outlive the HTTP request that started them.

Getting masks or features for a folder is ONE POST that can run for an hour. Locally that is fine;
behind a reverse proxy or tunnel it is not (Cloudflare drops a request after 100 s with a 524, and
the browser reports a failure while the server carries on). This middleware detaches any JSON POST
still running after `after` seconds: the client gets `202 {"job": id}` and polls `GET /api/jobs/{id}`,
which answers 202 while the job runs and then the job's own response, byte for byte. Only requests
carrying `X-Chevron-Detach` are detached: the frontend's `api()` helper sends it and does the polling,
so no endpoint changed, and a script calling the API keeps its plain synchronous answer.

Heavy endpoints (model inference, embedding, training) also queue for ONE server-wide slot, so two
people starting runs on two projects do not load two models into one GPU at once. The queue is
visible through `heavy_state()`, which /api/progress reports.

Pure ASGI rather than BaseHTTPMiddleware: the job must keep running after its request has been
answered, and the contextvars (who is acting, through which view) must travel with it.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time
import traceback
from dataclasses import dataclass, field

from . import context

HEAVY_PATHS = frozenset({
    "/api/propose", "/api/remask", "/api/compute_features", "/api/compute_raddino", "/api/sample",
    "/api/infer_dir", "/api/reinfer", "/api/preview_infer", "/api/scaled_pseudolabel",
    "/api/import_proposals", "/api/train/launch", "/api/train/overfit_check", "/api/shape_transfer",
    "/api/auto_refine_many",
})


@dataclass
class Job:
    id: str
    path: str
    actor: str | None
    owner: str | None
    project: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    status: int = 0
    headers: list = field(default_factory=list)
    body: bytes = b""


def _json(send, status: int, obj: dict):
    body = json.dumps(obj).encode()
    return _replay(send, status, [(b"content-type", b"application/json")], body)


async def _replay(send, status: int, headers: list, body: bytes):
    headers = [(k, v) for k, v in headers if k.lower() != b"content-length"]
    headers.append((b"content-length", str(len(body)).encode()))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class DetachMiddleware:
    def __init__(self, app, after: float = 20.0, keep: float = 900.0, heavy=None, holder=None):
        self.app, self.after, self.keep = app, after, keep
        self.heavy = HEAVY_PATHS if heavy is None else frozenset(heavy)
        if holder is not None:
            holder["mw"] = self                       # the app reads heavy_state() through it
        self.jobs: dict[str, Job] = {}
        self._slot: asyncio.Semaphore | None = None
        self._running: dict | None = None
        self._waiting: list[dict] = []

    def heavy_state(self) -> dict:
        return {"running": self._running, "waiting": list(self._waiting)}

    @staticmethod
    def _project() -> str | None:
        c = context.current()
        if c is None:
            return None
        return c.project or getattr(c.session, "pid", None)

    def running_for(self, pid) -> bool:
        """A job still working in project `pid`, detached or queued for the heavy slot."""
        running = self._running                       # read from a worker thread: copies, not live views
        return any(j.project == pid and j.finished is None for j in list(self.jobs.values())) or \
            any(w.get("project") == pid for w in list(self._waiting)) or \
            bool(running and running.get("project") == pid)

    @staticmethod
    def _owner() -> str | None:
        c = context.current()
        s = getattr(c, "session", None) if c else None
        return getattr(s, "id", None)

    def _gc(self) -> None:
        now = time.time()
        for k in [k for k, j in self.jobs.items() if j.finished and now - j.finished > self.keep]:
            del self.jobs[k]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path, method = scope["path"], scope["method"]
        if method == "GET" and path.startswith("/api/jobs/"):
            return await self._poll(path[len("/api/jobs/"):], send)
        ctype = dict(scope.get("headers") or []).get(b"content-type", b"").lower()
        if method != "POST" or not path.startswith("/api/") or ctype.startswith(b"multipart/"):
            return await self.app(scope, receive, send)
        # only clients that know how to poll get a 202 (the UI says so); a script keeps its plain answer
        if b"x-chevron-detach" not in dict(scope.get("headers") or []):
            if path in self.heavy:
                job = Job(secrets.token_urlsafe(12), path, context.actor(), self._owner(), self._project())
                return await self._heavy(job, scope, receive, send)
            return await self.app(scope, receive, send)
        return await self._detachable(scope, receive, send, path)

    async def _poll(self, jid: str, send):
        self._gc()
        job = self.jobs.get(jid)
        if job is None or (job.owner is not None and job.owner != self._owner()):
            return await _json(send, 404, {"detail": "no such job"})
        if job.finished is None:
            return await _json(send, 202, {"job": job.id, "running": True,
                                           "elapsed": round(time.time() - job.started, 1)})
        return await _replay(send, job.status, job.headers, job.body)

    async def _detachable(self, scope, receive, send, path):
        chunks = []
        while True:                                   # the whole body, so the job can replay it later
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        body, finished = b"".join(chunks), asyncio.Event()
        sent_body = False

        async def replay_receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            await finished.wait()                     # never a disconnect while the job still runs
            return {"type": "http.disconnect"}

        job = Job(secrets.token_urlsafe(12), path, context.actor(), self._owner(), self._project())
        out_chunks: list[bytes] = []

        async def capture(msg):
            if msg["type"] == "http.response.start":
                job.status, job.headers = msg["status"], list(msg.get("headers") or [])
            elif msg["type"] == "http.response.body":
                out_chunks.append(msg.get("body", b""))

        detached = False

        async def run():
            try:
                if path in self.heavy:
                    await self._heavy(job, scope, replay_receive, capture)
                else:
                    await self.app(scope, replay_receive, capture)
            except Exception as e:                    # the app's own handlers did not answer
                job.status, job.headers = 500, [(b"content-type", b"application/json")]
                out_chunks[:] = [json.dumps({"detail": f"{type(e).__name__}: {e}"}).encode()]
                if not detached:
                    raise                             # answered in time: normal error handling + logging
                traceback.print_exc()                 # nobody is waiting on this request to log it
            finally:
                job.body, job.finished = b"".join(out_chunks), time.time()
                finished.set()

        task = asyncio.create_task(run())
        await asyncio.wait({task}, timeout=self.after)
        if task.done():
            task.result()                             # re-raises what the app raised
            return await _replay(send, job.status, job.headers, job.body)
        detached = True
        self._gc()
        self.jobs[job.id] = job
        job._task = task                              # keep a reference: the loop holds tasks weakly
        return await _json(send, 202, {"job": job.id, "detached": True})

    async def _heavy(self, job: Job, scope, receive, send):
        if self._slot is None:
            self._slot = asyncio.Semaphore(1)
        me = {"path": job.path, "user": job.actor, "since": job.started, "project": job.project}
        self._waiting.append(me)
        try:
            await self._slot.acquire()
        finally:
            self._waiting.remove(me)
        self._running = {**me, "since": time.time()}
        try:
            await self.app(scope, receive, send)
        finally:
            self._running = None
            self._slot.release()
