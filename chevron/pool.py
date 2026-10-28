"""Which projects are resident, for a server where several people work at once.

The server used to hold exactly ONE engine: opening a project closed whatever was open, for every
browser connected. That is right for one person on a laptop and wrong for a team, where Ana opening
project B must not pull project A out from under Bo.

The pool keeps engines by project id. Each browser session points at the project it has open, and
the pool loads that project on first use. Memory stays bounded: beyond `max_open` resident projects,
the least recently used one that is not serving a request is flushed and closed; with `idle` set,
so is any project nobody has touched for that long. `max_open=1` with no idle timeout is the old
single-project behaviour, which is what local mode uses.
"""
from __future__ import annotations

import contextlib
import threading
import time


class ProjectClosing(Exception):
    """The project is being deleted or unlinked; it cannot be opened until that is done."""


class ProjectPool:
    def __init__(self, registry=None, *, max_open: int = 1, idle: float | None = None, clock=time.monotonic,
                 single_user: bool = True):
        self.registry, self.single_user = registry, single_user
        self.max_open, self.idle, self.clock = max(1, int(max_open)), idle, clock
        self._engines: dict[str | None, object] = {}
        self._used: dict[str | None, float] = {}
        self._inflight: dict[str | None, int] = {}
        self._pinned: set = set()
        self._lock = threading.RLock()
        self._opening: dict[str, threading.Lock] = {}
        self._retired: set = set()

    def adopt(self, engine, pid: str | None = None) -> None:
        """Bind an already-constructed engine that is never evicted (single-project mode, tests)."""
        with self._lock:
            self._engines[pid] = engine
            self._used[pid] = self.clock()
            self._pinned.add(pid)

    def peek(self, pid):
        with self._lock:
            return self._engines.get(pid)

    def open_ids(self) -> list:
        with self._lock:
            return list(self._engines)

    def get(self, pid):
        """The engine for `pid`, loading it if it is not resident."""
        from .engine import CuratorEngine
        with self._lock:
            if pid in self._retired:
                raise ProjectClosing(f"project {pid} is being removed")
            eng = self._engines.get(pid)
            if eng is not None:
                self._used[pid] = self.clock()
                return eng
            gate = self._opening.setdefault(pid, threading.Lock())
        with gate:                                    # one load per project, however many ask at once
            with self._lock:
                eng = self._engines.get(pid)
                if eng is not None:
                    return eng
            eng = CuratorEngine(self.registry.path_for(pid))
            with self._lock:
                if pid in self._retired:              # retired while it loaded: never hand it out
                    self._opening.pop(pid, None)
                    eng.close()
                    raise ProjectClosing(f"project {pid} is being removed")
                self._engines[pid] = eng
                self._used[pid] = self.clock()
                self._opening.pop(pid, None)
                self._evict(keep=pid)
        return eng

    def busy(self, pid) -> bool:
        with self._lock:
            return bool(self._inflight.get(pid))

    def release(self, pid) -> bool:
        """Close `pid` now unless a request is still using it (then the next sweep does)."""
        with self._lock:
            if self._inflight.get(pid) or pid in self._pinned:
                return False
            self._close_locked(pid)
            return True

    @contextlib.contextmanager
    def retired(self, pid, busy=lambda: False):
        """Close `pid` and keep it closed for the block (its directory is going away), checking and
        closing in one step so no request can open it in between. Yields False, changing nothing,
        while a request or `busy()` (a job) is still working on it."""
        with self._lock:
            if self._inflight.get(pid) or busy():
                ok = False
            else:
                ok = True
                self._retired.add(pid)
                self._close_locked(pid)
        try:
            yield ok
        finally:
            if ok:
                with self._lock:
                    self._retired.discard(pid)

    @contextlib.contextmanager
    def using(self, pid):
        """Mark `pid` busy for the duration of a request, so it is never evicted mid-request."""
        with self._lock:
            self._inflight[pid] = self._inflight.get(pid, 0) + 1
            if pid in self._engines:
                self._used[pid] = self.clock()        # every request counts as use, not just the load
        try:
            yield
        finally:
            with self._lock:                          # recency is when a request STARTED, so a long one
                self._inflight[pid] -= 1              # ending does not outrank a newer project

    def _evict(self, keep=None) -> int:
        """Close what is over budget or idle. Caller holds the lock."""
        now, n = self.clock(), 0
        free = [p for p in self._engines
                if p != keep and p not in self._pinned and not self._inflight.get(p)]
        free.sort(key=lambda p: self._used.get(p, 0))
        for p in free:
            over = len(self._engines) > self.max_open
            stale = self.idle is not None and now - self._used.get(p, now) > self.idle
            if over or stale:
                self._close_locked(p)
                n += 1
        return n

    def sweep(self) -> int:
        with self._lock:
            return self._evict()

    def _close_locked(self, pid) -> None:
        eng = self._engines.pop(pid, None)
        self._used.pop(pid, None)
        self._pinned.discard(pid)
        if eng is not None:
            try:
                eng.close()                           # flushes any write-behind state
            except Exception:
                pass
            if self.single_user:                      # one person: the image/crop LRUs held the closed
                from .engine import clear_image_caches   # project's working set, which must not evict the next
                clear_image_caches()

    def close(self, pid) -> None:
        with self._lock:
            self._close_locked(pid)

    def close_all(self) -> None:
        with self._lock:
            for p in list(self._engines):
                self._close_locked(p)
