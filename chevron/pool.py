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


class ProjectPool:
    def __init__(self, registry=None, *, max_open: int = 1, idle: float | None = None, clock=time.monotonic):
        self.registry = registry
        self.max_open, self.idle, self.clock = max(1, int(max_open)), idle, clock
        self._engines: dict[str | None, object] = {}
        self._used: dict[str | None, float] = {}
        self._inflight: dict[str | None, int] = {}
        self._pinned: set = set()
        self._lock = threading.RLock()
        self._opening: dict[str, threading.Lock] = {}

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
                self._engines[pid] = eng
                self._used[pid] = self.clock()
                self._opening.pop(pid, None)
                self._evict(keep=pid)
        return eng

    def release(self, pid) -> bool:
        """Close `pid` now unless a request is still using it (then the next sweep does)."""
        with self._lock:
            if self._inflight.get(pid) or pid in self._pinned:
                return False
            self._close_locked(pid)
            return True

    @contextlib.contextmanager
    def using(self, pid):
        """Mark `pid` busy for the duration of a request, so it is never evicted mid-request."""
        with self._lock:
            self._inflight[pid] = self._inflight.get(pid, 0) + 1
        try:
            yield
        finally:
            with self._lock:                          # recency is when it was ASKED for (get), so a long
                self._inflight[pid] -= 1              # request ending does not outrank a newer project

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
            if self.max_open == 1:                    # single project: the image/crop LRUs held its
                from .engine import clear_image_caches   # working set, which must not evict the next one
                clear_image_caches()

    def close(self, pid) -> None:
        with self._lock:
            self._close_locked(pid)

    def close_all(self) -> None:
        with self._lock:
            for p in list(self._engines):
                self._close_locked(p)
