"""Long requests detach into jobs; heavy ones queue for one slot (chevron.jobs).
Run: pytest tests/test_jobs.py -q
"""
from __future__ import annotations

import threading
import time

from fastapi import Body, FastAPI
from fastapi.testclient import TestClient

from chevron.jobs import DetachMiddleware


def _app(after=0.2):
    app = FastAPI()
    log = []

    @app.post("/api/fast")
    def fast(body: dict = Body(default={})):
        return {"echo": body.get("x")}

    @app.post("/api/slow")
    def slow(body: dict = Body(default={})):
        time.sleep(float(body.get("s", 0.6)))
        return {"done": body.get("x")}

    @app.post("/api/propose")
    def heavy(body: dict = Body(default={})):
        log.append(("start", body["x"], time.time()))
        time.sleep(0.5)
        log.append(("end", body["x"], time.time()))
        return {"ok": body["x"]}

    @app.post("/api/boom")
    def boom(body: dict = Body(default={})):
        time.sleep(0.4)
        raise RuntimeError("kaputt")

    mw = DetachMiddleware(app, after=after)
    return mw, log


def _wait(c, r):
    while r.status_code == 202:
        time.sleep(0.05)
        r = c.get(f"/api/jobs/{r.json()['job']}")
    return r


def test_fast_requests_answer_directly():
    mw, _ = _app()
    with TestClient(mw, headers={"X-Chevron-Detach": "1"}) as c:
        r = c.post("/api/fast", json={"x": 3})
        assert r.status_code == 200 and r.json() == {"echo": 3}


def test_slow_request_detaches_and_delivers_its_response():
    mw, _ = _app()
    with TestClient(mw, headers={"X-Chevron-Detach": "1"}) as c:
        r = c.post("/api/slow", json={"x": "a"})
        assert r.status_code == 202 and r.json()["detached"]
        jid = r.json()["job"]
        r = _wait(c, r)
        assert r.status_code == 200 and r.json() == {"done": "a"}
        assert c.get(f"/api/jobs/{jid}").json() == {"done": "a"}       # polling again is idempotent


def test_unknown_job_is_404():
    mw, _ = _app()
    with TestClient(mw, headers={"X-Chevron-Detach": "1"}) as c:
        assert c.get("/api/jobs/nope").status_code == 404


def test_a_crash_in_a_detached_job_comes_back_as_500():
    mw, _ = _app()
    with TestClient(mw, raise_server_exceptions=False, headers={"X-Chevron-Detach": "1"}) as c:
        r = _wait(c, c.post("/api/boom", json={}))
        assert r.status_code == 500


def test_heavy_jobs_run_one_at_a_time():
    mw, log = _app(after=0.1)
    with TestClient(mw, headers={"X-Chevron-Detach": "1"}) as c:
        res = {}

        def go(x):
            res[x] = _wait(c, c.post("/api/propose", json={"x": x})).json()
        ths = [threading.Thread(target=go, args=(x,)) for x in ("a", "b")]
        for t in ths:
            t.start()
        time.sleep(0.25)
        st = mw.heavy_state()
        assert st["running"] and len(st["waiting"]) == 1
        for t in ths:
            t.join()
    assert res == {"a": {"ok": "a"}, "b": {"ok": "b"}}
    (s1, x1, _), (e1, _, t_end1), (s2, x2, t_start2), _ = log
    assert (s1, e1, s2) == ("start", "end", "start") and t_start2 >= t_end1


def test_without_the_opt_in_header_a_slow_request_answers_plainly():
    mw, _ = _app()
    with TestClient(mw) as c:
        r = c.post("/api/slow", json={"x": "a", "s": 0.5})
        assert r.status_code == 200 and r.json() == {"done": "a"}
