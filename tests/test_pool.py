"""Resident projects for several people at once (chevron.pool).
Run: pytest tests/test_pool.py -q
"""
from __future__ import annotations

import threading

from chevron.pool import ProjectPool
from chevron.projects import ProjectRegistry


def _reg(tmp_path, n=3):
    reg = ProjectRegistry(tmp_path)
    ids = [reg.create(f"p{i}", {"images": {"root": str(tmp_path)}}).id for i in range(n)]
    return reg, ids


def test_single_slot_closes_the_previous_project(tmp_path):
    reg, (a, b, _) = _reg(tmp_path)
    pool = ProjectPool(reg, max_open=1)
    ea = pool.get(a)
    pool.get(b)
    assert pool.open_ids() == [b] and ea._save_stop.is_set()


def test_several_slots_keep_several_projects(tmp_path):
    reg, (a, b, c) = _reg(tmp_path)
    pool = ProjectPool(reg, max_open=2)
    ea = pool.get(a); pool.get(b)
    assert pool.get(a) is ea                           # resident: the same engine, not reloaded
    pool.get(c)                                        # over budget: the least recently used goes
    assert sorted(pool.open_ids()) == sorted([a, c])


def test_a_project_serving_a_request_is_never_evicted(tmp_path):
    reg, (a, b, _) = _reg(tmp_path)
    pool = ProjectPool(reg, max_open=1)
    pool.get(a)
    with pool.using(a):
        pool.get(b)
        assert set(pool.open_ids()) == {a, b}
    pool.sweep()
    assert pool.open_ids() == [b]


def test_idle_projects_are_closed(tmp_path):
    now = [0.0]
    reg, (a, b, _) = _reg(tmp_path)
    pool = ProjectPool(reg, max_open=5, idle=60, clock=lambda: now[0])
    pool.get(a)
    now[0] = 30
    pool.get(b)
    now[0] = 70
    pool.sweep()
    assert pool.open_ids() == [b]


def test_concurrent_opens_load_once(tmp_path):
    reg, (a, _, _) = _reg(tmp_path)
    pool = ProjectPool(reg, max_open=2)
    got = []
    ths = [threading.Thread(target=lambda: got.append(pool.get(a))) for _ in range(6)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert len({id(e) for e in got}) == 1


def test_adopted_engine_is_pinned(tmp_path):
    reg, (a, b, _) = _reg(tmp_path)
    pool = ProjectPool(reg, max_open=1)
    ea = pool.get(a)
    pool2 = ProjectPool(reg, max_open=1)
    pool2.adopt(ea, a)
    pool2.get(b)
    assert set(pool2.open_ids()) == {a, b}


def test_steady_use_keeps_a_project_resident(tmp_path):
    """Recency is every request, not the load: a project used every few minutes is never idle-closed."""
    now = [0.0]
    reg, (a, _, _) = _reg(tmp_path)
    pool = ProjectPool(reg, max_open=5, idle=60, clock=lambda: now[0])
    ea = pool.get(a)
    for t in range(0, 600, 30):
        now[0] = t
        with pool.using(a):
            pool.sweep()
        pool.sweep()
    assert pool.peek(a) is ea


def test_a_retired_project_cannot_be_reopened_until_it_is_gone(tmp_path):
    import pytest
    from chevron.pool import ProjectClosing
    reg, (a, _, _) = _reg(tmp_path)
    pool = ProjectPool(reg, max_open=2)
    pool.get(a)
    with pool.retired(a) as ok:
        assert ok and a not in pool.open_ids()
        with pytest.raises(ProjectClosing):
            pool.get(a)
    assert pool.get(a) is not None                     # the id is free again afterwards


def test_retiring_a_busy_project_changes_nothing(tmp_path):
    reg, (a, _, _) = _reg(tmp_path)
    pool = ProjectPool(reg, max_open=2)
    ea = pool.get(a)
    with pool.using(a), pool.retired(a) as ok:
        assert not ok and pool.peek(a) is ea
    with pool.retired(a, busy=lambda: True) as ok:
        assert not ok and pool.peek(a) is ea
