"""Undo is per person: you undo your own last step, never someone else's, and never over their newer work.
Run: pytest tests/test_history_actors.py -q
"""
from __future__ import annotations

import json

from chevron.context import Ctx, use

from test_views import _engine


def _cls(eng, u):
    c = eng.state.meta[u].assigned_class
    return eng.state.class_name(c) if c else None


def test_each_person_undoes_their_own_step(tmp_path):
    eng = _engine(tmp_path)
    ana, bo = Ctx("ana"), Ctx("bo")
    with use(ana):
        eng.assign(["u0"], "cat")
    with use(bo):
        eng.assign(["u1"], "dog")
    with use(ana):
        assert eng.undo() == "assign"
    assert _cls(eng, "u0") is None and _cls(eng, "u1") == "dog"


def test_undo_refused_when_someone_changed_the_same_instance_since(tmp_path):
    eng = _engine(tmp_path)
    ana, bo = Ctx("ana"), Ctx("bo")
    with use(ana):
        eng.assign(["u0"], "cat")
    with use(bo):
        eng.assign(["u0"], "dog")
    with use(ana):
        res = eng.undo()
        assert res is None and eng.undo_conflict and "bo" in eng.undo_conflict
    with use(bo):
        assert eng.undo_conflict is None               # the refusal is ana's, not bo's
    assert _cls(eng, "u0") == "dog"                    # bo's newer label stands
    with use(ana):
        assert eng.history.depths == (0, 0)            # the unundoable step is dropped, not stuck on top


def test_redo_is_per_person_too(tmp_path):
    eng = _engine(tmp_path)
    ana, bo = Ctx("ana"), Ctx("bo")
    with use(ana):
        eng.assign(["u0"], "cat")
        eng.undo()
    with use(bo):
        assert eng.redo() is None                      # nothing of bo's to redo
    with use(ana):
        assert eng.redo() == "assign"
    assert _cls(eng, "u0") == "cat"


def test_unrelated_work_by_others_does_not_block_undo(tmp_path):
    eng = _engine(tmp_path)
    ana, bo = Ctx("ana"), Ctx("bo")
    with use(ana):
        eng.assign(["u0"], "cat")
    with use(bo):
        eng.assign(["u5"], "bird")
    with use(ana):
        assert eng.undo() == "assign" and eng.undo_conflict is None


def test_audit_log_records_who(tmp_path):
    eng = _engine(tmp_path)
    with use(Ctx("ana")):
        eng.assign(["u0"], "cat")
    lines = [json.loads(x) for x in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert any(d.get("op") == "assign" and d.get("user") == "ana" for d in lines)


def test_without_context_undo_works_as_before(tmp_path):
    eng = _engine(tmp_path)
    eng.assign(["u0"], "cat")
    eng.assign(["u1"], "dog")
    assert eng.undo() == "assign" and _cls(eng, "u1") is None
    assert eng.undo() == "assign" and _cls(eng, "u0") is None
    assert eng.redo() == "assign" and _cls(eng, "u0") == "cat"
