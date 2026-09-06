"""Model-free unit tests for the curator backend core (ids/state/store/history/metrics).
Run: pytest tests/test_core.py -q   (from repo root)
"""
from __future__ import annotations

import numpy as np

from chevron import ids
from chevron.history import History
from chevron.metrics import filter_instances, partition_summary, sort_instances
from chevron.state import CuratorState, InstanceMeta
from chevron.store import Store


def _state(tmp_path, n=6):
    st = CuratorState(project_dir=str(tmp_path))
    for i in range(n):
        u = ids.new_uid()
        st.order.append(u)
        st.meta[u] = InstanceMeta(iuid=u, batch_id="b0", row=i, image_id=100 + (i % 2))
    st.rebuild_rows()
    return st


def _collection(st):
    recs = [{"score": 0.5 + 0.05 * i, "mask_area_frac": 0.01 * (i + 1), "row": i} for i in range(len(st.order))]
    return {"records": recs, "feats": {}, "n_images": 2}


def test_ids_unique():
    xs = {ids.new_uid() for _ in range(1000)}
    assert len(xs) == 1000
    assert ids.batch_id().startswith("b_") and ids.class_id().startswith("c_")


def test_state_roundtrip_and_classes(tmp_path):
    st = _state(tmp_path)
    cid = st.add_class("ett")
    assert st.add_class("ett") == cid                # idempotent by name
    st.meta[st.order[0]].assigned_class = cid
    d = st.to_dict()
    st2 = CuratorState.from_dict(d)
    assert st2.order == st.order
    assert st2.class_name(cid) == "ett"
    assert st2.meta[st.order[0]].assigned_class == cid
    st.assert_aligned(len(st.order))                 # invariant holds


def test_store_atomic_roundtrip(tmp_path):
    s = Store(tmp_path)
    s.ensure()
    st = _state(tmp_path)
    s.save_state(st)
    s.save_manifest({"schema_version": 1, "coll_version": 3, "processed_paths": ["a.png"], "n_instances": 6})
    s.save_collection({"records": [{"x": 1}], "feats": {}})
    assert s.is_project()
    assert s.load_state().order == st.order
    assert s.load_manifest()["coll_version"] == 3
    assert s.load_collection()["records"][0]["x"] == 1
    # history + refine + cache
    s.append_history({"op": "assign", "n": 1})
    assert s.read_history()[-1]["op"] == "assign"
    s.save_refine("u1", {"ops": ["otsu"]})
    assert s.load_refine("u1")["ops"] == ["otsu"]
    s.delete_refine("u1")
    assert s.load_refine("u1") is None
    s.save_cache("k1", np.zeros((6, 2), int), [3, 2], st.order)
    parts, counts, order = s.load_cache("k1")
    assert parts.shape == (6, 2) and counts == [3, 2] and order == st.order
    assert s.snapshot("test")


def test_history_undo_redo(tmp_path):
    s = Store(tmp_path); s.ensure()
    st = _state(tmp_path)
    h = History(s)
    cid = st.add_class("ngt")
    u = st.order[0]
    tok = h.begin(st, [u], [cid])
    st.meta[u].assigned_class = cid
    st.meta[u].assign_source = "manual"
    h.commit(st, tok, "assign", "assign ngt")
    assert st.meta[u].assigned_class == cid
    assert h.undo(st) == "assign"
    assert st.meta[u].assigned_class is None         # restored
    assert h.redo(st) == "assign"
    assert st.meta[u].assigned_class == cid          # re-applied
    h.barrier()
    assert h.depths == (0, 0)


def test_history_class_creation_undo(tmp_path):
    """Undo of an assign that created a new class also removes the class.
    Engine pattern: begin() BEFORE mutating, then append any newly-created class_id
    to the token's class_ids so the after-snapshot captures it."""
    s = Store(tmp_path); s.ensure()
    st = _state(tmp_path)
    u = st.order[0]
    H = History(s)
    tok = H.begin(st, [u], [])                       # before: no class exists yet
    cid = st.add_class("freshcls")
    tok["class_ids"].append(cid)                     # include new class in the after-snapshot
    st.meta[u].assigned_class = cid
    H.commit(st, tok, "assign_new", "assign freshcls (new)")
    assert "freshcls" in st.class_names()
    H.undo(st)
    assert "freshcls" not in st.class_names()         # class removed on undo
    assert st.meta[u].assigned_class is None
    H.redo(st)
    assert "freshcls" in st.class_names() and st.meta[u].assigned_class == cid


def test_metrics_partition_and_filter(tmp_path):
    st = _state(tmp_path, n=6)
    col = _collection(st)
    cid = st.add_class("a")
    # assign rows 0,1,2 to class a; leave 3,4,5 unassigned; partition labels: [0,0,0,1,1,1]
    for r in (0, 1, 2):
        st.meta[st.order[r]].assigned_class = cid
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([r["score"] for r in col["records"]], float)
    summ = partition_summary(labels, st, scores)
    assert summ[0]["size"] == 3 and summ[0]["purity"] == 1.0 and summ[0]["majority_class"] == cid
    assert summ[1]["frac_unassigned"] == 1.0 and summ[1]["purity"] is None
    # filters
    assert set(filter_instances(st, col, classes=[cid])) == {st.order[0], st.order[1], st.order[2]}
    assert len(filter_instances(st, col, only_unassigned=True)) == 3
    assert filter_instances(st, col, image_id=100) == [st.order[i] for i in (0, 2, 4)]
    srt = sort_instances(st.order, st, col, by="score", desc=True)
    assert srt[0] == st.order[5]                     # highest score
