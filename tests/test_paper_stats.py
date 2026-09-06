"""paper_stats: harvest a curation project's logs (history/merge_log/state/lineage) into the paper aggregates.
Pure-python, no model. Run: pytest chevron/tests/test_paper_stats.py -q
"""
from __future__ import annotations


def _project(tmp_path):
    from chevron.state import CuratorState, InstanceMeta
    from chevron.store import Store
    st = Store(tmp_path); st.ensure()
    state = CuratorState(project_dir=str(tmp_path))
    cA, cB = state.add_class("letters"), state.add_class("tube")
    # 6 instances: 2 manual, 2 one-click-partition, 1 classifier, 1 unassigned; + 1 rejected, 1 merge-child
    rows = [("u_man1", cA, "manual", None), ("u_man2", cA, "manual", None),
            ("u_part1", cB, "partition", None), ("u_part2", cB, "partition", None),
            ("u_clf1", cB, "classifier", 0.82), ("u_un1", None, None, None)]
    for i, (u, cid, src, sc) in enumerate(rows):
        state.meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=1000 + i,
                                     assigned_class=cid, assign_source=src, assign_score=sc)
        state.order.append(u)
    state.meta["u_bg"] = InstanceMeta(iuid="u_bg", batch_id="b", row=6, image_id=1006, is_background=True)
    state.meta["u_child"] = InstanceMeta(iuid="u_child", batch_id="b", row=7, image_id=1000, merged_into="u_man1")
    state.order += ["u_bg", "u_child"]
    st.save_state(state)
    # history: 3 assigns + 1 merge + 1 reject + 1 undo (non-action)
    for i, (op, n) in enumerate([("assign", 2), ("assign", 2), ("classifier-assign", 1), ("merge", 2), ("background", 1)]):
        st.append_history({"cmd_id": f"c{i}", "ts": 100.0 + i, "op": op, "n_instances": n, "iuids": []})
    st.append_history({"ts": 110.0, "op": "undo", "of": "background"})
    # merge events: 1 manual merge, 2 recommended merges, 1 recommended reject (with ts + source)
    st.append_merge_event({"kind": "merge", "iuids": ["a", "b"], "source": "manual", "ts": 101.0})
    st.append_merge_event({"kind": "merge", "iuids": ["c", "d"], "source": "recommended", "ts": 105.0})
    st.append_merge_event({"kind": "merge", "iuids": ["e", "f"], "source": "recommended", "ts": 106.0})
    st.append_merge_event({"kind": "reject", "iuids": ["g", "h"], "source": "recommended", "ts": 107.0})
    # lineage: one retrain+adopt record
    st.append_lineage_event({"ts": 200.0, "coll_version": 3, "n_assigned": 5,
                             "export_json": "exports/curated.json", "ckpt": "out/model_best.pth",
                             "metric_name": "segm/AP", "metric": 41.3})
    return st


def test_action_stats(tmp_path):
    from chevron import paper_stats as ps
    a = ps.action_stats(_project(tmp_path))
    assert a["n_actions"] == 5 and a["n_undo"] == 1 and a["n_redo"] == 0
    assert a["by_op"]["assign"] == 2 and a["by_op"]["merge"] == 1
    assert a["n_instances_touched"] == 8                              # 2+2+1+2+1
    assert len(a["timeline"]) == 5 and a["timeline"][-1][1] == 5      # cumulative reaches n_actions
    assert a["duration_s"] == 4.0                                     # ts 100..104


def test_source_and_effort(tmp_path):
    from chevron import paper_stats as ps
    st = _project(tmp_path)
    s = ps.source_stats(st.load_state())
    assert s["n_assigned"] == 5 and s["by_source"] == {"manual": 2, "partition": 2, "classifier": 1}
    assert s["n_unassigned"] == 1 and s["n_rejected"] == 1 and s["n_merged_children"] == 1
    out = ps.summarize(tmp_path)
    assert abs(out["actions_per_verified_instance"] - 5 / 5) < 1e-9   # 5 actions / 5 verified


def test_merge_recommender_accept_rate(tmp_path):
    from chevron import paper_stats as ps
    m = ps.merge_stats(_project(tmp_path))
    assert m["n_merge_events"] == 3 and m["n_reject_events"] == 1
    assert m["by_merge_source"] == {"manual": 1, "recommended": 2}
    rec = m["recommender"]
    assert rec["accepted"] == 2 and rec["rejected"] == 1 and abs(rec["accept_rate"] - 2 / 3) < 1e-9
    assert len(rec["timeline"]) == 3 and rec["timeline"][-1][1] == 3  # 3 recommender decisions on a timeline


def test_write_report_emits_files(tmp_path):
    from chevron import paper_stats as ps
    _project(tmp_path)
    out, od = ps.write_report(tmp_path, tmp_path / "rep")
    for f in ("summary.json", "action_mix.csv", "actions_timeline.csv", "source_breakdown.csv",
              "merge_recommender_timeline.csv", "lineage.csv"):
        assert (od / f).exists(), f
    assert len(out["lineage"]) == 1 and out["lineage"][0]["metric"] == 41.3


def test_summarize_tolerates_empty_project(tmp_path):
    from chevron import paper_stats as ps
    out = ps.summarize(tmp_path)                                      # no logs at all
    assert out["actions"]["n_actions"] == 0 and out["merges"]["n_merge_events"] == 0 and out["lineage"] == []
    assert "sources" not in out                                      # no state.json -> skipped, no crash
