"""Two people, one engine: shared data, separate views (chevron.context).
Run: pytest tests/test_views.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from chevron.context import Ctx, StaleCluster, use


def _engine(tmp_path, n=60):
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    rng = np.random.default_rng(0)
    cents = rng.normal(0, 4, (6, 8))
    feats = np.array([cents[i % 6] + rng.normal(0, 0.3, 8) for i in range(n)], np.float32)
    eng.collection = {"records": [{"iuid": f"u{i}", "row": i, "score": 0.6, "batch_id": "b0" if i < n // 2 else "b1"}
                                  for i in range(n)],
                      "n_images": n, "feats": {"decoder": feats}}
    eng.state.order = [f"u{i}" for i in range(n)]
    eng.state.meta = {f"u{i}": InstanceMeta(f"u{i}", "b0" if i < n // 2 else "b1", i, 1000 + i) for i in range(n)}
    eng.state.coll_version = 1
    return eng


def test_filters_are_per_view(tmp_path):
    eng = _engine(tmp_path)
    ana, bo = Ctx("ana"), Ctx("bo")
    with use(ana):
        eng.set_kind_filter(granularity="sample")
        assert eng._granularity_filter == {"sample"}
    with use(bo):
        assert eng._granularity_filter is None
    assert eng._granularity_filter is None             # outside any request: the default view


def test_scope_tokens_never_collide_across_views(tmp_path):
    eng = _engine(tmp_path)
    ana, bo = Ctx("ana"), Ctx("bo")
    with use(ana):
        eng.set_kind_filter(granularity="sample")
        ta = eng._scope_token
    with use(bo):
        eng.set_kind_filter(granularity="instance")
        tb = eng._scope_token
    assert ta != tb and ta and tb


def test_level_is_per_view_but_clustering_is_shared(tmp_path):
    eng = _engine(tmp_path)
    ana, bo = Ctx("ana"), Ctx("bo")
    with use(ana):
        eng.cluster({"decoder": 1.0})
        n_levels = len(eng._cluster["counts"])
        assert n_levels >= 2
        eng.set_level(0)
        ana_parts = len(eng.partition_view())
    with use(bo):
        assert eng._cluster is not None                # bo sees the clustering ana made
        eng.set_level(n_levels - 1)
        bo_parts = len(eng.partition_view())
    with use(ana):
        assert eng._cluster["level"] == 0              # bo's choice did not move ana
        assert len(eng.partition_view()) == ana_parts
    assert ana_parts > bo_parts


def test_partition_members_follow_each_views_level(tmp_path):
    eng = _engine(tmp_path)
    ana, bo = Ctx("ana"), Ctx("bo")
    with use(ana):
        eng.cluster({"decoder": 1.0})
        eng.set_level(0)
        a0 = set(eng.partition_iuids("0"))
    with use(bo):
        eng.set_level(len(eng._cluster["counts"]) - 1)
        b0 = set(eng.partition_iuids("0"))
    with use(ana):
        assert set(eng.partition_iuids("0")) == a0
    assert a0 != b0


def test_stale_partition_id_is_refused(tmp_path):
    eng = _engine(tmp_path)
    ana = Ctx("ana")
    with use(ana):
        eng.cluster({"decoder": 1.0})
        rev = eng._cluster.token()
    seen = Ctx("ana", views=ana.views, cluster_rev=rev)
    with use(seen):
        assert eng.partition_iuids("0")                # same clustering: fine
    with use(Ctx("bo")):
        eng.cluster({"decoder": 1.0})                  # someone re-clusters
    with use(seen), pytest.raises(StaleCluster):
        eng.reject_partition("0")
    assert not any(m.is_background for m in eng.state.meta.values())


def test_class_pseudo_partitions_need_no_cluster_rev(tmp_path):
    eng = _engine(tmp_path)
    eng.assign(["u0", "u1"], "cat")
    cid = eng.state.class_id_by_name("cat")
    with use(Ctx("ana", cluster_rev="12345:0")):
        assert set(eng.partition_iuids(f"class:{cid}")) == {"u0", "u1"}


def test_labels_are_shared(tmp_path):
    eng = _engine(tmp_path)
    with use(Ctx("ana")):
        eng.assign(["u0"], "cat")
    with use(Ctx("bo")):
        assert eng.state.meta["u0"].assigned_class == eng.state.class_id_by_name("cat")


def test_a_level_change_makes_old_partition_ids_stale(tmp_path):
    """Two tabs of one browser share a view: one switching level renumbers the other's partitions."""
    eng = _engine(tmp_path)
    tab = Ctx("ana")
    with use(tab):
        eng.cluster({"decoder": 1.0})
        eng.set_level(0)
        tok = eng._cluster.token()
        eng.set_level(1)
    with use(Ctx("ana", views=tab.views, cluster_rev=tok)), pytest.raises(StaleCluster):
        eng.partition_iuids("0")
