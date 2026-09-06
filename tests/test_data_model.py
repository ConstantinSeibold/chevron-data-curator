"""P2 data model: item granularity/modality, project mode, capabilities, and the kind facet.

An item is either a mask INSTANCE within a source or a whole SAMPLE. Both live in the same
row-aligned feature matrices, so only mask-specific tools care — and they ask `capabilities`.

Run: pytest tests/test_data_model.py -q
"""
from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from chevron import ids
from chevron.engine import CuratorEngine
from chevron.server import create_app
from chevron.state import CuratorState, InstanceMeta


def _engine(tmp_path, config=None):
    eng = CuratorEngine(tmp_path)
    eng.init_project(config or {"model": {}})
    return eng


def _add(eng, n=4, *, granularity="instance", modality="image", image_id=1):
    order, out = list(eng.state.order), []
    for _ in range(n):
        u = ids.new_uid()
        eng.state.meta[u] = InstanceMeta(iuid=u, batch_id="b", row=len(order), image_id=image_id,
                                         granularity=granularity, modality=modality)
        order.append(u); out.append(u)
    eng.state.order = order
    return out


# --------------------------------------------------------------------------- model
def test_defaults_and_roundtrip():
    m = InstanceMeta(iuid="u", batch_id="b", row=0, image_id=7)
    assert (m.granularity, m.modality) == ("instance", "image")
    back = InstanceMeta.from_dict(m.to_dict())
    assert (back.granularity, back.modality) == ("instance", "image")

    s = InstanceMeta(iuid="u", batch_id="b", row=0, image_id=7, granularity="sample", modality="text")
    assert InstanceMeta.from_dict(s.to_dict()).modality == "text"


def test_old_state_json_loads_as_instance_image():
    """The whole point of defaulting: existing projects need NO migration."""
    old = {"iuid": "u", "batch_id": "b", "row": 3, "image_id": 9, "assigned_class": None,
           "is_background": False, "assign_source": None, "assign_score": None,
           "merged_into": None, "merge_members": [], "refined": False, "rule_ops": None,
           "provenance": {}}
    m = InstanceMeta.from_dict(old)
    assert (m.granularity, m.modality, m.row) == ("instance", "image", 3)


def test_from_dict_tolerates_unknown_keys():
    """Forward skew: a project written by a newer Chevron must still open, not TypeError."""
    d = {"iuid": "u", "batch_id": "b", "row": 0, "image_id": 1, "some_future_field": 42}
    assert InstanceMeta.from_dict(d).iuid == "u"


def test_state_mode_helpers_and_capabilities():
    st = CuratorState(project_dir="/tmp/x")
    assert st.mode() == "instance" and st.modality() == "image" and not st.is_sample_mode()
    assert st.primary_extractor() is None
    caps = st.capabilities()
    assert all(caps[k] for k in ("masks", "refine", "merge", "substructure", "coco_export"))

    st.config = {"mode": "sample", "modality": "text", "primary_extractor": "clip"}
    assert st.is_sample_mode() and st.primary_extractor() == "clip"
    assert not any(st.capabilities()[k] for k in ("masks", "refine", "merge", "coco_export"))


# --------------------------------------------------------------------------- stamping
def test_ingest_stamps_the_project_mode(tmp_path):
    eng = _engine(tmp_path, {"model": {}, "mode": "sample", "modality": "text"})
    batch = {"records": [{"iuid": ids.new_uid(), "row": 0, "image_id": 1, "score": 0.5,
                          "batch_id": "b", "abs_path": "/x.png"}],
             "n_images": 1, "feats": {"decoder": np.zeros((1, 4), np.float32)}}
    eng._fold_batch(batch)
    m = next(iter(eng.state.meta.values()))
    assert (m.granularity, m.modality) == ("sample", "text")


def test_split_children_inherit_kind_and_record_their_parent(tmp_path):
    """A split child is the same kind as its parent, and now records which instance it came from."""
    import cv2
    from pycocotools import mask as mu
    eng = _engine(tmp_path, {"model": {}})
    img = tmp_path / "im.png"
    cv2.imwrite(str(img), np.zeros((64, 64, 3), np.uint8))
    m = np.zeros((64, 64), np.uint8)
    cv2.circle(m, (16, 16), 6, 1, -1)
    cv2.circle(m, (48, 48), 6, 1, -1)                       # two components -> splittable
    rle = mu.encode(np.asfortranarray(m)); rle["counts"] = rle["counts"].decode("ascii")
    u = ids.new_uid()
    eng.collection = {"records": [{"iuid": u, "row": 0, "image_id": 1, "H": 64, "W": 64, "score": 0.9,
                                   "rle": rle, "file_name": str(img), "abs_path": str(img), "batch_id": "b"}],
                      "n_images": 1, "feats": {"decoder": np.zeros((1, 4), np.float32)}}
    eng.state.order = [u]
    eng.state.meta = {u: InstanceMeta(iuid=u, batch_id="b", row=0, image_id=1)}
    eng.state.coll_version = 1

    assert eng.split_instances([u]) == 2
    children = [mm for iu, mm in eng.state.meta.items() if iu != u]
    assert all(c.granularity == "instance" and c.modality == "image" for c in children)
    assert all(c.provenance.get("split_from") == u for c in children)


# --------------------------------------------------------------------------- facet
def test_kind_facet_filters_every_view_through_in_scope(tmp_path):
    eng = _engine(tmp_path)
    inst = _add(eng, 3, granularity="instance", modality="image")
    samp = _add(eng, 2, granularity="sample", modality="text")

    assert all(eng._in_scope(u) for u in inst + samp)         # open by default

    eng.set_kind_filter(granularity="instance")
    assert all(eng._in_scope(u) for u in inst)
    assert not any(eng._in_scope(u) for u in samp)

    eng.set_kind_filter(modality="text")                      # granularity cleared, modality set
    assert not any(eng._in_scope(u) for u in inst)
    assert all(eng._in_scope(u) for u in samp)

    eng.set_kind_filter()                                     # clears both
    assert all(eng._in_scope(u) for u in inst + samp)


def test_kinds_reports_counts_and_a_single_mode_project_has_one_row(tmp_path):
    eng = _engine(tmp_path)
    _add(eng, 3)
    k = eng.kinds()
    assert k["kinds"] == [{"granularity": "instance", "modality": "image", "n": 3}]
    assert k["mode"] == "instance" and k["capabilities"]["refine"] is True

    _add(eng, 2, granularity="sample", modality="text")
    assert len(eng.kinds()["kinds"]) == 2


def test_kind_filter_busts_the_view_caches(tmp_path):
    """Same contract as the source facet: the index/projection caches must not survive a facet change."""
    eng = _engine(tmp_path)
    _add(eng, 2)
    before = eng._scope_token
    eng._index = object()
    eng.set_kind_filter(granularity="sample")
    assert eng._scope_token > before and eng._index is None


def test_open_and_reset_clear_the_facet(tmp_path):
    eng = _engine(tmp_path)
    _add(eng, 2)
    eng.save()
    eng.set_kind_filter(granularity="sample")
    eng.open()
    assert eng._granularity_filter is None and eng._modality_filter is None


# --------------------------------------------------------------------------- server
def test_state_exposes_mode_and_capabilities(tmp_path):
    eng = _engine(tmp_path, {"model": {}, "mode": "sample", "modality": "text",
                             "primary_extractor": "siglip2"})
    c = TestClient(create_app(engine=eng))
    st = c.get("/api/state").json()
    assert st["mode"] == "sample" and st["modality"] == "text"
    assert st["primary_extractor"] == "siglip2"
    assert st["capabilities"]["refine"] is False and st["capabilities"]["masks"] is False


def test_kind_endpoints(tmp_path):
    eng = _engine(tmp_path)
    _add(eng, 2)
    _add(eng, 1, granularity="sample", modality="text")
    c = TestClient(create_app(engine=eng))
    assert len(c.get("/api/kinds").json()["kinds"]) == 2
    r = c.post("/api/kind_filter", json={"granularity": "sample"}).json()
    assert r["active_granularity"] == ["sample"]
    assert c.post("/api/kind_filter", json={}).json()["active_granularity"] is None


def test_sample_mode_refuses_coco_export(tmp_path):
    eng = _engine(tmp_path, {"model": {}, "mode": "sample", "modality": "text"})
    c = TestClient(create_app(engine=eng))
    r = c.post("/api/export", json={})
    assert r.status_code == 400 and "sample mode" in r.json()["detail"]


def test_instance_mode_still_exports(tmp_path):
    """The guard must not touch the normal path."""
    eng = _engine(tmp_path)
    _add(eng, 1)
    eng.collection = {"records": [{"iuid": eng.state.order[0], "row": 0, "image_id": 1, "H": 4, "W": 4,
                                   "score": 0.5, "file_name": "a.png", "abs_path": "a.png",
                                   "rle": {"size": [4, 4], "counts": "0"}}],
                      "n_images": 1, "feats": {"decoder": np.zeros((1, 2), np.float32)}}
    c = TestClient(create_app(engine=eng))
    assert c.post("/api/export", json={}).status_code == 200


def test_project_summary_reads_mode_from_config(tmp_path):
    """P1's launcher cards show mode/modality; P2 is what makes them non-default."""
    from chevron.projects import ProjectRegistry
    reg = ProjectRegistry(tmp_path / "root")
    reg.create("Texty", {"mode": "sample", "modality": "text"})
    s = reg.summarize("texty")
    assert (s.mode, s.modality) == ("sample", "text")
    assert json.loads((reg.path_for("texty") / "state.json").read_text())["config"]["mode"] == "sample"
