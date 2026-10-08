"""Partitions 'Remove duplicates': per-image mask/box-IoU NMS over a rail scope, confirmed instances win.
Run: pytest tests/test_dedup_scope.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np

from test_overlap_merge import _overlap_engine


def _scores(eng, order, vals):
    for u, v in zip(order, vals):
        eng.collection["records"][eng.state.meta[u].row]["score"] = v


def test_pool_keeps_higher_score(tmp_path):
    eng, o = _overlap_engine(tmp_path)               # A/B mask IoU ~0.77; C frame + D its hole (box ~0.75, mask 0)
    _scores(eng, o, [0.6, 0.9, 0.5, 0.5])
    r = eng.scope_duplicates(o, 0.5, metric="mask")
    assert r["reject"] == [o[0]] and r["keep_of"] == {o[0]: o[1]} and r["n_images"] == 1


def test_box_metric_catches_frame_and_hole(tmp_path):
    eng, o = _overlap_engine(tmp_path)
    _scores(eng, o, [0.6, 0.9, 0.7, 0.5])
    assert set(eng.scope_duplicates(o, 0.5, metric="box")["reject"]) == {o[0], o[3]}
    assert o[3] not in eng.scope_duplicates(o, 0.5, metric="mask")["reject"]


def _review_mask(eng, u):
    """What the re-mask queue's pick / accept leaves behind."""
    eng._mask_cands[u] = {"backend": "x", "cands": [], "reviewed": True}


def test_reviewed_mask_wins_even_from_outside_the_scope(tmp_path):
    eng, o = _overlap_engine(tmp_path)
    _scores(eng, o, [0.3, 0.9, 0.5, 0.5])
    eng.assign([o[0]], "X", source="import"); eng.assign([o[1]], "Y", source="import")   # classes from a file
    _review_mask(eng, o[0])                          # a person accepted A's mask, despite its LOWER score
    scope = eng.partition_iuids("class:" + eng.state.meta[o[1]].assigned_class)
    assert scope == [o[1]]
    assert eng.scope_duplicates(scope, 0.5)["reject"] == [o[1]]
    assert eng.scope_duplicates([o[0]], 0.5)["reject"] == []          # a reviewed scope member never goes


def test_two_reviewed_copies_keep_one(tmp_path):
    # e.g. two boxes on the same object, each picked: still a duplicate — but an unreviewed mask never wins
    eng, o = _overlap_engine(tmp_path)
    _scores(eng, o, [0.6, 0.9, 0.5, 0.5])
    _review_mask(eng, o[0]); _review_mask(eng, o[1])
    assert eng.scope_duplicates(o, 0.5, metric="mask")["reject"] == [o[0]]
    assert eng.scope_duplicates([o[0]], 0.5, metric="mask")["reject"] == [o[0]]   # anchored from outside
    _review_mask(eng, o[1]); eng._mask_cands.pop(o[0])
    assert eng.scope_duplicates(o, 0.5, metric="mask")["reject"] == [o[0]]


def test_overlap_catches_a_fragment_and_keeps_the_larger(tmp_path):
    # a fragment inside the full object has a low IoU but is fully covered; equal scores -> the larger stays
    eng, o = _overlap_engine(tmp_path)
    m_full, m_frag = eng._mask(o[0]), eng._mask(o[0]).copy()
    ys, xs = np.nonzero(m_frag); m_frag[:, : int(np.median(xs))] = False; m_frag[: int(np.median(ys)), :] = False
    from pycocotools import mask as mu
    eng.collection["records"][eng.state.meta[o[1]].row]["rle"] = mu.encode(np.asfortranarray(m_frag.astype(np.uint8)))
    _scores(eng, o, [1.0, 1.0, 1.0, 1.0])
    assert o[1] not in eng.scope_duplicates(o[:2], 0.8, metric="mask")["reject"]
    assert eng.scope_duplicates(o[:2], 0.8, metric="overlap")["reject"] == [o[1]]


def test_mask_review_is_separate_from_class(tmp_path):
    # imported classes are trusted; their generated masks are not — until drawn or accepted
    eng, o = _overlap_engine(tmp_path)
    eng.assign(o, "X", source="import")
    assert not any(eng.mask_reviewed(u) for u in o)
    assert eng.stats()["n_mask_unreviewed"] == 4
    _review_mask(eng, o[0])
    m = eng.state.meta[o[1]]; m.refined = True; m.provenance = {"draw": {}}   # what set_mask leaves
    assert eng.mask_reviewed(o[0]) and eng.mask_reviewed(o[1]) and eng.stats()["n_mask_unreviewed"] == 2


def test_export_states_review_per_annotation(tmp_path):
    import json
    eng, o = _overlap_engine(tmp_path)
    eng.assign(o[:3], "X", source="import"); eng.assign(o[3:], "X", source="classifier")
    _review_mask(eng, o[0])
    for partial in (False, True):
        coco = json.loads(eng.export_coco(tmp_path / f"e{partial}.json", partial_labels=partial).read_text())
        by = {a["iuid"]: a for a in coco["annotations"]}
        assert by[o[0]]["mask_reviewed"] and by[o[0]]["class_reviewed"]
        assert not by[o[1]]["mask_reviewed"] and by[o[1]]["class_reviewed"]          # imported class is trusted
        assert not by[o[3]]["class_reviewed"]                                         # classifier is a guess
        assert coco["info"]["n_mask_unreviewed"] == 3 and coco["info"]["n_class_unreviewed"] == 1


def test_endpoint_preview_then_apply_is_one_undo(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, o = _overlap_engine(tmp_path)
    _scores(eng, o, [0.6, 0.9, 0.5, 0.5])
    c = TestClient(create_app(engine=eng))
    pv = c.post("/api/dedup_scope", json={"iuids": o, "thresh": 0.5}).json()
    assert pv["reject"] == [o[0]] and not eng.state.meta[o[0]].is_background
    ap = c.post("/api/dedup_scope", json={"iuids": o, "thresh": 0.5, "apply": True}).json()
    assert ap["n"] == 1 and eng.state.meta[o[0]].is_background
    eng.undo()
    assert not eng.state.meta[o[0]].is_background
    assert c.post("/api/dedup_scope", json={"iuids": o, "metric": "poly"}).status_code == 400
