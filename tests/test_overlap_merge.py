"""In-image model-free merge suggestions: group instances by mask/box IoU >= threshold, accept as one undo step.
Run: pytest tests/test_overlap_merge.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np

from test_merge_rec import _engine, _rle


def _box(y0, y1, x0, x1, H=128, W=128):
    m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True
    return m


def _overlap_engine(tmp_path):
    """One image, 4 instances: A/B overlap heavily as masks (IoU ~0.77); C is a frame and D fills its hole
    exactly — box IoU ~0.75 but mask IoU 0."""
    eng, order = _engine(tmp_path, n_img=1, per_img=4)
    frame = _box(60, 120, 60, 120); frame[64:116, 64:116] = False
    masks = [_box(10, 40, 10, 40), _box(12, 42, 12, 42), frame, _box(64, 116, 64, 116)]
    for u, m in zip(order, masks):
        eng.collection["records"][eng.state.meta[u].row]["rle"] = _rle(m)
    return eng, order


def test_mask_iou_groups(tmp_path):
    eng, order = _overlap_engine(tmp_path)
    groups = eng.overlap_merge_groups(1000, 0.5, metric="mask")
    assert [set(g["iuids"]) for g in groups] == [{order[0], order[1]}]
    assert 0.7 < groups[0]["prob"] < 0.8


def test_box_iou_groups_catch_nested(tmp_path):
    eng, order = _overlap_engine(tmp_path)
    groups = eng.overlap_merge_groups(1000, 0.5, metric="box")
    assert sorted(map(frozenset, (g["iuids"] for g in groups)), key=min) == \
        sorted([frozenset(order[:2]), frozenset(order[2:])], key=min)


def test_threshold_filters(tmp_path):
    eng, _ = _overlap_engine(tmp_path)
    assert eng.overlap_merge_groups(1000, 0.9, metric="mask") == []
    assert eng.overlap_merge_groups(1000, 0.9, metric="box") == []


def test_accept_groups_single_undo_and_source(tmp_path):
    eng, order = _overlap_engine(tmp_path)
    groups = [g["iuids"] for g in eng.overlap_merge_groups(1000, 0.5, metric="box")]
    assert eng.accept_merge_groups(groups) == 2
    assert {eng.state.meta[u].merged_into is not None for u in order} == {True, False}
    evs = [e for e in eng.store.read_merge_events() if e.get("kind") == "merge"]
    assert len(evs) == 2 and {e["source"] for e in evs} == {"overlap"}
    assert eng.overlap_merge_groups(1000, 0.5, metric="box") == []   # merged children leave the image set
    eng.undo()
    assert all(eng.state.meta[u].merged_into is None for u in order)  # both groups revert in one step
