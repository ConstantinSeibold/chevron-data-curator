"""Box-guided re-masking — new masks for boxes you already have, and a COCO patched with them.

Two kinds of backend fill a box. A promptable one (SAM / SAM-HQ / MedSAM) is handed the boxes as
prompts; any other one runs on the whole image and each box keeps the prediction that best matches
it. Both are exercised with stubs, so the bookkeeping is tested without model weights.

Run: pytest tests/test_boxguide.py -q
"""
from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient
from pycocotools import mask as mu

from chevron.backends import base as B
from chevron.backends.boxguide import propose_in_boxes
from chevron.engine import CuratorEngine
from chevron.server import create_app
from test_backends import _coco, _images

S = 64


def _rect(x0, y0, x1, y1):
    m = np.zeros((S, S), bool)
    m[y0:y1, x0:x1] = True
    return m


class _Detector:
    """Whole-image predictions: one per COCO box (shrunk, and the second one spilling out of its box),
    plus an image-sized blob that must match nothing."""
    name, label, requires = "stub_det", "Stub detector", "nothing"

    def available(self):
        return True, "ok"

    def propose(self, image_rgb, path=None, **cfg):
        spill = _rect(28, 10, 38, 24)
        spill[10:12, 38:60] = True                        # a sliver far outside the box's wiggle room
        return [B.Proposal(_rect(8, 10, 18, 24), 0.8), B.Proposal(spill, 0.7),
                B.Proposal(np.ones((S, S), bool), 0.99)]


class _Prompter:
    """Box prompts: returns the box shrunk by 2 px. `propose` must never be called."""
    name, label, requires = "stub_prompt", "Stub prompter", "nothing"

    def __init__(self):
        self.calls = []

    def available(self):
        return True, "ok"

    def propose(self, image_rgb, path=None, **cfg):
        raise AssertionError("a promptable backend must be prompted, not run on the whole image")

    def propose_boxes(self, image_rgb, boxes, **cfg):
        self.calls.append(len(boxes))
        return [B.Proposal(_rect(int(b[0]) + 2, int(b[1]) + 2, int(b[2]) - 2, int(b[3]) - 2), 0.77)
                for b in boxes]


def _register(*backends):
    for b in backends:
        B.register(b.name, lambda b=b: b)


def _unregister(*backends):
    for b in backends:
        B._REGISTRY.pop(b.name, None)


# --------------------------------------------------------------------------- matching
BOXES = [[6, 8, 20, 26], [26, 8, 40, 26]]                 # the two boxes `_coco` writes, as xyxy


def test_each_box_keeps_its_own_best_prediction_clipped_to_the_box():
    img = np.zeros((S, S, 3), np.uint8)
    got = propose_in_boxes(_Detector(), img, BOXES, pad=0.1)
    assert got[0] is not None and got[0].mask.sum() == _rect(8, 10, 18, 24).sum()
    assert got[1] is not None and got[1].score == 0.7
    assert not got[1].mask[:, 45:].any()                  # the sliver outside the padded box is cut


def test_one_prediction_cannot_serve_two_boxes():
    class _One(_Detector):
        def propose(self, image_rgb, path=None, **cfg):
            return [B.Proposal(_rect(7, 9, 19, 25), 0.9)]
    got = propose_in_boxes(_One(), np.zeros((S, S, 3), np.uint8), [BOXES[0], BOXES[0]])
    assert sum(g is not None for g in got) == 1


def test_a_box_nothing_matches_gets_none():
    class _Far(_Detector):
        def propose(self, image_rgb, path=None, **cfg):
            return [B.Proposal(_rect(50, 50, 60, 60), 0.9)]
    assert propose_in_boxes(_Far(), np.zeros((S, S, 3), np.uint8), BOXES) == [None, None]


def test_a_promptable_backend_is_prompted_in_box_order():
    pr = _Prompter()
    got = propose_in_boxes(pr, np.zeros((S, S, 3), np.uint8), BOXES)
    assert pr.calls == [2]
    assert [g.mask.sum() for g in got] == [10 * 14, 10 * 14]
    assert got[0].mask[12, 10] and not got[0].mask[12, 30]


# --------------------------------------------------------------------------- new project
def test_coco_import_remasked_by_a_prompter_keeps_every_annotation(tmp_path):
    paths = _images(tmp_path / "img")
    cj = _coco(paths, tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    pr = _Prompter(); _register(pr)
    try:
        res = eng.propose_instances("coco", coco_path=str(cj), remask_with="stub_prompt")
    finally:
        _unregister(pr)
    assert res.get("ok"), res
    recs = eng.collection["records"]
    assert len(recs) == 6                                 # overlapping boxes too: no NMS on this path
    assert sorted(r["src_ann_id"] for r in recs) == [1, 2, 3, 4, 5, 6]
    assert {r["remask"] for r in recs} == {"stub_prompt"}
    assert all(mu.area(r["rle"]) == 10 * 14 for r in recs)
    eng.close()


def test_coco_import_keeps_the_files_mask_where_nothing_matched(tmp_path):
    paths = _images(tmp_path / "img", n=1)
    cj = _coco(paths, tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})

    class _Far(_Detector):
        name = "stub_far"

        def propose(self, image_rgb, path=None, **cfg):
            return [B.Proposal(_rect(50, 50, 60, 60), 0.9)]
    far = _Far(); _register(far)
    try:
        eng.propose_instances("coco", coco_path=str(cj), remask_with="stub_far")
    finally:
        _unregister(far)
    recs = eng.collection["records"]
    assert {r["remask"] for r in recs} == {"kept"}
    assert all(mu.area(r["rle"]) > 200 for r in recs)     # the original 14x18 polygons
    eng.close()


def test_plain_coco_import_still_records_its_source_annotations(tmp_path):
    paths = _images(tmp_path / "img", n=1)
    cj = _coco(paths, tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    assert {r.get("src_ann_id") for r in eng.collection["records"]} == {1, 2}
    assert "remask" not in eng.collection["records"][0]
    eng.close()


# --------------------------------------------------------------------------- existing project
def test_remask_existing_instances_is_one_undo_step(tmp_path):
    paths = _images(tmp_path / "img", n=2)
    cj = _coco(paths, tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    before = {u: int(eng._mask(u).sum()) for u in eng.state.order}
    pr = _Prompter(); _register(pr)
    try:
        res = eng.remask_instances("stub_prompt")
    finally:
        _unregister(pr)
    assert {k: res[k] for k in ("n_remasked", "n_kept", "n_images")} == \
           {"n_remasked": 4, "n_kept": 0, "n_images": 2}
    assert pr.calls == [2] * 6                            # per image: all its boxes at once, x3 box sizes
    assert all(int(eng._mask(u).sum()) < before[u] for u in eng.state.order)
    eng.undo()
    assert {u: int(eng._mask(u).sum()) for u in eng.state.order} == before
    eng.close()


def test_api_remask_rejects_an_unavailable_backend(tmp_path):
    paths = _images(tmp_path / "img", n=1)
    cj = _coco(paths, tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))

    class _Off(_Prompter):
        name = "stub_off"

        def available(self):
            return False, "not here"
    off = _Off(); _register(off)
    try:
        c = TestClient(create_app(engine=eng))
        assert c.post("/api/remask", json={"backend": "stub_off"}).status_code == 400
        assert c.post("/api/remask", json={}).status_code == 400
    finally:
        _unregister(off)
    eng.close()


# --------------------------------------------------------------------------- patched export
def _gt(cj):
    return json.loads(cj.read_text())


def test_patched_export_changes_only_the_masks(tmp_path):
    paths = _images(tmp_path / "img", n=2)
    cj = _coco(paths, tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    pr = _Prompter(); _register(pr)
    try:
        eng.propose_instances("coco", coco_path=str(cj), remask_with="stub_prompt")
    finally:
        _unregister(pr)
    res = eng.export_patched_coco()
    assert res["ok"] and res["n_patched"] == 4 and res["n_unchanged"] == 0
    src, out = _gt(cj), json.loads(open(res["path"]).read())
    assert out["images"] == src["images"] and out["categories"] == src["categories"]
    for a, b in zip(src["annotations"], out["annotations"]):
        assert {k: v for k, v in a.items() if k not in ("segmentation", "area", "bbox")} == \
               {k: v for k, v in b.items() if k not in ("segmentation", "area", "bbox")}
        assert b["bbox"] == [a["bbox"][0] + 2, a["bbox"][1] + 2, 10.0, 14.0]   # recomputed
        assert b["area"] == 140.0
        assert isinstance(b["segmentation"], list)        # polygons in, polygons out
    eng.close()


def test_patched_export_leaves_untouched_masks_alone(tmp_path):
    paths = _images(tmp_path / "img", n=2)
    cj = _coco(paths, tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    u0 = eng.state.order[0]
    pr = _Prompter(); _register(pr)
    try:
        eng.remask_instances("stub_prompt", iuids=[u0])
    finally:
        _unregister(pr)
    res = eng.export_patched_coco()
    assert res["n_patched"] == 1 and res["n_unchanged"] == 3
    src, out = _gt(cj), json.loads(open(res["path"]).read())
    changed = [b for a, b in zip(src["annotations"], out["annotations"]) if a != b]
    assert len(changed) == 1
    assert changed[0]["id"] == eng.collection["records"][eng.state.meta[u0].row]["src_ann_id"]
    eng.close()


def test_patched_export_matches_by_overlap_when_the_annotation_id_is_unknown(tmp_path):
    """Projects ingested before annotations were tracked have no `src_ann_id` on their records."""
    paths = _images(tmp_path / "img", n=1)
    cj = _coco(paths, tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    for r in eng.collection["records"]:
        r.pop("src_ann_id"); r.pop("src_coco")
    pr = _Prompter(); _register(pr)
    try:
        eng.remask_instances("stub_prompt")
    finally:
        _unregister(pr)
    assert eng.export_patched_coco()["error"]              # no recorded source: must be told which
    res = eng.export_patched_coco(str(cj))
    assert res["n_patched"] == 2 and res["n_unmatched"] == 0
    out = json.loads(open(res["path"]).read())
    assert [a["bbox"][0] for a in out["annotations"]] == [8.0, 28.0]
    eng.close()


def test_patched_export_leaves_held_images_alone(tmp_path):
    paths = _images(tmp_path / "img", n=2)
    cj = _coco(paths, tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    pr = _Prompter(); _register(pr)
    try:
        eng.propose_instances("coco", coco_path=str(cj), remask_with="stub_prompt")
    finally:
        _unregister(pr)
    eng.set_release_policy("accepted_only")               # nothing is signed off -> all held
    res = eng.export_patched_coco()
    assert res["n_patched"] == 0 and res["held_back"] == 2
    eng.close()


def test_api_export_patched(tmp_path):
    paths = _images(tmp_path / "img", n=1)
    cj = _coco(paths, tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    c = TestClient(create_app(engine=eng))
    pr = _Prompter(); _register(pr)
    try:
        r = c.post("/api/propose", json={"backend": "coco", "coco_path": str(cj),
                                         "remask_with": "stub_prompt", "box_pad": 0.2})
    finally:
        _unregister(pr)
    assert r.status_code == 200, r.text
    r = c.post("/api/export_patched", json={})
    assert r.status_code == 200 and r.json()["n_patched"] == 2
    assert c.post("/api/export_patched", json={"coco_path": str(tmp_path / "nope.json")}).status_code == 400
    eng.close()


# --------------------------------------------------------------------------- SAM box prompts
class _FakePredictor:
    """Stands in for `SamPredictor`: three candidate masks per box with known scores."""

    def __init__(self):
        self.images, self.calls = [], []

    def set_image(self, img):
        self.images.append(img)

    def predict(self, box=None, multimask_output=True):
        self.calls.append(multimask_output)
        x0, y0, x1, y1 = (int(v) for v in box)
        full = _rect(x0, y0, x1, y1)
        cands = [full, _rect(x0, y0, x0 + 2, y0 + 2), _rect(x0, y0, x0 + 4, y0 + 4)]
        if not multimask_output:
            return np.stack(cands[:1]), np.array([0.5]), None
        return np.stack(cands), np.array([0.2, 0.9, 0.4]), None


def test_sam_boxes_embeds_once_and_keeps_the_best_scoring_mask(monkeypatch):
    from chevron import refine as rf
    fp = _FakePredictor()
    monkeypatch.setattr(rf, "_resolve_predictor", lambda ckpt=None, mt=None, model="auto": (fp, model))
    out = rf.sam_boxes(np.zeros((S, S, 3), np.uint8), BOXES, family="sam")
    assert len(fp.images) == 1 and fp.calls == [True, True]
    assert [int(m.sum()) for m, _ in out] == [4, 4] and [s for _, s in out] == [0.9, 0.9]


def test_medsam_gets_one_mask_on_a_stretched_image(monkeypatch):
    from chevron import refine as rf
    fp = _FakePredictor()
    monkeypatch.setattr(rf, "_resolve_predictor", lambda ckpt=None, mt=None, model="auto": (fp, model))
    img = np.full((S, S, 3), 100, np.uint8); img[0, 0] = 120
    out = rf.sam_boxes(img, BOXES[:1], family="medsam")
    assert fp.calls == [False] and fp.images[0].max() == 255 and fp.images[0].min() == 0
    assert int(out[0][0].sum()) == 14 * 18


def test_a_model_package_missing_a_dependency_is_a_400_not_a_500(tmp_path):
    """segment-anything-hq imports timm without declaring it; that must reach the user as a fix."""
    paths = _images(tmp_path / "img", n=1)
    cj = _coco(paths, tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))

    class _NoTimm(_Prompter):
        name = "stub_notimm"

        def prepare(self, **kw):
            raise ModuleNotFoundError("No module named 'timm'", name="timm")
    nt = _NoTimm(); _register(nt)
    try:
        r = TestClient(create_app(engine=eng)).post("/api/remask", json={"backend": "stub_notimm"})
    finally:
        _unregister(nt)
    assert r.status_code == 400 and "pip install timm" in r.json()["detail"]
    eng.close()


def test_samhq_needs_timm_to_count_as_installed(monkeypatch):
    import importlib.util

    from chevron import refine as rf
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a: None if n == "timm" else real(n, *a))
    assert rf.samhq_available() is False


# --------------------------------------------------------------------------- COCO categories as labels
def _two_category_coco(tmp_path, n=2):
    cj = _coco(_images(tmp_path / "img", n=n), tmp_path / "gt.json")
    gt = json.loads(cj.read_text())
    gt["categories"] = [{"id": 7, "name": "coin"}, {"id": 9, "name": "clip"}]
    for a in gt["annotations"]:
        a["category_id"] = 7 if a["id"] % 2 else 9
    cj.write_text(json.dumps(gt))
    return cj


def _labels(eng):
    recs = eng.collection["records"]
    return {recs[m.row]["src_ann_id"]: eng.state.class_name(m.assigned_class)
            for m in eng.state.meta.values()}


def test_coco_categories_become_assignments_when_asked(tmp_path):
    cj = _two_category_coco(tmp_path)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    res = eng.propose_instances("coco", coco_path=str(cj), assign_categories=True)
    assert res["n_assigned"] == 4
    assert _labels(eng) == {1: "coin", 2: "clip", 3: "coin", 4: "clip"}
    assert {m.assign_source for m in eng.state.meta.values()} == {"import"}
    pinned = {c.name: c.coco_cat_id for c in eng.state.taxonomy.values()}
    assert pinned == {"coin": 7, "clip": 9}                # an export writes the same ids back
    eng.undo()                                            # one step takes the labels back off
    assert set(_labels(eng).values()) == {None}
    eng.close()


def test_coco_categories_are_ignored_unless_asked(tmp_path):
    cj = _two_category_coco(tmp_path, n=1)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    res = eng.propose_instances("coco", coco_path=str(cj))
    assert res["n_assigned"] == 0 and set(_labels(eng).values()) == {None}
    eng.close()


def test_categories_and_remasking_combine(tmp_path):
    cj = _two_category_coco(tmp_path, n=1)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    pr = _Prompter(); _register(pr)
    try:
        r = TestClient(create_app(engine=eng)).post(
            "/api/propose", json={"backend": "coco", "coco_path": str(cj),
                                  "remask_with": "stub_prompt", "assign_categories": True})
    finally:
        _unregister(pr)
    assert r.status_code == 200 and r.json()["n_assigned"] == 2
    assert _labels(eng) == {1: "coin", 2: "clip"}
    eng.close()


def test_an_existing_class_is_reused_and_its_coco_id_left_alone(tmp_path):
    cj = _two_category_coco(tmp_path, n=1)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    cid = eng.state.add_class("coin"); eng.state.taxonomy[cid].coco_cat_id = 3
    eng.propose_instances("coco", coco_path=str(cj), assign_categories=True)
    assert eng.state.class_id_by_name("coin") == cid and eng.state.taxonomy[cid].coco_cat_id == 3
    eng.close()


# --------------------------------------------------------------------------- categories on an existing project
def test_existing_instances_take_the_files_current_categories(tmp_path):
    cj = _two_category_coco(tmp_path)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    gt = json.loads(cj.read_text())
    gt["annotations"][0]["category_id"] = 9               # edited after ingest: the file wins
    cj.write_text(json.dumps(gt))
    res = eng.assign_coco_categories()                    # path defaults to the recorded source
    assert res["n_assigned"] == 4 and res["n_unmatched"] == 0
    assert _labels(eng) == {1: "clip", 2: "clip", 3: "coin", 4: "clip"}
    eng.undo()
    assert set(_labels(eng).values()) == {None}
    eng.close()


def test_existing_labels_are_kept_unless_overwrite(tmp_path):
    cj = _two_category_coco(tmp_path, n=1)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    u1, u2 = eng.state.order
    eng.assign([u1], "mine"); eng.set_background([u2])
    assert eng.assign_coco_categories()["n_assigned"] == 0
    assert eng.assign_coco_categories(overwrite=True)["n_assigned"] == 2
    assert not eng.state.meta[u2].is_background
    eng.close()


def test_untracked_instances_are_matched_by_image_and_overlap(tmp_path):
    """A project ingested before annotations were tracked (or from another model) has no ann ids."""
    cj = _two_category_coco(tmp_path, n=1)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    for r in eng.collection["records"]:
        r.pop("src_ann_id"); r.pop("src_coco"); r.pop("src_category_id")
    assert eng.assign_coco_categories()["error"]           # nothing recorded: must be told which file
    res = eng.assign_coco_categories(str(cj))
    assert res["n_assigned"] == 2
    names = sorted(eng.state.class_name(m.assigned_class) for m in eng.state.meta.values())
    assert names == ["clip", "coin"]
    eng.close()


def test_api_assign_coco_categories(tmp_path):
    cj = _two_category_coco(tmp_path, n=1)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    c = TestClient(create_app(engine=eng))
    r = c.post("/api/assign_coco_categories", json={})
    assert r.status_code == 200 and r.json()["n_assigned"] == 2 and "coin" in r.json()["classes"]
    assert c.post("/api/assign_coco_categories",
                  json={"coco_path": str(tmp_path / "nope.json")}).status_code == 400
    eng.close()



# --------------------------------------------------------------------------- several candidates, reviewed
from chevron.backends.boxguide import candidates_in_boxes  # noqa: E402


class _MultiPrompter(_Prompter):
    """Like SAM: three masks per box prompt, with scores that rank them."""
    name = "stub_multi"

    def propose_boxes_multi(self, image_rgb, boxes, **cfg):
        self.calls.append(len(boxes))
        out = []
        for b in boxes:
            x0, y0, x1, y1 = (int(v) for v in b)
            out.append([B.Proposal(_rect(x0 + 2, y0 + 2, x1 - 2, y1 - 2), 0.9),
                        B.Proposal(_rect(x0 + 2, y0 + 2, x1 - 2, y1 - 2), 0.85),   # near-copy
                        B.Proposal(_rect(x0 + 2, y0 + 2, x0 + 6, y0 + 6), 0.5)])
        return out


def test_candidates_are_distinct_best_first_and_capped():
    pr = _MultiPrompter()
    got = candidates_in_boxes(pr, np.zeros((S, S, 3), np.uint8), BOXES, pad=0.3, k=3)
    assert pr.calls == [2, 2, 2]                          # tight, pad, 2*pad boxes
    for cs in got:
        assert len(cs) == 3
        assert [c.score for c in cs] == sorted((c.score for c in cs), reverse=True)
        for a in range(3):
            for b in range(a + 1, 3):
                inter = (cs[a].mask & cs[b].mask).sum(); uni = (cs[a].mask | cs[b].mask).sum()
                assert inter / uni < 0.9


def test_detector_candidates_start_with_the_one_to_one_pick():
    class _Two(_Detector):
        def propose(self, image_rgb, path=None, **cfg):
            return [B.Proposal(_rect(8, 10, 18, 24), 0.8), B.Proposal(_rect(6, 8, 20, 20), 0.6),
                    B.Proposal(_rect(50, 50, 60, 60), 0.99)]
    got = candidates_in_boxes(_Two(), np.zeros((S, S, 3), np.uint8), BOXES[:1])
    # ranked by how well each matches the box, not by the model's own score; the far blob never shows
    assert [c.score for c in got[0]] == [0.6, 0.8]


def _remasked(tmp_path, n=1):
    cj = _coco(_images(tmp_path / "img", n=n), tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    orig = {u: int(eng._mask(u).sum()) for u in eng.state.order}
    pr = _MultiPrompter(); _register(pr)
    try:
        res = eng.remask_instances("stub_multi")
    finally:
        _unregister(pr)
    return eng, orig, res


def test_remask_applies_the_best_and_queues_the_rest(tmp_path):
    eng, orig, res = _remasked(tmp_path)
    assert res["n_remasked"] == 2 and res["n_candidates"] >= 4
    q = eng.mask_candidates()
    assert q["total"] == 2 and q["n_unreviewed"] == 2
    it = q["items"][0]
    assert it["current"] == 1 and it["n"] >= 2 and not it["reviewed"]
    assert int(eng._mask(it["iuid"]).sum()) == 10 * 14    # the best candidate is live
    thumbs = eng.candidate_crops(it["iuid"])
    assert len(thumbs) == it["n"] + 1 and len({t.shape for t in thumbs}) == 1   # same framing
    eng.close()


def test_picking_a_candidate_or_the_original_is_undoable(tmp_path):
    eng, orig, _ = _remasked(tmp_path)
    it = eng.mask_candidates()["items"][0]
    u, last = it["iuid"], it["n"]                         # the lowest-scoring: the 4x4 sliver
    assert eng.pick_mask_candidate(u, last)["current"] == last
    assert int(eng._mask(u).sum()) == 16
    assert eng.pick_mask_candidate(u, 0)["current"] == 0
    assert int(eng._mask(u).sum()) == orig[u]
    eng.undo()
    assert eng.mask_candidates()["items"][0]["current"] == last
    q = eng.mask_candidates(only_unreviewed=True)
    assert q["n_unreviewed"] == 1 and [i["iuid"] for i in q["items"]] != [u]
    assert eng.pick_mask_candidate(u, 9)["error"]
    eng.close()


def test_keeping_the_current_mask_just_marks_it_reviewed(tmp_path):
    eng, orig, _ = _remasked(tmp_path)
    u = eng.state.order[0]
    before = int(eng._mask(u).sum())
    eng.pick_mask_candidate(u, None)
    assert int(eng._mask(u).sum()) == before and eng.mask_candidates()["n_unreviewed"] == 1
    eng.close()


def test_candidates_survive_a_reopen_and_a_rerun_keeps_the_first_original(tmp_path):
    eng, orig, _ = _remasked(tmp_path)
    u = eng.state.order[0]
    pr = _MultiPrompter(); _register(pr)
    try:
        eng.remask_instances("stub_multi", iuids=[u])     # prompted from the re-masked box this time
    finally:
        _unregister(pr)
    eng.save(); eng.close()
    eng2 = CuratorEngine(tmp_path / "proj"); eng2.open()
    assert eng2.mask_candidates()["total"] == 2
    eng2.pick_mask_candidate(u, 0)
    assert int(eng2._mask(u).sum()) == orig[u]
    eng2.close()


def test_picking_back_the_original_is_not_a_change_for_the_patched_export(tmp_path):
    eng, orig, _ = _remasked(tmp_path)
    for u in eng.state.order:
        eng.pick_mask_candidate(u, 0)
    assert eng.export_patched_coco()["n_patched"] == 0
    eng.close()


def test_api_candidate_review(tmp_path):
    eng, orig, _ = _remasked(tmp_path)
    c = TestClient(create_app(engine=eng))
    q = c.get("/api/mask_candidates", params={"only_unreviewed": 1}).json()
    u = q["items"][0]["iuid"]
    v = c.post("/api/mask_candidates/view", json={"iuid": u}).json()
    assert len(v["thumbs"]) == q["items"][0]["n"] + 1 and v["thumbs"][0].startswith("data:image/png")
    assert v["current"] == 1 and v["scores"][0] is None
    r = c.post("/api/mask_candidates/pick", json={"iuid": u, "index": 0})
    assert r.status_code == 200 and r.json()["current"] == 0
    assert c.post("/api/mask_candidates/pick", json={"iuid": "nope", "index": 0}).status_code == 400
    assert c.get("/api/mask_candidates", params={"only_unreviewed": 1}).json()["n_unreviewed"] == 1
    eng.close()


def test_undoing_a_mask_edit_restores_the_previous_mask_across_a_reopen(tmp_path):
    """History used to record only THAT an instance was refined, so undoing the second of two mask
    edits kept the second mask — and a reopen read it back from the overlay file."""
    eng, orig, _ = _remasked(tmp_path)
    it = eng.mask_candidates()["items"][0]
    u, live = it["iuid"], int(eng._mask(it["iuid"]).sum())
    eng.pick_mask_candidate(u, it["n"])
    eng.undo()
    assert int(eng._mask(u).sum()) == live
    eng.redo()
    assert int(eng._mask(u).sum()) == 16
    eng.undo(); eng.save(); eng.close()
    eng2 = CuratorEngine(tmp_path / "proj"); eng2.open()
    assert int(eng2._mask(u).sum()) == live
    eng2.close()


# --------------------------------------------------------------------------- box-only annotations
def _box_only_coco(tmp_path, n=1):
    cj = _coco(_images(tmp_path / "img", n=n), tmp_path / "gt.json")
    gt = json.loads(cj.read_text())
    gt["annotations"][0].pop("segmentation")             # first: bbox only
    gt["annotations"][1]["segmentation"] = [[26, 8, 40, 8, 33, 26]]   # second: a real (triangle) mask
    cj.write_text(json.dumps(gt))
    return cj


class _Nothing(_Prompter):
    name = "stub_nothing"

    def propose_boxes(self, image_rgb, boxes, **cfg):
        return [None] * len(boxes)


def test_a_box_only_annotation_nothing_matched_is_flagged_not_passed_off(tmp_path):
    cj = _box_only_coco(tmp_path)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    nb = _Nothing(); _register(nb)
    try:
        res = eng.propose_instances("coco", coco_path=str(cj), remask_with="stub_nothing")
    finally:
        _unregister(nb)
    by_ann = {r["src_ann_id"]: r for r in eng.collection["records"]}
    assert by_ann[1]["remask"] == "none_found" and by_ann[1]["box_only"]
    assert by_ann[2]["remask"] == "kept" and not by_ann[2]["box_only"]
    assert res["n_box_only"] == 1
    eng.close()


def test_a_found_mask_clears_the_box_only_flag(tmp_path):
    cj = _box_only_coco(tmp_path)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    pr = _Prompter(); _register(pr)
    try:
        res = eng.propose_instances("coco", coco_path=str(cj), remask_with="stub_prompt")
    finally:
        _unregister(pr)
    assert res["n_box_only"] == 0
    assert not any(r["box_only"] for r in eng.collection["records"])
    eng.close()


def test_only_box_remasks_just_the_box_shaped_and_labels_the_original(tmp_path):
    cj = _box_only_coco(tmp_path)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    assert eng.n_box_only() == 1
    pr = _Prompter(); _register(pr)
    try:
        res = eng.remask_instances("stub_prompt", only_box=True)
        assert res["n_remasked"] == 1 and res["n_box_only"] == 0
        assert eng.remask_instances("stub_prompt", only_box=True)["error"]
    finally:
        _unregister(pr)
    u = eng.mask_candidates()["items"][0]["iuid"]
    assert eng._mask_cands[u]["original_box_only"]
    eng.pick_mask_candidate(u, 0)                         # back to the box: it counts as box-only again
    assert eng.n_box_only() == 1
    v = TestClient(create_app(engine=eng)).post("/api/mask_candidates/view", json={"iuid": u}).json()
    assert v["original_box_only"] is True
    eng.close()


def test_older_projects_are_judged_by_shape(tmp_path):
    cj = _box_only_coco(tmp_path)
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    for r in eng.collection["records"]:
        r.pop("box_only")
    assert eng.n_box_only() == 1                          # the filled box, not the triangle
    eng.close()


def test_refine_starts_from_the_remasked_mask_not_the_box_it_replaced(tmp_path):
    eng, orig, _res = _remasked(tmp_path)
    u = eng.state.order[0]
    remasked = eng._mask(u).copy()
    assert int(remasked.sum()) < orig[u]
    assert np.array_equal(mu.decode(eng._refine_base_rle(u)).astype(bool), remasked)
    eng.apply_refine(u, [{"name": "erode", "kw": {"k": 3}}])
    eroded = eng._mask(u).copy()
    assert eroded.sum() and not (eroded & ~remasked).any()   # eroded the re-mask, not the box
    eng.apply_refine(u, [{"name": "erode", "kw": {"k": 3}}])
    assert np.array_equal(eng._mask(u), eroded)              # chains do not compound
    eng.close()
    eng2 = CuratorEngine(tmp_path / "proj"); eng2.open()
    assert np.array_equal(mu.decode(eng2._refine_base_rle(u)).astype(bool), remasked)
    eng2.revert_refine(u)
    assert int(eng2._mask(u).sum()) == orig[u]
    eng2.undo()
    assert np.array_equal(eng2._mask(u), eroded)
    assert np.array_equal(mu.decode(eng2._refine_base_rle(u)).astype(bool), remasked)
    eng2.close()


def test_undoing_a_refine_on_a_remask_keeps_the_remask_as_base(tmp_path):
    eng, orig, _res = _remasked(tmp_path)
    u = eng.state.order[0]
    remasked = eng._mask(u).copy()
    eng.apply_refine(u, [{"name": "erode", "kw": {"k": 3}}])
    eng.undo()
    assert np.array_equal(eng._mask(u), remasked)
    eng.undo()                                                # the re-mask itself
    assert int(eng._mask(u).sum()) == orig[u]
    assert int(mu.decode(eng._refine_base_rle(u)).sum()) == orig[u]
    eng.close()


def test_masks_remasked_before_set_rle_existed_still_refine_from_the_remask(tmp_path):
    eng, orig, _res = _remasked(tmp_path)
    u = eng.state.order[0]
    remasked = eng._mask(u).copy()
    eng.close()
    ov = eng.store.load_refine(u); ov.pop("set_rle"); eng.store.save_refine(u, ov)
    eng2 = CuratorEngine(tmp_path / "proj"); eng2.open()
    assert np.array_equal(mu.decode(eng2._refine_base_rle(u)).astype(bool), remasked)
    eng2.close()


# --------------------------------------------------------------------------- several models at once
def test_pooling_interleaves_models_and_puts_agreed_masks_first():
    from chevron.backends.boxguide import pool_candidates
    a1, a2 = _rect(0, 0, 10, 10), _rect(20, 20, 30, 30)
    b1, b2 = _rect(40, 40, 50, 50), _rect(0, 0, 10, 10)          # b2 == a1
    got = pool_candidates([("a", [B.Proposal(a1, 0.9), B.Proposal(a2, 0.8)]),
                           ("b", [B.Proposal(b1, 0.3), B.Proposal(b2, 0.2)])], k=10)
    assert [c.meta["by"] for c in got] == [["a", "b"], ["b"], ["a"]]
    assert np.array_equal(got[0].mask, a1) and np.array_equal(got[1].mask, b1)
    assert len(pool_candidates([("a", [B.Proposal(a1, 1), B.Proposal(a2, 1)]),
                                ("b", [B.Proposal(b1, 1)])], k=2)) == 2


def test_remask_with_several_models_pools_their_candidates(tmp_path):
    cj = _coco(_images(tmp_path / "img", n=1), tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    orig = {u: int(eng._mask(u).sum()) for u in eng.state.order}
    mp, pr = _MultiPrompter(), _Prompter(); _register(mp, pr)
    try:
        res = eng.remask_instances(["stub_multi", "stub_prompt"])
    finally:
        _unregister(mp, pr)
    assert res["backend"] == "stub_multi+stub_prompt" and res["n_remasked"] == 2
    assert mp.calls == pr.calls == [2, 2, 2]                    # same boxes, each model once per size
    it = eng.mask_candidates()["items"][0]
    assert it["by"][0] == ["stub_multi", "stub_prompt"]         # both found it: first, and applied
    assert {"stub_multi"} <= {n for by in it["by"] for n in by}
    assert it["current"] == 1
    eng.undo()                                                   # still one step
    assert {u: int(eng._mask(u).sum()) for u in eng.state.order} == orig
    eng.close()


def test_an_unusable_model_in_the_list_stops_the_run_before_anything_runs(tmp_path):
    cj = _coco(_images(tmp_path / "img", n=1), tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))

    class _Off(_Prompter):
        name = "stub_off"

        def available(self):
            return False, "not here"
    mp, off = _MultiPrompter(), _Off(); _register(mp, off)
    try:
        c = TestClient(create_app(engine=eng))
        assert c.post("/api/remask", json={"backend": ["stub_multi", "stub_off"]}).status_code == 400
        assert mp.calls == [] and not eng.mask_candidates()["total"]
    finally:
        _unregister(mp, off)
    eng.close()


def test_api_remask_takes_a_list_or_comma_separated_models(tmp_path):
    cj = _coco(_images(tmp_path / "img", n=1), tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    eng.propose_instances("coco", coco_path=str(cj))
    mp, pr = _MultiPrompter(), _Prompter(); _register(mp, pr)
    try:
        c = TestClient(create_app(engine=eng))
        r = c.post("/api/remask", json={"backend": "stub_multi, stub_prompt"}).json()
        assert r["backends"] == ["stub_multi", "stub_prompt"]
        r = c.post("/api/remask", json={"backend": ["stub_prompt", "stub_multi"]}).json()
        assert r["backend"] == "stub_prompt+stub_multi"
        v = c.post("/api/mask_candidates/view", json={"iuid": eng.state.order[0]}).json()
        assert v["by"][0] is None and v["by"][1] == ["stub_prompt", "stub_multi"]
        assert len(v["by"]) == len(v["thumbs"])
    finally:
        _unregister(mp, pr)
    eng.close()


def test_importing_with_several_models_queues_their_candidates(tmp_path):
    cj = _coco(_images(tmp_path / "img", n=2), tmp_path / "gt.json")
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    mp, pr = _MultiPrompter(), _Prompter(); _register(mp, pr)
    try:
        res = eng.propose_instances("coco", coco_path=str(cj), remask_with=["stub_multi", "stub_prompt"])
    finally:
        _unregister(mp, pr)
    assert res["n_instances"] == 4 and res["n_remasked"] == 4 and res["n_candidates"] >= 8
    assert eng.mask_candidates()["total"] == 4
    assert all(eng.collection["records"][eng.state.meta[u].row].get("src_ann_id") for u in eng.state.order)
    eng.close()
