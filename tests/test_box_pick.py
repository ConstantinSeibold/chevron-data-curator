"""Pick masks: instances of the SAME source box (one COCO annotation ingested / re-masked several times)
are shown side by side; one pick keeps a mask, marks it reviewed, rejects the rest — one undo step.
Run: pytest tests/test_box_pick.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np

from test_merge_rec import _engine, _rle


def _sq(y0, x0, s, H=128, W=128):
    m = np.zeros((H, W), bool); m[y0:y0 + s, x0:x0 + s] = True
    return m


def _ix(eng, k, gen):
    """Index of the choice `gen` made — choices are ranked, so tests address them by generator."""
    return next(i for i, c in enumerate(eng.box_choices(k)) if gen in c["by"])


def _boxes_engine(tmp_path):
    """One image, two source boxes x three generated masks; instance 0 also kept a re-mask alternative."""
    eng, o = _engine(tmp_path, n_img=1, per_img=6)
    masks = [_sq(10, 10, 30), _sq(10, 10, 30), _sq(12, 12, 20),          # box 1: two agree, one smaller
             _sq(70, 70, 30), _sq(72, 72, 26), _sq(60, 60, 50)]          # box 2: three different
    methods = ["samhq", "qseg", "hf_seg"] * 2
    for j, (u, m) in enumerate(zip(o, masks)):
        r = eng.collection["records"][eng.state.meta[u].row]
        r.update(rle=_rle(m), src_ann_id=1 + j // 3, src_category_id=7, src_coco=f"c{j % 2}.json")
        eng.state.meta[u].provenance = {"remask": {"backend": methods[j]}}
    eng.assign(o, "X", source="import")
    eng._mask_cands[o[0]] = {"backend": "samhq", "original": _rle(_sq(5, 5, 40)), "original_box_only": True,
                             "reviewed": False, "cands": [{"rle": _rle(masks[0]), "score": .9, "by": ["samhq"]}]}
    return eng, o


def test_groups_ignore_the_file_name(tmp_path):
    eng, o = _boxes_engine(tmp_path)
    assert sorted(map(frozenset, eng.box_groups().values()), key=min) == \
        sorted([frozenset(o[:3]), frozenset(o[3:])], key=min)


def test_choices_merge_agreeing_masks_and_include_alternatives(tmp_path):
    eng, o = _boxes_engine(tmp_path)
    k = next(k for k, v in eng.box_groups().items() if o[0] in v)
    ch = eng.box_choices(k)
    assert ch[0]["by"] == ["samhq", "qseg"]                       # identical masks: one choice, both credited
    assert [c["by"] for c in ch[1:]] == [["hf_seg"], ["box"]]      # + the box-only original, offered last
    assert len(eng.box_choice_crops(k)) == len(ch)


def test_pick_keeps_one_reviews_it_rejects_the_rest_and_undoes(tmp_path):
    eng, o = _boxes_engine(tmp_path)
    k = next(k for k, v in eng.box_groups().items() if o[3] in v)
    s0 = eng.stats()
    assert eng.boxes()["n_todo"] == 2
    r = eng.pick_box_choice(k, _ix(eng, k, "qseg"))                # qseg's mask, on instance 4
    assert r["kept"] == o[4] and set(r["rejected"]) == {o[3], o[5]}
    assert eng.mask_reviewed(o[4]) and all(eng.state.meta[u].is_background for u in (o[3], o[5]))
    assert eng.state.meta[o[4]].assigned_class                     # the class stays on the survivor
    assert eng.boxes()["n_todo"] == 1 and eng.box_pick_wins() == {"qseg": 1}
    assert eng.stats()["n_mask_unreviewed"] == s0["n_mask_unreviewed"] - 3
    eng.undo()
    assert not eng.mask_reviewed(o[4]) and not eng.state.meta[o[3]].is_background
    assert eng.boxes()["n_todo"] == 2


def test_picking_an_alternative_sets_that_mask(tmp_path):
    from pycocotools import mask as mu
    eng, o = _boxes_engine(tmp_path)
    k = next(k for k, v in eng.box_groups().items() if o[0] in v)
    eng.pick_box_choice(k, _ix(eng, k, "box"))                     # the box-only original of instance 0
    kept = next(u for u in o[:3] if not eng.state.meta[u].is_background)
    assert kept == o[0] and mu.area(eng._eff_rle(o[0])) == 40 * 40


def test_endpoints(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, o = _boxes_engine(tmp_path)
    c = TestClient(create_app(engine=eng))
    b = c.get("/api/boxes").json()
    assert b["n_boxes"] == 2 and b["classes"][0]["todo"] == 2 and len(b["items"]) == 2
    it = b["items"][0]
    png = c.get("/api/box_crop", params={"key": it["key"], "i": 0})
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert c.post("/api/box_pick", json={"key": it["key"], "index": 0}).json()["ok"]
    assert c.get("/api/boxes").json()["n_todo"] == 1
    assert c.post("/api/box_pick", json={"key": "nope", "index": 0}).status_code == 400


def test_mask_method_facet_filters_every_view(tmp_path):
    eng, o = _boxes_engine(tmp_path)
    assert {m["method"]: m["n"] for m in eng.sources()["methods"]} == {"samhq": 2, "qseg": 2, "hf_seg": 2}
    eng.set_method_filter(["qseg"])
    assert sorted(eng.partition_iuids("class:" + eng.state.meta[o[1]].assigned_class)) == sorted([o[1], o[4]])
    assert all(len(v) == 1 for v in eng.box_groups().values())
    eng.set_method_filter(None)
    k = next(k for k, v in eng.box_groups().items() if o[4] in v)
    eng.pick_box_choice(k, _ix(eng, k, "hf_seg"))                                  # hf_seg's mask, kept
    survivor = next(u for u in o[3:] if not eng.state.meta[u].is_background)
    assert eng._method_of(survivor) == "hf_seg"                                    # the pick, not the instance


def test_class_method_chips_ignore_the_active_filter(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, o = _boxes_engine(tmp_path)
    cid = eng.state.meta[o[0]].assigned_class
    c = TestClient(create_app(engine=eng))
    c.post("/api/source_filter", json={"methods": ["qseg"]})
    r = c.get("/api/class_subclusters", params={"cid": cid}).json()
    assert {m["method"]: m["n"] for m in r["methods"]} == {"samhq": 2, "qseg": 2, "hf_seg": 2}   # all offered
    assert r["methods_active"] == ["qseg"] and sum(x["size"] for x in r["rows"]) == 2           # subs filtered


def test_accept_masks_marks_reviewed_and_undoes(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, o = _boxes_engine(tmp_path)
    c = TestClient(create_app(engine=eng))
    n0 = eng.stats()["n_mask_unreviewed"]
    r = c.post("/api/accept_masks", json={"iuids": [o[0], o[1], o[0]]}).json()
    assert r["n"] == 2 and r["stats"]["n_mask_unreviewed"] == n0 - 2
    assert eng.mask_reviewed(o[0]) and eng.mask_reviewed(o[1])
    assert c.post("/api/accept_masks", json={"iuids": [o[0]]}).json()["n"] == 0     # already reviewed
    eng.undo()
    assert not eng.mask_reviewed(o[0]) and eng.stats()["n_mask_unreviewed"] == n0


def test_choices_rank_agreement_first_and_say_so(tmp_path):
    eng, o = _boxes_engine(tmp_path)
    k1 = next(k for k, v in eng.box_groups().items() if o[0] in v)
    k2 = next(k for k, v in eng.box_groups().items() if o[3] in v)
    assert eng.box_choices(k1)[0]["agree"] == 2                    # samhq + qseg made the same mask
    assert all(c["agree"] == 1 for c in eng.box_choices(k2))       # three different masks
    assert [it["key"] for it in eng.boxes(sort="agree")["items"]][0] == k1
    b = eng.boxes()
    assert b["n_agree"] == 1 and b["n_multi"] == 2


def test_collapse_rejects_duplicates_keeps_review_open_and_alternatives(tmp_path):
    eng, o = _boxes_engine(tmp_path)
    k2 = next(k for k, v in eng.box_groups().items() if o[3] in v)
    n_choices = len(eng.box_choices(k2))
    assert eng.collapse_boxes(apply=False) == {"n": 2}
    r = eng.collapse_boxes()
    assert r["n"] == 2 and r["n_rejected"] == 4
    assert all(len(v) == 1 for v in eng.box_groups().values())
    assert eng.boxes()["n_todo"] == 2                              # nothing reviewed: still to do
    (k2b,) = [k for k, v in eng.box_groups().items() if v[0] in o[3:]]
    assert len(eng.box_choices(k2b)) == n_choices                  # every alternative still on offer
    eng.undo()
    assert sum(len(v) for v in eng.box_groups().values()) == 6


def test_accept_agreeing_reviews_only_boxes_generators_agree_on(tmp_path):
    eng, o = _boxes_engine(tmp_path)
    assert eng.accept_agreeing_boxes(apply=False) == {"n": 1}
    r = eng.accept_agreeing_boxes()
    assert r["n"] == 1 and r["n_rejected"] == 2
    kept = [u for u in o[:3] if not eng.state.meta[u].is_background]
    assert len(kept) == 1 and eng.mask_reviewed(kept[0])
    assert not any(eng.mask_reviewed(u) for u in o[3:]) and eng.boxes()["n_todo"] == 1


def test_reject_box_and_endpoints(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, o = _boxes_engine(tmp_path)
    c = TestClient(create_app(engine=eng))
    assert c.post("/api/box_accept_agree", json={}).json()["n"] == 1          # dry run: a count
    assert c.post("/api/box_collapse", json={}).json()["n"] == 2
    k2 = next(k for k, v in eng.box_groups().items() if o[3] in v)
    assert c.post("/api/box_pick", json={"key": k2, "index": 0, "review": False}).json()["ok"]
    (survivor,) = [u for u in o[3:] if not eng.state.meta[u].is_background]
    assert not eng.mask_reviewed(survivor)
    (k2,) = [k for k, v in eng.box_groups().items() if survivor in v]
    assert c.post("/api/box_reject", json={"key": k2}).json()["ok"]
    assert eng.state.meta[survivor].is_background
    assert c.post("/api/box_accept_agree", json={"apply": True}).json()["n"] == 1
    assert c.get("/api/boxes").json()["n_todo"] == 0


def test_editor_window_covers_the_source_box(tmp_path):
    eng, o = _boxes_engine(tmp_path)
    r = eng.collection["records"][eng.state.meta[o[5]].row]
    r["box_xyxy"] = np.array([40, 40, 120, 120], np.float32)      # annotated box wider than the mask
    v = eng.edit_view(o[5], pad=0)
    assert v["box"][:2] == [40, 40] and v["src_box"] is not None


def test_a_reviewed_mask_survives_collapse(tmp_path):
    eng, o = _boxes_engine(tmp_path)
    eng.accept_masks([o[5]])                                       # hf_seg's mask on box 2, signed off
    k2 = next(k for k, v in eng.box_groups().items() if o[5] in v)
    assert eng.box_choices(k2)[0]["owner"] == o[5]
    eng.collapse_boxes()
    assert not eng.state.meta[o[5]].is_background and eng.mask_reviewed(o[5])
    assert all(eng.state.meta[u].is_background for u in (o[3], o[4]))


def test_finished_images_and_classes_are_marked(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, o = _boxes_engine(tmp_path)
    cid = eng.state.class_id_by_name("X")
    iid = eng.state.meta[o[0]].image_id
    assert eng.final_marks() == {"images": {iid: False}, "classes": {cid: False}}
    for k in list(eng.box_groups()):                               # one reviewed mask per box, duplicates rejected
        eng.pick_box_choice(k, 0)
    assert eng.final_marks() == {"images": {iid: True}, "classes": {cid: True}}
    c = TestClient(create_app(engine=eng))
    assert c.get("/api/images").json()["items"][0]["final"] is True
    assert next(r for r in c.get("/api/partitions").json()["rows"] if r["pid"] == f"class:{cid}")["final"]
    eng.undo()                                                     # one box open again -> neither is finished
    assert eng.final_marks() == {"images": {iid: False}, "classes": {cid: False}}


def test_image_view_counts_say_what_they_count(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.server import create_app
    eng, o = _boxes_engine(tmp_path)
    cid = eng.state.class_id_by_name("X")
    iid = eng.state.meta[o[0]].image_id
    eng.set_background([o[5]])
    c = TestClient(create_app(engine=eng))
    assert c.get("/api/image_classes", params={"image_id": iid}).json()["classes"] == {cid: 5}
    eng.assign([o[0]], "Y")                                         # a scoped picker count is that scope's only
    it = c.get("/api/images", params={"pid": f"class:{cid}"}).json()["items"][0]
    assert (it["n"], it["n_all"]) == (4, 6)
