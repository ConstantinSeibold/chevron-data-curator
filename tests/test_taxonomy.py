"""Nested taxonomy: seed_taxonomy loads superclass->concept->part leaves with pinned ids; taxonomy_tree
groups + buckets temp; release_qc part-rule gate; export excludes temp + emits real supercategory + mimic
crosswalk; the shipped seed_taxonomy.json is well-formed. Run: pytest tools/curator/tests/test_taxonomy.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _eng(tmp_path):
    from tools.curator.engine import CuratorEngine
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    return eng


def test_seed_taxonomy_loads_nested(tmp_path):
    eng = _eng(tmp_path)
    rep = eng.seed_taxonomy()
    assert rep["superclasses"] >= 10 and rep["concepts"] >= 60 and rep["leaves"] >= 80
    # pacemaker concept -> two part leaves, grouped under cardiac_implant
    assert "pacemaker_body" in eng.state.taxonomy and "pacemaker_lead" in eng.state.taxonomy
    body = eng.state.taxonomy["pacemaker_body"]
    assert body.concept == "pacemaker" and body.supercategory == "cardiac_implant" and not body.temp
    assert body.coco_cat_id and eng.state.taxonomy["pacemaker_lead"].coco_cat_id != body.coco_cat_id   # pinned, distinct
    # part-less concept is its own leaf
    assert "coin" in eng.state.taxonomy and eng.state.taxonomy["coin"].concept == "coin"
    # mimic crosswalk preserved on the concept
    assert eng.state.concepts["pacemaker"].mimic_family == "pacemaker"
    assert eng.state.concepts["iabp"].superclass == "cardiac_implant"      # the placement fix stuck


def test_taxonomy_tree_groups_and_temp_bucket(tmp_path):
    eng = _eng(tmp_path)
    eng.seed_taxonomy()
    # add a temp/scratch class (the placeholder case) + an ungrouped one
    from tools.curator.state import TaxonomyClass
    eng.state.taxonomy["scratch1"] = TaxonomyClass(class_id="scratch1", name="scratch1", temp=True)
    tree = eng.taxonomy_tree()
    sc = {s["id"]: s for s in tree["superclasses"]}
    assert "cardiac_implant" in sc
    pace = next(c for c in sc["cardiac_implant"]["concepts"] if c["id"] == "pacemaker")
    assert {lf["id"] for lf in pace["leaves"]} == {"pacemaker_body", "pacemaker_lead"}
    assert any(t["id"] == "scratch1" and t["temp"] for t in tree["temp"])   # temp lands in the scratch bucket


def test_assign_leaf_promotes_temp(tmp_path):
    """A temp/scratch class can be promoted into a concept (inherits its superclass, temp cleared)."""
    from tools.curator.state import TaxonomyClass
    eng = _eng(tmp_path)
    eng.seed_taxonomy()
    eng.state.taxonomy["scratch_can"] = TaxonomyClass(class_id="scratch_can", name="scratch_can", temp=True)
    assert eng.assign_leaf("scratch_can", "pacemaker")["ok"]
    t = eng.state.taxonomy["scratch_can"]
    assert t.concept == "pacemaker" and t.supercategory == "cardiac_implant" and t.temp is False
    assert eng.assign_leaf("scratch_can", None)["ok"] and eng.state.taxonomy["scratch_can"].concept is None


def test_release_qc_part_rule_gate(tmp_path):
    """An image with a pacemaker_body but no pacemaker_lead violates the completeness rule -> held back."""
    from tools.curator.state import InstanceMeta
    eng = _eng(tmp_path)
    eng.seed_taxonomy()
    # image 1: body + lead (ok); image 2: body only (violates body=>lead)
    eng.state.meta = {
        "a": InstanceMeta(iuid="a", batch_id="b", row=0, image_id=1, assigned_class="pacemaker_body"),
        "b": InstanceMeta(iuid="b", batch_id="b", row=1, image_id=1, assigned_class="pacemaker_lead"),
        "c": InstanceMeta(iuid="c", batch_id="b", row=2, image_id=2, assigned_class="pacemaker_body"),
    }
    qc = eng.release_qc()
    assert qc["n_violating"] == 1 and "2" in qc["violating_image_ids"] and "1" not in qc["violating_image_ids"]
    assert qc["violations"][0]["missing"] == ["pacemaker_lead"]


def test_export_excludes_temp_and_uses_supercategory(tmp_path):
    from tools.curator import export_coco as ex
    from tools.curator.state import InstanceMeta
    import cv2
    from pycocotools import mask as mu
    eng = _eng(tmp_path)
    eng.seed_taxonomy()
    m = np.zeros((32, 32), np.uint8); cv2.circle(m, (16, 16), 6, 1, -1)
    r = mu.encode(np.asfortranarray(m)); r["counts"] = r["counts"].decode("ascii")
    recs = [{"rle": r, "image_id": 1, "H": 32, "W": 32, "score": 0.9, "file_name": "/x/1.png", "row": 0},
            {"rle": r, "image_id": 1, "H": 32, "W": 32, "score": 0.9, "file_name": "/x/1.png", "row": 1}]
    eng.collection = {"records": recs, "feats": {}, "n_images": 1}
    from tools.curator.state import TaxonomyClass
    eng.state.taxonomy["scratch1"] = TaxonomyClass(class_id="scratch1", name="scratch1", temp=True)
    eng.state.order = ["a", "b"]
    eng.state.meta = {"a": InstanceMeta(iuid="a", batch_id="b", row=0, image_id=1, assigned_class="pacemaker_body"),
                      "b": InstanceMeta(iuid="b", batch_id="b", row=1, image_id=1, assigned_class="scratch1")}
    coco = ex.assemble_curated_coco(eng.collection, eng.state)
    names = {c["name"] for c in coco["categories"]}
    assert "scratch1" not in names                            # temp class excluded from the release
    cat = next(c for c in coco["categories"] if c["name"] == "Pulse generator")
    assert cat["supercategory"] == "Cardiac & vascular implants" and cat.get("mimic_family") == "pacemaker"
    assert len(coco["annotations"]) == 1                       # only the non-temp instance


def test_seed_json_wellformed():
    d = json.loads((Path(__file__).resolve().parents[1] / "taxonomy_seed.json").read_text())
    sc_ids = {s["id"] for s in d["superclasses"]}
    assert len(sc_ids) == len(d["superclasses"])               # unique superclass ids
    cids = [c["id"] for c in d["concepts"]]
    assert len(cids) == len(set(cids))                         # unique concept ids
    for c in d["concepts"]:
        assert c["superclass"] in sc_ids, c["id"]              # every concept points at a real superclass
        for r in c.get("part_rules", []):                      # rules reference this concept's own parts
            pids = {p["id"] for p in c.get("parts", [])}
            assert r["if"] in pids and all(t in pids for t in r["then"]), c["id"]
