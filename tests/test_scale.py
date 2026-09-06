"""Scalable pseudo-labeling: scale.py logic (batch->COCO, merge, reference assign) + the sharded engine
pipeline (RAM-bounded, model stubbed). Run: pytest tests/test_scale.py -q
"""
from __future__ import annotations

import json

import numpy as np


def _rle(h=32, w=32):
    import cv2
    from pycocotools import mask as mu
    m = np.zeros((h, w), np.uint8); cv2.circle(m, (16, 16), 6, 1, -1)
    r = mu.encode(np.asfortranarray(m)); r["counts"] = r["counts"].decode("ascii")
    return r


def _batch(n=4, img=1000):
    recs = [{"rle": _rle(), "image_id": img + (i % 2), "H": 32, "W": 32, "score": 0.5 + 0.1 * i,
             "file_name": f"/x/im{img + (i % 2)}.png"} for i in range(n)]
    return {"records": recs, "feats": {}, "n_images": 2}


def test_batch_to_coco_labels_and_drops():
    from chevron import scale as sc
    b = _batch(4)
    coco = sc.batch_to_coco(b, ["coin", None, "lead", "coin"], scores=[0.9, 0.0, 0.8, 0.7])
    assert len(coco["annotations"]) == 3                              # the None is dropped
    assert {c["name"] for c in coco["categories"]} == {"coin", "lead"}
    assert all("assign_score" in a and "segmentation" in a for a in coco["annotations"])
    ca = sc.batch_to_coco(b, ["coin", "lead", "coin", "lead"], class_agnostic=True)
    assert {c["name"] for c in ca["categories"]} == {"object"} and len(ca["annotations"]) == 4


def test_merge_cocos(tmp_path):
    from chevron import scale as sc
    p1 = tmp_path / "s0.json"; p2 = tmp_path / "s1.json"
    p1.write_text(json.dumps(sc.batch_to_coco(_batch(4, 1000), ["coin", "coin", None, "lead"])))
    p2.write_text(json.dumps(sc.batch_to_coco(_batch(4, 2000), ["lead", None, "tube", "tube"])))
    rep = sc.merge_cocos([str(p1), str(p2)], tmp_path / "merged.json")
    m = json.loads((tmp_path / "merged.json").read_text())
    assert rep["categories"] == 3 and {c["name"] for c in m["categories"]} == {"coin", "lead", "tube"}
    assert len({a["id"] for a in m["annotations"]}) == len(m["annotations"])   # ids reindexed unique
    assert len({im["id"] for im in m["images"]}) == len(m["images"])


def test_assign_by_reference():
    from chevron import scale as sc
    from chevron.reference_bank import ReferenceBank
    bank = ReferenceBank(np.eye(4, dtype=np.float32)[:3], ["coin", "coin", "lead"],
                         {"coin": "coin", "lead": "lead"}, [])
    emb = np.array([[1, 0.02, 0, 0], [0, 0, 1, 0.02]], np.float32)     # near 'coin' dir, near 'lead' dir
    names, scores = sc.assign_by_reference(emb, bank, thresh=-1e9)
    assert names[0] == "coin" and names[1] == "lead"


def test_scaled_pseudolabel_is_sharded_and_ram_bounded(tmp_path, monkeypatch):
    """The pipeline shards the model run, writes a COCO per shard + merged, and NEVER grows the main
    collection (RAM stays bounded to one shard). collect_batch stubbed."""
    from chevron import collect as _co
    from chevron.engine import CuratorEngine
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    eng.collection = {"records": [], "n_images": 0, "feats": {"decoder": np.zeros((0, 8), np.float32)}}
    monkeypatch.setattr(eng, "_ensure_model", lambda: (None, None, None))
    calls = {"n": 0}
    def fake_collect(model, cfg, d2_cfg, files, *, score_thresh, feature_cfg):
        calls["n"] += 1
        return _batch(len(files))                                      # 1 instance per image
    monkeypatch.setattr(_co, "collect_batch", fake_collect)
    files = [f"/x/im{i}.png" for i in range(25)]
    rep = eng.scaled_pseudolabel(image_paths=files, shard_size=10, method="raw",
                                 out_dir=str(tmp_path / "pl"), class_agnostic=True)
    assert rep["ok"] and rep["shards"] == 3 and calls["n"] == 3        # 25 imgs / 10 -> 3 shards
    assert rep["n_instances"] == 25 and rep["n_labeled"] == 25         # raw -> every detection labeled 'object'
    assert eng.collection["records"] == []                            # main collection NEVER grown (RAM bounded)
    merged = json.loads((tmp_path / "pl" / "merged.json").read_text())
    assert len(merged["annotations"]) == 25 and {c["name"] for c in merged["categories"]} == {"object"}
    assert len(list((tmp_path / "pl").glob("shard_*.json"))) == 3
