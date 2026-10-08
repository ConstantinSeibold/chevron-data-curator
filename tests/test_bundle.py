"""Project bundles: one zip with the project, its images and source COCO files; import on another
machine (the originals gone) and carry on — same labels, masks and log; images load.
Run: pytest tests/test_bundle.py -q  (from repo root)
"""
from __future__ import annotations

import json
import shutil
import zipfile

import numpy as np
import pytest

from test_merge_rec import _rle


def _project(tmp_path):
    """A project in tmp/proj whose images (and source COCO) live OUTSIDE it, in tmp/data."""
    import cv2
    from chevron import ids
    from chevron.engine import CuratorEngine
    from chevron.state import InstanceMeta
    data = tmp_path / "data"
    (data / "raw" / "a").mkdir(parents=True)
    coco = data / "coco.json"
    coco.write_text(json.dumps({"images": [], "annotations": [], "categories": []}))
    eng = CuratorEngine(tmp_path / "proj")
    eng.init_project({"images": {"root": str(data)}, "model": {"ckpt": "/nowhere/model.pth"}})
    recs, order, meta = [], [], {}
    for ii in range(2):
        p = data / "raw" / "a" / f"im{ii}.png"
        cv2.imwrite(str(p), np.full((64, 64, 3), 60 + 50 * ii, np.uint8))
        for j in range(3):
            m = np.zeros((64, 64), bool); m[10 + 15 * j:20 + 15 * j, 10:30] = True
            u = ids.new_uid(); row = len(recs)
            recs.append({"iuid": u, "row": row, "inst_id": row, "image_id": 1000 + ii, "H": 64, "W": 64,
                         "score": 0.9, "rle": _rle(m), "file_name": str(p), "abs_path": str(p),
                         "src_coco": str(coco), "batch_id": "b", "cx": .3, "cy": .3, "bw": .3, "bh": .2,
                         "box_area": .06, "mask_area_frac": float(m.mean())})
            order.append(u); meta[u] = InstanceMeta(iuid=u, batch_id="b", row=row, image_id=1000 + ii,
                                                    provenance={"file": str(p)})
    eng.collection = {"records": recs, "n_images": 2, "feats": {"decoder": np.random.rand(6, 4).astype(np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    eng.assign(order[:2], "pacemaker")
    eng.accept_masks(order[:1])
    eng.set_background(order[5:])
    eng.save(snapshot=True)
    return eng, order, data


def test_bundle_roundtrip_on_another_machine(tmp_path):
    from chevron import bundle
    from chevron.engine import CuratorEngine
    from chevron.projects import ProjectRegistry
    eng, order, data = _project(tmp_path)
    _live = lambda st: {k: v for k, v in st.items() if k not in ("serial", "undo", "redo")}  # per process
    before = _live(eng.stats())
    n_log = len((tmp_path / "proj" / "history.jsonl").read_text().splitlines())
    px = eng._rgb_by_image(1001).copy()
    eng.close()
    r = bundle.export_bundle(tmp_path / "proj", tmp_path / "out" / "p.zip")
    assert len(r["images"]) == 2 and len(r["sources"]) == 1 and not r["missing"]
    assert r["left_out"] == {"ckpt": "/nowhere/model.pth"}
    shutil.rmtree(data); shutil.rmtree(tmp_path / "proj")            # the other machine has neither

    info = ProjectRegistry(tmp_path / "root").import_bundle(tmp_path / "out" / "p.zip", "FB")
    dest = tmp_path / "root" / info.id
    assert info.name == "FB" and info.n_instances == 6
    b = CuratorEngine(dest); b.open()
    assert _live(b.stats()) == before
    assert b.mask_reviewed(order[0]) and b.state.meta[order[5]].is_background
    rec = b.collection["records"][0]
    assert rec["abs_path"] == str(dest / "images" / "raw" / "a" / "im0.png")
    assert rec["src_coco"] == str(dest / "sources" / "0_coco.json")
    assert b.state.config["images"]["root"] == str(dest / "images")
    assert b.state.meta[order[0]].provenance["file"].startswith(str(dest))
    assert np.array_equal(b._rgb_by_image(1001), px)                 # real pixels, not a black fallback
    assert str(data) not in (dest / "state.json").read_text()
    assert all(str(data) not in p.read_text() for p in dest.glob("snapshots/*/state.json"))
    assert len((dest / "history.jsonl").read_text().splitlines()) == n_log   # the audit log came along
    b.close()


def test_import_refuses_a_zip_that_escapes_the_folder(tmp_path):
    from chevron import bundle
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(bundle.BUNDLE_FILE, json.dumps({"format": 1, "images": {}, "sources": {}}))
        f.writestr("project/../../escaped.txt", "x")
    with pytest.raises(ValueError):
        bundle.import_bundle(z, tmp_path / "dest")
    assert not (tmp_path / "escaped.txt").exists()
    with pytest.raises(ValueError):                                   # not a bundle at all
        zipfile.ZipFile(tmp_path / "plain.zip", "w").close()
        bundle.read_manifest(tmp_path / "plain.zip")


def test_launcher_endpoints_export_scan_and_import(tmp_path):
    from fastapi.testclient import TestClient
    from chevron.projects import ProjectRegistry
    from chevron.server import create_app
    eng, order, data = _project(tmp_path)
    eng.close()
    root = tmp_path / "root"
    reg = ProjectRegistry(root)
    pid = reg.link(tmp_path / "proj", "FB").id
    c = TestClient(create_app(root=str(root)))
    res = c.get(f"/api/projects/{pid}/bundle")
    assert res.status_code == 200 and res.headers["content-type"] == "application/zip"
    zp = tmp_path / "dl.zip"; zp.write_bytes(res.content)
    found = c.post("/api/projects/scan", json={"path": str(zp)}).json()["found"]
    assert found[0]["bundle"] and found[0]["n_images"] == 2 and found[0]["left_out"] == ["ckpt"]
    shutil.rmtree(data)
    p = c.post("/api/projects/import", json={"path": str(zp)}).json()["project"]
    assert p["id"] != pid and p["n_instances"] == 6 and not p["linked"]
    assert c.post(f"/api/projects/{p['id']}/open").json()["ok"]
    assert c.get("/api/image_overlay", params={"image_id": 1000}).status_code == 200
