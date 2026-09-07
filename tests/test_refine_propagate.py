"""SAM-HQ checkpoint resolution + within-partition refinement propagation (Task 1). The RAD-DINO match
gate and op-replay are exercised with apply_refine_many / embeddings stubbed (no GPU, no image IO).
Run: pytest tests/test_refine_propagate.py -q
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest


def _rle(h=32, w=32):
    import cv2
    from pycocotools import mask as mu
    m = np.zeros((h, w), np.uint8); cv2.circle(m, (16, 16), 6, 1, -1)
    r = mu.encode(np.asfortranarray(m)); r["counts"] = r["counts"].decode("ascii")
    return r


def _fb(files, dim=8):
    from chevron import ids
    recs = [{"iuid": ids.new_uid(), "batch_id": "b", "abs_path": f, "file_name": f,
             "image_id": abs(hash(f)) % 1000000, "score": 0.9, "rle": _rle(), "H": 32, "W": 32}
            for f in files]
    return {"records": recs, "feats": {"decoder": np.zeros((len(files), dim), np.float32)}, "n_images": len(files)}


def _eng_with_instances(tmp_path, monkeypatch, n=4):
    from chevron import collect as _co
    from chevron.engine import CuratorEngine
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    eng.collection = {"records": [], "n_images": 0, "feats": {"decoder": np.zeros((0, 8), np.float32)}}
    monkeypatch.setattr(eng, "_ensure_model", lambda: (None, None, None))
    monkeypatch.setattr(_co, "collect_batch",
                        lambda m, c, d, fs, *, score_thresh, feature_cfg: _fb(fs))
    eng.ingest_paths([f"/x/{i}.png" for i in range(n)])
    return eng


# ---- SAM-HQ checkpoint resolution (no download) --------------------------------------------------------
def test_samhq_checkpoint_resolution(tmp_path, monkeypatch):
    from chevron import refine as rf
    monkeypatch.setattr(rf, "_sam_dir", lambda: tmp_path)
    (tmp_path / "sam_vit_b_01ec64.pth").write_bytes(b"x")               # only a VANILLA ckpt present
    assert rf.find_sam_checkpoint(family="samhq") == (None, None)       # never hand a vanilla file to the HQ arch
    (tmp_path / "sam_hq_vit_b.pth").write_bytes(b"x")                    # now a SAM-HQ ckpt is present
    ck2, mt2 = rf.find_sam_checkpoint(family="samhq")
    assert rf.detect_sam_family(ck2) == "samhq" and mt2 == "vit_b"
    assert set(rf._HQ_URLS) >= {"vit_b", "vit_l", "vit_h", "vit_tiny"}
    # ensure_* returns the cached HQ ckpt without downloading. It refuses outright when the package
    # is absent, so this last step needs the optional extra; the resolution logic above does not.
    pytest.importorskip("segment_anything_hq")
    assert rf.ensure_samhq_checkpoint("vit_b") == str(tmp_path / "sam_hq_vit_b.pth")


def test_the_checkpoint_download_reports_bytes(tmp_path, monkeypatch):
    """The caller renders "184 MB / 379 MB", so the hook must hand it BYTES, not a 0..1 fraction —
    a fraction shown as a byte count is a job that claims to have downloaded 0.4 bytes."""
    pytest.importorskip("segment_anything_hq")
    from chevron import refine as rf
    monkeypatch.setattr(rf, "_sam_dir", lambda: tmp_path)

    def _fake_urlretrieve(url, dest, hook):
        pathlib.Path(dest).write_bytes(b"weights")
        hook(0, 8192, 16384); hook(1, 8192, 16384); hook(2, 8192, 16384)

    import urllib.request as _u                      # imported inside the function under test
    monkeypatch.setattr(_u, "urlretrieve", _fake_urlretrieve)
    seen = []
    out = rf.ensure_samhq_checkpoint("vit_b", progress=lambda d, t: seen.append((d, t)))
    assert seen == [(0, 16384), (8192, 16384), (16384, 16384)]   # clamped: never past the total
    assert pathlib.Path(out).exists() and not list(tmp_path.glob("*.part"))


def test_vanilla_ensure_never_adopts_an_hq_checkpoint(tmp_path, monkeypatch):
    """With only HQ weights cached, a vanilla request must DOWNLOAD — not load sam_hq through the
    vanilla arch. `sam_hq_*` sorts first in the cache dir, so a family-less lookup picked it and the
    build died in torch.load."""
    pytest.importorskip("segment_anything")
    from chevron import refine as rf
    monkeypatch.setattr(rf, "_sam_dir", lambda: tmp_path)
    monkeypatch.delenv("CURATOR_SAM_CKPT", raising=False)
    (tmp_path / "sam_hq_vit_b.pth").write_bytes(b"hq")

    fetched = []

    def _fake_urlretrieve(url, dest, hook):
        fetched.append(url)
        pathlib.Path(dest).write_bytes(b"vanilla")

    import urllib.request as _u
    monkeypatch.setattr(_u, "urlretrieve", _fake_urlretrieve)
    out = rf.ensure_sam_checkpoint("vit_b")
    assert rf.detect_sam_family(out) != "samhq" and fetched, "an HQ file was handed to the vanilla arch"


def test_a_gpu_saved_checkpoint_loads_on_a_cpu_only_machine(monkeypatch):
    """The published SAM-HQ weights carry CUDA storages and `_build_sam` calls `torch.load(f)` with
    no map_location, which on a CPU-only box refuses to deserialize at all."""
    torch = pytest.importorskip("torch")
    from chevron import refine as rf
    seen = {}
    monkeypatch.setattr(torch, "load", lambda *a, **kw: seen.update(kw) or {})
    with rf.load_on_cpu():
        torch.load("some.pth")
    assert seen.get("map_location") == "cpu"
    # and the default is put back, so nothing else in the process silently loads onto the CPU
    seen.clear()
    torch.load("some.pth")
    assert "map_location" not in seen


def test_find_checkpoint_never_crosses_registry(tmp_path, monkeypatch):
    """With BOTH a vanilla and a sam_hq ckpt cached, vanilla requests must NOT pick sam_hq (which sorts
    first alphabetically) — loading HQ weights into the vanilla Sam arch raises 'Unexpected key(s)'."""
    from chevron import refine as rf
    monkeypatch.setattr(rf, "_sam_dir", lambda: tmp_path)
    (tmp_path / "sam_vit_b_01ec64.pth").write_bytes(b"x")
    (tmp_path / "sam_hq_vit_b.pth").write_bytes(b"x")
    assert rf.detect_sam_family(rf.find_sam_checkpoint(family="sam")[0]) == "sam"
    assert rf.detect_sam_family(rf.find_sam_checkpoint(family="medsam")[0]) == "sam"   # no medsam -> vanilla, not hq
    assert rf.detect_sam_family(rf.find_sam_checkpoint(family="samhq")[0]) == "samhq"


# ---- within-partition propagation (Task 1) -------------------------------------------------------------
def test_propagate_replays_chain_across_partition(tmp_path, monkeypatch):
    eng = _eng_with_instances(tmp_path, monkeypatch, n=4)
    ius = list(eng.state.meta)
    eng.assign(ius, "device")                                           # one class pseudo-partition
    cid = eng.state.class_id_by_name("device")
    ref = ius[0]
    eng.state.meta[ref].rule_ops = [{"name": "sam", "model": "samhq"}]  # the reference's recorded recipe
    seen = {}
    monkeypatch.setattr(eng, "apply_refine_many",
                        lambda iuids, ops: (seen.update(iuids=list(iuids), ops=ops), len(iuids))[1])

    r = eng.propagate_refinement(ref, pid=f"class:{cid}")               # ungated: all peers, ref excluded
    assert r["applied"] == 3 and set(seen["iuids"]) == set(ius[1:])
    assert seen["ops"] == [{"name": "sam", "model": "samhq"}] and r["matched"] is False

    # ops default comes from the reference; pid defaults to the reference's own partition
    seen.clear()
    r2 = eng.propagate_refinement(ref)
    assert r2["pid"] == f"class:{cid}" and r2["applied"] == 3


def test_propagate_raddino_gate_skips_dissimilar(tmp_path, monkeypatch):
    eng = _eng_with_instances(tmp_path, monkeypatch, n=4)
    ius = list(eng.state.meta)
    eng.assign(ius, "device")
    cid = eng.state.class_id_by_name("device")
    ref = ius[0]
    eng.state.meta[ref].rule_ops = [{"name": "dilate", "px": 1}]
    seen = {}
    monkeypatch.setattr(eng, "apply_refine_many",
                        lambda iuids, ops: (seen.update(iuids=list(iuids)), len(iuids))[1])
    emb = {ref: [1, 0], ius[1]: [1, 0], ius[2]: [0, 1], ius[3]: [0, 1]}  # only ius[1] resembles ref
    monkeypatch.setattr(eng, "_instance_ref_embeddings",
                        lambda iuids: np.array([emb[u] for u in iuids], np.float32))

    r = eng.propagate_refinement(ref, pid=f"class:{cid}", match_thresh=0.9)
    assert r["applied"] == 1 and r["skipped"] == 2 and seen["iuids"] == [ius[1]] and r["matched"] is True


def test_propagate_error_paths(tmp_path, monkeypatch):
    eng = _eng_with_instances(tmp_path, monkeypatch, n=2)
    ius = list(eng.state.meta)
    assert "error" in eng.propagate_refinement("nonexistent")
    # an instance with no recorded chain and no explicit ops -> clear error
    assert "error" in eng.propagate_refinement(ius[0], pid="class:none", ops=None)
