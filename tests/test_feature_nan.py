"""A feature method whose matrix contains NaN/inf must be flagged (so the classifier selector disables it)
and dropped from the classifier spec server-side (so it can't break sklearn). Model-free.
Run: pytest tools/curator/tests/test_feature_nan.py -q
"""
from __future__ import annotations

import numpy as np


def _rle(h=32, w=32):
    import cv2
    from pycocotools import mask as mu
    m = np.zeros((h, w), np.uint8); cv2.circle(m, (16, 16), 6, 1, -1)
    r = mu.encode(np.asfortranarray(m)); r["counts"] = r["counts"].decode("ascii")
    return r


def _fb(files, dim=8):
    from tools.curator import ids
    recs = [{"iuid": ids.new_uid(), "batch_id": "b", "abs_path": f, "file_name": f,
             "image_id": abs(hash(f)) % 1000000, "score": 0.9, "rle": _rle(), "H": 32, "W": 32}
            for f in files]
    return {"records": recs, "feats": {"decoder": np.zeros((len(files), dim), np.float32)}, "n_images": len(files)}


def _eng_with_nan_feature(tmp_path, monkeypatch, n=6):
    from tools.curator import collect as _co
    from tools.curator.engine import CuratorEngine
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"},
                      "features": {"model_features": ["decoder"]}})
    eng.collection = {"records": [], "n_images": 0, "feats": {"decoder": np.zeros((0, 8), np.float32)}}
    monkeypatch.setattr(eng, "_ensure_model", lambda: (None, None, None))
    monkeypatch.setattr(_co, "collect_batch",
                        lambda m, c, d, fs, *, score_thresh, feature_cfg: _fb(fs))
    eng.ingest_paths([f"/x/{i}.png" for i in range(n)])
    rows = eng.collection["feats"]["decoder"].shape[0]
    shape = np.ones((rows, 3), np.float32); shape[2, 1] = np.nan            # a NaN-tainted feature method
    eng.collection["feats"]["shape"] = shape
    eng.state.coll_version += 1                                             # bust the nan-methods cache
    return eng


def test_nan_feature_detected_and_excluded(tmp_path, monkeypatch):
    eng = _eng_with_nan_feature(tmp_path, monkeypatch)
    assert eng.feature_nan_methods() == {"shape"}                          # only the tainted method
    health = eng.feature_health()
    assert health["nan"] == ["shape"] and set(health["features"]) >= {"decoder", "shape"}

    clean, dropped = eng._clf_spec_clean({"decoder": 1.0, "shape": 1.0})    # NaN method dropped from the spec
    assert "decoder" in clean and "shape" not in clean and dropped == ["shape"]

    inf = eng.collection["feats"]["shape"]; inf[0, 0] = np.inf              # inf counts too
    eng.state.coll_version += 1
    assert "shape" in eng.feature_nan_methods()


def test_train_classifier_drops_nan_and_errors_if_only_nan(tmp_path, monkeypatch):
    eng = _eng_with_nan_feature(tmp_path, monkeypatch)
    # selecting ONLY the NaN feature -> clear error mentioning NaN, no sklearn crash
    rep = eng.train_classifier({"shape": 1.0})
    assert "error" in rep and "NaN" in rep["error"]
