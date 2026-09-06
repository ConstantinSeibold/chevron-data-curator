"""Model-free tests: shape-coordinate features + additive concat + sampling.
Run: pytest tests/test_collect.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np
import pytest

from chevron import collect, sample


def _line_mask(h=64, w=64):
    import cv2
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, (5, 10), (58, 55), 1, 2)
    return m.astype(bool)


def _blob_mask(h=64, w=64):
    import cv2
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (32, 32), 14, 1, -1)
    return m.astype(bool)


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


def test_shapecoord_shapes_and_discriminative():
    line, blob = _line_mask(), _blob_mask()
    vl, vb = collect.shapecoord_vector(line), collect.shapecoord_vector(blob)
    assert vl.shape == vb.shape == (5 + 16 + 8,)
    assert np.isfinite(vl).all() and np.isfinite(vb).all()
    # a line is far more elongated than a blob -> pca_ratio (minor/major) much smaller
    assert vl[2] < vb[2]
    # empty mask -> zeros, no crash
    assert collect.shapecoord_vector(np.zeros((32, 32), bool)).shape == (29,)


def test_attach_shapecoord():
    recs = [{"rle": _rle(_line_mask())}, {"rle": _rle(_blob_mask())}]
    col = {"records": recs, "feats": {}, "n_images": 2}
    collect.attach_shapecoord(col)
    assert col["feats"]["shapecoord"].shape == (2, 29)
    assert len(col["feats"]["_shapecoord_cols"]) == 29
    assert "f_shapecoord" in recs[0]


def _fake_col(n, dim=4, off=0.0):
    recs = [{"iuid": f"u{off+i}", "score": 0.5, "rle": _rle(_blob_mask())} for i in range(n)]
    feats = {"decoder": (np.arange(n * dim).reshape(n, dim) + off).astype(np.float32),
             "_shape_cols": ["a", "b"]}
    return {"records": recs, "feats": feats, "n_images": n}


def test_concat_collections():
    a, b = _fake_col(3, off=0), _fake_col(2, off=100)
    m = collect.concat_collections(a, b)
    assert len(m["records"]) == 5
    assert m["feats"]["decoder"].shape == (5, 4)
    assert m["feats"]["_shape_cols"] == ["a", "b"]            # column-name list carried
    assert [r["row"] for r in m["records"]] == [0, 1, 2, 3, 4]  # reindexed
    # from-empty
    m0 = collect.concat_collections(None, a)
    assert len(m0["records"]) == 3
    # method mismatch -> raises
    bad = _fake_col(2); bad["feats"]["extra"] = np.zeros((2, 2), np.float32)
    with pytest.raises(ValueError):
        collect.concat_collections(a, bad)


def test_path_image_id_stable():
    assert collect.path_image_id("/x/y.png") == collect.path_image_id("/x/y.png")
    assert collect.path_image_id("/x/y.png") != collect.path_image_id("/x/z.png")


def test_sampling(tmp_path):
    for i in range(10):
        (tmp_path / f"img{i}.png").write_bytes(b"x")
    files = sample.list_images(tmp_path)
    assert len(files) == 10
    s = sample.sample_random(files, 4, exclude={files[0]}, seed=0)
    assert len(s) == 4 and files[0] not in s
    low = sample.pick_lowest([(f, i) for i, f in enumerate(files)], 3)
    assert low == files[:3]
