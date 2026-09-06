"""The extractor registry — the "model dropdown".

An extractor's only job is `grid_batch(images) -> (B, C, g, g)`. The pooling that turns a grid into
per-instance features is shared, so it is tested here with a stub encoder: no weights downloaded, no
GPU, and the bookkeeping that actually breaks (row alignment, feature dims, the shared-space
projection hook) is what gets exercised.

Run: pytest tests/test_extractors.py -q
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pycocotools import mask as mu

from chevron import collect as _co
from chevron import ids
from chevron.engine import CuratorEngine
from chevron.extractors import base as E
from chevron.extractors import list_extractors
from chevron.server import create_app
from chevron.state import InstanceMeta

DIM, GRID = 12, 4


class _StubEncoder:
    """A deterministic stand-in for a ViT: a (C, g, g) grid whose values encode position, so a
    mask-pooled vector genuinely depends on WHERE the mask is."""
    name, label, modality, space, requires = "stub", "Stub", "image", None, "nothing"

    def available(self):
        return True, "ok"

    def grid_batch(self, images_rgb):
        import torch
        b = len(images_rgb)
        g = torch.arange(GRID * GRID, dtype=torch.float32).reshape(1, 1, GRID, GRID)
        return g.repeat(b, DIM, 1, 1) + torch.arange(DIM).reshape(1, DIM, 1, 1)


def _project(tmp_path, n=4):
    root = tmp_path / "img"; root.mkdir(parents=True)
    p = root / "a.png"
    cv2.imwrite(str(p), (np.random.default_rng(0).random((32, 32, 3)) * 200).astype(np.uint8))
    eng = CuratorEngine(tmp_path / "proj"); eng.init_project({})
    recs, order, meta = [], [], {}
    for i in range(n):
        m = np.zeros((32, 32), np.uint8)
        cv2.circle(m, (6 + 7 * i, 16), 4, 1, -1)              # different positions -> different pools
        rle = mu.encode(np.asfortranarray(m)); rle["counts"] = rle["counts"].decode()
        u = ids.new_uid()
        recs.append({"iuid": u, "row": i, "inst_id": i, "image_id": 1, "H": 32, "W": 32, "score": .9,
                     "rle": rle, "file_name": str(p), "abs_path": str(p), "batch_id": "b",
                     "cx": .5, "cy": .5, "bw": .3, "bh": .3, "box_area": .09, "mask_area_frac": .05})
        order.append(u); meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=1)
    eng.collection = {"records": recs, "n_images": 1,
                      "feats": {"shapecoord": np.zeros((n, 29), np.float32)}}
    eng.state.order, eng.state.meta, eng.state.coll_version = order, meta, 1
    return eng


# --------------------------------------------------------------------------- registry
def test_dropdown_lists_every_extractor_without_importing_torch():
    got = {e["name"]: e for e in list_extractors()}
    assert {"raddino", "dinov2", "clip", "siglip2"} <= set(got)
    for e in got.values():
        assert isinstance(e["available"], bool) and e["label"] and e["modality"] == "image"


def test_shared_space_is_declared_only_where_it_exists():
    """A `space` tag means "comparable with anything else carrying it" — the basis for a text query
    landing among image instances. RAD-DINO and DINOv2 have no text tower, so no tag."""
    got = {e["name"]: e for e in list_extractors()}
    assert got["clip"]["space"] == "clip" and got["siglip2"]["space"] == "siglip"
    assert got["raddino"]["space"] is None and got["dinov2"]["space"] is None


def test_unknown_extractor_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown extractor"):
        E.get("no_such_model")


# --------------------------------------------------------------------------- pooling
def test_pooling_writes_a_row_aligned_feature_column(tmp_path):
    pytest.importorskip("torch")
    eng = _project(tmp_path, n=4)
    _co.pool_by_path(eng.collection, _StubEncoder(), "stub")
    f = eng.collection["feats"]["stub"]
    assert f.shape == (4, DIM)
    assert np.isfinite(f).all(), "a NaN column would disable the feature globally"
    eng.state.assert_aligned(f.shape[0])
    # masks at different positions must pool to different vectors, or pooling is not using the mask
    assert len({tuple(np.round(r, 3)) for r in f}) > 1
    eng.close()


def test_bbox_pooling_skips_the_rle_decode_and_still_aligns(tmp_path):
    pytest.importorskip("torch")
    eng = _project(tmp_path, n=3)
    _co.pool_by_path(eng.collection, _StubEncoder(), "stub", pool="bbox")
    assert eng.collection["feats"]["stub"].shape == (3, DIM)
    eng.close()


def test_shared_space_extractors_get_their_projection_applied(tmp_path):
    """A CLIP-style extractor maps the pooled region into the image-text space; without that a text
    query and an instance would live in different spaces and could not be compared."""
    pytest.importorskip("torch")
    import torch

    class _Projected(_StubEncoder):
        space = "clip"
        def project_pooled(self, pooled):
            return torch.nn.functional.normalize(pooled.float(), dim=-1)   # unit-norm, like CLIP

    eng = _project(tmp_path, n=3)
    _co.pool_by_path(eng.collection, _Projected(), "clip")
    f = eng.collection["feats"]["clip"]
    assert f.shape == (3, DIM)
    assert np.allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-5), "projection was not applied"
    eng.close()


# --------------------------------------------------------------------------- engine + API
def test_compute_features_adds_a_selectable_column(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    eng = _project(tmp_path, n=3)
    monkeypatch.setattr(E, "get", lambda name: _StubEncoder())
    rep = eng.compute_features("stub")
    assert rep["ok"] and rep["n"] == 3
    assert "stub" in rep["available"] == eng.available_features()
    # a second call is a no-op unless forced — extraction is expensive
    assert "already present" in eng.compute_features("stub")["msg"]
    assert eng.compute_features("stub", force=True)["n"] == 3
    eng.close()


def test_compute_features_needs_a_collection(tmp_path):
    eng = CuratorEngine(tmp_path / "p"); eng.init_project({})
    assert "error" in eng.compute_features("raddino")
    eng.close()


def test_raddino_is_now_just_one_registry_entry(tmp_path, monkeypatch):
    """The old entry point must keep working — it is an alias now."""
    pytest.importorskip("torch")
    eng = _project(tmp_path, n=2)
    seen = {}
    monkeypatch.setattr(E, "get", lambda name: seen.setdefault("name", name) and _StubEncoder()
                        or _StubEncoder())
    eng.compute_raddino()
    assert seen["name"] == "raddino"
    eng.close()


def test_api_exposes_the_dropdown(tmp_path):
    eng = _project(tmp_path, n=2)
    c = TestClient(create_app(engine=eng))
    r = c.get("/api/extractors").json()
    assert {"raddino", "dinov2", "clip", "siglip2"} <= {e["name"] for e in r["extractors"]}
    assert "shapecoord" in r["present"]
    assert c.post("/api/compute_features", json={"extractor": "nope"}).status_code == 400
    eng.close()


def test_text_embedding_survives_the_transformers_api_change():
    """transformers 4.x returned a bare tensor from get_text_features; 5.x returns a
    BaseModelOutputWithPooling. Both must work — a version bump silently breaking text queries is
    exactly the kind of thing that is only noticed much later."""
    import torch

    from chevron.extractors.registry import ProjectedVisionExtractor as PVE

    bare = torch.ones(2, 512)
    assert PVE._as_tensor(bare) is bare

    class _Out:                                   # the 5.x shape
        pooler_output = torch.ones(2, 512) * 3
    assert torch.equal(PVE._as_tensor(_Out()), torch.ones(2, 512) * 3)

    class _Hidden:                                # a last-resort shape: take the first token
        pooler_output = None
        text_embeds = None
        last_hidden_state = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    assert PVE._as_tensor(_Hidden()).shape == (2, 4)

    class _Nothing:
        pass
    with pytest.raises(TypeError, match="cannot read a text embedding"):
        PVE._as_tensor(_Nothing())


# --------------------------------------------------------------------------- patch-grid derivation
def _stub(patch_size, drop_prefix=1):
    """An extractor whose config reports a patch size, without loading any weights."""
    from chevron.extractors.base import HFPatchGridExtractor

    class _Cfg:
        pass
    cfg = _Cfg(); cfg.patch_size = patch_size
    e = HFPatchGridExtractor("stub/model", "stub", "Stub", drop_prefix=drop_prefix)

    class _M:
        config = cfg
    e.model = _M()
    return e


def test_the_patch_grid_comes_from_the_patch_size_not_a_fixed_prefix():
    """Models disagree about how many tokens sit in front of the patches: DINOv2 prepends CLS alone,
    DINOv3 prepends CLS plus four registers. Both are real numbers observed from their HF configs."""
    assert _stub(14)._grid_shape((224, 224), 257) == (16, 16)      # DINOv2-base: 1 + 256
    assert _stub(16)._grid_shape((224, 224), 201) == (14, 14)      # DINOv3/16:   5 + 196
    assert _stub(16)._grid_shape((512, 512), 1029) == (32, 32)     # same, larger input


def test_assuming_one_prefix_token_would_have_broken_dinov3():
    """The regression this guards. With a hardcoded drop_prefix=1, DINOv3's 201 tokens leave 200,
    `round(sqrt(200))` is 14, and reshaping 200 tokens into 14x14=196 is an error — so the wrong
    answer here is a crash at best, and a silently mis-shaped grid at input sizes where the leftover
    count happens to be square."""
    import math
    leftover = 201 - 1
    assert int(round(math.sqrt(leftover))) ** 2 != leftover


def test_a_model_with_no_readable_patch_size_falls_back_to_drop_prefix():
    from chevron.extractors.base import HFPatchGridExtractor
    e = HFPatchGridExtractor("stub/model", "stub", "Stub", drop_prefix=1)

    class _M:
        config = None
    e.model = _M()
    assert e._grid_shape((224, 224), 257) == (16, 16)


def test_dinov3_is_offered_in_the_dropdown():
    from chevron.extractors import list_extractors
    by = {e["name"]: e for e in list_extractors()}
    assert {"dinov3", "dinov3b", "dinov3l"} <= set(by)
    assert by["dinov3"]["label"].startswith("DINOv3")
    # the weights are gated, so the picker has to say what to do about it
    assert "hf auth login" in by["dinov3"]["requires"]
