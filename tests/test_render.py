"""Executes the app's @gr.render bodies with a real (model-free) collection so a bad
component kwarg (e.g. gr.Image(show_download_button=...)) is caught by pytest — a plain
launch smoke does NOT exercise the render bodies (they only run when a partition/image is
selected). Run: pytest tools/curator/tests/test_render.py -q  (from repo root)
"""
from __future__ import annotations

import numpy as np


def _rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii")
    return r


def _engine(tmp_path):
    import cv2
    from tools.curator import ids
    from tools.curator.engine import CuratorEngine
    from tools.curator.state import InstanceMeta
    eng = CuratorEngine(tmp_path)
    eng.init_project({"images": {"root": str(tmp_path)}, "model": {"ckpt": "x"}, "features": {"model_features": ["decoder"]}})
    recs, dec, order, meta = [], [], [], {}
    for ii in range(2):
        p = tmp_path / f"im{ii}.png"
        cv2.imwrite(str(p), (np.random.default_rng(ii).random((512, 512, 3)) * 120 + 40).astype(np.uint8))
        for j in range(3):
            m = np.zeros((512, 512), np.uint8)
            cv2.line(m, (40, 40), (460, 400), 1, 6) if j == 0 else cv2.circle(m, (150 + 90 * j, 250), 40, 1, -1)
            mb = m > 0; ys, xs = np.where(mb); u = ids.new_uid(); row = len(recs)
            recs.append({"iuid": u, "row": row, "inst_id": row, "image_id": 1000 + ii, "H": 512, "W": 512, "score": 0.6,
                         "rle": _rle(mb), "file_name": str(p), "abs_path": str(p),
                         "cx": float(xs.mean() / 512), "cy": float(ys.mean() / 512), "bw": 0.3, "bh": 0.3,
                         "box_area": 0.09, "mask_area_frac": float(mb.mean())})
            dec.append(np.random.default_rng(row).normal(0, 1, 8)); order.append(u)
            meta[u] = InstanceMeta(iuid=u, batch_id="b", row=row, image_id=1000 + ii)
    eng.collection = {"records": recs, "n_images": 2, "feats": {"decoder": np.array(dec, np.float32)}}
    eng.state.order = order; eng.state.meta = meta; eng.state.coll_version = 1
    eng.store.save_collection(eng.collection); eng.save()
    return eng


def test_render_bodies_execute(tmp_path):
    from gradio.context import LocalContext
    from tools.curator import app
    eng = _engine(tmp_path)
    app.ENG = eng
    eng.cluster({"decoder": 1.0})
    pid = eng.partition_view()[0]["pid"]
    iid = eng.image_ids()[0]
    demo = app.build_app(str(tmp_path))

    def args_for(r):
        n = len(r.inputs)
        if n == 4:                                                                  # partition grid
            return (pid, True, "crop", 0)
        if n == 3:                                                                  # refine preview
            return ({"kind": "partition", "pid": pid}, [{"name": "dilate", "kw": {"k": 2, "max_contrast": 0.2}}], True)
        # two 2-input renderables: in-image grid (first input = Dropdown) vs classifier preview (first = State)
        if type(r.inputs[0]).__name__ == "State":
            return ([], 12)                                                         # classifier preview (preds, pred_n)
        return (iid, 0)                                                             # in-image grid (image_id, nonce)

    tok = LocalContext.blocks_config.set(demo.default_config)
    try:
        with demo:
            ran = 0
            for r in demo.renderables:
                r.apply(*args_for(r))                                               # raises on bad kwargs
                ran += 1
        assert ran == 4                                                             # partition, in-image, refine, classifier
    finally:
        LocalContext.blocks_config.reset(tok)
        app.ENG = None
