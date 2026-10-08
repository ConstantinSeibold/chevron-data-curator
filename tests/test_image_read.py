"""A failed image read shows black but is NOT cached: once the file reads again, crops are real again
(it used to stay black for the rest of the session), and the browser is told not to keep the black one.
Run: pytest tests/test_image_read.py -q  (from repo root)
"""
from __future__ import annotations

import shutil

from test_merge_rec import _engine


def test_black_stand_in_heals_once_the_image_reads(tmp_path):
    from fastapi.testclient import TestClient
    from chevron import engine as E
    from chevron.server import create_app
    E.clear_image_caches()
    eng, o = _engine(tmp_path, n_img=1, per_img=2)
    img = tmp_path / "im0.png"
    shutil.move(img, tmp_path / "away.png")                       # e.g. a volume not mounted yet
    assert eng.crop(o[0], mask_overlay=False).max() == 0 and not eng.image_ok(o[0])
    c = TestClient(create_app(engine=eng))
    assert c.get("/api/crop", params={"iuid": o[0]}).headers["cache-control"] == "no-store"
    shutil.move(tmp_path / "away.png", img)
    assert eng.crop(o[0], mask_overlay=False).max() > 0 and eng.image_ok(o[0])
    assert c.get("/api/crop", params={"iuid": o[0]}).headers["cache-control"].startswith("max-age")
