"""`chevron models`: lists what this machine can run + what is cached, and prefetches named weights.
Run: pytest tests/test_models_cli.py -q  (from repo root)
"""
from __future__ import annotations

from chevron import models, refine as rf


def test_status_lists_backends_and_embeddings(capsys, monkeypatch):
    monkeypatch.setattr(rf, "find_sam_checkpoint", lambda *a, **k: (None, None))
    assert models.main([]) == 0
    out = capsys.readouterr().out
    assert "Mask backends" in out and "Embedding models" in out
    assert "coco" in out and "raddino" in out


def test_download_fetches_through_chevrons_own_loader(capsys, monkeypatch):
    got = []
    monkeypatch.setattr(rf, "samhq_available", lambda: True)
    monkeypatch.setattr(rf, "find_sam_checkpoint", lambda *a, **k: (None, None))
    monkeypatch.setattr(rf, "ensure_samhq_checkpoint", lambda progress=None: got.append("hq") or "/x/sam_hq_vit_b.pth")
    assert models.main(["--download", "samhq_auto", "coco"]) == 0
    out = capsys.readouterr().out
    assert got == ["hq"] and "✓ samhq_auto" in out and "no model" in out


def test_unknown_or_unusable_names_fail(capsys):
    assert models.main(["--download", "nope"]) == 1
    assert "unknown" in capsys.readouterr().out
