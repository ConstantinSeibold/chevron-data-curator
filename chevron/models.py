"""`chevron models`: what this machine can run, what is already downloaded, and fetching the rest.

Every model Chevron uses downloads on first use, which is fine on a fast line and miserable when the
first ingest sits on a 2.5 GB download. This lists the same backends and embedding models the Set up
pane does, says which are usable here and which have their weights cached, and downloads the ones
you name ahead of time, each through the loader Chevron itself uses, so the cache it fills is the
one Chevron reads.

    chevron models                              # status
    chevron models --download samhq_auto raddino
    chevron models --download all               # every usable model that downloads its weights
"""
from __future__ import annotations

import argparse
import sys

# Backends with nothing to fetch: they read a file, the images themselves, or a project checkpoint.
_NO_WEIGHTS = {"coco": "reads a COCO file — no model", "whole_image": "one item per image — no model",
               "qseg": "uses the project's own trained checkpoint"}


def _hf_cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    p = try_to_load_from_cache(repo, "config.json")
    return isinstance(p, str)


def _hf_id(obj) -> str | None:
    return getattr(obj, "hf_id", None) or getattr(obj, "HF_ID", None) or getattr(obj, "model_id", None)


def _torchvision_cached() -> bool:
    try:
        import os
        import torch
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
        url = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.url
        return os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(url)))
    except Exception:
        return False


def _weights(kind: str, name: str, obj) -> tuple[str, bool]:
    """(where the weights stand, whether `--download` can fetch them)."""
    from . import refine as rf
    if name in _NO_WEIGHTS:
        return _NO_WEIGHTS[name], False
    if name in ("sam_auto", "samhq_auto"):
        ckpt, _ = rf.find_sam_checkpoint(family="samhq" if name == "samhq_auto" else "sam")
        return (f"cached  {ckpt}", False) if ckpt else ("not downloaded", True)
    if name == "medsam_box":
        return "manual: point CURATOR_MEDSAM_CKPT at the .pth (no auto-download)", False
    if name == "torchvision_maskrcnn":
        return ("cached", False) if _torchvision_cached() else ("not downloaded", True)
    hf = _hf_id(obj)
    if hf:
        return (f"cached  {hf}", False) if _hf_cached(hf) else (f"not downloaded  {hf}", True)
    return "downloads on first use", True


def rows() -> list[dict]:
    from .backends import base as bb
    from .extractors import base as eb
    out = []
    for kind, reg, listing in (("masks", bb._REGISTRY, bb.list_backends()),
                               ("embedding", eb._REGISTRY, eb.list_extractors())):
        for r in listing:
            try:
                obj = reg[r["name"]]()
            except Exception:
                obj = None
            weights, fetchable = _weights(kind, r["name"], obj) if r["available"] else ("", False)
            out.append({"kind": kind, "name": r["name"], "label": r["label"], "usable": r["available"],
                        "why": r["detail"], "requires": r.get("requires", ""), "weights": weights,
                        "fetchable": fetchable, "obj": obj})
    return out


def _download(r: dict) -> None:
    from . import refine as rf
    name, obj = r["name"], r["obj"]

    def _bar(done, total):
        mb = lambda b: f"{b / 1e6:.0f} MB"
        sys.stdout.write(f"\r  {mb(done)} / {mb(total)}" if total else f"\r  {mb(done)}")
        sys.stdout.flush()

    if name == "sam_auto":
        print(f"  -> {rf.ensure_sam_checkpoint(progress=_bar)}")
    elif name == "samhq_auto":
        print(f"  -> {rf.ensure_samhq_checkpoint(progress=_bar)}")
    else:
        obj._load()                       # the component's own loader: fetches exactly what it reads


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="chevron models",
                                 description="List the models this machine can run and download their weights.")
    ap.add_argument("--download", nargs="+", metavar="NAME",
                    help="names from the list (e.g. samhq_auto raddino), or 'all' for every usable one")
    args = ap.parse_args(argv)

    from .device import describe
    d = describe()
    print(f"Device: {d['device']}" + (f"  (torch {d['torch']})" if d["torch"] else "  (torch not installed)")
          + ("  — set by CHEVRON_DEVICE" if d.get("override") else ""))
    rs = rows()
    if not args.download:
        for kind, title in (("masks", "Mask backends"), ("embedding", "Embedding models")):
            print(f"\n{title}")
            for r in (x for x in rs if x["kind"] == kind):
                if r["usable"]:
                    gated = "  [gated: accept the licence on HF, then `hf auth login`]" \
                        if r["fetchable"] and "gated" in r["requires"] else ""
                    print(f"  ✓ {r['name']:<22} {r['weights']}{gated}")
                else:
                    print(f"  ✗ {r['name']:<22} {r['why']}" + (f"  [{r['requires']}]" if r["requires"] else ""))
        todo = [r["name"] for r in rs if r["usable"] and r["fetchable"]]
        if todo:
            print(f"\nNot downloaded yet: {', '.join(todo)}\n"
                  f"Fetch the ones you will use:  chevron models --download {todo[0]} ...   (or: --download all)")
        return 0

    by = {r["name"]: r for r in rs}
    names = [r["name"] for r in rs if r["usable"] and r["fetchable"]] if args.download == ["all"] else args.download
    failed = 0
    for n in names:
        r = by.get(n)
        if r is None:
            print(f"✗ {n}: unknown — run `chevron models` for the names"); failed += 1; continue
        if not r["usable"]:
            print(f"✗ {n}: not usable here — {r['why']}"); failed += 1; continue
        if not r["fetchable"]:
            print(f"· {n}: {r['weights']}"); continue
        print(f"↓ {n}")
        try:
            _download(r)
            print(f"\n✓ {n}")
        except Exception as e:                    # gated repo without `hf auth login`, offline, disk full
            print(f"\n✗ {n}: {type(e).__name__}: {e}"); failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
