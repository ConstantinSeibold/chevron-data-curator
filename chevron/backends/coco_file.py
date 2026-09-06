"""Masks from a COCO file — the one proposer that needs no ML stack at all.

Distinct from `engine.import_proposals_coco`, which adds a SECOND tagged source to a project that
already has instances, matching by image basename. This one BOOTSTRAPS: it reads the COCO's own
`images` list, resolves each file against an image root, and produces the project's first collection.

Which makes it the answer to "I have masks from somewhere else, can I just curate them" — and the
reason a fresh clone is useful without installing torch.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .base import Proposal, register


class CocoFileBackend:
    name = "coco"
    label = "COCO file (masks you already have — no model needed)"
    requires = "nothing"

    def __init__(self):
        self.path: str | None = None
        self.image_root: str | None = None
        self._by_file: dict[str, list] = {}

    def available(self) -> tuple[bool, str]:
        return True, "reads a COCO json; no model or GPU"

    # ---- the framework drives per-image; this backend is per-file, so it indexes up front ----
    def load(self, path: str, image_root: str | None = None) -> list[str]:
        """Index the COCO and return the absolute image paths it references (in file order)."""
        with open(path) as f:
            coco = json.load(f)
        self.path, self.image_root = path, image_root
        root = image_root or os.path.dirname(os.path.abspath(path))
        by_id = {im["id"]: im for im in coco.get("images", [])}
        anns: dict[int, list] = {}
        for a in coco.get("annotations", []):
            anns.setdefault(a.get("image_id"), []).append(a)

        paths, self._by_file = [], {}
        for iid, im in by_id.items():
            fn = str(im.get("file_name") or "")
            if not fn:
                continue
            p = fn if os.path.isabs(fn) else os.path.join(root, fn)
            if not os.path.isfile(p):
                p2 = os.path.join(root, os.path.basename(fn))     # flat image roots are common
                if not os.path.isfile(p2):
                    continue
                p = p2
            got = anns.get(iid, [])
            if not got:
                continue
            self._by_file[os.path.abspath(p)] = got
            paths.append(os.path.abspath(p))
        return paths

    def propose(self, image_rgb: np.ndarray, path: str | None = None, **cfg) -> list[Proposal]:
        """Annotations for the image at `path`, decoded to masks. Keyed by path rather than by call
        order, so it does not care how the framework iterates."""
        from ..engine import CuratorEngine
        H, W = image_rgb.shape[:2]
        out = []
        for ann in self._by_file.get(os.path.abspath(str(path)), []):
            m = CuratorEngine._decode_ann_mask(ann, H, W)   # RLE | polygons | bbox-only
            if m is None or not m.any():
                continue
            out.append(Proposal(mask=m, score=float(ann.get("score", 1.0))))
        return out


def build_coco_collection(path: str, *, image_root: str | None = None, batch_id: str = "coco",
                          score_thresh: float = 0.0, progress=None) -> tuple[dict, dict]:
    """Bootstrap a collection straight from a COCO. Returns (collection, report)."""
    from .base import build_collection

    be = CocoFileBackend()
    paths = be.load(path, image_root)
    if not paths:
        return ({"records": [], "n_images": 0, "feats": {}},
                {"error": "no COCO images could be resolved on disk — check the image root",
                 "path": path, "image_root": image_root})
    col = build_collection(be, paths, score_thresh=score_thresh, batch_id=batch_id, progress=progress)
    return col, {"path": path, "n_images": col["n_images"], "n_instances": len(col["records"])}


register("coco", CocoFileBackend)
