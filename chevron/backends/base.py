"""Proposal backends: where class-agnostic instance masks come from.

Chevron curates proposals; it does not care which model produced them. A backend's whole job is
`propose(image) -> [Proposal]`. Everything after that — ids, records, geometry features, NMS, the
row-alignment invariant — is done once here rather than five times in five backends.

Labels are deliberately DISCARDED. A COCO-pretrained detector's 80 classes are not the label space
you are curating; the point is the masks, and the human supplies the taxonomy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


@dataclass
class Proposal:
    """One class-agnostic instance: a boolean mask over the image, and how sure the proposer is."""
    mask: np.ndarray                      # bool (H, W)
    score: float = 1.0


@runtime_checkable
class ProposalBackend(Protocol):
    name: str
    label: str
    requires: str                         # human-readable install hint, shown when unavailable

    def available(self) -> tuple[bool, str]:
        """(usable_here, why_not). Never raises — the UI lists every backend with its status."""

    def propose(self, image_rgb: np.ndarray, **cfg) -> list[Proposal]:
        """Masks for ONE image. `cfg` carries `path=` (the file it came from) plus backend knobs;
        accept `**cfg` and ignore what you do not use."""

    # OPTIONAL — `prepare(*, progress=None, stage=None, **cfg)`. A backend whose first `propose`
    # would download or load weights should define it: the caller runs it before the image loop and
    # gives it a phase of its own, so a cold cache reads "184 MB / 379 MB" instead of an image
    # counter frozen at 0/80 that the UI is right to call stalled. `progress(done_bytes,
    # total_bytes)`; `stage(text, stall_after_seconds)` names the current step and says how long it
    # may legitimately go quiet. Take **cfg so a caller passing neither hook still works, and omit
    # the method entirely when there is nothing to fetch.


# --------------------------------------------------------------------------- registry
_REGISTRY: dict[str, Callable[[], ProposalBackend]] = {}


def register(name: str, factory: Callable[[], ProposalBackend]) -> None:
    _REGISTRY[name] = factory


def get(name: str) -> ProposalBackend:
    if name not in _REGISTRY:
        raise KeyError(f"unknown proposal backend {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_backends() -> list[dict]:
    """Every backend with its availability — so the UI can show what is installable, not just what
    is installed. Construction failures are reported, never raised."""
    out = []
    for name in sorted(_REGISTRY):
        try:
            b = _REGISTRY[name]()
            ok, why = b.available()
            out.append({"name": name, "label": b.label, "available": bool(ok),
                        "requires": b.requires, "detail": why})
        except Exception as e:                       # a broken optional import must not hide the list
            out.append({"name": name, "label": name, "available": False,
                        "requires": "", "detail": f"{type(e).__name__}: {e}"})
    return out


# --------------------------------------------------------------------------- shared assembly
# The `coords` feature block is built from these record fields, in this exact column order, by
# core.collection._stack_feats. Backend proposals must use the SAME convention as the qseg path or
# `coords` would silently mean different things for different sources in one project — hence the
# canonical _coord_feats (box CENTRE, not mask centroid) rather than a second implementation.
COORD_COLS = ("cx", "cy", "bw", "bh", "box_area", "mask_area_frac")


def _geometry(mask: np.ndarray, H: int, W: int) -> dict:
    from ..core.collection import _coord_feats
    ys, xs = np.where(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    return {**_coord_feats(box, mask, H, W), "_box": box}


def build_collection(backend: ProposalBackend, image_paths, *, score_thresh: float = 0.0,
                     min_area_frac: float = 0.0, batch_id: str = "b",
                     progress: Callable[[int, int, str], None] | None = None,
                     **cfg) -> dict:
    """Run `backend` over the images and assemble a Chevron collection.

    Produces only the model-INDEPENDENT features — `shapecoord` (mask geometry) and `coords` (box
    position/size). Detector-internal features (decoder/maskpool/roialign) exist only for the qseg
    backend; a project mixing sources zero-fills them, exactly as the COCO import already does, so
    the cross-source space stays finite and the global NaN check never trips.
    """
    import cv2
    from pycocotools import mask as mu

    from .. import collect as _co
    from .. import ids as _ids

    paths = [str(p) for p in image_paths]
    records: list[dict] = []
    feats: dict[str, list] = {"shapecoord": [], "coords": []}
    n_images = 0
    for i, path in enumerate(paths):
        if progress:
            progress(i, len(paths), os.path.basename(path))
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        n_images += 1
        image_id = _co.path_image_id(path)
        for p in backend.propose(rgb, path=path, **cfg):
            m = np.asarray(p.mask, bool)
            if m.shape != (H, W) or not m.any():
                continue
            if float(p.score) < score_thresh or m.mean() < min_area_frac:
                continue
            g = _geometry(m, H, W)
            g.pop("_box")
            rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            records.append({"iuid": _ids.new_uid(), "row": 0, "inst_id": 0, "image_id": image_id,
                            "H": H, "W": W, "score": float(p.score), "rle": rle,
                            "file_name": path, "abs_path": path, "batch_id": batch_id, **g})
            feats["shapecoord"].append(_co.shapecoord_vector(m))
            feats["coords"].append([g[c] for c in COORD_COLS])

    col = {"records": records, "n_images": n_images,
           "feats": {k: (np.asarray(v, np.float32) if v else np.zeros((0, 1), np.float32))
                     for k, v in feats.items()}}
    for i, r in enumerate(col["records"]):
        r["row"] = r["inst_id"] = i
    return col
