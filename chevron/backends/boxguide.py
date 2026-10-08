"""Box-guided re-masking: new masks for boxes you already have.

The boxes come from a COCO file (a new project) or from the project's own instances (an existing one);
any registered backend then supplies the mask inside each box:

- a PROMPTABLE backend (SAM / SAM-HQ / MedSAM) defines `propose_boxes(image, boxes_xyxy)` and is
  handed the boxes directly as prompts;
- every other backend runs as usual on the whole image, and each box keeps the prediction that best
  matches it — one-to-one, so two neighbouring boxes cannot both claim the same object — clipped to
  the box plus some wiggle room.

A box nothing matches yields None, and the caller keeps the mask it already had: re-masking is allowed
to fail to improve a mask, never to lose one.
"""
from __future__ import annotations

import numpy as np

from .base import Proposal


def xywh_to_xyxy(b) -> np.ndarray:
    x, y, w, h = (float(v) for v in b[:4])
    return np.array([x, y, x + w, y + h], float)


def mask_box(mask: np.ndarray) -> np.ndarray | None:
    """Tight [x0, y0, x1, y1) of a mask, or None when it is empty."""
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def pad_box(box, H: int, W: int, pad: float) -> tuple[int, int, int, int]:
    """`box` grown by `pad` × its own width/height on every side (at least 4 px), clamped to the image."""
    x0, y0, x1, y1 = (float(v) for v in box)
    px, py = max(4.0, pad * (x1 - x0)), max(4.0, pad * (y1 - y0))
    return (max(0, int(np.floor(x0 - px))), max(0, int(np.floor(y0 - py))),
            min(W, int(np.ceil(x1 + px))), min(H, int(np.ceil(y1 + py))))


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _clip(mask: np.ndarray, win) -> np.ndarray:
    x0, y0, x1, y1 = win
    out = np.zeros_like(mask, bool)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def is_promptable(backend) -> bool:
    return callable(getattr(backend, "propose_boxes", None))


def propose_in_boxes(backend, image_rgb: np.ndarray, boxes_xyxy, *, pad: float = 0.1,
                     min_iou: float = 0.3, path: str | None = None, **cfg) -> list[Proposal | None]:
    """Exactly one entry per box, in box order: the new mask for that box, or None if there is none."""
    boxes = [np.asarray(b, float) for b in boxes_xyxy]
    if not boxes:
        return []
    H, W = image_rgb.shape[:2]
    wins = [pad_box(b, H, W, pad) for b in boxes]

    if is_promptable(backend):
        got = backend.propose_boxes(image_rgb, boxes, path=path, **cfg)
        out: list[Proposal | None] = []
        for g, win in zip(got, wins):
            if g is None:
                out.append(None)
                continue
            m = _clip(np.asarray(g.mask, bool), win)
            out.append(Proposal(mask=m, score=float(g.score)) if m.any() else None)
        return out

    preds = []
    for p in backend.propose(image_rgb, path=path, **cfg):
        m = np.asarray(p.mask, bool)
        if m.shape != (H, W):
            continue
        pb = mask_box(m)
        if pb is not None:
            preds.append((m, float(p.score), pb, float(m.sum())))

    # every (box, prediction) pair worth considering, best first: how well the boxes agree, times how
    # much of the prediction actually falls inside the box's wiggle room
    pairs = []
    for bi, (b, win) in enumerate(zip(boxes, wins)):
        x0, y0, x1, y1 = win
        for pi, (m, _s, pb, area) in enumerate(preds):
            iou = box_iou(b, pb)
            if iou < min_iou:
                continue
            inside = float(m[y0:y1, x0:x1].sum()) / area
            pairs.append((iou * inside, bi, pi))
    pairs.sort(key=lambda t: -t[0])

    out = [None] * len(boxes)
    used: set[int] = set()
    for _q, bi, pi in pairs:
        if out[bi] is not None or pi in used:
            continue
        m = _clip(preds[pi][0], wins[bi])
        if m.any():
            out[bi] = Proposal(mask=m, score=preds[pi][1])
            used.add(pi)
    return out


class BoxGuidedCocoBackend:
    """A COCO file's annotations, re-masked inside their boxes by `refiner`.

    Behaves as an ordinary backend towards `build_collection`, so a new project gets progress,
    geometry and features exactly as any other ingest does. Every annotation yields one proposal — the
    refiner's mask when it found one, the file's own mask otherwise — tagged with `remask` so the two
    can be told apart afterwards.
    """
    name = "coco_boxguided"
    requires = "nothing"

    def __init__(self, coco_backend, refiner, *, refiner_name: str, pad: float = 0.1,
                 min_iou: float = 0.3):
        self.coco, self.refiner, self.refiner_name = coco_backend, refiner, refiner_name
        self.pad, self.min_iou = float(pad), float(min_iou)
        self.label = f"COCO boxes re-masked by {refiner_name}"

    def available(self) -> tuple[bool, str]:
        return self.refiner.available()

    def prepare(self, **kw) -> None:
        prep = getattr(self.refiner, "prepare", None)
        if callable(prep):
            prep(**kw)

    def propose(self, image_rgb: np.ndarray, path: str | None = None, **cfg) -> list[Proposal]:
        anns = self.coco.annotations(image_rgb, path)
        if not anns:
            return []
        boxes = []
        for ann, m in anns:
            b = ann.get("bbox")
            boxes.append(xywh_to_xyxy(b) if b and len(b) >= 4 and b[2] > 0 and b[3] > 0
                         else mask_box(m))
        new = propose_in_boxes(self.refiner, image_rgb, boxes, pad=self.pad, min_iou=self.min_iou,
                               path=path, **cfg)
        out = []
        for (ann, m), p in zip(anns, new):
            src = self.coco.src_meta(ann)
            # nothing found: a real mask is kept as it was; a box-only annotation stays the filled
            # box, flagged so it can be found and re-masked with another model rather than pass as a mask
            meta = {**src, "remask": (self.refiner_name if p is not None
                                      else "none_found" if src["box_only"] else "kept")}
            if p is not None:
                meta["box_only"] = False
            if p is None:
                out.append(Proposal(mask=m, score=float(ann.get("score", 1.0)), meta=meta))
            else:
                out.append(Proposal(mask=p.mask, score=p.score, meta=meta))
        return out


# --------------------------------------------------------------------------- several candidates per box
def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = float((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def _dedupe(cands: list[Proposal], k: int, same_iou: float) -> list[Proposal]:
    """Keep order, drop a candidate that is a near-copy of one already kept, stop at `k`."""
    kept: list[Proposal] = []
    for c in cands:
        if not c.mask.any() or any(_mask_iou(c.mask, o.mask) >= same_iou for o in kept):
            continue
        kept.append(c)
        if len(kept) >= k:
            break
    return kept


def candidates_in_boxes(backend, image_rgb: np.ndarray, boxes_xyxy, *, pad: float = 0.1,
                        min_iou: float = 0.3, k: int = 5, same_iou: float = 0.9,
                        path: str | None = None, **cfg) -> list[list[Proposal]]:
    """Up to `k` distinct candidate masks per box, best first — for a reviewer to choose between.

    - Promptable backends are prompted with three versions of each box (tight, `pad`, 2×`pad` larger),
      keeping every mask they return for each (SAM gives three per prompt, MedSAM one).
    - Any other backend contributes every prediction that matches the box, ranked by how well it
      matches; the first is the same one-to-one pick `propose_in_boxes` would make.

    Masks are clipped to the loosest box's wiggle room; near-duplicates (mask IoU ≥ `same_iou`)
    collapse into the better-ranked one. A box with no candidate gets an empty list."""
    boxes = [np.asarray(b, float) for b in boxes_xyxy]
    if not boxes:
        return []
    H, W = image_rgb.shape[:2]
    wins = [pad_box(b, H, W, 2 * pad) for b in boxes]

    if is_promptable(backend):
        variants = [boxes] + [[np.asarray(pad_box(b, H, W, f * pad), float) for b in boxes]
                              for f in (1, 2)] if pad > 0 else [boxes]
        multi = getattr(backend, "propose_boxes_multi", None)
        per_box: list[list[Proposal]] = [[] for _ in boxes]
        for vb in variants:
            got = (multi(image_rgb, vb, path=path, **cfg) if callable(multi)
                   else [[g] if g is not None else [] for g in
                         backend.propose_boxes(image_rgb, vb, path=path, **cfg)])
            for bi, cs in enumerate(got):
                per_box[bi] += [Proposal(mask=_clip(np.asarray(c.mask, bool), wins[bi]),
                                         score=float(c.score)) for c in cs if c is not None]
        # best first overall; a stable sort keeps the tight box ahead of the looser ones on ties
        return [_dedupe(sorted(cs, key=lambda c: -c.score), k, same_iou) for cs in per_box]

    preds = []
    for p in backend.propose(image_rgb, path=path, **cfg):
        m = np.asarray(p.mask, bool)
        if m.shape == (H, W) and (pb := mask_box(m)) is not None:
            preds.append((m, float(p.score), pb, float(m.sum())))
    firsts = propose_in_boxes(_Fixed(preds), image_rgb, boxes, pad=2 * pad, min_iou=min_iou)
    out = []
    for b, win, first in zip(boxes, wins, firsts):
        x0, y0, x1, y1 = win
        ranked = []
        for m, s, pb, area in preds:
            iou = box_iou(b, pb)
            if iou >= min_iou:
                ranked.append((iou * float(m[y0:y1, x0:x1].sum()) / area, m, s))
        ranked.sort(key=lambda t: -t[0])
        cs = ([first] if first is not None else []) + [Proposal(mask=_clip(m, win), score=s)
                                                       for _q, m, s in ranked]
        out.append(_dedupe(cs, k, same_iou))
    return out


class _Fixed:
    """Replays already-computed predictions, so the one-to-one pick needs no second model run."""

    def __init__(self, preds):
        self._p = [Proposal(mask=m, score=s) for m, s, _pb, _a in preds]

    def propose(self, image_rgb, **cfg):
        return self._p


def pool_candidates(per_backend: list[tuple[str, list[Proposal]]], *, k: int,
                    same_iou: float = 0.9) -> list[Proposal]:
    """One candidate list out of several models' lists for the same box, best first.

    Scores from different models are not comparable, so the lists are interleaved by rank (each
    model's best, then each one's second, …, in the order the models were given). A mask several
    models found — near-copies at mask IoU ≥ `same_iou` — is kept once, credited to all of them in
    `meta["by"]`, and ranked ahead of masks fewer models agree on. At most `k` are kept."""
    order: list[tuple[str, Proposal]] = []
    for r in range(max((len(cs) for _n, cs in per_backend), default=0)):
        order += [(n, cs[r]) for n, cs in per_backend if r < len(cs)]
    kept: list[Proposal] = []
    for n, c in order:
        if not c.mask.any():
            continue
        twin = next((o for o in kept if _mask_iou(c.mask, o.mask) >= same_iou), None)
        if twin is not None:
            if n not in twin.meta["by"]:
                twin.meta["by"].append(n)
            continue
        kept.append(Proposal(mask=c.mask, score=c.score, meta={**c.meta, "by": [n]}))
    kept.sort(key=lambda c: -len(c.meta["by"]))           # stable: rank order within equal agreement
    return kept[:k]
