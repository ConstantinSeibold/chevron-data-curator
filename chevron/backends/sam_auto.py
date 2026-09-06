"""SAM automatic mask generation — proposals with no trained model at all.

This is the backend that makes Chevron useful on a dataset nobody has trained on: SAM segments
everything it can find, you curate which of those are the objects you care about.

Reuses the checkpoint plumbing already built for the refine ops (`chevron.refine`): discovery in the
cache dir, arch detection from the filename, auto-download, and the SAM / SAM-HQ / MedSAM family
split. Nothing here duplicates that.
"""
from __future__ import annotations

import numpy as np

from .base import Proposal, register


class SamAutoBackend:
    name = "sam_auto"
    label = "SAM — automatic masks (no trained model needed)"
    requires = "pip install 'chevron-curator[sam]'  (checkpoint auto-downloads on first use)"

    def __init__(self, model_type: str = "vit_b", family: str = "sam"):
        self.model_type, self.family = model_type, family
        self._gen = self._sam = None
        self.device = "cpu"
        if family == "samhq":
            # the two variants are registered separately and now appear side by side in a dropdown;
            # sharing the class-level label rendered them as two identical, indistinguishable rows
            self.label = "SAM-HQ — automatic masks (sharper boundaries, no trained model needed)"
            self.requires = "pip install segment-anything-hq  (checkpoint auto-downloads on first use)"

    def available(self) -> tuple[bool, str]:
        from .. import refine as rf
        if self.family == "samhq" and not rf.samhq_available():
            return False, "segment-anything-hq is not installed"
        if self.family != "samhq" and not rf.sam_available():
            return False, "segment-anything is not installed"
        ckpt, _ = rf.find_sam_checkpoint(family=self.family)
        return True, ("checkpoint ready" if ckpt else
                      "no checkpoint cached yet — it downloads on first use")

    def _generator(self, **cfg):
        if self._gen is not None:
            return self._gen
        from .. import refine as rf
        ckpt, mt = rf.find_sam_checkpoint(family=self.family)
        if not ckpt:
            ckpt = (rf.ensure_samhq_checkpoint(self.model_type) if self.family == "samhq"
                    else rf.ensure_sam_checkpoint(self.model_type))
            mt = self.model_type
        if self.family == "samhq":
            from segment_anything_hq import SamAutomaticMaskGenerator, sam_model_registry
        else:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        from ..device import move_to, resolve_device
        self.device = resolve_device()
        sam = self._sam = move_to(sam_model_registry[mt or self.model_type](checkpoint=ckpt),
                                  self.device)
        self._gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=int(cfg.get("points_per_side", 32)),
            pred_iou_thresh=float(cfg.get("pred_iou_thresh", 0.88)),
            stability_score_thresh=float(cfg.get("stability_score_thresh", 0.92)),
            min_mask_region_area=int(cfg.get("min_mask_region_area", 0)),
        )
        return self._gen

    def _demote_to_cpu(self) -> None:
        if self._sam is not None:
            self._sam.to("cpu")                  # in-place for parameters, so `self._gen` follows
        self.device = "cpu"

    def propose(self, image_rgb: np.ndarray, **cfg) -> list[Proposal]:
        from ..device import run_or_fallback
        gen = self._generator(**cfg)
        anns = run_or_fallback(lambda: gen.generate(image_rgb), device=self.device,
                               demote=self._demote_to_cpu, what="SAM automatic mask generation")
        # SAM's own score is predicted_iou; stability_score is the more selective one but the
        # generator has already thresholded on it, so predicted_iou is what ranks what survives.
        return [Proposal(mask=np.asarray(a["segmentation"], bool),
                         score=float(a.get("predicted_iou", 1.0)))
                for a in anns]


register("sam_auto", SamAutoBackend)
register("samhq_auto", lambda: SamAutoBackend(family="samhq"))
