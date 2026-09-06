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
        self._gen = None

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
        import torch
        sam = sam_model_registry[mt or self.model_type](checkpoint=ckpt)
        if torch.cuda.is_available():
            sam.to("cuda")
        self._gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=int(cfg.get("points_per_side", 32)),
            pred_iou_thresh=float(cfg.get("pred_iou_thresh", 0.88)),
            stability_score_thresh=float(cfg.get("stability_score_thresh", 0.92)),
            min_mask_region_area=int(cfg.get("min_mask_region_area", 0)),
        )
        return self._gen

    def propose(self, image_rgb: np.ndarray, **cfg) -> list[Proposal]:
        gen = self._generator(**cfg)
        # SAM's own score is predicted_iou; stability_score is the more selective one but the
        # generator has already thresholded on it, so predicted_iou is what ranks what survives.
        return [Proposal(mask=np.asarray(a["segmentation"], bool),
                         score=float(a.get("predicted_iou", 1.0)))
                for a in gen.generate(image_rgb)]


register("sam_auto", SamAutoBackend)
register("samhq_auto", lambda: SamAutoBackend(family="samhq"))
