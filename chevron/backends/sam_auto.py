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


def _coords_in_float32(gen):
    """Make the generator's point grid float32 before it reaches torch.

    `SamAutomaticMaskGenerator` builds its grid in numpy and hands it straight to `torch.as_tensor`
    with no dtype, so it arrives as float64 — which Metal does not support in any form ("Cannot
    convert a MPS Tensor to float64"). The transform is the single place that dtype is decided, so
    casting there keeps a two-hour job on the GPU instead of demoting it to the CPU. Pixel
    coordinates lose nothing in float32, and the cast is a no-op everywhere else.
    """
    tr = getattr(getattr(gen, "predictor", None), "transform", None)
    apply = getattr(tr, "apply_coords", None)
    if apply is None or getattr(apply, "_chevron_f32", False):
        return gen                                   # nothing to patch, or already patched

    def _f32(coords, original_size, _apply=apply):
        return np.asarray(_apply(coords, original_size), np.float32)

    _f32._chevron_f32 = True
    tr.apply_coords = _f32
    return gen


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
            # `requires` is NOT overridden: the [sam] extra installs both forks, so the class-level
            # hint is already the right one for HQ too.

    def available(self) -> tuple[bool, str]:
        from .. import refine as rf
        if self.family == "samhq" and not rf.samhq_available():
            return False, "segment-anything-hq is not installed"
        if self.family != "samhq" and not rf.sam_available():
            return False, "segment-anything is not installed"
        ckpt, _ = rf.find_sam_checkpoint(family=self.family)
        return True, ("checkpoint ready" if ckpt else
                      "no checkpoint cached yet — it downloads on first use")

    def prepare(self, *, progress=None, stage=None, **cfg) -> None:
        """Fetch the checkpoint and build the generator BEFORE the image loop.

        Left to the first `propose`, a cold cache spends ten minutes downloading a few hundred MB
        inside "image 0 of 80" — a counter that cannot move, which the UI correctly reads as a
        stalled job. `progress(done_bytes, total_bytes)` reports the download; `stage(text, budget)`
        says which part is running, since a stopped download is broken within seconds and a silent
        ViT load is not.
        """
        self._generator(_dl_progress=progress, _stage=stage, **cfg)

    def _generator(self, *, _dl_progress=None, _stage=None, **cfg):
        if self._gen is not None:
            return self._gen
        from .. import refine as rf
        say = _stage or (lambda *a, **k: None)
        ckpt, mt = rf.find_sam_checkpoint(family=self.family)
        if not ckpt:
            say("downloading the checkpoint", 45)
            ckpt = (rf.ensure_samhq_checkpoint(self.model_type, progress=_dl_progress)
                    if self.family == "samhq"
                    else rf.ensure_sam_checkpoint(self.model_type, progress=_dl_progress))
            mt = self.model_type
        say("loading the model", 600)
        # The registry follows the CHECKPOINT's arch, not the requested family — the same rule the
        # refine path already keeps (`refine._sam_predictor`). A sam_hq_* file loaded through the
        # vanilla builder is not a near miss: it dies in `torch.load`, or on "Unexpected key(s)".
        if rf.detect_sam_family(ckpt) == "samhq":
            from segment_anything_hq import SamAutomaticMaskGenerator, sam_model_registry
        else:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        from ..device import move_to, resolve_device
        self.device = resolve_device()
        with rf.load_on_cpu():                       # published HQ weights carry CUDA storages
            sam = sam_model_registry[mt or self.model_type](checkpoint=ckpt)
        sam = self._sam = move_to(sam, self.device)
        self._gen = _coords_in_float32(SamAutomaticMaskGenerator(
            sam,
            points_per_side=int(cfg.get("points_per_side", 32)),
            pred_iou_thresh=float(cfg.get("pred_iou_thresh", 0.88)),
            stability_score_thresh=float(cfg.get("stability_score_thresh", 0.92)),
            min_mask_region_area=int(cfg.get("min_mask_region_area", 0)),
        ))
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
