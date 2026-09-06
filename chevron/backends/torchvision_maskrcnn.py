"""Torchvision Mask R-CNN (COCO-pretrained), labels stripped.

The lightest possible proposer: one pip dependency, runs on CPU. Its 80 COCO classes are irrelevant
to whatever you are curating, so they are discarded and only the masks and scores are kept.
"""
from __future__ import annotations

import numpy as np

from .base import Proposal, register


class TorchvisionMaskRCNNBackend:
    name = "torchvision_maskrcnn"
    label = "Torchvision Mask R-CNN (COCO weights, labels dropped)"
    requires = "pip install torch torchvision"

    def __init__(self):
        self._model = None
        self.device = "cpu"

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except Exception as e:
            return False, f"torchvision is not installed ({e})"
        return True, "weights download on first use"

    def _load(self):
        if self._model is not None:
            return self._model
        from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_Weights,
                                                  maskrcnn_resnet50_fpn)
        from ..device import move_to, resolve_device
        m = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        m.eval()
        self.device = resolve_device()
        self._model = move_to(m, self.device)
        return self._model

    def _demote_to_cpu(self) -> None:
        self._model.to("cpu")
        self.device = "cpu"

    def propose(self, image_rgb: np.ndarray, **cfg) -> list[Proposal]:
        import torch
        from ..device import run_or_fallback
        m = self._load()

        def _run():
            # detection heads reach for torchvision's custom ops (nms, roi_align), which is exactly
            # where Metal coverage is thinnest — so the whole forward retries, not just the backbone
            x = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div_(255).to(self.device)
            with torch.inference_mode():
                return m([x])[0]

        out = run_or_fallback(_run, device=self.device, demote=self._demote_to_cpu,
                              what="Mask R-CNN")
        thr = float(cfg.get("mask_thresh", 0.5))
        masks = out["masks"].squeeze(1).detach().cpu().numpy()      # (N, H, W) soft
        scores = out["scores"].detach().cpu().numpy()
        return [Proposal(mask=(mk > thr), score=float(s)) for mk, s in zip(masks, scores)]


register("torchvision_maskrcnn", TorchvisionMaskRCNNBackend)
