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
        import torch
        from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_Weights,
                                                  maskrcnn_resnet50_fpn)
        m = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        m.eval()
        if torch.cuda.is_available():
            m.to("cuda")
        self._model = m
        return m

    def propose(self, image_rgb: np.ndarray, **cfg) -> list[Proposal]:
        import torch
        m = self._load()
        dev = next(m.parameters()).device
        x = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div_(255).to(dev)
        with torch.inference_mode():
            out = m([x])[0]
        thr = float(cfg.get("mask_thresh", 0.5))
        masks = out["masks"].squeeze(1).detach().cpu().numpy()      # (N, H, W) soft
        scores = out["scores"].detach().cpu().numpy()
        return [Proposal(mask=(mk > thr), score=float(s)) for mk, s in zip(masks, scores)]


register("torchvision_maskrcnn", TorchvisionMaskRCNNBackend)
