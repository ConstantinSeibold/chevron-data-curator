"""Hugging Face universal segmentation (Mask2Former / OneFormer), labels stripped.

Any `AutoModelForUniversalSegmentation` checkpoint works; the default is Mask2Former trained on COCO
instances. Reuses the same `transformers` stack as the embedding extractors, so a project that has
`chevron[embed]` already has this.
"""
from __future__ import annotations

import numpy as np

from .base import Proposal, register

DEFAULT_MODEL = "facebook/mask2former-swin-base-coco-instance"


class HFSegBackend:
    name = "hf_seg"
    label = "Hugging Face Mask2Former / OneFormer (labels dropped)"
    requires = "pip install 'chevron-curator[embed]'  (torch + transformers)"

    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self._proc = self._model = None

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as e:
            return False, f"transformers/torch not installed ({e})"
        return True, f"{self.model_id} downloads on first use"

    def _load(self):
        if self._model is not None:
            return self._proc, self._model
        import torch
        from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
        self._proc = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForUniversalSegmentation.from_pretrained(self.model_id).eval()
        if torch.cuda.is_available():
            self._model.to("cuda")
        return self._proc, self._model

    def propose(self, image_rgb: np.ndarray, **cfg) -> list[Proposal]:
        import torch
        proc, model = self._load()
        inputs = proc(images=image_rgb, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs)
        H, W = image_rgb.shape[:2]
        res = proc.post_process_instance_segmentation(
            out, target_sizes=[(H, W)], threshold=float(cfg.get("score_thresh", 0.5)))[0]
        seg, info = res["segmentation"], res["segments_info"]
        if seg is None or not info:
            return []
        seg = seg.cpu().numpy()
        # `segmentation` is an id map; each segment's id indexes segments_info. The predicted
        # label_id is deliberately ignored — COCO's 80 classes are not the label space being curated.
        return [Proposal(mask=(seg == s["id"]), score=float(s.get("score", 1.0))) for s in info]


register("hf_seg", HFSegBackend)
