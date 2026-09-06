"""Whole images as items — the backend for sample mode.

This is the payoff of the design the whole port rests on: *a sample is an instance with a trivial
mask*. Labelling whole images needs no second data model, no second grid, no second selection and no
second feature pipeline — just a proposer that returns one all-ones mask per image.

Everything downstream then works unchanged, because it reads `feats[name]` and never touches masks:
clustering, the classifier, the projection, the map, kNN, the reference bank. An extractor pooled
over an all-ones mask IS a whole-image embedding.
"""
from __future__ import annotations

import numpy as np

from .base import Proposal, register


class WholeImageBackend:
    name = "whole_image"
    label = "Whole images (sample mode — label images, not masks)"
    requires = "nothing"

    def available(self) -> tuple[bool, str]:
        return True, "one item per image; no model or GPU"

    def propose(self, image_rgb: np.ndarray, path: str | None = None, **cfg) -> list[Proposal]:
        H, W = image_rgb.shape[:2]
        return [Proposal(mask=np.ones((H, W), bool), score=1.0)]


register("whole_image", WholeImageBackend)
