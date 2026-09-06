"""Embedding extractors: one encoder pass per image, pooled per item.

The shape every extractor shares is RAD-DINO's: run a ViT once per image to get a patch-token grid
`(C, g, g)`, then pool it. Pool by the instance mask and you get an instance embedding; pool globally
and you get a whole-image (sample) embedding — the same code, which is why sample mode needs no
second feature pipeline.

So an extractor only has to produce grids. The pooling, batching, threading and matrix assembly are
done once in `chevron.collect`, and every extractor lands in `collection["feats"][<name>]` where
clustering, the classifier, the projection and kNN already read from.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Extractor(Protocol):
    name: str                 # the feats key it writes
    label: str
    modality: str             # "image" | "text" | "video"
    space: str | None         # shared-embedding-space tag; extractors sharing one can be compared
    requires: str

    def available(self) -> tuple[bool, str]:
        """(usable_here, why_not). Never raises."""

    def grid_batch(self, images_rgb: list):
        """Patch grids for a LIST of images in ONE forward: tensor (B, C, g, g)."""


_REGISTRY: dict[str, Callable[[], Extractor]] = {}


def register(name: str, factory: Callable[[], Extractor]) -> None:
    _REGISTRY[name] = factory


def get(name: str) -> Extractor:
    if name not in _REGISTRY:
        raise KeyError(f"unknown extractor {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_extractors() -> list[dict]:
    """Every extractor with availability — the model dropdown. Never raises."""
    out = []
    for name in sorted(_REGISTRY):
        try:
            e = _REGISTRY[name]()
            ok, why = e.available()
            out.append({"name": name, "label": e.label, "modality": e.modality,
                        "space": e.space, "available": bool(ok), "requires": e.requires,
                        "detail": why})
        except Exception as ex:
            out.append({"name": name, "label": name, "modality": "image", "space": None,
                        "available": False, "requires": "", "detail": f"{type(ex).__name__}: {ex}"})
    return out


class HFPatchGridExtractor:
    """Any HF vision encoder whose `last_hidden_state` is [CLS] + patch tokens.

    Covers the whole DINO family and most ViTs. `grid_batch` mirrors RAD-DINO's: one forward per
    batch, with the device, the autocast dtype and the memory format all decided by
    `chevron.device` — so the same code runs on CUDA, Apple MPS and CPU, and an operator MPS has no
    kernel for costs speed rather than the ingest.
    """
    modality = "image"
    space = None
    requires = "pip install 'chevron-curator[embed]'  (torch + transformers)"

    def __init__(self, hf_id: str, name: str, label: str, *, drop_prefix: int = 1,
                 device: str | None = None):
        self.hf_id, self.name, self.label = hf_id, name, label
        self.drop_prefix = drop_prefix          # tokens before the patches ([CLS], registers, ...)
        self._device = device                   # the PREFERENCE; the real one is resolved at load
        self.device, self._channels_last = "cpu", False
        self.proc = self.model = None

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as e:
            return False, f"torch/transformers not installed ({e})"
        return True, f"{self.hf_id} downloads on first use"

    def _load(self):
        if self.model is not None:
            return
        import os
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from ..device import move_to, prefers_channels_last, resolve_device
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        dev = resolve_device(self._device)
        self.proc = AutoImageProcessor.from_pretrained(self.hf_id)
        self.model = move_to(AutoModel.from_pretrained(self.hf_id), dev).eval()
        self.device = dev
        self._channels_last = prefers_channels_last(dev)
        if self._channels_last:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                self._channels_last = False

    def _demote_to_cpu(self) -> None:
        self.model = self.model.to("cpu")
        self.device, self._channels_last = "cpu", False

    def grid_batch(self, images_rgb: list):
        import math
        import torch
        from ..device import autocast_ctx, run_or_fallback
        if not images_rgb:
            return torch.empty(0)
        self._load()
        px_cpu = self.proc(images=list(images_rgb), return_tensors="pt")["pixel_values"]

        def _forward():
            # inside the closure so a retry re-places the inputs on the DEMOTED device
            px = px_cpu.to(self.device)
            if self._channels_last:
                px = px.to(memory_format=torch.channels_last)
            with torch.inference_mode(), autocast_ctx(self.device):
                return self._forward_tokens(px)

        tok = run_or_fallback(_forward, device=self.device, demote=self._demote_to_cpu,
                              what=f"the {self.name!r} extractor")
        tok = tok.float()[:, self.drop_prefix:, :]           # drop CLS/register tokens
        B, P, C = tok.shape
        g = int(round(math.sqrt(P)))
        return tok.reshape(B, g, g, C).permute(0, 3, 1, 2).contiguous()

    def _forward_tokens(self, px):
        return self.model(px).last_hidden_state
