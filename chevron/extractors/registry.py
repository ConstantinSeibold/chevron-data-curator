"""The extractor line-up — this is the "model dropdown".

Adding one is a registry entry, not a subsystem: `engine.available_features()` is already the single
source of truth for every feature selector, so a new key propagates to clustering, the classifier,
the projection spec and retrieval on its own.
"""
from __future__ import annotations

from .base import HFPatchGridExtractor, register


class _Named(HFPatchGridExtractor):
    """HFPatchGridExtractor with a fixed shared-space tag."""
    def __init__(self, hf_id, name, label, *, space=None, drop_prefix=1, requires=None):
        super().__init__(hf_id, name, label, drop_prefix=drop_prefix)
        self.space = space
        if requires:
            self.requires = requires


class ProjectedVisionExtractor(_Named):
    """CLIP / SigLIP-style dual encoders.

    The patch tokens live in the vision tower's own space, while the SHARED image-text space is what
    the visual projection maps into. Pooling patch tokens and then applying that projection is the
    MaskCLIP-style approximation: it puts a mask-pooled region into the same space as a text query,
    which is what makes "type a phrase, find the instances" work (P6). It is an approximation —
    the projection was trained on the pooled/CLS token, not on arbitrary sub-regions — and is
    labelled as such rather than presented as exact.
    """

    def _load(self):
        if self.model is not None:
            return
        from transformers import AutoModel, AutoProcessor
        from ..device import move_to, prefers_channels_last, resolve_device
        dev = resolve_device(self._device)
        self.proc = AutoProcessor.from_pretrained(self.hf_id)
        self.model = move_to(AutoModel.from_pretrained(self.hf_id), dev).eval()
        self.device, self._channels_last = dev, prefers_channels_last(dev)

    def _forward_tokens(self, px):
        vision = getattr(self.model, "vision_model", None)
        out = vision(px) if vision is not None else self.model(px)
        return out.last_hidden_state

    @staticmethod
    def _as_tensor(out):
        """transformers 4.x returned a bare tensor from get_text_features; 5.x returns a
        BaseModelOutputWithPooling whose `pooler_output` is the projected embedding. Handle both —
        the contract changed under us and a version bump should not silently break text queries."""
        import torch
        if torch.is_tensor(out):
            return out
        for attr in ("pooler_output", "text_embeds", "last_hidden_state"):
            v = getattr(out, attr, None)
            if v is not None:
                return v[:, 0] if attr == "last_hidden_state" else v
        raise TypeError(f"cannot read a text embedding out of {type(out).__name__}")

    def embed_text(self, texts: list[str]):
        """Text in the SAME space the pooled image features are projected into."""
        import torch
        from ..device import run_or_fallback
        self._load()

        def _run():
            inputs = self.proc(text=list(texts), return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                return self._as_tensor(self.model.get_text_features(**inputs))

        feats = run_or_fallback(_run, device=self.device, demote=self._demote_to_cpu,
                                what=f"the {self.name!r} text encoder")
        return torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy()

    def project_pooled(self, pooled):
        """Map mask-pooled patch features into the shared image-text space."""
        import torch
        from ..device import run_or_fallback
        proj = getattr(self.model, "visual_projection", None)
        if proj is None:
            return pooled

        def _run():                              # `proj` is a submodule, so a demote moves it too
            with torch.inference_mode():
                return torch.nn.functional.normalize(proj(pooled.to(self.device)).float(), dim=-1)

        return run_or_fallback(_run, device=self.device, demote=self._demote_to_cpu,
                               what=f"the {self.name!r} projection")


def _raddino():
    from .raddino import RadDinoExtractor

    class _Rad:
        name, label = "raddino", "RAD-DINO (chest X-ray, ViT-B/14)"
        modality, space = "image", None
        requires = "pip install 'chevron-curator[embed]'  (torch + transformers)"

        def __init__(self):
            self._e = None

        def available(self):
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
            except Exception as e:
                return False, f"torch/transformers not installed ({e})"
            return True, "microsoft/rad-dino — domain-matched to chest X-rays"

        def grid_batch(self, images_rgb):
            if self._e is None:
                self._e = RadDinoExtractor()          # device resolved in chevron.device
            return self._e.grid_batch(images_rgb)

    return _Rad()


register("raddino", _raddino)
register("dinov2", lambda: _Named("facebook/dinov2-base", "dinov2",
                                  "DINOv2 ViT-B/14 (general purpose)"))

# DINOv3 (LVD-1689M). Stronger dense features than DINOv2 and the better default for anything
# non-medical: colour endoscopy and surgical video in particular. The weights are HF-GATED, so a
# first use needs the licence accepted on the model page and `hf auth login`; `available()` cannot
# see that in advance, which is why the detail line says so rather than letting the download fail
# with an opaque 401. Three sizes, because the jump from S to L is a real quality/^cost trade.
_DINOV3 = "facebook/dinov3-{}-pretrain-lvd1689m"
_GATED = ("gated on Hugging Face: accept the licence on the model page, then `hf auth login`")
register("dinov3", lambda: _Named(_DINOV3.format("vits16"), "dinov3",
                                  "DINOv3 ViT-S/16 (general purpose, fast)", requires=_GATED))
register("dinov3b", lambda: _Named(_DINOV3.format("vitb16"), "dinov3b",
                                   "DINOv3 ViT-B/16 (general purpose)", requires=_GATED))
register("dinov3l", lambda: _Named(_DINOV3.format("vitl16"), "dinov3l",
                                   "DINOv3 ViT-L/16 (general purpose, strongest)", requires=_GATED))
register("clip", lambda: ProjectedVisionExtractor(
    "openai/clip-vit-base-patch32", "clip", "CLIP ViT-B/32 (image + text)", space="clip"))
register("siglip2", lambda: ProjectedVisionExtractor(
    "google/siglip-base-patch16-224", "siglip2", "SigLIP ViT-B/16 (image + text)", space="siglip"))
