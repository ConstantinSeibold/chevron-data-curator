"""RAD-DINO feature extractor — vendored from qseg's `notebooks/qseg_playground.py`.

`microsoft/rad-dino` (ViT-B/14, chest-X-ray pretrained), loaded frozen and once. `grid()` returns
the patch-token grid `(C, g, g)`; `add_raddino_features` soft mask-pools that grid per instance to
produce `collection["feats"]["raddino"]`.

This is the reference implementation of Chevron's extractor shape: ONE encoder pass per image, then
pool by mask. Pool with an all-ones mask instead and the same code yields a whole-image (sample)
embedding — which is why sample mode needs no second feature pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

class RadDinoExtractor:
    """microsoft/rad-dino (ViT-B/14, chest-xray pretrained). Loaded frozen, once.
    `grid(img_rgb)` returns the patch-token grid (C, g, g) with C=768, g=37 at the
    processor's default 518px. Domain-matched + TASK-INDEPENDENT (not trained on
    RANZCR), so it's a feature axis orthogonal to the M2F head's own features."""
    HF_ID = "microsoft/rad-dino"

    def __init__(self, device: str = "cuda"):
        import torch  # noqa
        from transformers import AutoModel, AutoImageProcessor
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.proc = AutoImageProcessor.from_pretrained(self.HF_ID)
        self.model = AutoModel.from_pretrained(self.HF_ID).to(device).eval()
        self.device = device
        # SPEED: bf16 autocast on CUDA (~1.5-2x, negligible change to pooled features) + channels_last (helps
        # the patch-embed conv). Optional torch.compile via CURATOR_RADDINO_COMPILE=1 (warmup cost, amortizes
        # over many images).
        self._amp = (device == "cuda" and torch.cuda.is_available())
        if self._amp:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                pass
        if os.environ.get("CURATOR_RADDINO_COMPILE") == "1":
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

    def _grids_from_pixels(self, px):
        import math
        import torch
        if self._amp:
            px = px.to(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._amp):
            tok = self.model(px).last_hidden_state[:, 1:, :]      # drop CLS -> (B, P, C)
        tok = tok.float()
        B, P, C = tok.shape
        g = int(round(math.sqrt(P)))
        return tok.reshape(B, g, g, C).permute(0, 3, 1, 2).contiguous()   # (B, C, g, g)

    def grid(self, img_rgb: np.ndarray):
        px = self.proc(images=img_rgb, return_tensors="pt")["pixel_values"].to(self.device)
        return self._grids_from_pixels(px)[0]                     # (C, g, g)

    def grid_batch(self, imgs_rgb: list):
        """Patch grids for a LIST of images in ONE forward (much higher GPU utilization than per-image).
        Returns (B, C, g, g)."""
        if not imgs_rgb:
            import torch
            return torch.empty(0)
        px = self.proc(images=list(imgs_rgb), return_tensors="pt")["pixel_values"].to(self.device)
        return self._grids_from_pixels(px)                        # (B, C, g, g)


def add_raddino_features(collection: dict, image_root: str | Path,
                         device: str = "cuda", verbose: bool = True) -> dict:
    """Augment an existing collection with RAD-DINO features (one RAD-DINO pass per
    image; soft mask-pooled per instance). Adds rec['f_raddino'] + feats['raddino'].
    Works on a cached collection — no M2F re-inference."""
    import torch
    import torch.nn.functional as F
    from collections import defaultdict
    ext = RadDinoExtractor(device)
    records = collection["records"]
    by_img: dict = defaultdict(list)
    for i, r in enumerate(records):
        by_img[r["image_id"]].append(i)
    cdim = None
    for n, (iid, idxs) in enumerate(by_img.items()):
        img = load_image(records[idxs[0]], image_root)
        if img is None:
            for i in idxs:
                records[i]["f_raddino"] = None
            print(f"[raddino] WARN no image for id {iid} ({records[idxs[0]].get('file_name')})")
            continue
        grid = ext.grid(img)                                   # (C, g, g)
        C, g, _ = grid.shape; cdim = C
        gf = grid.reshape(C, -1)                               # (C, g*g)
        masks = torch.stack([torch.from_numpy(decode_mask(records[i])).float() for i in idxs])
        soft = F.interpolate(masks.unsqueeze(1), size=(g, g), mode="bilinear",
                             align_corners=False).squeeze(1)   # (n, g, g) fractional coverage
        sf = soft.reshape(len(idxs), -1).to(grid.device)       # (n, g*g)
        denom = sf.sum(1, keepdim=True)
        pooled = (sf @ gf.t()) / denom.clamp_min(1e-6)         # (n, C)
        for j, i in enumerate(idxs):
            if denom[j].item() < 1e-4:                         # mask smaller than a patch -> centroid token
                px_ = min(int(records[i]["cx"] * g), g - 1)
                py_ = min(int(records[i]["cy"] * g), g - 1)
                pooled[j] = gf[:, py_ * g + px_]
            records[i]["f_raddino"] = pooled[j].detach().cpu().numpy().astype(np.float32)
        if verbose and (n + 1) % 20 == 0:
            print(f"  ...{n + 1}/{len(by_img)} images")
    collection["feats"]["raddino"] = np.stack(
        [r["f_raddino"] for r in records if r.get("f_raddino") is not None]).astype(np.float32)
    if verbose:
        print(f"[raddino] added {cdim}-d CXR features for {len(records)} instances "
              f"({len(by_img)} RAD-DINO passes)")
    return collection


# --------------------------------------------------------------------------- #
# Persistence + mask decode
# --------------------------------------------------------------------------- #
