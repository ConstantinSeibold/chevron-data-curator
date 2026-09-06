"""Inference core for the label-free shape-prior OOD evaluator.

A denoising-autoencoder shape prior (trained separately by the standalone
``shape_prior/`` toolkit on PaxRay++ GT masks) scores how far a predicted per-class
mask is from the learned manifold of plausible anatomical shapes — with NO ground
truth. See CLAUDE.md "Out-of-distribution shape-prior evaluation".

This module is pure torch/numpy/cv2/scipy (no detectron2) so it imports anywhere.
The ConvDAE architecture MUST stay identical to ``shape_prior/model.py`` (the
trainer), else the saved ``dae_cls<id>.pth`` weights won't load.

Vendored from qseg `src/qseg/evaluation/shape_prior_model.py` — the ConvDAE shape-prior
inference core (canonicalize / implausibility_detail / load_priors_by_catid) used by Chevron's
auto-refine reward. Pure torch/numpy/cv2/scipy. The ConvDAE architecture MUST stay identical to the
trainer that produced the `dae_cls<id>.pth` weights.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def _enc_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout),
                         nn.LeakyReLU(0.2, True))


def _dec_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout),
                         nn.ReLU(True))


class ConvDAE(nn.Module):
    def __init__(self, ch: int = 48, zdim: int = 256, size: int = 128):
        super().__init__()
        self.size = size
        self.enc = nn.Sequential(_enc_block(1, ch), _enc_block(ch, ch * 2),
                                 _enc_block(ch * 2, ch * 4), _enc_block(ch * 4, ch * 8))
        self._feat = ch * 8
        self._fmap = size // 16
        flat = self._feat * self._fmap * self._fmap
        self.to_z = nn.Linear(flat, zdim)
        self.from_z = nn.Linear(zdim, flat)
        self.dec = nn.Sequential(_dec_block(ch * 8, ch * 4), _dec_block(ch * 4, ch * 2),
                                 _dec_block(ch * 2, ch), nn.ConvTranspose2d(ch, 1, 4, 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.enc(x).reshape(b, -1)
        h = self.from_z(self.to_z(h)).reshape(b, self._feat, self._fmap, self._fmap)
        return self.dec(h)


def canonicalize(mask: np.ndarray, size: int = 128, margin: float = 0.15) -> Optional[np.ndarray]:
    """Pose-normalise: crop to bbox (+margin), pad to square, resize. Makes the
    score depend on SHAPE, not position/scale, and matches what the prior saw."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    my, mx = int(h * margin), int(w * margin)
    y0, x0 = max(0, y0 - my), max(0, x0 - mx)
    y1, x1 = min(mask.shape[0] - 1, y1 + my), min(mask.shape[1] - 1, x1 + mx)
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    ch, cw = crop.shape
    s = max(ch, cw)
    pad = np.zeros((s, s), np.uint8)
    oy, ox = (s - ch) // 2, (s - cw) // 2
    pad[oy:oy + ch, ox:ox + cw] = crop
    return cv2.resize(pad, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def dice(a: np.ndarray, b: np.ndarray, eps: float = 1.0) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return float((2 * (a & b).sum() + eps) / (a.sum() + b.sum() + eps))


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _boundary(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.uint8)
    return (m > 0) & (cv2.erode(m, _disk(1)) == 0)


def boundary_band_score(m: np.ndarray, r: np.ndarray, w: int = 4) -> float:
    b = (_boundary(m) | _boundary(r)).astype(np.uint8)
    if b.sum() == 0:
        return 0.0
    near = cv2.dilate(b, _disk(w)) > 0
    return 1.0 - dice((m > 0) & near, (r > 0) & near)


def contour_distance(m: np.ndarray, r: np.ndarray) -> float:
    bm, br = _boundary(m), _boundary(r)
    if bm.sum() == 0 or br.sum() == 0:
        return float(max(m.shape))
    return float(0.5 * (distance_transform_edt(~br)[bm].mean()
                        + distance_transform_edt(~bm)[br].mean()))


@torch.no_grad()
def reconstruct(model: ConvDAE, mask01: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(mask01[None, None].astype("float32")).to(device)
    return (torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def implausibility_detail(model: ConvDAE, mask01: np.ndarray, device: str,
                          band_w: int = 4) -> Tuple[float, float, float]:
    """(global, boundary_band, contour_distance_px). Combined score = mean of the
    first two. Low = plausible (AE ~ identity), high = off-manifold. No GT."""
    recon = reconstruct(model, mask01, device)
    return (1.0 - dice(mask01, recon), boundary_band_score(mask01, recon, band_w),
            contour_distance(mask01, recon))


def load_priors_by_catid(prior_dir: str, catids, device: str) -> Dict[int, ConvDAE]:
    """Load dae_cls<catid>.pth for each requested coco category id that exists."""
    out: Dict[int, ConvDAE] = {}
    for cid in catids:
        path = os.path.join(prior_dir, f"dae_cls{int(cid)}.pth")
        if not os.path.isfile(path):
            continue
        m = ConvDAE().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        out[int(cid)] = m
    return out


def available_prior_catids(prior_dir: str) -> list:
    return sorted(int(os.path.basename(f)[len("dae_cls"):-len(".pth")])
                  for f in glob.glob(os.path.join(prior_dir, "dae_cls*.pth")))
