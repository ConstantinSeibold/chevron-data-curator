"""Self-supervised contrastive refinement of instance FEATURES (feature-space SimCLR / NT-Xent), to
surface SUBSTRUCTURE within a partition/class. A tiny MLP encoder is trained to be invariant to feature
augmentations (dropout/noise/scale) while discriminating instances; the learned, L2-normalized embeddings
are then clustered (FINCH) to find sub-modes. Operates on the per-instance feature vectors already in the
collection (no raw-image augmentation). Pure torch; CPU by default (the net is tiny).
"""
from __future__ import annotations

import numpy as np


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    return (X - mu) / sd


def _nt_xent(z1, z2, temperature: float):
    """SimCLR NT-Xent over a batch of paired (augmented) views; z1/z2 are L2-normalized (B, d)."""
    import torch
    import torch.nn.functional as F
    b = z1.shape[0]
    z = torch.cat([z1, z2], 0)                       # (2B, d)
    sim = (z @ z.t()) / float(temperature)
    sim.fill_diagonal_(float("-inf"))                # no self-similarity
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)  # each view's positive
    return F.cross_entropy(sim, targets)


def _knn_index(Xs: np.ndarray, k: int) -> np.ndarray:
    """k nearest neighbours (cosine, excluding self) per row -> (N, k) indices."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xs)
    return nn.kneighbors(Xs, return_distance=False)[:, 1:]


def train_embeddings(X: np.ndarray, *, dim: int = 64, hidden: int = 256, epochs: int = 150,
                     batch: int = 256, temperature: float = 0.3, lr: float = 1e-3, drop: float = 0.1,
                     noise: float = 0.05, scale: float = 0.1, knn: int = 5, device: str = "cpu",
                     seed: int = 0, min_n: int = 8) -> np.ndarray:
    """Train a feature-space NEIGHBOURHOOD-contrastive encoder on X (N, D) and return L2-normalized
    embeddings (N, dim). Positives are an instance paired with a random one of its k feature-space nearest
    neighbours (both feature-augmented: dropout + gaussian noise + scaling); negatives are the rest of the
    batch (NT-Xent). Unlike plain instance-discrimination (which spreads instances UNIFORMLY and washes out
    coarse structure), aligning neighbourhoods SHARPENS the density structure, so FINCH on the embeddings
    surfaces sub-modes. With < min_n samples (too few negatives) we skip training and return the
    standardized features (normalized)."""
    X = np.asarray(X, np.float32)
    N, D = X.shape
    Xs = _standardize(X).astype(np.float32)
    if N < int(min_n):
        return (Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from .device import resolve_device, run_or_fallback

    k = max(1, min(int(knn), N - 1))
    nbrs = _knn_index(Xs, k)                             # sklearn, CPU, device-independent

    def _fit(dev_name: str):
        torch.manual_seed(int(seed))                     # inside, so a CPU retry is still deterministic
        dev = torch.device(dev_name)
        xt = torch.from_numpy(Xs).to(dev)
        neigh = torch.from_numpy(nbrs).long().to(dev)                  # (N, k)
        # LayerNorm (not BatchNorm) so tiny/last batches and eval behave identically.
        enc = nn.Sequential(nn.Linear(D, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                            nn.Linear(hidden, int(dim))).to(dev)
        proj = nn.Sequential(nn.Linear(int(dim), int(dim)), nn.ReLU(),
                             nn.Linear(int(dim), max(16, int(dim) // 2))).to(dev)
        opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=float(lr))

        def aug(x):
            if drop > 0:
                x = x * (torch.rand_like(x) > drop).float() / (1.0 - drop)
            if noise > 0:
                x = x + torch.randn_like(x) * noise
            if scale > 0:
                x = x * (1.0 + (torch.rand(x.shape[0], 1, device=x.device) * 2 - 1) * scale)
            return x

        bs = min(int(batch), N)
        enc.train(); proj.train()
        with torch.enable_grad():                        # immune to an ambient no_grad (e.g. eval elsewhere)
            for _ in range(int(epochs)):
                perm = torch.randperm(N, device=dev)
                for s in range(0, N, bs):
                    idx = perm[s:s + bs]
                    if idx.numel() < 2:
                        continue
                    pos = neigh[idx, torch.randint(0, k, (idx.numel(),), device=dev)]  # a random NN per anchor
                    z1 = F.normalize(proj(enc(aug(xt[idx]))), dim=1)
                    z2 = F.normalize(proj(enc(aug(xt[pos]))), dim=1)
                    loss = _nt_xent(z1, z2, temperature)
                    opt.zero_grad(); loss.backward(); opt.step()
        enc.eval()
        with torch.no_grad():
            return F.normalize(enc(xt), dim=1).cpu().numpy()

    # nothing persists between calls here, so "demote" just re-runs the whole (small) fit on the CPU
    picked = {"dev": resolve_device(device)}
    emb = run_or_fallback(lambda: _fit(picked["dev"]), device=picked["dev"],
                          demote=lambda: picked.update(dev="cpu"),
                          what="contrastive substructure training")
    return emb.astype(np.float32)
