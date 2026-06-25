"""Reference exemplar bank: a labeled COCO of foreign-object reference crops (e.g. the fb-reference DB, 66
device classes) embedded with RAD-DINO, used to SUGGEST a fine class for the curator's unassigned instances.

Plain cosine-NN to the bank collapses onto a few populous reference classes (a hubness artifact, amplified by
the reference-photo -> CXR domain gap). Two fixes here, both pure-numpy + testable:
  - CSLS (cross-domain similarity local scaling): 2·cos − r_query − r_ref, where r_ref penalizes references
    that are universally close to many queries (the hubs) — so a query maps to its DISTINCTIVE nearest
    reference, not the popular one.
  - kNN class vote: aggregate the top neighbours per class (max CSLS) instead of a single top-1.

The RAD-DINO embedding is SYMMETRIC across both sides (bbox-crop → grid → MAX-pool for the curator's own
instances AND for the references), so the same object embeds the same way — see
`engine._instance_ref_embeddings`. It lives in the engine; this module is the bank container + the
retrieval math.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def csls_matrix(Q: np.ndarray, B: np.ndarray, *, knn: int = 10) -> np.ndarray:
    """CSLS similarity (n_query, n_bank). Q (instances), B (bank) need not be normalized — done here.
    De-hubs the cross-domain NN so retrieval doesn't collapse onto a handful of popular references."""
    Q, B = _l2(Q), _l2(B)
    S = Q @ B.T                                                  # cosine
    if S.size == 0:
        return S
    kq = min(int(knn), B.shape[0]); kb = min(int(knn), Q.shape[0])
    rq = np.sort(S, axis=1)[:, -kq:].mean(1, keepdims=True)      # each query's mean cos to its knn references
    rb = np.sort(S, axis=0)[-kb:, :].mean(0, keepdims=True)      # each reference's mean cos to its knn queries
    return 2.0 * S - rq - rb                                     # hub references (high rb) get penalized


def suggest(Q: np.ndarray, B: np.ndarray, labels, *, topk: int = 3, knn: int = 8, use_csls: bool = True):
    """Per query row -> ranked [(class_label, score)] by kNN class vote (max CSLS per class among the knn
    nearest bank crops). `labels` is the bank's per-row class label (any hashable). Returns a list of lists."""
    labels = list(labels)
    if Q.size == 0 or B.size == 0:
        return [[] for _ in range(len(Q))]
    S = csls_matrix(Q, B, knn=knn) if use_csls else (_l2(Q) @ _l2(B).T)
    out = []
    k = min(int(knn), B.shape[0])
    for row in S:
        nn = np.argsort(-row)[:max(k, topk)]
        best: dict = {}
        for i in nn:
            c = labels[int(i)]
            best[c] = max(best.get(c, -1e9), float(row[int(i)]))   # strongest evidence for each class
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:int(topk)]
        out.append(ranked)
    return out


def rank_instances_for_class(Q: np.ndarray, B: np.ndarray, labels, cls, *, knn: int = 8,
                             use_csls: bool = True):
    """The INVERSE of `suggest`: fix a reference CLASS, rank the query rows (the curator's own instances) by
    how strongly each resembles it — per query, the max (CSLS-de-hubbed) similarity over the bank rows
    labelled `cls`. Returns (order, scores): `order` = query indices best-first, `scores` aligned to the
    ORIGINAL query rows. Lets you find which instances/partitions are nearest a presented reference sample
    WITHOUT preselecting a partition. CSLS is computed against the FULL bank (so hubness is measured across
    all classes, same as `suggest`), then sliced to the target class."""
    labels = list(labels)
    cols = [i for i, l in enumerate(labels) if l == cls]
    if Q.size == 0 or B.size == 0 or not cols:
        return np.array([], dtype=int), np.zeros(len(Q), np.float32)
    S = csls_matrix(Q, B, knn=knn) if use_csls else (_l2(Q) @ _l2(B).T)
    scores = S[:, cols].max(axis=1).astype(np.float32)
    return np.argsort(-scores), scores


class ReferenceBank:
    """In-memory bank: emb (N, D) L2-normalizable, labels (N,) class ids/names, class_names map, and per-row
    exemplar metadata (file_name, bbox, class) for the visual panel. Persists to npz + json."""

    def __init__(self, emb: np.ndarray, labels: list, class_names: dict, exemplars: list, pool: str = "max"):
        self.emb = np.asarray(emb, np.float32)
        self.labels = list(labels)
        self.class_names = dict(class_names)                    # label -> display name
        self.exemplars = list(exemplars)                        # [{file_name, bbox, cls}]
        self.pool = pool                                        # patch pooling used (max > mean, verified); guards the cache

    @property
    def n(self) -> int:
        return len(self.labels)

    def classes(self) -> list[str]:
        return sorted({self.class_names.get(l, str(l)) for l in self.labels})

    def exemplars_for(self, name: str, limit: int = 8) -> list[dict]:
        return [e for e in self.exemplars if e.get("cls") == name][:limit]

    def add(self, emb: np.ndarray, labels: list, exemplars: list | None = None) -> None:
        emb = np.asarray(emb, np.float32).reshape(len(labels), -1)
        self.emb = np.vstack([self.emb, emb]) if self.emb.size else emb
        self.labels += list(labels)
        for l in labels:
            self.class_names.setdefault(l, str(l))
        if exemplars:
            self.exemplars += list(exemplars)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(path.with_suffix(".npz"), emb=self.emb,
                            labels=np.asarray(self.labels, dtype=object))
        path.with_suffix(".json").write_text(json.dumps(
            {"class_names": {str(k): v for k, v in self.class_names.items()},
             "exemplars": self.exemplars, "pool": self.pool}))

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceBank | None":
        path = Path(path)
        npz, meta = path.with_suffix(".npz"), path.with_suffix(".json")
        if not (npz.exists() and meta.exists()):
            return None
        z = np.load(npz, allow_pickle=True)
        m = json.loads(meta.read_text())
        return cls(z["emb"], list(z["labels"]), m.get("class_names", {}), m.get("exemplars", []),
                   pool=m.get("pool", "mean"))
