"""FAISS-backed kNN classifier search: the Flat (small-ref) path must be EXACT (match sklearn cosine
distances), and KNNClassifier.proba must run + stay finite. The win is at scale (large background): see the
benchmark in the commit msg — sklearn brute balloons to ~16 min at 50k background vs ~26 s for FAISS.
Run: pytest tests/test_knn_faiss.py -q
"""
from __future__ import annotations

import numpy as np


def test_faiss_flat_matches_sklearn_cosine_distance():
    from sklearn.metrics.pairwise import cosine_distances
    from chevron import classify as C
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((50, 16)).astype(np.float32)         # small -> Flat (exact) path
    Q = rng.standard_normal((20, 16)).astype(np.float32)
    d = C._knn_dist(C._knn_index(ref, "cosine"), Q, 5, "cosine")   # (20, 5) nearest cosine dists
    exact = np.sort(cosine_distances(Q, ref), axis=1)[:, :5]       # sklearn ground truth
    assert d.shape == (20, 5)
    assert np.allclose(np.sort(d, axis=1), exact, atol=1e-4)       # cosine_dist = L2^2/2 on unit vectors


def test_faiss_euclidean_matches_sklearn():
    from sklearn.metrics.pairwise import euclidean_distances
    from chevron import classify as C
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((40, 8)).astype(np.float32)
    Q = rng.standard_normal((10, 8)).astype(np.float32)
    d = C._knn_dist(C._knn_index(ref, "euclidean"), Q, 3, "euclidean")
    exact = np.sort(euclidean_distances(Q, ref), axis=1)[:, :3]
    assert np.allclose(np.sort(d, axis=1), exact, atol=1e-3)


def test_knn_classifier_proba_runs_and_is_bounded():
    from chevron.classify import KNNClassifier
    rng = np.random.default_rng(2)
    Xc = [rng.standard_normal((30, 16)).astype(np.float32) + 3 * i for i in range(3)]   # 3 separable classes
    bg = rng.standard_normal((40, 16)).astype(np.float32)
    clf = KNNClassifier(["a", "b", "c"], Xc, bg, k=5, metric="cosine")
    p = clf.proba(rng.standard_normal((25, 16)).astype(np.float32))
    assert p.shape == (25, 3) and np.isfinite(p).all() and (p >= 0).all()
    # a query AT class 0's centre scores class 0 highest
    q0 = (Xc[0].mean(0))[None]
    assert int(np.argmax(clf.proba(q0)[0])) == 0


def test_ref_cap_subsamples_large_reference():
    from chevron import classify as C
    rng = np.random.default_rng(3)
    big = rng.standard_normal((C._KNN_REF_CAP + 5000, 8)).astype(np.float32)
    kind, idx, n = C._knn_index(big, "cosine")
    assert n == C._KNN_REF_CAP                                     # capped (kNN reject needs only a sample)
