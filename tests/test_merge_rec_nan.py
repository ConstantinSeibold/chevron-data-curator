"""merge-rec trains on PAIR features (pair_features), whose GEOMETRY columns (orientation/elongation/area
ratios, bbox gap) divide by per-instance shape and yield NaN/inf for degenerate instances — independent of
the feature spec, so it bit even with clean raddino feats ("Input contains NaN"). The pair matrix is now
sanitized. Model-free (pair_features stubbed).
Run: pytest tools/curator/tests/test_merge_rec_nan.py -q
"""
from __future__ import annotations

import numpy as np


def test_finite_sanitizer():
    from tools.curator.merge_rec import _finite
    Y = _finite(np.array([[1.0, np.nan], [np.inf, -np.inf]], np.float64))
    assert Y.dtype == np.float32 and np.isfinite(Y).all()
    assert Y[0, 0] == 1.0 and Y[0, 1] == 0.0 and Y[1, 0] == 0.0 and Y[1, 1] == 0.0


def _state(n=4, image_id=7):
    from tools.curator.state import CuratorState, InstanceMeta
    st = CuratorState(project_dir="/tmp/mr")
    ius = [f"u{i}" for i in range(n)]
    for i, u in enumerate(ius):
        st.meta[u] = InstanceMeta(iuid=u, batch_id="b", row=i, image_id=image_id)
    st.order = list(ius)
    return st, ius


def test_build_pair_xy_sanitizes_nan_and_train_runs(monkeypatch):
    from tools.curator import merge_rec as MR

    class _FakeP:
        def pair_features(self, collection, parr, methods=("decoder",)):
            D = 5
            X = np.ones((len(parr), D), np.float32)
            X[0, 2] = np.nan                                   # degenerate-geometry NaN
            X[-1, 3] = np.inf                                  # and an inf
            return X, [f"f{j}" for j in range(D)]
    monkeypatch.setattr(MR, "_P", lambda: _FakeP())

    st, ius = _state(4)
    coll = {"records": [{} for _ in ius], "feats": {"decoder": np.zeros((4, 5), np.float32)}}
    X, y, parr, rep = MR.build_pair_xy(coll, st, [[ius[0], ius[1]]], [], {"decoder": 1.0})

    assert np.isfinite(X).all()                                # NaN/inf gone
    assert X.shape[0] == len(y) == len(parr) and rep["n_pos"] >= 1
    if len(set(y.tolist())) >= 2:
        MR.train(X, y)                                         # no "Input contains NaN"
