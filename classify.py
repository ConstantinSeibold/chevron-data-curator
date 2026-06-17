"""Train a classifier on assigned instances, score the unassigned, and assign by a
confidence threshold. Features come from a fusion spec (reuses P.fuse_features)."""
from __future__ import annotations

import numpy as np

from .cluster import fused_matrix, normalize_spec
from .state import CuratorState


def build_xy(collection: dict, state: CuratorState, spec) -> tuple[np.ndarray, list[str], list[str]]:
    """X = fused rows of ASSIGNED, non-background, non-merge-child instances; y = class_id; iuids."""
    X_all = fused_matrix(collection, normalize_spec(spec))
    rows, y, iuids = [], [], []
    for u, m in state.meta.items():
        if m.assigned_class and not m.is_background and m.merged_into is None:
            rows.append(m.row); y.append(m.assigned_class); iuids.append(u)
    if not rows:
        return np.zeros((0, X_all.shape[1]), np.float32), [], []
    return X_all[rows], y, iuids


def _make(algo: str):
    if algo == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, class_weight="balanced")


def train(X: np.ndarray, y: list[str], *, algo: str = "logreg", calibrate: bool = True):
    """Returns (clf, report). report: classes, per-class precision/recall (CV-OOF), cv_acc."""
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import cross_val_predict
    classes = sorted(set(y))
    yi = np.array([classes.index(c) for c in y])
    n_min = min(np.bincount(yi)) if len(classes) > 1 else 0
    base = _make(algo)
    report = {"classes": classes, "n": len(y), "n_classes": len(classes)}
    if len(classes) >= 2 and n_min >= 2:
        cv = int(min(5, n_min))
        try:
            oof = cross_val_predict(base, X, yi, cv=cv)
            p, r, f, _ = precision_recall_fscore_support(yi, oof, labels=range(len(classes)), zero_division=0)
            report["precision"] = p.tolist(); report["recall"] = r.tolist(); report["f1"] = f.tolist()
            report["cv_acc"] = float((oof == yi).mean())
        except Exception as e:
            report["cv_error"] = str(e)
    clf = base.fit(X, yi)
    if calibrate and len(classes) >= 2 and n_min >= 3:
        from sklearn.calibration import CalibratedClassifierCV
        try:
            clf = CalibratedClassifierCV(_make(algo), cv=int(min(3, n_min))).fit(X, yi)
        except Exception:
            pass
    clf._curator_classes = classes
    return clf, report


def predict_unassigned(clf, collection: dict, state: CuratorState, spec):
    """Returns (iuids, proba [M, C], classes) over unassigned non-background non-child instances."""
    classes = clf._curator_classes
    X_all = fused_matrix(collection, normalize_spec(spec))
    iuids = [u for u, m in state.meta.items()
             if m.assigned_class is None and not m.is_background and m.merged_into is None]
    if not iuids:
        return [], np.zeros((0, len(classes)), np.float32), classes
    rows = [state.meta[u].row for u in iuids]
    proba = clf.predict_proba(X_all[rows])
    return iuids, proba, classes


def threshold_assign(iuids, proba, classes, thresh: float) -> list[tuple[str, str, float]]:
    """Assign each instance to its argmax class if max proba >= thresh."""
    out = []
    if len(iuids) == 0:
        return out
    best = proba.argmax(1)
    conf = proba.max(1)
    for u, b, c in zip(iuids, best, conf):
        if c >= thresh:
            out.append((u, classes[int(b)], float(c)))
    return out


def pr_curve(X: np.ndarray, y: list[str], *, algo: str = "logreg") -> dict:
    """CV-OOF one-vs-rest precision/recall-vs-threshold per class (for the UI plot)."""
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import cross_val_predict
    classes = sorted(set(y))
    if len(classes) < 2:
        return {"classes": classes, "curves": {}}
    yi = np.array([classes.index(c) for c in y])
    n_min = int(min(np.bincount(yi)))
    if n_min < 2:
        return {"classes": classes, "curves": {}}
    cv = int(min(5, n_min))
    try:
        proba = cross_val_predict(_make(algo), X, yi, cv=cv, method="predict_proba")
    except Exception as e:
        return {"classes": classes, "curves": {}, "error": str(e)}
    curves = {}
    for ci, c in enumerate(classes):
        p, r, t = precision_recall_curve((yi == ci).astype(int), proba[:, ci])
        curves[c] = {"precision": p.tolist(), "recall": r.tolist(), "thresholds": t.tolist()}
    return {"classes": classes, "curves": curves}
