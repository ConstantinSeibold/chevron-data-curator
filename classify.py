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


def threshold_assign(iuids, proba, classes, thresh: float, only_class=None) -> list[tuple[str, str, float]]:
    """Assign each instance to its argmax class if max proba >= thresh. If only_class is given, keep
    only instances whose argmax class IS that class (so a per-class threshold can be applied alone)."""
    out = []
    if len(iuids) == 0:
        return out
    best = proba.argmax(1)
    conf = proba.max(1)
    for u, b, c in zip(iuids, best, conf):
        cid = classes[int(b)]
        if only_class is not None and cid != only_class:
            continue
        if c >= thresh:
            out.append((u, cid, float(c)))
    return out


# --------------------------------------------------------------------------- #
# Factored open-set classifier:  score_c = P(c vs not-c) * P(c vs other classes)
# The "vs not-c" detector uses the BACKGROUND + UNASSIGNED pool as negatives, so an
# instance that matches no class scores low on every class and stays unassigned
# (a closed-set multinomial would force it into the nearest known class).
# --------------------------------------------------------------------------- #
class FactoredClassifier:
    def __init__(self, classes, term2, term1):
        self.classes = classes      # ordered class_ids
        self.term2 = term2          # multinomial over assigned classes (this vs OTHER classes)
        self.term1 = term1          # {class_id: binary clf} (this vs NOT-this, incl. bg/unassigned)
        self._curator_classes = classes

    def proba(self, X: np.ndarray) -> np.ndarray:
        """[N, C] where col c = P1_c (vs not-c) * P2_c (vs other classes)."""
        if len(X) == 0:
            return np.zeros((0, len(self.classes)), np.float32)
        p2 = self.term2.predict_proba(X)            # columns aligned to 0..C-1 == self.classes order
        out = np.zeros((len(X), len(self.classes)), np.float32)
        for j, c in enumerate(self.classes):
            p1 = self.term1[c].predict_proba(X)[:, 1]
            out[:, j] = p1 * p2[:, j]
        return out


def build_pools(collection: dict, state: CuratorState, spec):
    """Returns (X_all, assigned_rows, y_classids, background_rows, unassigned_rows)."""
    spec = normalize_spec(spec)
    feats = collection.get("feats", {})
    present = {m: w for m, w in spec.items() if m in feats}
    if not present:
        raise ValueError(f"none of the selected features {sorted(spec)} are present; "
                         f"available: {[k for k in feats if not k.startswith('_')]}")
    X = fused_matrix(collection, present)
    a_rows, y, bg_rows, un_rows = [], [], [], []
    for u, m in state.meta.items():
        if m.merged_into is not None:
            continue
        if m.is_background:
            bg_rows.append(m.row)
        elif m.assigned_class:
            a_rows.append(m.row); y.append(m.assigned_class)
        else:
            un_rows.append(m.row)
    return X, a_rows, y, bg_rows, un_rows


def _fit_factored(X, a_rows, y, neg_extra_rows, classes, algo) -> FactoredClassifier:
    Xa = X[a_rows]
    yi = np.array([classes.index(c) for c in y])
    Xneg = X[neg_extra_rows] if neg_extra_rows else np.zeros((0, X.shape[1]), X.dtype)
    term2 = _make(algo).fit(Xa, yi)                 # this vs OTHER assigned classes
    # ensure term2 columns map to classes order (sklearn sorts classes_ -> 0..C-1 already)
    term1 = {}
    for ci, c in enumerate(classes):
        pos = Xa[yi == ci]
        neg = np.vstack([Xa[yi != ci], Xneg]) if len(Xneg) else Xa[yi != ci]
        Xb = np.vstack([pos, neg])
        yb = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        term1[c] = _make(algo).fit(Xb, yb)          # this vs NOT-this (incl. bg/unassigned)
    return FactoredClassifier(classes, term2, term1)


def train_factored(collection: dict, state: CuratorState, spec, *, algo: str = "logreg",
                   use_unassigned_negatives: bool = True, max_unassigned_neg: int = 4000):
    """Returns (FactoredClassifier|None, report)."""
    try:
        X, a_rows, y, bg_rows, un_rows = build_pools(collection, state, spec)
    except ValueError as e:
        return None, {"error": str(e)}
    from collections import Counter
    cnt = Counter(y)
    classes = sorted(c for c, n in cnt.items() if n >= 2)            # trainable: >= 2 instances
    skipped = sorted(c for c, n in cnt.items() if n < 2)            # under-sampled classes -> excluded from training
    if len(classes) < 2:
        return None, {"error": f"need >=2 classes with >=2 assigned instances each (counts: {dict(cnt)})"}
    keep = set(classes)                                            # drop singleton-class rows (neither pos nor neg)
    a_rows, y = map(list, zip(*[(r, c) for r, c in zip(a_rows, y) if c in keep]))
    neg_extra = list(bg_rows)
    if use_unassigned_negatives and un_rows:
        rng = np.random.default_rng(0)
        k = min(len(un_rows), max_unassigned_neg)
        neg_extra += [int(r) for r in rng.choice(un_rows, k, replace=False)]
    clf = _fit_factored(X, a_rows, y, neg_extra, classes, algo)
    report = {"classes": classes, "n": len(y), "n_classes": len(classes),
              "n_background": len(bg_rows), "n_unassigned_neg": len(neg_extra) - len(bg_rows),
              "per_class": {c: int(sum(1 for v in y if v == c)) for c in classes},
              "skipped_classes": skipped}
    report["pr"] = pr_curve_factored(X, a_rows, y, bg_rows, neg_extra, classes, algo=algo)
    return clf, report


def pr_curve_factored(X, a_rows, y, bg_rows, neg_extra, classes, *, algo="logreg", n_splits=3) -> dict:
    """CV-OOF precision/recall of the factored product score per class (positives = assigned-c;
    negatives = other assigned + background — so it reflects the open-set 'none' rejection)."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.model_selection import StratifiedKFold
    yi = np.array([classes.index(c) for c in y])
    n_min = int(min(np.bincount(yi)))
    if n_min < 2:
        return {"classes": classes, "curves": {}}
    a_rows = np.asarray(a_rows)
    oof = np.zeros((len(a_rows), len(classes)), np.float32)
    skf = StratifiedKFold(n_splits=int(min(n_splits, n_min)), shuffle=True, random_state=0)
    for tr, va in skf.split(a_rows, yi):
        clf = _fit_factored(X, list(a_rows[tr]), [y[i] for i in tr], neg_extra, classes, algo)
        oof[va] = clf.proba(X[a_rows[va]])
    full = _fit_factored(X, list(a_rows), y, neg_extra, classes, algo)
    bg_scores = full.proba(X[bg_rows]) if bg_rows else np.zeros((0, len(classes)), np.float32)
    curves = {}
    for ci, c in enumerate(classes):
        pos = oof[yi == ci, ci]
        neg = np.concatenate([oof[yi != ci, ci], bg_scores[:, ci]]) if len(bg_scores) else oof[yi != ci, ci]
        scores = np.concatenate([pos, neg]); labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        if labels.sum() and (labels == 0).any():
            p, r, t = precision_recall_curve(labels, scores)
            fpr, tpr, rt = roc_curve(labels, scores)            # Youden's J = TPR - FPR (best operating point)
            jt = rt[np.argmax(tpr - fpr)]
            youden = float(min(max(jt, 0.0), 1.0)) if np.isfinite(jt) else 0.5
            curves[c] = {"precision": p.tolist(), "recall": r.tolist(), "thresholds": t.tolist(), "youden": youden}
    return {"classes": classes, "curves": curves}


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
