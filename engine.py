"""CuratorEngine — the UI-agnostic facade tying the curator modules together.

Holds the authoritative mutable project (collection + feature matrices + masks + the
overlay state). The Gradio app binds to one server-side instance. All renders return
numpy RGB images; all mutations go through the History for undo/redo + autosave.
"""
from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any

import numpy as np

from . import classify as _clf
from . import cluster as _cl
from . import collect as _co
from . import export_coco as _ex
from . import sample as _sa
from . import similar as _sim
from .history import History
from .metrics import filter_instances, partition_summary, sort_instances
from .state import CuratorState
from .store import Store

_AUTOSNAP_EVERY = 20


def _color(i: int):
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.85, 1.0)
    return np.array([r * 255, g * 255, b * 255])


class CuratorEngine:
    def __init__(self, project_dir: str | Path):
        self.store = Store(project_dir)
        self.state = CuratorState(project_dir=str(project_dir))
        self.collection: dict | None = None
        self.history = History(self.store)
        self.model = self.cfg = self.d2_cfg = self.scan = None
        self._overlay_rle: dict[str, dict] = {}        # iuid -> effective RLE (refine/merge)
        self._cluster: dict | None = None              # {spec, distance, per_image, partitions, counts, level}
        self._commits = 0
        if self.store.is_project():
            self.open()

    # ---- project lifecycle -------------------------------------------------
    def open(self) -> None:
        self.state = self.store.load_state()
        if self.store.has_collection():
            self.collection = self.store.load_collection()
        self.history = History(self.store)
        self._overlay_rle = {}
        for p in sorted(self.store.refine_dir.glob("*.pkl")):
            ov = self.store.load_refine(p.stem)
            if ov and "result_rle" in ov:
                self._overlay_rle[p.stem] = ov["result_rle"]
        self._cluster = None

    def init_project(self, config: dict) -> None:
        self.state = CuratorState(project_dir=str(self.store.dir), config=dict(config))
        self.collection = None
        self.store.ensure()
        self.save()

    def save(self, *, snapshot: bool = False) -> None:
        self.store.save_state(self.state)
        man = self.store.load_manifest()
        man.update({"coll_version": self.state.coll_version,
                    "n_instances": len(self.state.order)})
        self.store.save_manifest(man)
        if self.collection is not None and self.state.collection_dirty is False:
            pass  # collection saved explicitly on append (below)
        if snapshot:
            self.store.snapshot()

    # ---- sampling + extraction (additive) ---------------------------------
    def _ensure_model(self):
        if self.model is None:
            from ._bootstrap import get_P
            P = get_P()
            mc = self.state.config.get("model", {})
            P.setup_env(gpu=str(mc.get("gpu", "0")))
            self.model, self.cfg, self.d2_cfg, self.scan = P.load_model(
                ckpt=mc.get("ckpt"), config_name=mc.get("config_name", "experiments/synthfb_arch3"),
                overrides=mc.get("overrides"))
        return self.model, self.cfg, self.d2_cfg

    def sample_more(self, n: int, *, smart: bool = False, seed: int | None = None) -> dict:
        model, cfg, d2_cfg = self._ensure_model()
        root = self.state.config["images"]["root"]
        man = self.store.load_manifest()
        processed = set(man.get("processed_paths", []))
        files = _sa.list_images(root)
        new_files = _sa.sample_random(files, n, exclude=processed, seed=seed)
        if not new_files:
            return {"n_new_images": 0, "n_new_instances": 0, **self.stats()}
        feat_cfg = self.state.config.get("features_runtime", _default_feat_cfg(self.state.config))
        batch = _co.collect_batch(model, cfg, d2_cfg, new_files,
                                  score_thresh=self.state.config["model"].get("score_thresh", 0.3),
                                  feature_cfg=feat_cfg)
        n_new = len(batch["records"])
        self.collection = _co.concat_collections(self.collection, batch)
        # create overlay meta for new instances; rebuild order/rows
        self.state.order = [r["iuid"] for r in self.collection["records"]]
        from .state import InstanceMeta
        for r in batch["records"]:
            self.state.meta[r["iuid"]] = InstanceMeta(
                iuid=r["iuid"], batch_id=r["batch_id"], row=r["row"], image_id=int(r["image_id"]),
                provenance={"file": r.get("abs_path", ""), "src_score": float(r["score"]),
                            "ckpt": self.state.config["model"].get("ckpt", "")})
        self.state.rebuild_rows()
        self.state.assert_aligned(self.collection["feats"][_any_method(self.collection)].shape[0])
        self.state.coll_version += 1
        self.state.collection_dirty = True
        processed.update(new_files)
        man["processed_paths"] = sorted(processed)
        self.store.save_manifest(man)
        self.store.save_collection(self.collection)
        self.history.barrier()                         # additive sampling = undo barrier
        self.save()
        return {"n_new_images": len(new_files), "n_new_instances": n_new, **self.stats()}

    # ---- clustering --------------------------------------------------------
    def cluster(self, spec, *, distance: str = "cosine", per_image: bool = False, level: int | None = None) -> dict:
        partitions, counts = _cl.cluster_with_cache(self.collection, self.state, self.store, spec,
                                                    distance=distance, per_image=per_image)
        lvl = level if level is not None else _default_level(counts)
        self._cluster = {"spec": _cl.normalize_spec(spec), "distance": distance, "per_image": per_image,
                         "partitions": partitions, "counts": counts, "level": lvl}
        self.state.collection_dirty = False
        self.save()
        return {"counts": counts, "level": lvl, "n_levels": len(counts)}

    def set_level(self, level: int) -> None:
        if self._cluster:
            self._cluster["level"] = max(0, min(level, len(self._cluster["counts"]) - 1))

    def _labels(self) -> np.ndarray:
        return _cl.labels_at_level(self._cluster["partitions"], self._cluster["level"])

    def _scores(self) -> np.ndarray:
        return np.array([self.collection["records"][self.state.meta[u].row]["score"]
                         for u in self.state.order], np.float32)

    def partition_view(self) -> list[dict]:
        labels = self._labels()
        summ = partition_summary(labels, self.state, self._scores())
        rows = []
        for pid, s in summ.items():
            rows.append({**s, "majority_class": self.state.class_name(s["majority_class"])})
        return sorted(rows, key=lambda r: -r["size"])

    def partition_iuids(self, pid: int) -> list[str]:
        return _cl.partition_iuids(self.state, self._labels(), int(pid))

    # ---- rendering ---------------------------------------------------------
    def _mask(self, iuid: str) -> np.ndarray:
        from pycocotools import mask as mu
        rle = self._overlay_rle.get(iuid) or self.collection["records"][self.state.meta[iuid].row]["rle"]
        return mu.decode(rle).astype(bool)

    def _rgb(self, iuid: str) -> np.ndarray:
        import cv2
        rec = self.collection["records"][self.state.meta[iuid].row]
        img = cv2.imread(rec.get("abs_path") or rec["file_name"], cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((rec["H"], rec["W"], 3), np.uint8)

    def crop(self, iuid: str, *, mask_overlay: bool = True, pad: int = 10) -> np.ndarray:
        import cv2
        img = self._rgb(iuid).copy()
        m = self._mask(iuid)
        ys, xs = np.where(m)
        H, W = m.shape
        if len(xs) == 0:
            return img
        x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x2, y2 = min(W, xs.max() + pad + 1), min(H, ys.max() + pad + 1)
        out = img.copy()
        if mask_overlay:
            c = _color(self.state.meta[iuid].row)
            out[m] = (0.5 * out[m] + 0.5 * c).astype(np.uint8)
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, tuple(int(v) for v in c), 1)
        return out[y1:y2, x1:x2]

    def partition_crops(self, pid: int, *, mask_overlay: bool = True, limit: int = 60):
        iuids = self.partition_iuids(pid)[:limit]
        return [(self.crop(u, mask_overlay=mask_overlay),
                 f"{u[:6]} s={self.collection['records'][self.state.meta[u].row]['score']:.2f}"
                 + (f" [{self.state.class_name(self.state.meta[u].assigned_class)}]"
                    if self.state.meta[u].assigned_class else "")) for u in iuids], iuids

    def image_ids(self) -> list[int]:
        from collections import Counter
        c = Counter(m.image_id for m in self.state.meta.values())
        return [iid for iid, _ in c.most_common()]

    def image_overlay(self, image_id: int, *, color_by: str = "partition") -> np.ndarray:
        import cv2
        iuids = [u for u, m in self.state.meta.items() if m.image_id == image_id]
        if not iuids:
            return np.zeros((512, 512, 3), np.uint8)
        out = self._rgb(iuids[0]).copy().astype(np.float32)
        labels = self._labels() if (self._cluster and color_by == "partition") else None
        for u in iuids:
            m = self._mask(u)
            if color_by == "partition" and labels is not None:
                key = int(labels[self.state.meta[u].row])
            elif color_by == "class":
                key = abs(hash(self.state.meta[u].assigned_class or "")) % 997
            else:
                key = self.state.meta[u].row
            c = _color(key)
            out[m] = 0.5 * out[m] + 0.5 * c
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, tuple(int(v) for v in c), 1)
        return out.astype(np.uint8)

    def keypoints_overlay(self, image_id: int) -> np.ndarray:
        from ._bootstrap import get_P
        P = get_P()
        iuids = [u for u, m in self.state.meta.items() if m.image_id == image_id]
        if not iuids:
            return np.zeros((512, 512, 3), np.uint8)
        img = self._rgb(iuids[0])
        recs = self.collection["records"]
        kp = [recs[self.state.meta[u].row]["keypoints"] for u in iuids
              if "keypoints" in recs[self.state.meta[u].row]]
        vis = [recs[self.state.meta[u].row].get("keypoint_vis") for u in iuids
               if "keypoints" in recs[self.state.meta[u].row]]
        return P.overlay_keypoints(img, kp, vis, vis_thresh=0.3) if kp else img

    # ---- assignment / curation --------------------------------------------
    def new_class(self, name: str) -> str:
        tok = self.history.begin(self.state, [], [])
        cid = self.state.add_class(name)
        tok["class_ids"].append(cid)
        self.history.commit(self.state, tok, "new_class", f"new class {name}")
        self.save()
        return cid

    def assign(self, iuids: list[str], class_name: str, *, source: str = "manual",
               scores: dict | None = None) -> None:
        if not iuids:
            return
        existing = self.state.class_id_by_name(class_name)
        tok = self.history.begin(self.state, iuids, [existing] if existing else [])
        cid = self.state.add_class(class_name)
        if not existing:
            tok["class_ids"].append(cid)
        for u in iuids:
            m = self.state.meta[u]
            m.assigned_class = cid; m.is_background = False
            m.assign_source = source
            m.assign_score = (scores or {}).get(u)
        self.history.commit(self.state, tok, "assign", f"assign {len(iuids)}→{class_name}")
        self._after_mutation()

    def assign_partition(self, pid: int, class_name: str) -> None:
        self.assign(self.partition_iuids(pid), class_name, source="partition")

    def remove_from_class(self, iuids: list[str]) -> None:
        tok = self.history.begin(self.state, iuids, [])
        for u in iuids:
            self.state.meta[u].assigned_class = None
            self.state.meta[u].assign_source = None
            self.state.meta[u].assign_score = None
        self.history.commit(self.state, tok, "remove", f"unassign {len(iuids)}")
        self._after_mutation()

    def set_background(self, iuids: list[str]) -> None:
        tok = self.history.begin(self.state, iuids, [])
        for u in iuids:
            self.state.meta[u].is_background = True
            self.state.meta[u].assigned_class = None
        self.history.commit(self.state, tok, "background", f"reject {len(iuids)}")
        self._after_mutation()

    # ---- within-image merge ------------------------------------------------
    def merge_preview(self, image_id: int, *, dist_kind: str = "mask_gap", method: str = "decoder",
                      thresh: float = 0.05, max_group_size: int | None = None):
        from ._bootstrap import get_P
        P = get_P()
        groups, _ = P.merge_groups_in_image(self.collection, image_id, dist_kind=dist_kind, method=method,
                                            thresh=thresh, max_group_size=max_group_size)
        before, _ = P.merge_groups_in_image(self.collection, image_id, dist_kind=dist_kind, thresh=0.0)
        img = self._rgb_by_image(image_id)
        return (P.overlay_groups(img, self.collection, before),
                P.overlay_groups(img, self.collection, groups), groups)

    def _rgb_by_image(self, image_id: int) -> np.ndarray:
        iuids = [u for u, m in self.state.meta.items() if m.image_id == image_id]
        return self._rgb(iuids[0]) if iuids else np.zeros((512, 512, 3), np.uint8)

    def commit_merge(self, image_id: int, groups: list[list[int]]) -> None:
        """groups are GLOBAL row indices (from merge_preview). For each multi-member group,
        the highest-score member is the representative (union mask); the rest become children."""
        from pycocotools import mask as mu
        order = self.state.order
        touched = []
        for g in groups:
            if len(g) < 2:
                continue
            iuids = [order[r] for r in g]
            rep = max(iuids, key=lambda u: self.collection["records"][self.state.meta[u].row]["score"])
            union = None
            for u in iuids:
                m = self._mask(u)
                union = m if union is None else (union | m)
            rle = mu.encode(np.asfortranarray(union.astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
            touched.extend(iuids)
            tok = self.history.begin(self.state, iuids, [])
            for u in iuids:
                self.state.meta[u].merged_into = (None if u == rep else rep)
            self.state.meta[rep].merge_members = [u for u in iuids if u != rep]
            self.history.commit(self.state, tok, "merge", f"merge {len(iuids)}→{rep[:6]}")
            self._overlay_rle[rep] = rle
            self.store.save_refine(rep, {"iuid": rep, "base_rle": self.collection["records"][self.state.meta[rep].row]["rle"],
                                         "ops": [{"name": "merge", "members": iuids}], "result_rle": rle})
        self._after_mutation()

    # ---- refinement --------------------------------------------------------
    def refine_preview(self, iuid: str, ops: list[dict]):
        from .refine import apply_ops, to_gray
        img = self._rgb(iuid)
        base = self.collection["records"][self.state.meta[iuid].row]["rle"]
        from pycocotools import mask as mu
        base_m = mu.decode(base).astype(bool)
        refined = apply_ops(to_gray(img), base_m, ops)
        return self._crop_mask(img, base_m), self._crop_mask(img, refined)

    def _crop_mask(self, img, m, pad=12):
        import cv2
        ys, xs = np.where(m); H, W = m.shape
        out = img.copy()
        if len(xs):
            out[m] = (0.5 * out[m] + 0.5 * np.array([40, 220, 40])).astype(np.uint8)
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, (40, 220, 40), 1)
            x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad)
            x2, y2 = min(W, xs.max() + pad + 1), min(H, ys.max() + pad + 1)
            return out[y1:y2, x1:x2]
        return out

    def apply_refine(self, iuid: str, ops: list[dict]) -> None:
        from .refine import apply_ops, to_gray
        from pycocotools import mask as mu
        rec = self.collection["records"][self.state.meta[iuid].row]
        base = rec["rle"]
        refined = apply_ops(to_gray(self._rgb(iuid)), mu.decode(base).astype(bool), ops)
        rle = mu.encode(np.asfortranarray(refined.astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
        tok = self.history.begin(self.state, [iuid], [])
        self.state.meta[iuid].refined = True
        self.history.commit(self.state, tok, "refine", f"refine {iuid[:6]}")
        self._overlay_rle[iuid] = rle
        self.store.save_refine(iuid, {"iuid": iuid, "base_rle": base, "ops": ops, "result_rle": rle})
        if "shapecoord" in self.collection["feats"]:
            self.collection["feats"]["shapecoord"][self.state.meta[iuid].row] = _co.shapecoord_vector(refined)
        self._after_mutation()

    def revert_refine(self, iuid: str) -> None:
        self._overlay_rle.pop(iuid, None)
        self.store.delete_refine(iuid)
        tok = self.history.begin(self.state, [iuid], [])
        self.state.meta[iuid].refined = False
        self.history.commit(self.state, tok, "revert_refine", f"revert {iuid[:6]}")
        self._after_mutation()

    # ---- classifier / similar ---------------------------------------------
    def train_classifier(self, spec, *, algo: str = "logreg") -> dict:
        X, y, _ = _clf.build_xy(self.collection, self.state, spec)
        if len(y) < 4 or len(set(y)) < 2:
            return {"error": "need >=2 classes with >=2 assigned instances each"}
        self._clf, report = _clf.train(X, y, algo=algo)
        self._clf_spec = spec
        report["pr"] = _clf.pr_curve(X, y, algo=algo)
        return report

    def predict_and_threshold(self, thresh: float):
        iuids, proba, classes = _clf.predict_unassigned(self._clf, self.collection, self.state, self._clf_spec)
        return _clf.threshold_assign(iuids, proba, classes, thresh)  # [(iuid, class_id, conf)]

    def apply_predictions(self, thresh: float) -> int:
        preds = self.predict_and_threshold(thresh)
        by_class: dict[str, list[str]] = {}
        scores = {}
        for u, cid, conf in preds:
            by_class.setdefault(cid, []).append(u); scores[u] = conf
        for cid, us in by_class.items():
            self.assign(us, self.state.class_name(cid), source="classifier", scores=scores)
        return len(preds)

    def find_similar(self, iuid: str, *, k: int = 20, spec=None):
        return _sim.find_similar(self.collection, self.state, iuid, k=k,
                                 spec=spec or (self._cluster["spec"] if self._cluster else {"decoder": 1.0}))

    # ---- export / import ---------------------------------------------------
    def export_coco(self, out_path=None, **kw):
        out_path = Path(out_path) if out_path else (self.store.export_dir / "curated.json")
        return _ex.export(self.collection, self.state, out_path, rle_override=self._overlay_rle, **kw)

    def import_coco(self, path):
        tok_iuids = list(self.state.meta.keys())
        tok = self.history.begin(self.state, tok_iuids, [])
        rep = _ex.import_coco(path, self.collection, self.state)
        for cid in self.state.taxonomy:
            tok["class_ids"].append(cid)
        self.history.commit(self.state, tok, "import", f"import {rep['matched']}")
        self.save()
        return rep

    # ---- undo / redo / stats ----------------------------------------------
    def undo(self):
        op = self.history.undo(self.state); self.save(); return op

    def redo(self):
        op = self.history.redo(self.state); self.save(); return op

    def embed2d(self, *, method: str = "pca", color_by: str = "cluster"):
        from ._bootstrap import get_P
        P = get_P()
        spec = self._cluster["spec"] if self._cluster else {"decoder": 1.0}
        X = _cl.fused_matrix(self.collection, spec)
        xy = P.embed2d(X, method)
        labels = self._labels() if (self._cluster and color_by == "cluster") else \
            np.array([abs(hash(self.state.meta[u].assigned_class or "")) % 997 for u in self.state.order])
        return xy, labels, list(self.state.order)

    def stats(self) -> dict:
        n_assigned = sum(1 for m in self.state.meta.values() if m.assigned_class and not m.is_background)
        n_bg = sum(1 for m in self.state.meta.values() if m.is_background)
        u, r = self.history.depths
        return {"n_images": len(set(m.image_id for m in self.state.meta.values())),
                "n_instances": len(self.state.order), "n_assigned": n_assigned,
                "n_background": n_bg, "n_unassigned": len(self.state.order) - n_assigned - n_bg,
                "n_classes": len(self.state.taxonomy), "dirty": self.state.collection_dirty,
                "undo": u, "redo": r, "coll_version": self.state.coll_version}

    def _after_mutation(self):
        self._commits += 1
        self.save(snapshot=(self._commits % _AUTOSNAP_EVERY == 0))


def _any_method(collection: dict) -> str:
    for k in collection["feats"]:
        if not k.startswith("_"):
            return k
    raise ValueError("collection has no feature methods")


def _default_level(counts: list[int]) -> int:
    """Pick a mid-granularity level (closest to ~sqrt(N-ish) clusters)."""
    if len(counts) <= 1:
        return 0
    target = max(counts) ** 0.5
    return int(np.argmin([abs(c - target) for c in counts]))


def _default_feat_cfg(config: dict) -> dict:
    f = config.get("features", {})
    mf = set(f.get("model_features", ["decoder", "maskpool", "roialign"]))
    return {"with_features": bool(mf & {"decoder", "maskpool", "roialign", "backbone"}),
            "with_shape": bool(f.get("handcrafted", {}).get("shape", True)),
            "with_backbone": "backbone" in mf,
            "backbone_level": f.get("backbone_level", "p16"),
            "shapecoord": bool(f.get("handcrafted", {}).get("shape_coords_extra", True)),
            "raddino": bool(f.get("raddino", False))}
