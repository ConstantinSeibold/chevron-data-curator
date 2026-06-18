"""CuratorEngine — the UI-agnostic facade tying the curator modules together.

Holds the authoritative mutable project (collection + feature matrices + masks + the
overlay state). The Gradio app binds to one server-side instance. All renders return
numpy RGB images; all mutations go through the History for undo/redo + autosave.
"""
from __future__ import annotations

import colorsys
from collections import OrderedDict
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

# Bounded LRU of decoded RGB source images keyed by abs path. Source images never change, so no
# invalidation — just eviction. WITHOUT this, every crop re-imread()s the full-res JPEG, and a grid
# render does up to _GRID_CAP disk reads → the app stalls (see plan v5.6). Returned arrays are shared
# (read-only); every mutating caller (crop/_crop_mask/image_overlay) copies before drawing.
_IMG_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_IMG_CACHE_MAX = 24


def _load_rgb(path: str, fallback_hw: tuple[int, int] | None = None) -> np.ndarray:
    import cv2
    cached = _IMG_CACHE.get(path)
    if cached is not None:
        _IMG_CACHE.move_to_end(path)
        return cached
    img = cv2.imread(path, cv2.IMREAD_COLOR) if path else None
    rgb = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None
           else np.zeros((*(fallback_hw or (512, 512)), 3), np.uint8))
    _IMG_CACHE[path] = rgb
    _IMG_CACHE.move_to_end(path)
    while len(_IMG_CACHE) > _IMG_CACHE_MAX:
        _IMG_CACHE.popitem(last=False)
    return rgb


def _color(i: int):
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.85, 1.0)
    return np.array([r * 255, g * 255, b * 255])


def _downscale(img: np.ndarray, max_side: int = 220) -> np.ndarray:
    """Shrink to <= max_side on the longest side — keeps gallery payloads small (browser RAM)."""
    import cv2
    h, w = img.shape[:2]
    s = max_side / max(h, w) if max(h, w) > max_side else 1.0
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return img


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

    def compute_raddino(self) -> dict:
        """On-demand RAD-DINO features for the CURRENT collection (no re-detection): soft mask-pool
        each existing instance's mask over the RAD-DINO patch grid (reuses collect._raddino_by_path),
        adding feats['raddino'] aligned to existing rows → 'raddino' becomes selectable. GPU/HF, opt-in."""
        if not self.collection or not self.collection.get("records"):
            return {"error": "no collection — Sample & extract first"}
        if "raddino" in self.collection["feats"]:
            return {"ok": True, "msg": "raddino already present", "available": self.available_features()}
        from ._bootstrap import get_P
        _co._raddino_by_path(self.collection, get_P())
        if "raddino" not in self.collection["feats"]:
            return {"error": "RAD-DINO extraction produced no features"}
        self.state.assert_aligned(self.collection["feats"]["raddino"].shape[0])
        self.state.coll_version += 1
        self.state.collection_dirty = True
        self.store.save_collection(self.collection)
        self.save()
        return {"ok": True, "n": int(self.collection["feats"]["raddino"].shape[0]),
                "available": self.available_features()}

    # ---- clustering --------------------------------------------------------
    def available_features(self) -> list[str]:
        """Feature methods actually present in the collection (single source of truth for the UI
        selectors). Excludes `_`-prefixed metadata keys (e.g. _shape_cols)."""
        if not self.collection or not self.collection.get("feats"):
            return []
        return sorted(k for k in self.collection["feats"] if not k.startswith("_"))

    def _present_spec(self, spec) -> dict:
        """Spec restricted to feature methods present in the collection (drops absent ones)."""
        avail = set(self.available_features())
        return {m: w for m, w in _cl.normalize_spec(spec).items() if m in avail}

    def _pool_iuids(self) -> list[str]:
        """The curation pool that gets clustered: unassigned, non-background, non-merge-child."""
        return [u for u in self.state.order
                if self.state.meta[u].assigned_class is None
                and not self.state.meta[u].is_background and self.state.meta[u].merged_into is None]

    def cluster(self, spec, *, distance: str = "cosine", per_image: bool = False, level: int | None = None,
                req_clust: int | None = None) -> dict:
        """FINCH-cluster ONLY the unassigned pool — already-assigned instances are not reclustered
        (each class becomes its own standalone pseudo-partition in partition_view)."""
        from ._bootstrap import get_P
        P = get_P()
        pool = self._pool_iuids()
        spec = self._present_spec(spec)
        if not spec:
            raise ValueError(f"none of the selected features are present; available: {self.available_features()}")
        if len(pool) < 2:
            partitions, counts = np.zeros((len(pool), 1), int), [max(1, len(pool))]
        else:
            X = _cl.fused_matrix(self.collection, spec)[[self.state.meta[u].row for u in pool]]
            if req_clust:
                labels = np.asarray(P.cluster(X, "finch", req_clust=int(req_clust), distance=distance))
                partitions, counts = labels.reshape(-1, 1), [int(len(set(labels.tolist())))]
            else:
                partitions, counts = P.finch_hierarchy(X, distance=distance)
        lvl = level if level is not None else _default_level(counts)
        self._cluster = {"spec": spec, "distance": distance, "partitions": partitions,
                         "counts": counts, "level": lvl, "pool": pool}
        self.state.collection_dirty = False
        self.save()
        return {"counts": counts, "level": lvl, "n_levels": len(counts)}

    def set_level(self, level: int) -> None:
        if self._cluster:
            self._cluster["level"] = max(0, min(level, len(self._cluster["counts"]) - 1))

    def _pool_labels(self) -> np.ndarray:
        return _cl.labels_at_level(self._cluster["partitions"], self._cluster["level"])

    def _is_pool(self, u: str) -> bool:
        m = self.state.meta[u]
        return m.assigned_class is None and not m.is_background and m.merged_into is None

    def _view_sig(self):
        """O(1) signature of everything partition_view depends on — changes on any mutation (history
        push), undo/redo (depth), (re)cluster (new _cluster object), set_level, or append (coll_version)."""
        u, r = self.history.depths
        return (u, r, self.state.coll_version, id(self._cluster),
                self._cluster["level"] if self._cluster else -1, len(self.state.taxonomy))

    def partition_view(self) -> list[dict]:
        """Per-class pseudo-partitions (assigned instances, pid='class:<cid>') first, then the FINCH
        partitions of the still-unassigned pool (pid=str int), filtered to current membership.
        Memoized on _view_sig() — called 2-3x per interaction; recompute only when state actually changes."""
        sig = self._view_sig()
        cached = getattr(self, "_pv_cache", None)
        if cached is not None and cached[0] == sig:
            return cached[1]
        rows = []
        for cid in self.state.taxonomy:
            members = [u for u, m in self.state.meta.items()
                       if m.assigned_class == cid and not m.is_background and m.merged_into is None]
            if members:
                sc = [self.collection["records"][self.state.meta[u].row]["score"] for u in members]
                rows.append({"pid": f"class:{cid}", "size": len(members), "purity": 1.0,
                             "mean_score": round(float(np.mean(sc)), 2), "majority_class": self.state.class_name(cid)})
        if self._cluster:
            labels, pool = self._pool_labels(), self._cluster["pool"]
            for pid in sorted(set(int(x) for x in labels)):
                members = [pool[i] for i in np.where(labels == pid)[0] if self._is_pool(pool[i])]
                if members:
                    sc = [self.collection["records"][self.state.meta[u].row]["score"] for u in members]
                    rows.append({"pid": str(pid), "size": len(members), "purity": None,
                                 "mean_score": round(float(np.mean(sc)), 2), "majority_class": ""})
        rows.sort(key=lambda r: (not str(r["pid"]).startswith("class:"), -r["size"]))
        self._pv_cache = (sig, rows)
        return rows

    def partition_iuids(self, pid) -> list[str]:
        pid = str(pid)
        if pid.startswith("class:"):
            cid = pid[len("class:"):]
            return [u for u, m in self.state.meta.items()
                    if m.assigned_class == cid and not m.is_background and m.merged_into is None]
        if not self._cluster:
            return []
        labels, pool = self._pool_labels(), self._cluster["pool"]
        try:
            target = int(pid)
        except ValueError:
            return []
        return [pool[i] for i in np.where(labels == target)[0] if self._is_pool(pool[i])]

    # ---- rendering ---------------------------------------------------------
    def _eff_rle(self, iuid: str) -> dict:
        # overlay (refine/merge result) is used ONLY when the meta flag is set, so undo/redo of
        # refine/merge — which toggle those flags — actually revert the effective mask.
        m = self.state.meta[iuid]
        if (m.refined or m.merge_members) and iuid in self._overlay_rle:
            return self._overlay_rle[iuid]
        return self.collection["records"][m.row]["rle"]

    def _mask(self, iuid: str) -> np.ndarray:
        from pycocotools import mask as mu
        return mu.decode(self._eff_rle(iuid)).astype(bool)

    def _rgb(self, iuid: str) -> np.ndarray:
        rec = self.collection["records"][self.state.meta[iuid].row]
        return _load_rgb(rec.get("abs_path") or rec["file_name"], (int(rec["H"]), int(rec["W"])))

    def crop(self, iuid: str, *, mask_overlay: bool = True, pad: int = 10, context: bool = False,
             max_side: int = 512) -> np.ndarray:
        """Thumbnail crop of the instance (default) or the WHOLE source image with the instance
        highlighted (context=True), downscaled to <= max_side. Works on the instance's BBOX sub-region
        only (not a full-image copy/overlay) so cost is independent of the source resolution."""
        import cv2
        rgb = self._rgb(iuid)                               # cached; never mutate in place
        m = self._mask(iuid)
        H, W = m.shape
        c = _color(self.state.meta[iuid].row)
        x, y, w, h = cv2.boundingRect(m.astype(np.uint8))   # (0,0,0,0) when empty
        if context or w == 0:                               # whole image (rare; "in context" view)
            out = rgb.copy()
            if w:
                cv2.rectangle(out, (max(0, x - 2), max(0, y - 2)), (min(W, x + w + 2), min(H, y + h + 2)),
                              tuple(int(v) for v in c), 2)
            return _downscale(out, max_side)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
        sub = rgb[y1:y2, x1:x2].copy()                      # SMALL region only
        if mask_overlay:
            subm = m[y1:y2, x1:x2]
            sub[subm] = (0.5 * sub[subm] + 0.5 * c).astype(np.uint8)
            cont, _ = cv2.findContours(subm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sub, cont, -1, tuple(int(v) for v in c), 1)
        return _downscale(sub, max_side)                    # source name in the UI caption, not pixels

    def _src_name(self, iuid: str) -> str:
        return Path(self.collection["records"][self.state.meta[iuid].row].get("file_name", "")).name

    def _caption(self, iuid: str) -> str:
        m = self.state.meta[iuid]
        cls = f" [{self.state.class_name(m.assigned_class)}]" if m.assigned_class else ""
        return f"{self._src_name(iuid)} · {iuid[:6]} s={self.collection['records'][m.row]['score']:.2f}{cls}"

    def partition_crops(self, pid: int, *, mask_overlay: bool = True, limit: int = 48, context: bool = False):
        iuids = self.partition_iuids(pid)[:limit]
        return [(self.crop(u, mask_overlay=mask_overlay, context=context), self._caption(u)) for u in iuids], iuids

    def image_ids(self) -> list[int]:
        from collections import Counter
        c = Counter(m.image_id for m in self.state.meta.values())
        return [iid for iid, _ in c.most_common()]

    def image_overlay(self, image_id: int, *, color_by: str = "partition", max_side: int = 900) -> np.ndarray:
        """Whole-image overlay for the In-image tab, computed on a DOWNSCALED canvas (it's shown ~440px),
        so cost is independent of the source resolution (was full-res float ops per instance)."""
        import cv2
        iuids = self.image_instance_iuids(image_id)        # excludes merge children (rep shows the union)
        if not iuids:
            return np.zeros((512, 512, 3), np.uint8)
        rgb = self._rgb(iuids[0]); H, W = rgb.shape[:2]
        s = max_side / max(H, W) if max(H, W) > max_side else 1.0
        out = (cv2.resize(rgb, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
               if s < 1.0 else rgb.copy()).astype(np.float32)
        h2, w2 = out.shape[:2]
        labels = self._label_for_order(color_by) if (self._cluster and color_by == "partition") else None
        for u in iuids:
            m = self._mask(u)
            if s < 1.0:
                m = cv2.resize(m.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST) > 0
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
        img = self._rgb(iuids[0]).copy()                 # cached array is shared; overlay_keypoints may draw in place
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
        return self._rgb(iuids[0]).copy() if iuids else np.zeros((512, 512, 3), np.uint8)  # overlay_groups draws in place

    def _merge_mask(self, iuids: list[str], mode: str):
        """Combine the group's masks per mode (a/b = highest/2nd-highest score):
        union=OR, intersection=AND, pref_a=top instance's mask, pref_b=2nd instance's mask.
        Returns (result_mask, ordered_iuids) with ordered[0] the representative."""
        ordered = sorted(iuids, key=lambda u: -self.collection["records"][self.state.meta[u].row]["score"])
        masks = [self._mask(u) for u in ordered]
        if mode == "intersection":
            res = masks[0].copy()
            for m in masks[1:]:
                res &= m
        elif mode == "pref_a":
            res = masks[0]
        elif mode == "pref_b":
            res = masks[1] if len(masks) > 1 else masks[0]
        else:                                                   # union (default)
            res = masks[0].copy()
            for m in masks[1:]:
                res |= m
        return res, ordered

    def merge_result_preview(self, iuids: list[str], mode: str = "union", *, max_side: int = 256) -> np.ndarray:
        """Render what merging `iuids` with `mode` would look like (green mask on the group's bbox crop),
        WITHOUT committing. For the in-image / merge-rec previews."""
        import cv2
        iuids = [u for u in iuids if u in self.state.meta]
        if len(iuids) < 2:
            return np.zeros((64, 64, 3), np.uint8)
        res, ordered = self._merge_mask(iuids, mode)
        union = None                                            # always crop to the union bbox (stable framing)
        for u in ordered:
            m = self._mask(u); union = m if union is None else (union | m)
        rgb = self._rgb(ordered[0]); H, W = res.shape
        x, y, w, h = cv2.boundingRect(union.astype(np.uint8))
        if w == 0:
            return _downscale(rgb.copy(), max_side)
        x1, y1 = max(0, x - 10), max(0, y - 10); x2, y2 = min(W, x + w + 10), min(H, y + h + 10)
        sub = rgb[y1:y2, x1:x2].copy(); subm = res[y1:y2, x1:x2]
        if subm.any():
            sub[subm] = (0.5 * sub[subm] + 0.5 * np.array([40, 220, 40])).astype(np.uint8)
            cont, _ = cv2.findContours(subm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sub, cont, -1, (40, 220, 40), 1)
        return _downscale(sub, max_side)

    def _merge_group_nohist(self, iuids: list[str], mode: str = "union") -> str:
        """Merge a group into the highest-score representative (no history). Returns rep."""
        from pycocotools import mask as mu
        res, ordered = self._merge_mask(iuids, mode)
        rep = ordered[0]
        rle = mu.encode(np.asfortranarray(res.astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
        for u in iuids:
            self.state.meta[u].merged_into = (None if u == rep else rep)
        self.state.meta[rep].merge_members = [u for u in iuids if u != rep]
        self._overlay_rle[rep] = rle
        self.store.save_refine(rep, {"iuid": rep, "base_rle": self.collection["records"][self.state.meta[rep].row]["rle"],
                                     "ops": [{"name": "merge", "mode": mode, "members": iuids}], "result_rle": rle})
        return rep

    def _commit_merge_groups(self, groups_iuids: list[list[str]], label: str, mode: str = "union") -> int:
        groups_iuids = [g for g in groups_iuids if len(g) >= 2]
        if not groups_iuids:
            return 0
        all_iuids = [u for g in groups_iuids for u in g]
        tok = self.history.begin(self.state, all_iuids, [])
        for g in groups_iuids:
            self._merge_group_nohist(g, mode)
            self.store.append_merge_event({"kind": "merge", "iuids": list(g), "mode": mode,   # positive training signal
                                           "image_id": int(self.state.meta[g[0]].image_id)})
        self.history.commit(self.state, tok, "merge", label)
        self._after_mutation()
        return len(groups_iuids)

    def commit_merge(self, image_id: int, groups: list[list[int]], mode: str = "union") -> None:
        """groups are GLOBAL row indices (from merge_preview)."""
        order = self.state.order
        self._commit_merge_groups([[order[r] for r in g] for g in groups], f"merge img {image_id}", mode)

    def merge_instances(self, iuids: list[str], mode: str = "union") -> None:
        """Manual merge of an explicit instance set into one (in-image crop-select / canvas-click)."""
        self._commit_merge_groups([list(iuids)], f"merge {len(iuids)} instances", mode)

    def merge_partition_by_image(self, pid: int) -> int:
        """Merge all same-image instances within a partition (small partitions with dup regions)."""
        from collections import defaultdict
        by_img = defaultdict(list)
        for u in self.partition_iuids(pid):
            by_img[self.state.meta[u].image_id].append(u)
        return self._commit_merge_groups([g for g in by_img.values() if len(g) >= 2],
                                         f"merge same-image in partition {pid}")

    def dedup_current(self, iou: float = 0.8) -> int:
        """Mark near-duplicate (mask-IoU >= iou) lower-score instances per image as background
        (reversible). For already-collected sets."""
        from collections import defaultdict
        from pycocotools import mask as mu
        by_img = defaultdict(list)
        for u, m in self.state.meta.items():
            if not m.is_background and m.merged_into is None:
                by_img[m.image_id].append(u)
        to_bg = []
        for iuids in by_img.values():
            order = sorted(iuids, key=lambda u: -self.collection["records"][self.state.meta[u].row]["score"])
            kept = []
            for u in order:
                rle = self._eff_rle(u)
                if kept and float(np.max(mu.iou([rle], kept, [0] * len(kept)))) >= iou:
                    to_bg.append(u)
                else:
                    kept.append(rle)
        if to_bg:
            tok = self.history.begin(self.state, to_bg, [])
            for u in to_bg:
                self.state.meta[u].is_background = True
            self.history.commit(self.state, tok, "dedup", f"dedup→bg {len(to_bg)}")
            self._after_mutation()
        return len(to_bg)

    def instance_at_pixel(self, image_id: int, x: int, y: int) -> str | None:
        """Highest-score instance in the image whose effective mask covers pixel (x,y)."""
        cands = [u for u, m in self.state.meta.items() if m.image_id == image_id]
        cands.sort(key=lambda u: -self.collection["records"][self.state.meta[u].row]["score"])
        for u in cands:
            m = self._mask(u)
            if 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1] and m[int(y), int(x)]:
                return u
        return None

    def image_instance_gallery(self, image_id: int, *, mask_overlay: bool = True):
        iuids = self.image_instance_iuids(image_id)
        return [(self.crop(u, mask_overlay=mask_overlay), self._caption(u)) for u in iuids], iuids

    def image_instance_iuids(self, image_id: int) -> list[str]:
        # hide merge CHILDREN (merged_into set) — a merged group collapses to its representative,
        # whose effective mask is the union, so the in-image grid updates after a merge.
        return [u for u, m in self.state.meta.items() if m.image_id == image_id and m.merged_into is None]

    def background_iuids(self) -> list[str]:
        return [u for u, m in self.state.meta.items() if m.is_background]

    def unreject(self, iuids: list[str]) -> int:
        """Send rejected (background) instances back to UNASSIGNED. Reversible."""
        bg = [u for u in iuids if u in self.state.meta and self.state.meta[u].is_background]
        if not bg:
            return 0
        tok = self.history.begin(self.state, bg, [])
        for u in bg:
            self.state.meta[u].is_background = False
            self.state.meta[u].assigned_class = None
        self.history.commit(self.state, tok, "unreject", f"unreject {len(bg)}")
        self._after_mutation()
        return len(bg)

    def reset(self, *, keep_config: bool = True) -> None:
        """Drop everything (collection, instances, assignments, classes, overlays, caches,
        processed-image list); keep only the config. Destructive, NOT undoable."""
        cfg = dict(self.state.config) if keep_config else {}
        if self.store.collection_path.exists():
            self.store.collection_path.unlink()
        self.store.clear_cache()
        for f in self.store.refine_dir.glob("*.pkl"):
            f.unlink()
        man = self.store.load_manifest()
        man.update({"processed_paths": [], "coll_version": 0, "n_instances": 0})
        self.store.save_manifest(man)
        self.state = CuratorState(project_dir=str(self.store.dir), config=cfg)
        self.collection = None
        self._overlay_rle = {}
        self._cluster = None
        self._clf = None
        self.history.barrier()
        self.save()

    # ---- refinement --------------------------------------------------------
    def refine_preview(self, iuid: str, ops: list[dict], *, mask_overlay: bool = True):
        from .refine import apply_ops, to_gray
        img = self._rgb(iuid)
        base = self.collection["records"][self.state.meta[iuid].row]["rle"]
        from pycocotools import mask as mu
        base_m = mu.decode(base).astype(bool)
        refined = apply_ops(to_gray(img), base_m, ops)
        return (self._crop_mask(img, base_m, mask_overlay=mask_overlay),
                self._crop_mask(img, refined, mask_overlay=mask_overlay))

    def _crop_mask(self, img, m, pad=12, *, mask_overlay=True, max_side=512):
        import cv2
        ys, xs = np.where(m); H, W = m.shape
        out = img.copy()
        if len(xs) == 0:
            return _downscale(out, max_side)
        if mask_overlay:                                  # green overlay only when toggled on
            out[m] = (0.5 * out[m] + 0.5 * np.array([40, 220, 40])).astype(np.uint8)
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, (40, 220, 40), 1)
        x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad)   # always crop to the instance
        x2, y2 = min(W, xs.max() + pad + 1), min(H, ys.max() + pad + 1)
        return _downscale(out[y1:y2, x1:x2], max_side)

    def _refine_one_nohist(self, iuid: str, ops: list[dict]) -> None:
        from .refine import apply_ops, to_gray
        from pycocotools import mask as mu
        rec = self.collection["records"][self.state.meta[iuid].row]
        base = rec["rle"]
        refined = apply_ops(to_gray(self._rgb(iuid)), mu.decode(base).astype(bool), ops)
        rle = mu.encode(np.asfortranarray(refined.astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
        self.state.meta[iuid].refined = True
        self._overlay_rle[iuid] = rle
        self.store.save_refine(iuid, {"iuid": iuid, "base_rle": base, "ops": ops, "result_rle": rle})
        if "shapecoord" in self.collection["feats"]:
            self.collection["feats"]["shapecoord"][self.state.meta[iuid].row] = _co.shapecoord_vector(refined)

    def apply_refine(self, iuid: str, ops: list[dict]) -> None:
        tok = self.history.begin(self.state, [iuid], [])
        self._refine_one_nohist(iuid, ops)
        self.history.commit(self.state, tok, "refine", f"refine {iuid[:6]}")
        self._after_mutation()

    def apply_refine_partition(self, pid: int, ops: list[dict]) -> int:
        """Apply the op stack to EVERY instance in a partition (one undoable command)."""
        iuids = self.partition_iuids(pid)
        if not iuids:
            return 0
        tok = self.history.begin(self.state, iuids, [])
        for u in iuids:
            self._refine_one_nohist(u, ops)
        self.history.commit(self.state, tok, "refine_partition", f"refine partition {pid} ({len(iuids)})")
        self._after_mutation()
        return len(iuids)

    def apply_refine_many(self, iuids: list[str], ops: list[dict]) -> int:
        """Refine an explicit set of instances in one undoable command."""
        iuids = [u for u in iuids if u in self.state.meta]
        if not iuids:
            return 0
        tok = self.history.begin(self.state, iuids, [])
        for u in iuids:
            self._refine_one_nohist(u, ops)
        self.history.commit(self.state, tok, "refine_many", f"refine {len(iuids)} instances")
        self._after_mutation()
        return len(iuids)

    def refine_partition_preview(self, pid: int, ops: list[dict], n: int = 6):
        """Before/after crops for the first n instances of a partition (no persistence)."""
        befores, afters = [], []
        for u in self.partition_iuids(pid)[:n]:
            o, r = self.refine_preview(u, ops)
            befores.append((o, u[:6])); afters.append((r, u[:6]))
        return befores, afters

    def split_instances(self, iuids: list[str], *, min_area_frac: float = 0.002, connectivity: int = 8) -> int:
        """Split each instance's effective mask into its connected components, appending ONE new
        unassigned instance per component (new iuid/record/feats row) and sending the originals to
        background. New instances copy the parent's model features and recompute geometry/shapecoord;
        appending instances is an undo BARRIER (like sample_more), so this clears the undo stack.
        Re-cluster to see the children in partitions (they show in In-image immediately)."""
        import cv2
        from pycocotools import mask as mu

        from . import ids as _ids
        from .state import InstanceMeta
        feats = self.collection["feats"]
        recs = self.collection["records"]
        methods = [k for k in feats if not k.startswith("_")]
        new_records: list[dict] = []
        new_feats: dict[str, list] = {k: [] for k in methods}
        parents: list[str] = []
        for u in list(iuids):
            if u not in self.state.meta:
                continue
            m = self._mask(u)
            H, W = m.shape
            n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=connectivity)
            comps = [(lab == c) for c in range(1, n)]
            comps = [c for c in comps if int(c.sum()) >= max(1, int(min_area_frac * m.size))]
            if len(comps) < 2:                                    # nothing to split
                continue
            parents.append(u)
            prow = self.state.meta[u].row
            base = recs[prow]
            for c in comps:
                ys, xs = np.where(c)
                rle = mu.encode(np.asfortranarray(c.astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
                nu = _ids.new_uid()
                r = dict(base); r.pop("keypoints", None); r.pop("keypoint_vis", None)
                r.update({"iuid": nu, "rle": rle, "batch_id": f"{base.get('batch_id', 'b')}/split",
                          "cx": float(xs.mean() / W), "cy": float(ys.mean() / H),
                          "bw": float((xs.max() - xs.min() + 1) / W), "bh": float((ys.max() - ys.min() + 1) / H),
                          "box_area": float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) / (W * H)),
                          "mask_area_frac": float(c.mean())})
                new_records.append(r)
                for k in methods:
                    new_feats[k].append(_co.shapecoord_vector(c) if k == "shapecoord" else feats[k][prow].copy())
        if not new_records:
            return 0
        batch = {"records": new_records, "n_images": 0,
                 "feats": {k: np.asarray(v, dtype=feats[k].dtype) for k, v in new_feats.items()}}
        self.collection = _co.concat_collections(self.collection, batch)
        self.state.order = [r["iuid"] for r in self.collection["records"]]
        for r in new_records:
            self.state.meta[r["iuid"]] = InstanceMeta(
                iuid=r["iuid"], batch_id=r["batch_id"], row=r["row"], image_id=int(r["image_id"]),
                provenance={"split_from": "", "file": r.get("abs_path", "")})
        for u in parents:
            self.state.meta[u].is_background = True
            self.state.meta[u].assigned_class = None
        self.state.rebuild_rows()
        self.state.assert_aligned(self.collection["feats"][_any_method(self.collection)].shape[0])
        self.state.coll_version += 1
        self.state.collection_dirty = True
        self.store.save_collection(self.collection)
        self.history.barrier()
        self.save()
        return len(new_records)

    def revert_refine(self, iuid: str) -> None:
        self._overlay_rle.pop(iuid, None)
        self.store.delete_refine(iuid)
        tok = self.history.begin(self.state, [iuid], [])
        self.state.meta[iuid].refined = False
        self.history.commit(self.state, tok, "revert_refine", f"revert {iuid[:6]}")
        self._after_mutation()

    # ---- classifier / similar ---------------------------------------------
    def train_classifier(self, spec, *, algo: str = "logreg", use_unassigned_negatives: bool = True,
                         knn_k: int = 5, knn_metric: str = "cosine", knn_weights: str = "distance") -> dict:
        """algo 'logreg'/'rf' -> factored open-set classifier (P(c vs not-c)*P(c vs others), needs >=2
        per class); algo 'knn' -> distance vote over k nearest assigned (+background as reject neighbours),
        works with >=1 per class. Both expose .classes/.proba so predict/apply are identical downstream."""
        spec = self._present_spec(spec)
        if not spec:
            return {"error": f"none of the selected features are present; available: {self.available_features()}"}
        if algo == "knn":
            clf, report = _clf.train_knn(self.collection, self.state, spec, k=int(knn_k), metric=knn_metric,
                                         weights=knn_weights, use_unassigned_negatives=use_unassigned_negatives)
        else:
            clf, report = _clf.train_factored(self.collection, self.state, spec, algo=algo,
                                              use_unassigned_negatives=use_unassigned_negatives)
        if report.get("skipped_classes"):
            report["skipped_names"] = [self.state.class_name(c) for c in report["skipped_classes"]]
        if clf is None:
            return report
        self._clf = clf
        self._clf_spec = spec
        return report

    def predict_and_threshold(self, thresh: float, only_class: str | None = None):
        iuids = self.state.unassigned_iuids()                   # only ever scores not-yet-classified instances
        if not iuids or getattr(self, "_clf", None) is None:
            return []
        X = _cl.fused_matrix(self.collection, _cl.normalize_spec(self._clf_spec))
        rows = [self.state.meta[u].row for u in iuids]
        proba = self._clf.proba(X[rows])
        return _clf.threshold_assign(iuids, proba, self._clf.classes, float(thresh), only_class=only_class)

    def apply_predictions(self, thresh: float, only_class: str | None = None, exclude=None) -> int:
        preds = self.predict_and_threshold(thresh, only_class=only_class)
        exclude = set(exclude or [])                            # instances the user removed in the preview
        preds = [(u, c, conf) for u, c, conf in preds if u not in exclude]
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

    # ---- merge recommender (learns from past in-image merges) --------------
    def _merge_pos_groups(self) -> list[list[str]]:
        """Positive merge groups: logged merge events (survive undo/unmerge) + current merge_members,
        deduplicated by member-set (a still-active merge appears in BOTH sources)."""
        seen, groups = set(), []
        def _add(g):
            g = [u for u in g if u in self.state.meta]
            key = frozenset(g)
            if len(g) >= 2 and key not in seen:
                seen.add(key); groups.append(g)
        for ev in self.store.read_merge_events():
            if ev.get("kind") == "merge":
                _add(ev.get("iuids", []))
        for u, m in self.state.meta.items():
            if m.merge_members:
                _add([u] + list(m.merge_members))
        return groups

    def _merge_rejected_groups(self) -> list[list[str]]:
        return [[u for u in ev.get("iuids", []) if u in self.state.meta]
                for ev in self.store.read_merge_events() if ev.get("kind") == "reject"]

    def train_merge_recommender(self, spec, *, algo: str = "logreg") -> dict:
        from . import merge_rec as _mr
        spec = self._present_spec(spec)
        pos = self._merge_pos_groups()
        if not pos:
            return {"error": "no merges recorded yet — merge some instances in the In-image tab first"}
        X, y, _, rep = _mr.build_pair_xy(self.collection, self.state, pos, self._merge_rejected_groups(), spec)
        if rep["n_pos"] == 0 or rep["n_neg"] == 0:
            return {"error": f"not enough training pairs (positives={rep['n_pos']}, negatives={rep['n_neg']})"}
        self._merge_clf = _mr.train(X, y, algo=algo)
        self._merge_spec = spec
        rep.update(_mr.pr_youden(X, y, algo=algo))
        rep["n_merge_events"] = len(pos)
        return rep

    def recommend_merges(self, thresh: float, *, max_groups: int = 20) -> list[dict]:
        from . import merge_rec as _mr
        if getattr(self, "_merge_clf", None) is None:
            return []
        return _mr.candidate_groups(self.collection, self.state, self._merge_clf, self._merge_spec,
                                    float(thresh), max_groups=max_groups)

    def accept_merge(self, iuids: list[str], mode: str = "union") -> None:
        self.merge_instances(list(iuids), mode=mode)       # logs a merge event via _commit_merge_groups

    def reject_merge(self, iuids: list[str]) -> None:
        iuids = [u for u in iuids if u in self.state.meta]
        if len(iuids) >= 2:
            self.store.append_merge_event({"kind": "reject", "iuids": list(iuids),
                                           "image_id": int(self.state.meta[iuids[0]].image_id)})

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
        return xy, self._label_for_order(color_by), list(self.state.order)

    def _label_for_order(self, color_by: str) -> np.ndarray:
        """Per-order integer label for the Map: pool->FINCH label, assigned->1000+classidx,
        background->-1 (color_by='class' colors only by assigned class)."""
        order = self.state.order
        cids = list(self.state.taxonomy)
        if color_by == "class" or not self._cluster:
            return np.array([(1000 + cids.index(self.state.meta[u].assigned_class))
                             if self.state.meta[u].assigned_class in cids else
                             (-1 if self.state.meta[u].is_background else 0) for u in order])
        labels, pool = self._pool_labels(), self._cluster["pool"]
        pool_lab = {pool[i]: int(labels[i]) for i in range(len(pool))}
        out = []
        for u in order:
            m = self.state.meta[u]
            if u in pool_lab and self._is_pool(u):
                out.append(pool_lab[u])
            elif m.is_background:
                out.append(-1)
            elif m.assigned_class in cids:
                out.append(1000 + cids.index(m.assigned_class))
            else:
                out.append(-2)
        return np.array(out)

    def embed_thumbnails(self, *, method: str = "pca", color_by: str = "cluster",
                         max_pts: int = 250, thumb: int = 64):
        """embed2d + a small base64 JPG thumbnail per point (for the Map hover JS) + source names.
        Thumbnails computed for up to max_pts points (others empty) — each thumb is a crop (disk read,
        now cached); the cap keeps 'Compute map' to seconds instead of minutes on big projects."""
        import base64
        import cv2
        xy, labels, order = self.embed2d(method=method, color_by=color_by)
        n = len(order)
        idxs = range(n) if n <= max_pts else set(np.linspace(0, n - 1, max_pts).astype(int).tolist())
        thumbs, names = [""] * n, [""] * n
        for i in range(n):
            u = order[i]
            names[i] = Path(self.collection["records"][self.state.meta[u].row].get("file_name", "")).name
            if i in idxs:
                cr = cv2.resize(self.crop(u, mask_overlay=True), (thumb, thumb))
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(cr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    thumbs[i] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        return xy, labels, order, thumbs, names

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
            "raddino": bool(f.get("raddino", False)),
            "nms_iou": float(config.get("model", {}).get("nms_iou", 0.8))}
