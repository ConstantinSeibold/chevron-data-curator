"""On-disk project layout + atomic IO for a curation project.

    <project_dir>/
      state.json          CuratorState.to_dict() — config + taxonomy + meta(overlay) + order  (source of truth)
      manifest.json       {schema_version, coll_version, processed_paths[], n_instances}
      collection.pkl      the heavy {records, feats, n_images} collection (pickle)
      history.jsonl       append-only command/audit log (undo/redo + provenance)
      refine/<iuid>.pkl   reversible refine overlay {base_rle, ops[], result_rle, shape}
      cluster_cache/<k>.npz  cached FINCH partitions, keyed by (spec, distance, ..., coll_version)
      snapshots/<ts>/     coarse restore points (state.json + manifest.json copies)
      exports/            COCO exports

All small-file writes are tmp -> os.replace (atomic on POSIX).
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .state import CuratorState

SCHEMA_VERSION = 1


class Store:
    def __init__(self, project_dir: str | Path):
        self.dir = Path(project_dir)
        self.refine_dir = self.dir / "refine"
        self.cache_dir = self.dir / "cluster_cache"
        self.snap_dir = self.dir / "snapshots"
        self.export_dir = self.dir / "exports"

    # ---- layout ------------------------------------------------------------
    def ensure(self) -> None:
        for d in (self.dir, self.refine_dir, self.cache_dir, self.snap_dir, self.export_dir):
            d.mkdir(parents=True, exist_ok=True)

    def is_project(self) -> bool:
        return (self.dir / "state.json").exists()

    @property
    def state_path(self) -> Path: return self.dir / "state.json"
    @property
    def manifest_path(self) -> Path: return self.dir / "manifest.json"
    @property
    def collection_path(self) -> Path: return self.dir / "collection.pkl"
    @property
    def history_path(self) -> Path: return self.dir / "history.jsonl"

    # ---- atomic primitives -------------------------------------------------
    @staticmethod
    def _tmp(path: Path) -> Path:
        # UNIQUE temp name per write: the threaded server runs requests in parallel, so a fixed
        # "<file>.tmp" lets two concurrent saves collide (one os.replace moves the shared tmp, the
        # other then FileNotFoundErrors). os.replace stays atomic; the unique name avoids the race.
        return path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")

    def _write_json(self, path: Path, obj: Any) -> None:
        tmp = self._tmp(path)
        tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
        os.replace(tmp, path)

    def _write_bytes(self, path: Path, data: bytes) -> None:
        tmp = self._tmp(path)
        tmp.write_bytes(data)
        os.replace(tmp, path)

    # ---- state / manifest --------------------------------------------------
    def save_state(self, state: CuratorState) -> None:
        self.ensure()
        self._write_json(self.state_path, state.to_dict())

    def load_state(self) -> CuratorState:
        return CuratorState.from_dict(json.loads(self.state_path.read_text()))

    def save_manifest(self, manifest: dict) -> None:
        self.ensure()
        self._write_json(self.manifest_path, manifest)

    def load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {"schema_version": SCHEMA_VERSION, "coll_version": 0, "processed_paths": [], "n_instances": 0}

    # ---- collection (heavy pickle) ----------------------------------------
    def save_collection(self, collection: dict) -> None:
        self.ensure()
        tmp = self.collection_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(collection, f, protocol=4)
        os.replace(tmp, self.collection_path)

    def load_collection(self) -> dict:
        with open(self.collection_path, "rb") as f:
            return pickle.load(f)

    def has_collection(self) -> bool:
        return self.collection_path.exists()

    # ---- history (jsonl) ---------------------------------------------------
    def append_history(self, record: dict) -> None:
        self.ensure()
        with open(self.history_path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def read_history(self, limit: int | None = None) -> list[dict]:
        if not self.history_path.exists():
            return []
        lines = self.history_path.read_text().splitlines()
        if limit:
            lines = lines[-limit:]
        return [json.loads(ln) for ln in lines if ln.strip()]

    # ---- merge log (jsonl) — training signal for the merge recommender (survives undo/unmerge) ----
    @property
    def merge_log_path(self) -> Path: return self.dir / "merge_log.jsonl"

    def append_merge_event(self, record: dict) -> None:
        self.ensure()
        with open(self.merge_log_path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def read_merge_events(self) -> list[dict]:
        if not self.merge_log_path.exists():
            return []
        return [json.loads(ln) for ln in self.merge_log_path.read_text().splitlines() if ln.strip()]

    # ---- refine overlays ---------------------------------------------------
    def refine_path(self, iuid: str) -> Path:
        return self.refine_dir / f"{iuid}.pkl"

    def save_refine(self, iuid: str, obj: dict) -> None:
        self.refine_dir.mkdir(parents=True, exist_ok=True)
        with open(self.refine_path(iuid), "wb") as f:
            pickle.dump(obj, f, protocol=4)

    def load_refine(self, iuid: str) -> dict | None:
        p = self.refine_path(iuid)
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def delete_refine(self, iuid: str) -> None:
        p = self.refine_path(iuid)
        if p.exists():
            p.unlink()

    # ---- cluster cache (npz) ----------------------------------------------
    def cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npz"

    def save_cache(self, key: str, partitions: np.ndarray, counts: list[int], order: list[str]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cache_path(key), partitions=partitions,
                            counts=np.asarray(counts), order=np.asarray(order, dtype=object))

    def load_cache(self, key: str):
        p = self.cache_path(key)
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=True)
        return z["partitions"], [int(x) for x in z["counts"]], list(z["order"])

    def clear_cache(self) -> None:
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.npz"):
                f.unlink()

    # ---- snapshots ---------------------------------------------------------
    def snapshot(self, tag: str = "") -> str:
        """Copy the small state files into snapshots/<ts><_tag>/ . Returns the snapshot dir name."""
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}{('_' + tag) if tag else ''}"
        dst = self.snap_dir / name
        dst.mkdir(exist_ok=True)
        for p in (self.state_path, self.manifest_path):
            if p.exists():
                shutil.copy2(p, dst / p.name)
        return name


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")
