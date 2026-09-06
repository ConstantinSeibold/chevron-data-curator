"""Project discovery, summaries and lifecycle for the starter UI.

A Chevron *project* is a self-contained directory (see `store.Store`): `state.json` is the source of
truth, so projects are already fully isolated from one another on disk. This module adds the registry
around them — enumerate what exists, describe each one cheaply enough to render a card, and create or
delete them.

Summaries are computed WITHOUT constructing a `CuratorEngine`: opening an engine loads the heavy
`collection.pkl`, and the launcher would then pay that for every project it lists. Instead the summary
reads `manifest.json` (+ `state.json` for class/assignment counts) and is memoised in
`<root>/.registry.json` keyed by `state.json`'s (mtime, size), so a big project is parsed once per
change rather than once per page load.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .store import Store

REGISTRY_FILE = ".registry.json"


@dataclass
class ProjectInfo:
    """One launcher card."""
    id: str                                   # directory name — the stable handle used by the API
    name: str                                 # display name (editable, defaults to the id)
    path: str
    n_instances: int = 0
    n_assigned: int = 0
    n_rejected: int = 0
    n_unassigned: int = 0
    n_classes: int = 0
    n_images: int = 0
    pct_curated: float = 0.0
    modified: float = 0.0                     # unix ts of the most recent state write
    clustered: bool = False
    sources: list[str] = field(default_factory=list)
    mode: str = "instance"
    modality: str = "image"
    error: str | None = None                  # set when the project dir is unreadable/corrupt

    def to_dict(self) -> dict:
        return asdict(self)


def slugify(name: str) -> str:
    """Filesystem-safe project id. Collapses runs of non-alphanumerics to single dashes."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "").strip()).strip("-").lower()
    return s or "project"


def _summarize_state(state_path: Path) -> dict[str, Any]:
    """Class/assignment counts straight from `state.json` — no engine, no collection.pkl.

    Mirrors `CuratorState`'s own definitions: a merge CHILD (`merged_into` set) is not live, an
    instance is rejected when `is_background`, and assigned when it carries a class and is not
    rejected.
    """
    d = json.loads(state_path.read_text())
    meta = d.get("meta", {}) or {}
    assigned = rejected = unassigned = 0
    images: set = set()
    for m in meta.values():
        if m.get("merged_into") is not None:
            continue                                        # merge child — represented by its parent
        images.add(m.get("image_id"))
        if m.get("is_background"):
            rejected += 1
        elif m.get("assigned_class") is not None:
            assigned += 1
        else:
            unassigned += 1
    live = assigned + rejected + unassigned
    cfg = d.get("config", {}) or {}
    # a `temp` leaf is a scratch class, excluded from the exported taxonomy — don't advertise it
    taxonomy = d.get("taxonomy", {}) or {}
    return {
        "n_assigned": assigned,
        "n_rejected": rejected,
        "n_unassigned": unassigned,
        "n_classes": sum(1 for c in taxonomy.values() if not c.get("temp")),
        "n_images": len(images),
        "pct_curated": round(100.0 * (assigned + rejected) / live, 1) if live else 0.0,
        "mode": str(cfg.get("mode", "instance")),
        "modality": str(cfg.get("modality", "image")),
    }


class ProjectRegistry:
    """Projects under one root directory, plus a summary cache."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- registry file (display names + memoised summaries) ----------------
    @property
    def _registry_path(self) -> Path:
        return self.root / REGISTRY_FILE

    def _load_registry(self) -> dict:
        try:
            return json.loads(self._registry_path.read_text())
        except Exception:
            return {}                                        # a corrupt/absent cache is never fatal

    def _save_registry(self, reg: dict) -> None:
        tmp = self._registry_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(reg, indent=2))
        os.replace(tmp, self._registry_path)                 # atomic on POSIX

    # ---- discovery ---------------------------------------------------------
    def path_for(self, pid: str) -> Path:
        """Resolve a project id to its directory, refusing anything outside the root."""
        p = (self.root / pid).resolve()
        if p.parent != self.root:
            raise ValueError(f"invalid project id: {pid!r}")
        return p

    def exists(self, pid: str) -> bool:
        try:
            return Store(self.path_for(pid)).is_project()
        except ValueError:
            return False

    def ids(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir()
                      if d.is_dir() and Store(d).is_project())

    def summarize(self, pid: str) -> ProjectInfo:
        """Card data for one project, memoised on `state.json`'s (mtime, size)."""
        path = self.path_for(pid)
        store = Store(path)
        reg = self._load_registry()
        entry = reg.get(pid, {})
        info = ProjectInfo(id=pid, name=entry.get("name") or pid, path=str(path))

        sp = store.state_path
        try:
            st = sp.stat()
        except OSError:
            info.error = "state.json missing"
            return info
        info.modified = st.st_mtime

        stamp = [st.st_mtime, st.st_size]
        cached = entry.get("summary")
        if cached and entry.get("stamp") == stamp:
            for k, v in cached.items():
                setattr(info, k, v)
            return info

        try:
            man = store.load_manifest()
            summary = {"n_instances": int(man.get("n_instances", 0)), **_summarize_state(sp)}
            summary["clustered"] = bool(list(store.cache_dir.glob("*.npz"))) if store.cache_dir.is_dir() else False
            summary["sources"] = _read_sources(path)
        except Exception as e:                               # never let one bad project break the list
            info.error = f"{type(e).__name__}: {e}"
            return info

        for k, v in summary.items():
            setattr(info, k, v)
        reg[pid] = {**entry, "name": info.name, "stamp": stamp, "summary": summary}
        self._save_registry(reg)
        return info

    def list(self) -> list[ProjectInfo]:
        """All projects, most recently modified first."""
        out = [self.summarize(p) for p in self.ids()]
        out.sort(key=lambda i: i.modified, reverse=True)
        return out

    # ---- lifecycle ---------------------------------------------------------
    def create(self, name: str, config: dict | None = None) -> ProjectInfo:
        """Create an empty project directory and register its display name.

        The engine is NOT constructed here — the caller opens it, so creation stays cheap and a
        failure to open cannot leave a half-built project behind.
        """
        base = slugify(name)
        pid, n = base, 2
        while (self.root / pid).exists():                     # never silently adopt an existing dir
            pid, n = f"{base}-{n}", n + 1
        path = self.root / pid

        from .engine import CuratorEngine
        eng = CuratorEngine(path)
        eng.init_project(dict(config or {}))
        eng.close()

        reg = self._load_registry()
        reg[pid] = {"name": name or pid, "created": time.time()}
        self._save_registry(reg)
        return self.summarize(pid)

    def rename(self, pid: str, name: str) -> ProjectInfo:
        if not self.exists(pid):
            raise KeyError(pid)
        reg = self._load_registry()
        reg.setdefault(pid, {})["name"] = name
        self._save_registry(reg)
        return self.summarize(pid)

    def delete(self, pid: str) -> None:
        """Permanently remove a project directory and its registry entry."""
        path = self.path_for(pid)
        if not Store(path).is_project():
            raise KeyError(pid)
        shutil.rmtree(path)
        reg = self._load_registry()
        reg.pop(pid, None)
        self._save_registry(reg)


def _read_sources(path: Path) -> list[str]:
    """Distinct proposal sources recorded in the ingest registry (best-effort)."""
    f = path / "ingests.jsonl"
    if not f.is_file():
        return []
    out: list[str] = []
    try:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            s = (json.loads(line) or {}).get("source")
            if s and s not in out:
                out.append(str(s))
    except Exception:
        return out
    return out
