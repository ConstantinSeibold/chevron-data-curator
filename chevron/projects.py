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

Projects do not have to live under the root. A project directory anywhere on disk can be LINKED: the
registry stores its absolute path under an id, and from then on it lists, opens and summarises exactly
like a local one. Nothing is copied or moved, so a project that predates the launcher — or one kept on
another volume next to its images — is adopted in place rather than relocated.
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
    sources: list[str] = field(default_factory=list)
    mode: str = "instance"
    modality: str = "image"
    linked: bool = False                      # lives outside the root; registry holds its absolute path
    error: str | None = None                  # set when the project dir is unreadable/corrupt

    def to_dict(self) -> dict:
        return asdict(self)


def slugify(name: str) -> str:
    """Filesystem-safe project id. Collapses runs of non-alphanumerics to single dashes."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "").strip()).strip("-").lower()
    return s or "project"


def _normalize_config(config: dict | None) -> dict:
    """A new project's config, with the image folder moved to where the project reads it.

    Callers (the launcher dialog, the API) reasonably write a flat `image_root`; everything that
    resolves images reads `images.root`. Folding one into the other here means a folder typed at
    creation is actually used, instead of sitting unread in `state.json`.
    """
    cfg = dict(config or {})
    root = str(cfg.pop("image_root", "") or "").strip()
    if root:
        imgs = dict(cfg.get("images") or {})
        imgs.setdefault("root", root)
        cfg["images"] = imgs
    return cfg


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
    def _link_path(self, pid: str, reg: dict | None = None) -> Path | None:
        """The absolute path recorded for a linked project, or None if `pid` is a local one."""
        raw = ((reg if reg is not None else self._load_registry()).get(pid) or {}).get("path")
        return Path(raw).expanduser() if raw else None

    def path_for(self, pid: str) -> Path:
        """Resolve a project id to its directory.

        An id normally names a subdirectory of the root, and anything that escapes it is refused. A
        LINKED id instead resolves to the absolute path the registry recorded for it, which is how a
        project kept elsewhere on disk still gets a stable, URL-safe handle.
        """
        link = self._link_path(pid)
        if link is not None:
            return link
        p = (self.root / pid).resolve()
        if p.parent != self.root:
            raise ValueError(f"invalid project id: {pid!r}")
        return p

    def exists(self, pid: str) -> bool:
        try:
            return Store(self.path_for(pid)).is_project()
        except (ValueError, OSError):
            return False

    def ids(self) -> list[str]:
        """Every project the launcher knows: the root's own subdirectories, plus linked ones.

        A linked id is listed even when its path has gone missing — `summarize` turns that into a card
        carrying the error, which is what lets the user unlink a stale entry instead of it silently
        vanishing from the launcher.
        """
        reg = self._load_registry()
        out = {d.name for d in self.root.iterdir() if d.is_dir() and Store(d).is_project()}
        out |= {pid for pid, e in reg.items() if (e or {}).get("path")}
        return sorted(out)

    def summarize(self, pid: str) -> ProjectInfo:
        """Card data for one project, memoised on `state.json`'s (mtime, size)."""
        path = self.path_for(pid)
        store = Store(path)
        reg = self._load_registry()
        entry = reg.get(pid, {})
        info = ProjectInfo(id=pid, name=entry.get("name") or pid, path=str(path),
                           linked=bool(entry.get("path")))

        sp = store.state_path
        try:
            st = sp.stat()
        except OSError:
            # a linked folder can be renamed, deleted or on an unmounted volume — say which it is, so
            # the card offers the right fix (re-link or unlink) rather than looking corrupt
            info.error = "linked folder is missing or no longer a project" if info.linked \
                else "state.json missing"
            return info
        info.modified = st.st_mtime

        stamp = [st.st_mtime, st.st_size]
        cached = entry.get("summary")
        if cached and entry.get("stamp") == stamp:
            # A registry written by an older Chevron can carry keys for fields that no longer exist
            # (`clustered` was one); skipping them keeps a stale cache from growing stray
            # attributes.
            for k, v in cached.items():
                if k in ProjectInfo.__dataclass_fields__:
                    setattr(info, k, v)
            return info

        try:
            man = store.load_manifest()
            summary = {"n_instances": int(man.get("n_instances", 0)), **_summarize_state(sp)}
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
        taken = set(self._load_registry())                    # linked ids reserve a name too
        pid, n = base, 2
        while pid in taken or (self.root / pid).exists():      # never silently adopt an existing dir
            pid, n = f"{base}-{n}", n + 1
        path = self.root / pid

        from .engine import CuratorEngine
        eng = CuratorEngine(path)
        eng.init_project(_normalize_config(config))
        eng.close()

        reg = self._load_registry()
        reg[pid] = {"name": name or pid, "created": time.time()}
        self._save_registry(reg)
        return self.summarize(pid)

    def discover(self, path: str | Path) -> list[dict]:
        """Existing projects at `path`, for the "add existing" picker — nothing is registered.

        Covers both shapes of what a user types: the project directory itself, or the folder they keep
        projects in. Only immediate children are scanned; a path box must never kick off a deep walk of
        a home directory. `known_as` marks entries the launcher already lists, so the picker can show
        them as already-added instead of offering a duplicate.
        """
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"not a directory: {p}")

        known: dict[str, str] = {}
        for pid in self.ids():
            try:
                known[str(self.path_for(pid))] = pid
            except (ValueError, OSError):
                continue
        cands = [p] if Store(p).is_project() else sorted(c for c in p.iterdir() if c.is_dir())
        return [{"path": str(d), "name": d.name, "known_as": known.get(str(d))}
                for d in cands if Store(d).is_project()]

    def link(self, path: str | Path, name: str | None = None) -> ProjectInfo:
        """Adopt an EXISTING project directory in place, without copying or moving it.

        A directory that already sits inside the root is not linked — it is discoverable there anyway,
        so this only names it. Re-linking a path that is already registered returns the existing entry,
        which keeps the picker idempotent when the user adds the same folder twice.
        """
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"not a directory: {p}")
        if not Store(p).is_project():
            raise ValueError(f"not a Chevron project (no state.json): {p}")

        reg = self._load_registry()
        if p.parent == self.root:
            if name:
                reg.setdefault(p.name, {})["name"] = name
                self._save_registry(reg)
            return self.summarize(p.name)

        for pid, e in reg.items():
            if (e or {}).get("path") and Path(e["path"]).expanduser() == p:
                return self.summarize(pid)

        base = slugify(name or p.name)
        taken = set(reg) | {d.name for d in self.root.iterdir() if d.is_dir()}
        pid, n = base, 2
        while pid in taken:
            pid, n = f"{base}-{n}", n + 1
        reg[pid] = {"name": name or p.name, "path": str(p), "created": time.time()}
        self._save_registry(reg)
        return self.summarize(pid)

    def unlink(self, pid: str) -> None:
        """Forget a linked project. Its directory and everything in it are left untouched."""
        reg = self._load_registry()
        if not (reg.get(pid) or {}).get("path"):
            raise KeyError(pid)
        reg.pop(pid, None)
        self._save_registry(reg)

    def rename(self, pid: str, name: str) -> ProjectInfo:
        if not self.exists(pid):
            raise KeyError(pid)
        reg = self._load_registry()
        reg.setdefault(pid, {})["name"] = name
        self._save_registry(reg)
        return self.summarize(pid)

    def delete(self, pid: str) -> None:
        """Permanently remove a project directory and its registry entry.

        Refuses linked projects: their directory is somewhere the user chose to keep it, and dropping
        one from the launcher must not `rmtree` a path outside the root. `unlink` is the operation for
        those.
        """
        if (self._load_registry().get(pid) or {}).get("path"):
            raise ValueError(f"{pid!r} lives outside the projects root — unlink it instead of deleting")
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
