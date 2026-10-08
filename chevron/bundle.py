"""Project bundles: one zip holding a project AND its images, which another machine imports and keeps
working on from exactly where it was left.

A project directory alone is not portable: it records ABSOLUTE paths (each record's image file, the
image root, the COCO file its boxes came from), so a copy opened anywhere else shows black images.
A bundle therefore carries the images and source COCO files too, plus a manifest of the paths they
had, and import rewrites every one of those paths to where the files landed.

    <bundle>.zip
      chevron-bundle.json     format, name, the old image root / project dir, old path -> member map
      project/...             the project directory as is (state, collection, refine overlays,
                              history, snapshots, mask candidates, projection cache, exports)
      images/<rel>            every image a record points at, relative to the old image root
                              (a file outside it goes under images/_external/<n>/)
      sources/<n>_<name>      the COCO files the instances were imported from

Imported, the images sit in `<project>/images/` and the project's image root points there, so the
project is self-contained from then on: copying the folder moves everything. Image ids are kept as
they were (they are the project's join key), so labels, history and undo carry over unchanged.

Not bundled: the proposal-model checkpoint (`config.model.ckpt`, often GBs) and its training json —
they are only needed to run that model on new images, and are listed in the manifest as left out.
"""
from __future__ import annotations

import json
import os
import pickle
import time
import zipfile
from pathlib import Path
from typing import Callable

BUNDLE_FILE = "chevron-bundle.json"
FORMAT = 1
_SKIP_NAMES = {".DS_Store", "Thumbs.db"}
_SKIP_SUFFIXES = (".tmp", ".part")
# already-compressed formats: deflating them again costs time and saves nothing
_STORED = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".npz", ".zip", ".mp4", ".dcm", ".joblib"}


def _records(project: Path) -> list[dict]:
    """Every record the project holds: collection.pkl plus any not-yet-merged ingest shards."""
    out: list[dict] = []
    files = [project / "collection.pkl", *sorted((project / "collection_shards").glob("shard_*.pkl"))]
    for p in files:
        if p.is_file():
            with open(p, "rb") as f:
                out += list((pickle.load(f) or {}).get("records") or [])
    return out


def _skip(p: Path) -> bool:
    return p.name in _SKIP_NAMES or p.name.endswith(_SKIP_SUFFIXES) or ".tmp" in p.suffixes


def plan(project: str | Path) -> dict:
    """What a bundle of `project` would hold, without writing anything: the image and source files
    (old absolute path -> member name), which of them are missing on disk, and the total size."""
    project = Path(project).resolve()
    state = json.loads((project / "state.json").read_text())
    cfg = state.get("config") or {}
    root = str((cfg.get("images") or {}).get("root") or "")
    root = os.path.abspath(os.path.expanduser(root)) if root else ""
    images: dict[str, str] = {}
    sources: dict[str, str] = {}
    inside: set[str] = set()                              # images already in the project folder
    n_ext = 0
    for r in _records(project):
        for key in ("abs_path", "file_name"):
            f = r.get(key)
            if not f or not os.path.isabs(str(f)) or str(f) in images:
                continue
            f = str(f)
            if f.startswith(str(project) + os.sep):
                inside.add(f)
                continue                                  # inside the project: travels as project/...
            if root and f.startswith(root.rstrip(os.sep) + os.sep):
                images[f] = "images/" + os.path.relpath(f, root).replace(os.sep, "/")
            else:
                images[f] = f"images/_external/{n_ext}/{os.path.basename(f)}"
                n_ext += 1
        s = r.get("src_coco")
        if s and str(s) not in sources:
            sources[str(s)] = f"sources/{len(sources)}_{os.path.basename(str(s))}"
    missing = [f for f in [*images, *sources] if not os.path.isfile(f)]
    size = sum(os.path.getsize(f) for f in [*images, *sources] if os.path.isfile(f))
    size += sum(p.stat().st_size for p in project.rglob("*") if p.is_file() and not _skip(p))
    left_out = {k: v for k, v in (cfg.get("model") or {}).items() if isinstance(v, str) and os.path.isabs(v)}
    return {"project": str(project), "image_root": root, "images": images, "sources": sources,
            "n_images": len(images) + len(inside),
            "missing": missing, "bytes": size, "left_out": left_out}


def export_bundle(project: str | Path, out: str | Path, *, name: str | None = None,
                  progress: Callable[[int, int], None] | None = None) -> dict:
    """Write `project` + its images + source COCO files to the zip `out`. The project should be
    flushed first (an open engine writes behind). Returns the plan plus the zip's path and size."""
    pl = plan(project)
    project = Path(pl["project"])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [(p, "project/" + p.relative_to(project).as_posix())
             for p in sorted(project.rglob("*")) if p.is_file() and not _skip(p)
             and out.resolve() != p.resolve()]
    files += [(Path(f), m) for f, m in {**pl["images"], **pl["sources"]}.items() if os.path.isfile(f)]
    manifest = {"format": FORMAT, "name": name or project.name, "exported": time.time(),
                "project_dir": str(project), "image_root": pl["image_root"],
                "images": {f: m for f, m in pl["images"].items() if os.path.isfile(f)},
                "sources": {f: m for f, m in pl["sources"].items() if os.path.isfile(f)},
                "n_images": pl["n_images"], "missing": pl["missing"], "left_out": pl["left_out"]}
    total, done = sum(p.stat().st_size for p, _ in files), 0
    tmp = out.with_suffix(out.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        z.writestr(BUNDLE_FILE, json.dumps(manifest, indent=1))
        for p, m in files:
            z.write(p, m, compress_type=zipfile.ZIP_STORED if p.suffix.lower() in _STORED else zipfile.ZIP_DEFLATED)
            done += p.stat().st_size
            if progress:
                progress(done, total)
    os.replace(tmp, out)
    return {**pl, "zip": str(out), "zip_bytes": out.stat().st_size, "n_files": len(files)}


def is_bundle_dir(path: str | Path) -> bool:
    """An UNPACKED bundle: Safari ("open safe files after downloading") and Finder unzip a download into
    a folder holding chevron-bundle.json, project/, images/, sources/ — import takes that as well."""
    p = Path(path)
    return p.is_dir() and (p / BUNDLE_FILE).is_file() and (p / "project" / "state.json").is_file()


def read_manifest(zip_path: str | Path) -> dict:
    if is_bundle_dir(zip_path):
        m = json.loads((Path(zip_path) / BUNDLE_FILE).read_text())
    else:
        with zipfile.ZipFile(zip_path) as z:
            if BUNDLE_FILE not in z.namelist():
                raise ValueError(f"{zip_path} is not a Chevron project bundle (no {BUNDLE_FILE})")
            m = json.loads(z.read(BUNDLE_FILE))
    if int(m.get("format", 0)) > FORMAT:
        raise ValueError(f"bundle format {m.get('format')} is newer than this Chevron understands ({FORMAT})")
    return m


def _remapper(man: dict, dest: Path) -> Callable[[str], str]:
    """old absolute path -> where import put it. Exact file matches first; anything else under the old
    image root or project dir is re-rooted the same way, so paths in provenance and snapshots follow."""
    exact = {old: str(dest / m) for old, m in {**man["images"], **man["sources"]}.items()}
    prefixes = []
    if man.get("project_dir"):
        prefixes.append((man["project_dir"].rstrip(os.sep), str(dest)))
    if man.get("image_root"):
        prefixes.append((man["image_root"].rstrip(os.sep), str(dest / "images")))
    prefixes.sort(key=lambda kv: -len(kv[0]))                 # the more specific prefix wins

    def remap(s: str) -> str:
        if s in exact:
            return exact[s]
        for old, new in prefixes:
            if s == old:
                return new
            if s.startswith(old + os.sep):
                return new + s[len(old):]
        return s
    return remap


def _deep(o, remap):
    if isinstance(o, str):
        return remap(o) if os.path.isabs(o) else o
    if isinstance(o, dict):
        return {k: _deep(v, remap) for k, v in o.items()}
    if isinstance(o, list):
        return [_deep(v, remap) for v in o]
    return o


def _rewrite_json(p: Path, remap) -> None:
    d = json.loads(p.read_text())
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(_deep(d, remap), separators=(",", ":")))
    os.replace(tmp, p)


def _rewrite_pickle(p: Path, remap) -> None:
    with open(p, "rb") as f:
        c = pickle.load(f)
    if not isinstance(c, dict) or "records" not in c:
        return
    for r in c["records"]:
        for k, v in list(r.items()):
            if isinstance(v, str) and os.path.isabs(v):
                r[k] = remap(v)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(c, f, protocol=4)
    os.replace(tmp, p)


def import_bundle(zip_path: str | Path, dest: str | Path,
                  progress: Callable[[int, int], None] | None = None) -> dict:
    """Unpack a bundle into the NEW directory `dest` (the project; images land in `dest/images`) and
    point every recorded path at the unpacked files. Returns the manifest plus `dest`."""
    man = read_manifest(zip_path)
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} already exists and is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    if is_bundle_dir(zip_path):                           # already unpacked: copy instead of extracting
        _copy_unpacked(Path(zip_path), dest, progress)
    else:
        _extract(zip_path, dest, progress)
    return _relocate(man, dest)


def _copy_unpacked(src: Path, dest: Path, progress) -> None:
    import shutil
    files = [p for p in sorted(src.rglob("*")) if p.is_file() and p.name != BUNDLE_FILE and not _skip(p)]
    total, done = sum(p.stat().st_size for p in files), 0
    for p in files:
        rel = p.relative_to(src).as_posix()
        rel = rel[len("project/"):] if rel.startswith("project/") else rel
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        done += p.stat().st_size
        if progress:
            progress(done, total)


def _extract(zip_path, dest: Path, progress) -> None:
    with zipfile.ZipFile(zip_path) as z:
        infos = [i for i in z.infolist() if not i.is_dir() and i.filename != BUNDLE_FILE]
        total, done = sum(i.file_size for i in infos), 0
        for i in infos:
            name = i.filename
            rel = name[len("project/"):] if name.startswith("project/") else name   # images/, sources/
            target = (dest / rel).resolve()
            if dest != target and dest not in target.parents:                        # zip-slip guard
                raise ValueError(f"bundle member escapes the project folder: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            done += i.file_size
            if progress:
                progress(done, total)


def _relocate(man: dict, dest: Path) -> dict:
    """Point every path recorded in the unpacked project at where its files landed."""
    remap = _remapper(man, dest)
    for p in [dest / "state.json", dest / "manifest.json", *dest.glob("snapshots/*/*.json"),
              *dest.glob("dr/*.json")]:
        if p.is_file():
            _rewrite_json(p, remap)
    for p in [dest / "collection.pkl", *dest.glob("collection_shards/shard_*.pkl")]:
        if p.is_file():
            _rewrite_pickle(p, remap)
    # the image root is the bundled folder now, whatever it was called before
    sp = dest / "state.json"
    d = json.loads(sp.read_text())
    d.setdefault("config", {}).setdefault("images", {})["root"] = str(dest / "images")
    d["project_dir"] = str(dest)
    sp.write_text(json.dumps(d, separators=(",", ":")))
    return {**man, "dest": str(dest)}


def main(argv=None) -> int:
    """`chevron export <project> [out.zip]` and `chevron import <bundle.zip> [--root DIR] [--name N]`."""
    import argparse
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])

    last = [-1]

    def _bar(done, total):                         # one update per percent, not per file
        pct = int(100 * done / total) if total else 100
        if pct != last[0]:
            last[0] = pct
            sys.stdout.write(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB ({pct}%)")
            sys.stdout.flush()

    if cmd == "export":
        ap = argparse.ArgumentParser(prog="chevron export",
                                     description="Zip a project with its images, to import on another machine.")
        ap.add_argument("project", help="the project folder (holds state.json)")
        ap.add_argument("out", nargs="?", help="zip to write (default: <project>.chevron.zip next to it)")
        a = ap.parse_args(rest)
        proj = Path(a.project).expanduser().resolve()
        if not (proj / "state.json").is_file():
            print(f"not a Chevron project (no state.json): {proj}"); return 1
        out = Path(a.out).expanduser() if a.out else proj.parent / f"{proj.name}.chevron.zip"
        r = export_bundle(proj, out, progress=_bar)
        print(f"\n{out}  ({r['zip_bytes'] / 1e6:.0f} MB, {len(r['images'])} images, {len(r['sources'])} COCO files)")
        if r["missing"]:
            print(f"  {len(r['missing'])} referenced files were missing on disk and are NOT in the zip, e.g. {r['missing'][0]}")
        if r["left_out"]:
            print(f"  not included (model files): {', '.join(r['left_out'].values())}")
        print("  stop the server or close the project first if it is open elsewhere — its last edits may not be on disk yet")
        return 0
    if cmd == "import":
        from .projects import ProjectRegistry
        ap = argparse.ArgumentParser(prog="chevron import", description="Import a project bundle as a new project.")
        ap.add_argument("bundle", help="a .zip written by Export zip / chevron export, or the folder it unpacked to")
        ap.add_argument("--root", default="~/.chevron/projects", help="projects folder (default ~/.chevron/projects)")
        ap.add_argument("--name", help="display name (default: the exported project's)")
        a = ap.parse_args(rest)
        info = ProjectRegistry(Path(a.root).expanduser()).import_bundle(Path(a.bundle).expanduser(), a.name,
                                                                       progress=_bar)
        print(f"\n{info.name} -> {info.path}  ({info.n_instances} instances, {info.pct_curated:.0f}% curated)")
        print(f"  open it: chevron --root {a.root}   (or chevron --project {info.path})")
        return 0
    print("usage: chevron export <project> [out.zip] | chevron import <bundle.zip> [--root DIR]")
    return 2
