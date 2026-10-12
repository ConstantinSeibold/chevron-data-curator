"""Hosted mode: every path a client sends must resolve inside the data or project roots.

Locally, typing any path on your own machine is the point (an image folder, a COCO file, a
checkpoint). On a shared server the same endpoints would let any signed-in user read whatever the
server process can. Rather than auditing each of the endpoints that take a path, one check runs on
every request: any value under a path-like key, at any depth of the JSON body or in the query
string, is resolved the way the engine will resolve it (`~` expanded, symlinks followed, relative to
the working directory) and must land inside an allowed root.
"""
from __future__ import annotations

import os
from pathlib import Path

PATH_KEYS = frozenset({
    "path", "paths", "root", "image_root", "extra_image_root", "coco_path", "dir", "directory",
    "out_dir", "ckpt", "json_val", "json_test", "extra_train_json", "train_json",
})


class PathGuard:
    def __init__(self, roots):
        self.roots = [os.path.realpath(os.path.expanduser(str(r))) for r in roots]

    def allowed(self, p: str) -> bool:
        real = os.path.realpath(os.path.expanduser(str(p)))
        return any(real == r or real.startswith(r.rstrip(os.sep) + os.sep) for r in self.roots)

    def violations(self, obj, _key: str | None = None) -> list[str]:
        out: list[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                out += self.violations(v, str(k))
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                out += self.violations(v, _key)
        elif isinstance(obj, str) and _key in PATH_KEYS and obj.strip() and not self.allowed(obj.strip()):
            out.append(obj.strip())
        return out

    def describe(self) -> str:
        return ", ".join(str(Path(r)) for r in self.roots)
