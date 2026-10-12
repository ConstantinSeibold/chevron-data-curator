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
    "out_dir", "ckpt", "json_val", "json_test", "extra_train_json", "train_json", "prior_dir",
})
# a Hugging Face Hub id OR a local folder: only checked when it names something on this disk
MAYBE_PATH_KEYS = frozenset({"model_id"})
# free text that reaches a model config or a training command line (`MODEL.WEIGHTS=/anywhere`,
# `--config-name ../../x`): not a path we can resolve, so a hosted client may not set it at all
DENIED_KEYS = frozenset({"overrides", "config_name"})


class PathGuard:
    def __init__(self, roots):
        self.roots = [os.path.realpath(os.path.expanduser(str(r))) for r in roots]

    def allowed(self, p: str) -> bool:
        real = os.path.realpath(os.path.expanduser(str(p)))
        for r in self.roots:
            if real == r:
                return True
            if real.startswith(r.rstrip(os.sep) + os.sep):
                # dot-files are the server's own (accounts, sessions, the registry), never data
                return not any(part.startswith(".") for part in real[len(r):].split(os.sep) if part)
        return False

    def violations(self, obj, _key: str | None = None) -> list[str]:
        out: list[str] = []
        if _key in DENIED_KEYS:
            return [f"{_key}={obj}"] if obj not in (None, "", [], {}) else []
        if _key in PATH_KEYS:                             # a path, or a list of paths: nothing else
            items = obj if isinstance(obj, (list, tuple)) else [obj]
            for v in items:
                if v is None or v == "":
                    continue
                if not isinstance(v, str):                # e.g. {"paths": {"/etc/x": 1}} iterates to keys
                    out.append(repr(v))
                elif not self.allowed(v.strip()):
                    out.append(v.strip())
            return out
        if _key in MAYBE_PATH_KEYS and isinstance(obj, str):
            v = obj.strip()
            local = v.startswith(("/", "~", ".")) or os.path.exists(os.path.expanduser(v))
            return [v] if v and local and not self.allowed(v) else []
        if isinstance(obj, dict):
            for k, v in obj.items():
                out += self.violations(v, str(k))
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                out += self.violations(v, None)
        return out

    def describe(self) -> str:
        return ", ".join(str(Path(r)) for r in self.roots)
