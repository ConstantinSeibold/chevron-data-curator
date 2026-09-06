"""Undo/redo over the curation OVERLAY only.

Every mutating engine op declares which instances/classes it touches; the history
snapshots their `before` slice, the op mutates the `CuratorState`, then the history
snapshots the `after` slice. Undo restores `before`; redo restores `after`. Neither
ever touches the heavy `feats`/masks — only the small per-iuid meta + taxonomy.

Undo/redo stacks are in-memory per session (a fresh session resumes from the
persisted `state.json` with empty stacks). A compact audit line is appended to
`history.jsonl` for provenance. `sample_more` is an undo BARRIER (clears stacks)
because it irreversibly grows the collection.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from .state import CuratorState, InstanceMeta, TaxonomyClass


def _snapshot(state: CuratorState, iuids, class_ids) -> dict:
    return {
        "meta": {u: (state.meta[u].to_dict() if u in state.meta else None) for u in iuids},
        "tax": {c: (state.taxonomy[c].to_dict() if c in state.taxonomy else None) for c in class_ids},
    }


def _restore(state: CuratorState, snap: dict) -> None:
    for u, d in snap["meta"].items():
        if d is None:
            state.meta.pop(u, None)
        else:
            state.meta[u] = InstanceMeta.from_dict(d)
    for c, d in snap["tax"].items():
        if d is None:
            state.taxonomy.pop(c, None)
        else:
            state.taxonomy[c] = TaxonomyClass.from_dict(d)


class History:
    def __init__(self, store, max_depth: int = 200):
        self.store = store
        self.max_depth = max_depth
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []

    def begin(self, state: CuratorState, iuids, class_ids=()) -> dict:
        """Capture the `before` slice. Returns a token to pass to commit()."""
        iuids = list(dict.fromkeys(iuids))
        class_ids = list(dict.fromkeys(class_ids))
        return {"iuids": iuids, "class_ids": class_ids, "before": _snapshot(state, iuids, class_ids)}

    def commit(self, state: CuratorState, token: dict, op: str, label: str = "") -> None:
        # Backfill ids that were CREATED during the op (appended to the token after begin):
        # they did not exist before, so their "before" value is None.
        for u in token["iuids"]:
            token["before"]["meta"].setdefault(u, None)
        for c in token["class_ids"]:
            token["before"]["tax"].setdefault(c, None)
        after = _snapshot(state, token["iuids"], token["class_ids"])
        cmd = {
            "cmd_id": uuid.uuid4().hex[:12], "ts": time.time(), "op": op, "label": label,
            "iuids": token["iuids"], "class_ids": token["class_ids"],
            "before": token["before"], "after": after,
        }
        self.undo_stack.append(cmd)
        if len(self.undo_stack) > self.max_depth:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.store.append_history({
            "cmd_id": cmd["cmd_id"], "ts": cmd["ts"], "op": op, "label": label,
            "n_instances": len(token["iuids"]), "iuids": token["iuids"][:50],
        })

    def undo(self, state: CuratorState) -> str | None:
        if not self.undo_stack:
            return None
        cmd = self.undo_stack.pop()
        _restore(state, cmd["before"])
        self.redo_stack.append(cmd)
        self.store.append_history({"ts": time.time(), "op": "undo", "of": cmd["op"], "cmd_id": cmd["cmd_id"]})
        return cmd["op"]

    def redo(self, state: CuratorState) -> str | None:
        if not self.redo_stack:
            return None
        cmd = self.redo_stack.pop()
        _restore(state, cmd["after"])
        self.undo_stack.append(cmd)
        self.store.append_history({"ts": time.time(), "op": "redo", "of": cmd["op"], "cmd_id": cmd["cmd_id"]})
        return cmd["op"]

    def barrier(self) -> None:
        """Irreversible boundary (e.g. additive sampling): drop undo/redo history."""
        self.undo_stack.clear()
        self.redo_stack.clear()

    @property
    def depths(self) -> tuple[int, int]:
        return len(self.undo_stack), len(self.redo_stack)
