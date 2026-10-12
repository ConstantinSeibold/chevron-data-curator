"""Undo/redo over the curation OVERLAY only.

Every mutating engine op declares which instances/classes it touches; the history
snapshots their `before` slice, the op mutates the `CuratorState`, then the history
snapshots the `after` slice. Undo restores `before`; redo restores `after`. Neither
ever touches the heavy `feats`/masks — only the small per-iuid meta + taxonomy.

Undo/redo stacks are in-memory and kept PER ACTOR (`chevron.context.actor()`; None outside a
request, which is the single-user case). You undo your own last step, not whoever acted last, and an
undo is refused when someone else has since changed an instance or class that step touched: putting
back the `before` slice would silently erase their newer work. A fresh process resumes from the
persisted `state.json` with empty stacks. A compact audit line, with the actor, is appended to
`history.jsonl` for provenance. `sample_more` is an undo BARRIER (clears stacks)
because it irreversibly grows the collection.
"""
from __future__ import annotations

import itertools
import time
import uuid
from collections import deque
from typing import Any

from . import context

_AGED_OUT = object()                                  # "its history is gone": not a user name

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
    def __init__(self, store, max_depth: int = 200, overlays=None, bases=None):
        self.store = store
        self.max_depth = max_depth
        self._stacks: dict[str | None, tuple[list[dict], list[dict]]] = {}
        # every applied step (commit, undo, redo) in order, with what it touched: the conflict check
        self._seq = itertools.count(1)
        self._log: deque = deque(maxlen=20000)
        self._conflicts: dict[str | None, str | None] = {}   # per actor: why their last undo/redo was refused
        # `overlays()` -> the engine's iuid -> effective-mask RLE map. The meta snapshot only records
        # THAT an instance is refined, not WHICH mask it has, so without this undoing the second of two
        # mask edits kept the second mask. `last_restored` is the slice the latest undo/redo put back,
        # for the engine to re-apply the overlays from.
        self.overlays = overlays
        # `bases()` -> iuid -> the mask a draw / re-mask / transfer SET, which refine chains start from
        self.bases = bases
        self.last_restored: dict | None = None

    @property
    def conflict(self) -> str | None:
        return self._conflicts.get(context.actor())

    def _mine(self) -> tuple[list[dict], list[dict]]:
        return self._stacks.setdefault(context.actor(), ([], []))

    @property
    def undo_stack(self) -> list[dict]:
        return self._mine()[0]

    @property
    def redo_stack(self) -> list[dict]:
        return self._mine()[1]

    def _applied(self, cmd: dict) -> int:
        seq = next(self._seq)
        self._log.append((seq, context.actor(), set(cmd["iuids"]), set(cmd["class_ids"])))
        return seq

    def _overwritten_by(self, cmd: dict) -> str | None:
        """Who else has touched what `cmd` touched since it was last applied (None: nobody)."""
        me, iu, cl = context.actor(), set(cmd["iuids"]), set(cmd["class_ids"])
        if self._log and self._log[0][0] > cmd["seq"] + 1:
            return _AGED_OUT                          # its successors fell out of the log: cannot tell, refuse
        for seq, who, u, c in reversed(self._log):
            if seq <= cmd["seq"]:
                break
            if who != me and (u & iu or c & cl):
                return who or "someone"
        return None

    def _audit(self, rec: dict) -> None:
        who = context.actor()
        self.store.append_history({**rec, "user": who} if who else rec)

    def _ov(self, iuids) -> dict:
        if self.overlays is None:
            return {}
        ov = self.overlays()
        return {u: ov.get(u) for u in iuids}

    def _sb(self, iuids) -> dict:
        if self.bases is None:
            return {}
        sb = self.bases()
        return {u: sb.get(u) for u in iuids}

    def begin(self, state: CuratorState, iuids, class_ids=()) -> dict:
        """Capture the `before` slice. Returns a token to pass to commit()."""
        iuids = list(dict.fromkeys(iuids))
        class_ids = list(dict.fromkeys(class_ids))
        return {"iuids": iuids, "class_ids": class_ids,
                "before": {**_snapshot(state, iuids, class_ids), "ov": self._ov(iuids),
                           "sb": self._sb(iuids)}}


    def commit(self, state: CuratorState, token: dict, op: str, label: str = "") -> None:
        # Backfill ids that were CREATED during the op (appended to the token after begin):
        # they did not exist before, so their "before" value is None.
        for u in token["iuids"]:
            token["before"]["meta"].setdefault(u, None)
        for c in token["class_ids"]:
            token["before"]["tax"].setdefault(c, None)
        for k in ("ov", "sb"):
            token["before"].setdefault(k, {})
            for u in token["iuids"]:
                token["before"][k].setdefault(u, None)
        after = {**_snapshot(state, token["iuids"], token["class_ids"]), "ov": self._ov(token["iuids"]),
                 "sb": self._sb(token["iuids"])}
        cmd = {
            "cmd_id": uuid.uuid4().hex[:12], "ts": time.time(), "op": op, "label": label,
            "iuids": token["iuids"], "class_ids": token["class_ids"],
            "before": token["before"], "after": after,
        }
        cmd["seq"] = self._applied(cmd)
        undo, redo = self._mine()
        undo.append(cmd)
        if len(undo) > self.max_depth:
            undo.pop(0)
        redo.clear()
        self._audit({
            "cmd_id": cmd["cmd_id"], "ts": cmd["ts"], "op": op, "label": label,
            "n_instances": len(token["iuids"]), "iuids": token["iuids"][:50],
        })

    def _step(self, state: CuratorState, src: list, dst: list, side: str, op: str) -> str | None:
        self._conflicts[context.actor()] = None
        if not src:
            return None
        cmd = src.pop()
        who = self._overwritten_by(cmd)
        if who is not None:                       # dropped: it can never be applied cleanly again
            self._conflicts[context.actor()] = ("too much has happened since to undo it safely" if who is _AGED_OUT
                                                else f"{who} has changed these since")
            self.last_restored = None
            return None
        _restore(state, cmd[side])
        self.last_restored = cmd[side]
        cmd["seq"] = self._applied(cmd)
        dst.append(cmd)
        self._audit({"ts": time.time(), "op": op, "of": cmd["op"], "cmd_id": cmd["cmd_id"]})
        return cmd["op"]

    def undo(self, state: CuratorState) -> str | None:
        undo, redo = self._mine()
        return self._step(state, undo, redo, "before", "undo")

    def redo(self, state: CuratorState) -> str | None:
        undo, redo = self._mine()
        return self._step(state, redo, undo, "after", "redo")

    def barrier(self) -> None:
        """Irreversible boundary (e.g. additive sampling): drop everyone's undo/redo history."""
        self._stacks.clear()

    @property
    def depths(self) -> tuple[int, int]:
        undo, redo = self._mine()
        return len(undo), len(redo)
