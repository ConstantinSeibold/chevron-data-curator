"""In-memory curation overlay — the small, JSON-serializable state that sits on
top of the heavy instance collection (records + feature matrices + masks).

The collection itself (from `qseg_playground.collect_instances`) is held by the
engine/store and saved as a pickle. This module models ONLY the mutable curation
layer keyed by the immutable `iuid`, plus the row-order invariant.

THE INVARIANT: `CuratorState.order` is the list of `iuid`s in exactly the same
order as the rows of every `feats` matrix and of `collection["records"]`. Every
append must preserve it; every per-row feature lookup goes iuid -> meta.row.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class InstanceMeta:
    iuid: str
    batch_id: str
    row: int                                  # index into feats matrices / records (derived, rewritten on load/append)
    image_id: int
    assigned_class: str | None = None         # class_id, or None if unassigned
    is_background: bool = False               # explicit reject/background bin
    assign_source: str | None = None          # manual | partition | classifier | merge | import
    assign_score: float | None = None         # classifier confidence when assign_source == "classifier"
    merged_into: str | None = None            # iuid of the merge representative (this is a child)
    merge_members: list[str] = field(default_factory=list)  # children iuids (this is a representative)
    refined: bool = False                     # a reversible refine overlay exists at refine/<iuid>.pkl
    provenance: dict[str, Any] = field(default_factory=dict)  # model ckpt, score_thresh, source file, etc.

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InstanceMeta":
        return cls(**d)


@dataclass
class TaxonomyClass:
    class_id: str
    name: str
    color: list[int] = field(default_factory=lambda: [200, 60, 60])  # RGB for display
    coco_cat_id: int | None = None            # pinned COCO id (else assigned 1..K at export)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaxonomyClass":
        return cls(**d)


@dataclass
class CuratorState:
    project_dir: str
    config: dict[str, Any] = field(default_factory=dict)
    taxonomy: dict[str, TaxonomyClass] = field(default_factory=dict)   # class_id -> TaxonomyClass
    meta: dict[str, InstanceMeta] = field(default_factory=dict)        # iuid -> InstanceMeta
    order: list[str] = field(default_factory=list)                    # iuid order == feats row order (INVARIANT)
    coll_version: int = 0
    collection_dirty: bool = False                                    # clustering stale (instances changed)

    # ---- class helpers -----------------------------------------------------
    def class_name(self, class_id: str | None) -> str | None:
        if class_id is None:
            return None
        c = self.taxonomy.get(class_id)
        return c.name if c else class_id

    def class_id_by_name(self, name: str) -> str | None:
        for cid, c in self.taxonomy.items():
            if c.name == name:
                return cid
        return None

    def class_names(self) -> list[str]:
        return [c.name for c in self.taxonomy.values()]

    def add_class(self, name: str, *, color: list[int] | None = None) -> str:
        """Idempotent by name; returns the class_id."""
        from .ids import class_id as _cid
        existing = self.class_id_by_name(name)
        if existing:
            return existing
        cid = _cid()
        self.taxonomy[cid] = TaxonomyClass(class_id=cid, name=name,
                                           color=color or _auto_color(len(self.taxonomy)))
        return cid

    # ---- row/order helpers -------------------------------------------------
    def rebuild_rows(self) -> None:
        """Set every meta.row to its index in `order` (call after any append/reorder)."""
        for i, iuid in enumerate(self.order):
            if iuid in self.meta:
                self.meta[iuid].row = i

    def assert_aligned(self, n_rows: int) -> None:
        if not (len(self.order) == len(self.meta) == n_rows):
            raise AssertionError(
                f"row-alignment broken: order={len(self.order)} meta={len(self.meta)} feats_rows={n_rows}")

    def iuids_with_class(self, class_id: str) -> list[str]:
        return [u for u, m in self.meta.items() if m.assigned_class == class_id and not m.is_background]

    def unassigned_iuids(self) -> list[str]:
        return [u for u, m in self.meta.items()
                if m.assigned_class is None and not m.is_background and m.merged_into is None]

    # ---- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "project_dir": self.project_dir,
            "config": self.config,
            "taxonomy": {k: v.to_dict() for k, v in self.taxonomy.items()},
            "meta": {k: v.to_dict() for k, v in self.meta.items()},
            "order": self.order,
            "coll_version": self.coll_version,
            "collection_dirty": self.collection_dirty,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CuratorState":
        return cls(
            project_dir=d["project_dir"],
            config=d.get("config", {}),
            taxonomy={k: TaxonomyClass.from_dict(v) for k, v in d.get("taxonomy", {}).items()},
            meta={k: InstanceMeta.from_dict(v) for k, v in d.get("meta", {}).items()},
            order=d.get("order", []),
            coll_version=int(d.get("coll_version", 0)),
            collection_dirty=bool(d.get("collection_dirty", False)),
        )


def _auto_color(i: int) -> list[int]:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.85, 1.0)
    return [int(r * 255), int(g * 255), int(b * 255)]
