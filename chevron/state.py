"""In-memory curation overlay — the small, JSON-serializable state that sits on
top of the heavy instance collection (records + feature matrices + masks).

The collection itself (from a proposal backend) is held by the
engine/store and saved as a pickle. This module models ONLY the mutable curation
layer keyed by the immutable `iuid`, plus the row-order invariant.

THE INVARIANT: `CuratorState.order` is the list of `iuid`s in exactly the same
order as the rows of every `feats` matrix and of `collection["records"]`. Every
append must preserve it; every per-row feature lookup goes iuid -> meta.row.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
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
    rule_ops: list | None = None              # the refine rule-chain recorded on this instance (last applied)
    provenance: dict[str, Any] = field(default_factory=dict)  # model ckpt, score_thresh, source file, etc.

    def to_dict(self) -> dict:
        # flat field copy (not dataclasses.asdict): asdict deep-COPIES + recurses every field, which is
        # the dominant cost when serializing N instances on every mutation. These fields are JSON-ready
        # scalars/lists/dicts, so a shallow getattr map is ~5x faster and equivalent for serialization.
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "InstanceMeta":
        return cls(**d)


@dataclass
class TaxonomyClass:
    """A LEAF — the annotatable category instances are assigned to + exported. A part of a concept (e.g.
    'pacemaker_body') or a part-less concept (e.g. 'coin'). `concept`/`supercategory` place it in the
    nested taxonomy (superclass -> concept -> this leaf)."""
    class_id: str
    name: str
    color: list[int] = field(default_factory=lambda: [200, 60, 60])  # RGB for display
    coco_cat_id: int | None = None            # pinned COCO id (stable label space across exports)
    concept: str | None = None                # parent concept id (None = ungrouped)
    supercategory: str | None = None          # superclass id (None = ungrouped)
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    structure_type: str = ""                  # thin | tubular | compact | linear | external | mixed
    temp: bool = False                        # scratch/placeholder class — usable in the tool but EXCLUDED
                                              # from the exported taxonomy + the released COCO

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaxonomyClass":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})   # tolerant of old/extra keys


@dataclass
class Superclass:
    """A top-level family (COCO supercategory)."""
    id: str
    name: str
    color: list[int] = field(default_factory=lambda: [150, 150, 150])
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Superclass":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Concept:
    """A device/object CONCEPT ('pacemaker'), kept whole; its annotatable LEAVES are the TaxonomyClass entries
    whose `concept` == this id. `part_rules` are per-image release-completeness gates, e.g.
    {if:'pacemaker_body', then:['pacemaker_lead'], desc:...}."""
    concept_id: str
    name: str
    superclass: str | None = None
    description: str = ""
    structure_type: str = ""
    aliases: list[str] = field(default_factory=list)
    part_rules: list = field(default_factory=list)
    mimic_family: str | None = None           # CROSSWALK to a mimic-fb-bench device family (roll-up for eval);
                                              # advisory only — mimic is text/different-data, not a constraint

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Concept":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CuratorState:
    project_dir: str
    config: dict[str, Any] = field(default_factory=dict)
    taxonomy: dict[str, TaxonomyClass] = field(default_factory=dict)   # leaf class_id -> TaxonomyClass (annotatable)
    meta: dict[str, InstanceMeta] = field(default_factory=dict)        # iuid -> InstanceMeta
    order: list[str] = field(default_factory=list)                    # iuid order == feats row order (INVARIANT)
    class_rules: dict[str, list] = field(default_factory=dict)        # class_id -> refine rule-chain (the class recipe)
    superclasses: dict[str, Superclass] = field(default_factory=dict)  # superclass id -> Superclass (taxonomy L1)
    concepts: dict[str, Concept] = field(default_factory=dict)         # concept id -> Concept (taxonomy L2; leaves group under)
    coll_version: int = 0
    collection_dirty: bool = False                                    # clustering stale (instances changed)
    release_gate: dict[str, str] = field(default_factory=dict)        # image_id(str) -> "accepted"|"rejected" (image-level RELEASE gate, separate from instance is_background)

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
        # de-dup by name: two ids can carry the same display name (free-form add_class vs taxonomy-leaf id),
        # which would otherwise list a class twice in the pickers. assign-by-name resolves to the first id.
        return list(dict.fromkeys(c.name for c in self.taxonomy.values()))

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
        # snapshot the dict/list containers with list(...) before iterating: the background state-saver
        # (engine write-behind) may call this on its own thread while a request thread is structurally
        # mutating meta/order (ingest/split). `list(d.items())` is atomic under the CPython GIL, so this
        # never raises "changed size during iteration"; a field updated mid-snapshot is simply captured
        # on the next flush (the engine keeps the dirty flag set).
        return {
            "project_dir": self.project_dir,
            "config": self.config,
            "taxonomy": {k: v.to_dict() for k, v in list(self.taxonomy.items())},
            "meta": {k: v.to_dict() for k, v in list(self.meta.items())},
            "order": list(self.order),
            "class_rules": {k: list(v) for k, v in list(self.class_rules.items())},
            "superclasses": {k: v.to_dict() for k, v in list(self.superclasses.items())},
            "concepts": {k: v.to_dict() for k, v in list(self.concepts.items())},
            "coll_version": self.coll_version,
            "collection_dirty": self.collection_dirty,
            "release_gate": dict(self.release_gate),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CuratorState":
        return cls(
            project_dir=d["project_dir"],
            config=d.get("config", {}),
            taxonomy={k: TaxonomyClass.from_dict(v) for k, v in d.get("taxonomy", {}).items()},
            meta={k: InstanceMeta.from_dict(v) for k, v in d.get("meta", {}).items()},
            order=d.get("order", []),
            class_rules=d.get("class_rules", {}),
            superclasses={k: Superclass.from_dict(v) for k, v in d.get("superclasses", {}).items()},
            concepts={k: Concept.from_dict(v) for k, v in d.get("concepts", {}).items()},
            coll_version=int(d.get("coll_version", 0)),
            collection_dirty=bool(d.get("collection_dirty", False)),
            release_gate=dict(d.get("release_gate", {})),
        )


def _auto_color(i: int) -> list[int]:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.85, 1.0)
    return [int(r * 255), int(g * 255), int(b * 255)]
