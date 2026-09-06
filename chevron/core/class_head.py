"""Extend a trained classification head to new classes (class-incremental).

Generic over ANY qseg / MaskDINO checkpoint. Pads the class-indexed tensors
(`class_embed`, `label_enc`) from C_old to C_new, copying every old class row into
its new slot — matched BY NAME, so partial overlap, reordering and pure-append are
all handled — and initialising genuinely-new rows. Old rows are preserved
bit-exact, so the model keeps its existing predictions; only the new logits are
added. The rest of the head (mask/box embeds, decoder, queries) is class-agnostic
and untouched.

Why this is needed: qseg's `init_weights` loader DROPS shape-mismatched tensors and
re-initialises them — which would wipe ALL old class rows when num_classes grows.
Pre-expanding the checkpoint so the class tensors already match the new model means
the trained old rows load intact.

Pure torch (no detectron2) so it is importable in the CPU smoke env and from CLI.

Vendored from qseg `src/qseg/models/class_extend.py`. Chevron's retrain loop uses
`class_head_fg_count` + `collapse_checkpoint` for its class-agnostic warm-start gate: curation
produces class-agnostic proposals, so a warm-start from a multi-class base must COLLAPSE the
class head rather than let the loader re-initialise it at random (a random head made retrains
worse than baseline even in-domain). Pure torch — no qseg checkout needed.
"""
from __future__ import annotations

from typing import Any

import torch

# Substrings of state-dict keys whose leading dim is class-indexed. Add to this if
# a future head introduces another per-class tensor.
CLASS_TENSOR_KEYS: tuple[str, ...] = ("class_embed", "label_enc")


def _norm(name: str) -> str:
    """Whitespace/case-insensitive class-name key (handles 'left pulmonary artery ')."""
    return " ".join(str(name).strip().lower().split())


def class_names_in_contiguous_order(json_path: str) -> list[str]:
    """Class names in detectron2 contiguous-index order for a COCO json.

    This is the order the model's class logits are in, so it is the correct basis
    for matching old vs new rows. Uses the pure-python scanner (no detectron2).
    """
    from ..data.utils import scan_coco_json

    scan = scan_coco_json(json_path)
    return [scan.cat_id_to_name[scan.contig_to_cat_id[i]] for i in range(scan.num_classes)]


def build_class_row_map(old_names: list[str], new_names: list[str]) -> list[int | None]:
    """For each NEW contiguous index, the OLD index to copy its row from (None if new)."""
    old_idx: dict[str, int] = {}
    for i, n in enumerate(old_names):
        old_idx.setdefault(_norm(n), i)
    return [old_idx.get(_norm(n)) for n in new_names]


def expand_class_state_dict(
    sd: dict[str, Any],
    row_map: list[int | None],
    *,
    n_old: int,
    class_keys: tuple[str, ...] = CLASS_TENSOR_KEYS,
    init_std: float | None = None,
    bias_init: str = "old_mean",
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copy of `sd` with class-indexed tensors expanded to len(row_map).

    A tensor is treated as class-indexed iff its key contains one of `class_keys`
    AND its leading dim is `n_old` (instance mode) or `n_old + 1` (a trailing
    no-object/background row, which is preserved as the last row). New weight rows
    are drawn N(0, init_std) — defaulting to the std of the old rows so they are
    in-distribution and learnable; new bias entries default to the mean of the old
    biases (`bias_init='old_mean'`, or 'zeros').
    """
    n_new = len(row_map)
    out = dict(sd)
    gen = torch.Generator().manual_seed(seed)
    expanded: list[dict[str, Any]] = []
    for k, v in sd.items():
        if not any(ck in k for ck in class_keys):
            continue
        if not torch.is_tensor(v) or v.dim() < 1:
            continue
        extra = int(v.shape[0]) - n_old      # 0 (no bg) or 1 (trailing no-object)
        if extra not in (0, 1):
            continue                          # not class-indexed at this n_old
        new_t = v.new_zeros((n_new + extra,) + tuple(v.shape[1:]))
        if v.dim() >= 2:                      # weight matrix
            std = init_std if init_std is not None else (float(v[:n_old].float().std()) or 0.01)
            new_t.normal_(0.0, std, generator=gen)
        elif bias_init == "old_mean":         # 1-D bias
            new_t.fill_(float(v[:n_old].float().mean()))
        for new_i, old_i in enumerate(row_map):
            if old_i is not None:
                new_t[new_i] = v[old_i]
        if extra == 1:                        # carry the no-object/bg row to the new tail
            new_t[n_new] = v[n_old]
        new_t = new_t.to(v.dtype)
        out[k] = new_t
        expanded.append({"key": k, "old_shape": tuple(v.shape), "new_shape": tuple(new_t.shape)})
    report = {
        "n_old": n_old,
        "n_new": n_new,
        "n_copied": sum(1 for m in row_map if m is not None),
        "n_new_rows": sum(1 for m in row_map if m is None),
        "expanded": expanded,
    }
    return out, report


def collapse_class_state_dict(
    sd: dict[str, Any],
    *,
    n_old: int,
    agg: str = "mean",
    class_keys: tuple[str, ...] = CLASS_TENSOR_KEYS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collapse class-indexed tensors from `n_old` foreground classes to ONE ('object') — a class-agnostic
    head warm-started from a multi-class model instead of RE-INITIALISED.

    The trap this fixes: loading a C_old-class checkpoint into a `num_classes=1` model is a shape mismatch,
    so `init_weights` DROPS the class head and re-initialises it randomly — destroying the base model's
    learned object/no-object calibration (verified: a 120-class synth M2F → 2-class curator produced
    worse-than-baseline detections even in-domain). Instead, the single foreground row is the aggregate
    (`agg`: mean | sum) of the C_old foreground rows — the shared 'objectness' direction — and the trailing
    no-object/void row (M2F softmax head, `extra==1`) is preserved exactly. Symmetric to
    `expand_class_state_dict`'s `extra in (0,1)` handling.
    """
    out = dict(sd)
    collapsed: list[dict[str, Any]] = []
    for k, v in sd.items():
        if not any(ck in k for ck in class_keys):
            continue
        if not torch.is_tensor(v) or v.dim() < 1:
            continue
        extra = int(v.shape[0]) - n_old           # 0 (no void) or 1 (trailing no-object row)
        if extra not in (0, 1):
            continue                              # not class-indexed at this n_old
        fg = v[:n_old].float()
        obj = fg.sum(0) if agg == "sum" else fg.mean(0)
        new_t = v.new_zeros((1 + extra,) + tuple(v.shape[1:]))
        new_t[0] = obj.to(v.dtype)
        if extra == 1:                            # preserve the void/no-object row at the new tail
            new_t[1] = v[n_old]
        out[k] = new_t
        collapsed.append({"key": k, "old_shape": tuple(v.shape), "new_shape": tuple(new_t.shape)})
    return out, {"n_old": n_old, "agg": agg, "collapsed": collapsed}


def collapse_checkpoint(in_path: str, out_path: str, *, n_old: int, agg: str = "mean") -> dict[str, Any]:
    """Load a checkpoint, collapse its class head(s) `n_old` foreground classes -> 1 ('object'), save a clean
    class-agnostic warm-start to `out_path` (expands `model` + `ema`; drops stale optimizer/scheduler)."""
    ckpt = torch.load(in_path, map_location="cpu")
    reports: dict[str, Any] = {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        ckpt["model"], reports["model"] = collapse_class_state_dict(ckpt["model"], n_old=n_old, agg=agg)
        if isinstance(ckpt.get("ema"), dict):
            ckpt["ema"], reports["ema"] = collapse_class_state_dict(ckpt["ema"], n_old=n_old, agg=agg)
        for stale in ("optimizer", "scheduler"):
            ckpt.pop(stale, None)
        out_obj: Any = ckpt
    else:
        out_obj, reports["model"] = collapse_class_state_dict(ckpt, n_old=n_old, agg=agg)
    torch.save(out_obj, out_path)
    return reports


def class_head_fg_count(ckpt_path: str) -> int | None:
    """Foreground-class count of a checkpoint's class head (class_embed leading dim minus the void row if
    the head carries one). None if no class_embed tensor is found. Lets a caller decide whether a collapse
    is needed before loading a base into a class-agnostic model."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    for k, v in sd.items():
        if "class_embed" in k and torch.is_tensor(v) and v.dim() >= 2:
            d = int(v.shape[0])
            return d - 1 if d >= 2 else d         # M2F class_embed = num_classes + 1 (trailing void)
    return None


def expand_checkpoint(
    in_path: str,
    out_path: str,
    *,
    old_names: list[str],
    new_names: list[str],
    init_std: float | None = None,
    bias_init: str = "old_mean",
    seed: int = 0,
) -> dict[str, Any]:
    """Load a checkpoint, expand its class head(s) to `new_names`, save to `out_path`.

    Handles both a raw state_dict and a qseg/detectron2 checkpoint dict (expands
    `model` and, if present, the `ema` checkpointable; drops stale optimizer state
    so the result is a clean warm-start checkpoint for `train.init_weights`).
    """
    ckpt = torch.load(in_path, map_location="cpu")
    row_map = build_class_row_map(old_names, new_names)
    n_old = len(old_names)

    def _expand(sd: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return expand_class_state_dict(
            sd, row_map, n_old=n_old, init_std=init_std, bias_init=bias_init, seed=seed
        )

    reports: dict[str, Any] = {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        ckpt["model"], reports["model"] = _expand(ckpt["model"])
        if isinstance(ckpt.get("ema"), dict):
            ckpt["ema"], reports["ema"] = _expand(ckpt["ema"])
        for stale in ("optimizer", "scheduler"):   # shapes changed; warm-start only
            ckpt.pop(stale, None)
        out_obj: Any = ckpt
    else:
        out_obj, reports["model"] = _expand(ckpt)

    torch.save(out_obj, out_path)
    reports["dropped_old_classes"] = sorted(
        {_norm(n) for n in old_names} - {_norm(n) for n in new_names}
    )
    reports["row_map"] = row_map
    return reports
