"""The `P` facade that `_bootstrap.get_P()` returns.

Historically the curator reached into qseg's `notebooks/qseg_playground.py` for everything. Most of
what it used is generic and now lives in `chevron.core.collection` (+ `chevron.extractors.raddino`);
only three functions genuinely need a qseg checkout, and those delegate lazily.

Keeping the `P.<name>` call shape means the engine/collect/cluster call sites are unchanged, so the
extraction is behaviour-preserving.
"""
from __future__ import annotations

from ..extractors.raddino import RadDinoExtractor, add_raddino_features
from .collection import (  # noqa: F401  (re-exported as the P surface)
    _coord_feats,
    _img_indices,
    _pairwise_dist,
    _stack_feats,
    _zscore,
    build_matrix,
    busiest_image,
    candidate_pairs,
    cluster,
    cluster_per_image,
    cluster_per_image_X,
    decode_mask,
    embed2d,
    finch_hierarchy,
    fuse_features,
    load_collection,
    load_image,
    merge_groups_in_image,
    overlay_groups,
    overlay_keypoints,
    pair_features,
    pair_labels,
    save_collection,
    shape_descriptors,
    union_group_mask,
)

__all__ = [
    "RadDinoExtractor", "add_raddino_features", "build_matrix", "busiest_image", "candidate_pairs",
    "cluster", "cluster_per_image", "cluster_per_image_X", "collect_instances", "decode_mask",
    "embed2d", "finch_hierarchy", "fuse_features", "load_collection", "load_image", "load_model",
    "merge_groups_in_image", "overlay_groups", "overlay_keypoints", "pair_features", "pair_labels",
    "save_collection", "setup_env", "shape_descriptors", "union_group_mask",
]


# --------------------------------------------------------------------------- #
# qseg-only — delegated to an external qseg checkout (see chevron.backends.qseg)
# --------------------------------------------------------------------------- #
def _real() -> object:
    from .._bootstrap import real_playground
    return real_playground()


def setup_env(*args, **kwargs):
    """Pin the inference GPU + offline HF and put MaskDINO on sys.path (qseg backend only)."""
    return _real().setup_env(*args, **kwargs)


def load_model(*args, **kwargs):
    """Load a trained qseg Mask2Former/MaskDINO checkpoint (qseg backend only)."""
    return _real().load_model(*args, **kwargs)


def collect_instances(*args, **kwargs):
    """Run qseg inference and harvest per-instance features in one forward pass (qseg backend only)."""
    return _real().collect_instances(*args, **kwargs)
