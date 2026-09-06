"""The qseg proposal backend — the only backend that needs an external qseg checkout.

qseg (MaskDINO / Mask2Former) is ONE optional proposal source among several. Everything here
resolves lazily through `chevron._bootstrap`, so Chevron installs, imports and tests without qseg,
detectron2, MaskDINO or torch present.

Point it at a checkout with `CHEVRON_QSEG_ROOT` or `chevron._bootstrap.set_qseg_root(...)`.

The full `ProposalBackend` protocol (`propose` / `train`) lands in P5 alongside the off-the-shelf
backends (SAM auto-mask, HF Mask2Former/OneFormer, torchvision Mask R-CNN, detectron2 zoo). Today
the qseg inference path still runs through `chevron.collect.collect_batch` → `get_P()`.

NB the class-head surgery this backend used to own now lives in `chevron.core.class_head`: it is
pure torch and Chevron's own retrain warm-start gate depends on it, so it must work with no qseg
checkout present.
"""
from __future__ import annotations

NAME = "qseg"


def available() -> bool:
    """Whether a usable qseg checkout is configured (never raises)."""
    from .._bootstrap import qseg_root
    root = qseg_root()
    return bool(root and (root / "notebooks" / "qseg_playground.py").is_file())


def load_model(ckpt=None, config_name: str = "experiments/synthfb_arch3", overrides=None):
    """Load a trained qseg Mask2Former/MaskDINO checkpoint (+ its cfg/d2_cfg/scan)."""
    from .._bootstrap import get_P
    return get_P().load_model(ckpt=ckpt, config_name=config_name, overrides=overrides)


def setup_env(gpu: str = "0", offline_hf: bool = True) -> None:
    """Pin the inference GPU + offline HF and put MaskDINO on sys.path."""
    from .._bootstrap import get_P
    return get_P().setup_env(gpu=gpu, offline_hf=offline_hf)
