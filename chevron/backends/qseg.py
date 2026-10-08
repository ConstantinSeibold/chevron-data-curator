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


# --------------------------------------------------------------------------- checkpoint-only metadata
def checkpoint_shapes(ckpt: str) -> dict:
    """What qseg's model build needs, read off the checkpoint's own tensor shapes: foreground class
    count (`class_embed` rows minus the trailing no-object row), query count (`query_feat` rows) and
    the keypoint head's output width. `weights_only` + mmap: no pickle code runs, no 600 MB copy."""
    import torch
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=True, mmap=True)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    out = {"num_classes": None, "num_queries": None, "kpt_out": None}
    kpt_last = None
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith("predictor.class_embed.weight"):
            out["num_classes"] = int(v.shape[0]) - 1
        elif k.endswith("predictor.query_feat.weight"):
            out["num_queries"] = int(v.shape[0])
        elif k.startswith("kpt_head.mlp.") and k.endswith(".weight") and v.dim() == 2:
            i = int(k.split(".")[-2]) if k.split(".")[-2].isdigit() else -1
            if kpt_last is None or i > kpt_last[0]:
                kpt_last = (i, int(v.shape[0]))
    if kpt_last:
        out["kpt_out"] = kpt_last[1]
    return out


def stub_train_json(ckpt: str, out_dir, *, simcc_bins: int = 64, coord: str = "mask_centroid_simcc") -> tuple[str, dict]:
    """A training-JSON stand-in built from the checkpoint alone, for class-agnostic use: qseg scans
    its training json only to SIZE the model (classes, keypoint slots); the names, skeletons and
    images in it are training/eval concerns Chevron never touches. Placeholder categories
    `class_0..N-1`, each with K_max keypoints when the checkpoint has a keypoint head; no images.
    Returns (path, shapes). Written once per checkpoint geometry, then reused."""
    import json
    from pathlib import Path
    shp = checkpoint_shapes(ckpt)
    n = shp["num_classes"]
    if not n or n < 1:
        raise ValueError(f"{ckpt} has no Mask2Former/MaskDINO class head (predictor.class_embed)")
    k = 0
    if shp["kpt_out"]:
        per = (2 * int(simcc_bins) + 1) if "simcc" in str(coord) else 3
        if shp["kpt_out"] % per == 0:
            k = shp["kpt_out"] // per
    shp["max_keypoints"] = k
    cats = [{"id": i + 1, "name": f"class_{i}", "supercategory": "object",
             **({"keypoints": [f"k{j}" for j in range(k)], "skeleton": []} if k else {})}
            for i in range(n)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"stub_{n}c_{k}k.json"
    if not p.exists():
        p.write_text(json.dumps({"images": [], "annotations": [], "categories": cats}))
    return str(p), shp


# --------------------------------------------------------------------------- registered backend
class QsegBackend:
    """A trained qseg checkpoint as a proposal source.

    Unlike the other backends the model is the PROJECT's (`state.config['model']`: ckpt + config_name
    + overrides), so the engine binds its own loader in (`bind`). Ingest does not go through `propose`
    at all — `Engine.propose_instances` routes qseg to `collect_batch`, which keeps the detector-internal
    feature blocks (decoder / maskpool / roialign / backbone). `propose` is the per-image path the
    box-guided re-masker uses; its labels are dropped like every other backend's.
    """
    name = NAME
    label = "qseg checkpoint (MaskDINO / Mask2Former — keeps decoder features)"
    requires = ("a qseg checkout (CHEVRON_QSEG_ROOT) with detectron2 + MaskDINO installed, "
                "and a checkpoint path")

    def __init__(self):
        self._loader = None

    def bind(self, loader) -> "QsegBackend":
        """`loader() -> (model, cfg, d2_cfg)` — the engine's cached project model."""
        self._loader = loader
        return self

    def available(self) -> tuple[bool, str]:
        import importlib.util
        from .._bootstrap import qseg_root
        root = qseg_root()
        if not root:
            return False, "no qseg checkout configured (set CHEVRON_QSEG_ROOT)"
        if not available():
            return False, f"{root} has no notebooks/qseg_playground.py"
        missing = [m for m in ("detectron2", "hydra", "qseg") if importlib.util.find_spec(m) is None]
        if missing:
            return False, f"missing Python packages: {', '.join(missing)}"
        return True, f"checkout {root}; uses the project's checkpoint"

    def propose(self, image_rgb, **cfg):
        from .. import collect as _co
        from ..core.collection import decode_mask
        from .base import Proposal
        path = cfg.get("path")
        if self._loader is None or not path:
            raise RuntimeError("the qseg backend needs the engine's model loader and an image path")
        model, mcfg, d2_cfg = self._loader()
        col = _co.collect_batch(model, mcfg, d2_cfg, [path],
                                score_thresh=float(cfg.get("score_thresh", 0.0) or 0.0),
                                feature_cfg={"with_features": False, "with_backbone": False,
                                             "shapecoord": False})
        return [Proposal(mask=decode_mask(r), score=float(r["score"]))
                for r in col["records"]]


def _factory():
    return QsegBackend()


from .base import register  # noqa: E402

register(NAME, _factory)
