"""CuratorEngine — the UI-agnostic facade tying the curator modules together.

Holds the authoritative mutable project (collection + feature matrices + masks + the
overlay state). The Gradio app binds to one server-side instance. All renders return
numpy RGB images; all mutations go through the History for undo/redo + autosave.
"""
from __future__ import annotations

import colorsys
import functools
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

_TIMING = bool(os.environ.get("CURATOR_TIMING"))   # set CURATOR_TIMING=1 to log server-side op durations


def _timed(fn):
    """Env-gated wall-clock logger (zero overhead when CURATOR_TIMING is unset). Logs to stderr the
    duration of the wrapped op when it exceeds ~10ms, so the slow interaction at scale can be pinned to
    a concrete server op (vs. browser overhead, which logs nothing here)."""
    if not _TIMING:
        return fn

    @functools.wraps(fn)
    def w(*a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        dt = (time.perf_counter() - t) * 1000.0
        if dt >= 10.0:
            print(f"[CURATOR_TIMING] {getattr(fn, '__qualname__', fn.__name__)}: {dt:.0f} ms",
                  file=sys.stderr, flush=True)
        return r
    return w

from . import classify as _clf
from . import cluster as _cl
from . import collect as _co
from . import export_coco as _ex
from . import sample as _sa
from . import similar as _sim
from .history import History
from .metrics import filter_instances, partition_summary, sort_instances
from .state import CuratorState
from .store import Store

_AUTOSNAP_EVERY = 20

# Bounded LRU of decoded RGB source images keyed by abs path. Source images never change, so no
# invalidation — just eviction. WITHOUT this, every crop re-imread()s the full-res JPEG, and a grid
# render does up to _GRID_CAP disk reads → the app stalls (see plan v5.6). Returned arrays are shared
# (read-only); every mutating caller (crop/_crop_mask/image_overlay) copies before drawing.
_IMG_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_IMG_CACHE_MAX = 24

# Bounded LRU of finished crop thumbnails keyed by (iuid, mask_token, params). The web grids re-request
# crop() for every visible instance on each reload; caching makes a post-merge reload recompute only the
# crops whose mask actually changed. Keyed by mask_token, so it self-invalidates on merge/refine/split.
_CROP_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_CROP_CACHE_MAX = 128


def _load_rgb(path: str, fallback_hw: tuple[int, int] | None = None) -> np.ndarray:
    import cv2
    cached = _IMG_CACHE.get(path)
    if cached is not None:
        _IMG_CACHE.move_to_end(path)
        return cached
    img = cv2.imread(path, cv2.IMREAD_COLOR) if path else None
    rgb = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None
           else np.zeros((*(fallback_hw or (512, 512)), 3), np.uint8))
    _IMG_CACHE[path] = rgb
    _IMG_CACHE.move_to_end(path)
    while len(_IMG_CACHE) > _IMG_CACHE_MAX:
        _IMG_CACHE.popitem(last=False)
    return rgb


def _color(i: int):
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.85, 1.0)
    return np.array([r * 255, g * 255, b * 255])


def _downscale(img: np.ndarray, max_side: int = 220) -> np.ndarray:
    """Shrink to <= max_side on the longest side — keeps gallery payloads small (browser RAM)."""
    import cv2
    h, w = img.shape[:2]
    s = max_side / max(h, w) if max(h, w) > max_side else 1.0
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return img


class CuratorEngine:
    def __init__(self, project_dir: str | Path):
        self.store = Store(project_dir)
        self.state = CuratorState(project_dir=str(project_dir))
        self.collection: dict | None = None
        self.history = History(self.store)
        self.model = self.cfg = self.d2_cfg = self.scan = None
        self._overlay_rle: dict[str, dict] = {}        # iuid -> effective RLE (refine/merge)
        self._cluster: dict | None = None              # {spec, distance, per_image, partitions, counts, level}
        self._subcluster: dict | None = None           # within-class substructure: {target, iuids, partitions, counts, level}
        self._train_job: dict | None = None            # background qseg-train job (pid/proc/log/output_dir)
        self._fused_cache: dict[tuple, np.ndarray] = {}  # (spec_key, coll_version) -> fused feature matrix
        self._commits = 0
        if self.store.is_project():
            self.open()

    # ---- project lifecycle -------------------------------------------------
    def open(self) -> None:
        self.state = self.store.load_state()
        if self.store.has_collection():
            self.collection = self.store.load_collection()
        self.history = History(self.store)
        self._overlay_rle = {}
        for p in sorted(self.store.refine_dir.glob("*.pkl")):
            ov = self.store.load_refine(p.stem)
            if ov and "result_rle" in ov:
                self._overlay_rle[p.stem] = ov["result_rle"]
        self._cluster = None

    def init_project(self, config: dict) -> None:
        self.state = CuratorState(project_dir=str(self.store.dir), config=dict(config))
        self.collection = None
        self.store.ensure()
        self.save()

    def save(self, *, snapshot: bool = False) -> None:
        self.store.save_state(self.state)
        man = self.store.load_manifest()
        man.update({"coll_version": self.state.coll_version,
                    "n_instances": len(self.state.order)})
        self.store.save_manifest(man)
        if self.collection is not None and self.state.collection_dirty is False:
            pass  # collection saved explicitly on append (below)
        if snapshot:
            self.store.snapshot()

    # ---- sampling + extraction (additive) ---------------------------------
    def _ensure_model(self):
        if self.model is None:
            from ._bootstrap import get_P
            P = get_P()
            mc = self.state.config.get("model", {})
            P.setup_env(gpu=str(mc.get("gpu", "0")))
            self.model, self.cfg, self.d2_cfg, self.scan = P.load_model(
                ckpt=mc.get("ckpt"), config_name=mc.get("config_name", "experiments/synthfb_arch3"),
                overrides=mc.get("overrides"))
        return self.model, self.cfg, self.d2_cfg

    # ---- training-loop orchestration (launch qseg-train, watch, adopt) ----
    def _unload_inference_model(self) -> None:
        """Free the GPU the curator's inference model holds (so a same-GPU training job has room)."""
        self.model = self.cfg = self.d2_cfg = self.scan = None
        import gc
        import sys
        gc.collect()                                        # drop the model's tensors before freeing CUDA cache
        torch = sys.modules.get("torch")                    # only touch torch if it's ALREADY imported —
        if torch is not None:                               # never trigger a first import here (a fresh import
            try:                                            # in a request worker thread can partially init torch)
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _qseg_train_bin(self):
        import sys
        return Path(sys.executable).with_name("qseg-train")

    # ---- disk hygiene + checkpoint integrity (Gate 0: a run must not die on a full disk, and a corrupt /
    #      regressed checkpoint must never be silently adopted for inference) -----------------------------
    def _free_gb(self) -> float:
        import shutil
        try:
            return shutil.disk_usage(str(self.store.dir)).free / 1e9
        except Exception:
            return float("inf")

    # ---- progress (read by /api/progress while a long inference runs in another worker thread) ----
    def _set_progress(self, phase: str, done: int, total: int) -> None:
        self._progress = {"phase": phase, "done": int(done), "total": int(total), "active": True}

    def _clear_progress(self) -> None:
        self._progress = {"phase": "", "done": 0, "total": 0, "active": False}

    def progress(self) -> dict:
        return getattr(self, "_progress", {"phase": "", "done": 0, "total": 0, "active": False})

    _RUN_DUMP_DIRS = ("inference", "inference_val", "inference_test", "val", "test")   # heavy eval dumps

    def _prune_run_dir(self, run_dir: Path, *, keep_ckpt: str | None = None) -> int:
        """Drop a finished run's heavy, regenerable artifacts: eval prediction dumps (GBs — the curator
        re-infers itself, so it never needs them), intermediate model_<iter>.pth, and the resume-bloated
        model_final once a best/kept checkpoint exists. Keep model_best.pth (or the adopted ckpt), the
        logs/metrics/config. Returns bytes freed."""
        import shutil
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            return 0
        freed = 0
        for d in self._RUN_DUMP_DIRS:
            p = run_dir / d
            if p.is_dir():
                freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p, ignore_errors=True)
        for f in run_dir.glob("predictions.json"):
            freed += f.stat().st_size; f.unlink()
        kept = {Path(keep_ckpt).name} if keep_ckpt else {"model_best.pth"}
        has_kept = any((run_dir / k).exists() for k in kept)
        for ck in run_dir.glob("model_*.pth"):
            if ck.name in kept:
                continue
            if ck.name == "model_final.pth" and not has_kept:
                continue                                          # keep model_final only when nothing better exists
            freed += ck.stat().st_size; ck.unlink()
        return freed

    def _prune_old_runs(self, keep: int = 2) -> None:
        """Keep the most recent `keep` train-run dirs' checkpoints; strip dumps from the rest entirely."""
        import shutil
        runs = sorted((Path(self.store.dir) / "train_runs").glob("round_*"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for r in runs[keep:]:
            shutil.rmtree(r, ignore_errors=True)
        for r in runs[:keep]:
            self._prune_run_dir(r)

    def _warmstart_init(self, ckpt: str, *, class_agnostic: bool) -> str:
        """Init checkpoint for a fine-tune. For a CLASS-AGNOSTIC run whose base is MULTI-class (e.g. the
        120-class synth M2F), COLLAPSE the class head 119-fg -> 1 'object' (mean) + preserved void, writing a
        warm-start ckpt — instead of letting init_weights silently DROP the mismatched head and re-init it
        random (which made the retrain worse than baseline even in-domain). A base already at 1 fg (a prior
        class-agnostic curator round) is used as-is, so iterative chaining still works (and Gate 1's
        regression guard keeps the chain from drifting down)."""
        if not class_agnostic:
            return ckpt
        try:
            from qseg.models.class_extend import class_head_fg_count, collapse_checkpoint
            fg = class_head_fg_count(ckpt)
        except Exception:
            return ckpt                                      # can't inspect -> defer to the loader (unchanged)
        if fg is None or fg <= 1:
            return ckpt                                      # already class-agnostic / no class head -> no surgery
        out = Path(self.store.dir) / "exports" / "warmstart_collapsed.pth"
        collapse_checkpoint(ckpt, str(out), n_old=int(fg))
        return str(out)

    @staticmethod
    def _ckpt_ok(path: str | Path) -> bool:
        """True iff the checkpoint loads and carries model weights — catches the disk-full TRUNCATED write
        (PytorchStreamReader 'failed reading zip archive') that would otherwise be adopted as garbage."""
        import os
        if not path or not os.path.exists(path) or os.path.getsize(path) < 1024:
            return False
        try:
            import torch
            sd = torch.load(str(path), map_location="cpu")
            return len(sd.get("model", sd)) > 0
        except Exception:
            return False

    def launch_training(self, *, mode: str = "finetune", epochs=None, config_name=None, image_root=None,
                        json_val=None, json_test=None, partial: bool = True, class_agnostic: bool = False,
                        extra_train_json=None, extra_image_root=None) -> dict:
        """Export the curated COCO and spawn `qseg-train` on it as a DETACHED background process (not in
        this process). Unloads the inference model first (same-GPU). If `extra_train_json` is given (e.g.
        a synthfb COCO with complete masks), it is MERGED with the curated export into one train json
        (synth images marked exhaustive). Returns the job/command; watch via training_status()."""
        import json
        import os
        import subprocess
        import time
        job = getattr(self, "_train_job", None)
        if job and job.get("proc") is not None and job["proc"].poll() is None:
            return {"error": "a training job is already running"}
        binp = self._qseg_train_bin()
        if not Path(binp).exists():
            return {"error": f"qseg-train not found at {binp}"}
        self._prune_old_runs(keep=2)                             # reclaim space from prior rounds first
        for stale in (Path(self.store.dir) / "exports").glob("train_merged*.json"):
            stale.unlink()                                       # the merged train json is regenerated below
        free_gb = self._free_gb()
        if free_gb < 8.0:                                        # a run writes ~1.2GB ckpt + eval; refuse if tight
            return {"error": f"only {free_gb:.1f} GB free on the project disk — training writes >1 GB of "
                             f"checkpoints + eval dumps and WILL corrupt the checkpoint if the disk fills "
                             f"mid-write (Errno 28). Free space, then retry."}
        mc = self.state.config.get("model", {})
        config_name = config_name or mc.get("config_name", "experiments/synthfb_arch3")
        image_root = image_root or self.state.config.get("images", {}).get("root", "")
        export_path = self.export_coco(partial_labels=bool(partial), class_agnostic=bool(class_agnostic))
        train_json = export_path
        if extra_train_json:
            from . import export_coco as _ex
            if not Path(extra_train_json).exists():
                return {"error": f"extra train json not found: {extra_train_json}"}
            merged = _ex.merge_coco_sources(str(export_path), str(extra_train_json),
                                            class_agnostic=bool(class_agnostic),
                                            extra_image_root=(str(extra_image_root) if extra_image_root else None))
            train_json = Path(self.store.dir) / "exports" / "train_merged.json"
            train_json.write_text(json.dumps(merged))
        repo_root = Path(__file__).resolve().parents[2]
        out_dir = Path(self.store.dir) / "train_runs" / f"round_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(binp), "--config-name", str(config_name),
               f"data.json_train={train_json}", f"data.image_root={image_root}",
               f"train.output_dir={out_dir}", "train.checkpoint_period=100000000"]   # only model_final + model_best
        # only override val/test when explicitly given, so the config's defaults (e.g. a held-out / synthfb
        # eval) apply instead of silently evaluating on the training export.
        if json_val:
            cmd.append(f"data.json_val={json_val}")
        if json_test:
            cmd.append(f"data.json_test={json_test}")
        if epochs:
            cmd.append(f"train.max_epochs={int(epochs)}")
        if mode == "finetune" and mc.get("ckpt"):
            init_ckpt = self._warmstart_init(mc["ckpt"], class_agnostic=bool(class_agnostic))
            cmd.append(f"train.init_weights={init_ckpt}")
        return self._spawn_train(cmd, out_dir, repo_root, {"export": str(export_path), "train_json": str(train_json),
                                                           "coll_version": int(self.state.coll_version),
                                                           "n_assigned": self._n_assigned()})

    def _spawn_train(self, cmd, out_dir, repo_root, meta: dict) -> dict:
        """Spawn qseg-train detached (own session) with the MaskDINO PYTHONPATH + alloc env; unload the
        inference model first (same GPU); stash the job. Shared by launch_training + launch_overfit_check."""
        import os
        import subprocess
        import time
        out_dir = Path(out_dir)
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{repo_root / 'third_party' / 'MaskDINO'}:{env.get('PYTHONPATH', '')}"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # reduce fragmentation OOMs
        self._unload_inference_model()                       # give the GPU to training
        logf = open(out_dir / "train.log", "w")             # noqa: SIM115 (handed to the child for its lifetime)
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=logf,
                                stderr=subprocess.STDOUT, start_new_session=True)
        self._train_job = {"pid": proc.pid, "proc": proc, "log": str(out_dir / "train.log"),
                           "output_dir": str(out_dir), "config_name": str(cmd[cmd.index("--config-name") + 1]),
                           "started": time.time(), "cmd": " ".join(cmd), **meta}
        return {"ok": True, "pid": proc.pid, "output_dir": str(out_dir), "log": str(out_dir / "train.log"),
                "cmd": " ".join(cmd), **{k: meta[k] for k in ("export", "train_json", "overfit") if k in meta}}

    def overfit_iuids(self, n: int = 12) -> list[str]:
        """Up to n HUMAN-verified instances (assign_source manual|partition, not classifier-propagated) for
        the overfit sanity check — the labels must be trustworthy, since the test asserts the model can fit
        exactly them."""
        out = []
        for u, m in self.state.meta.items():
            if (m.assigned_class is not None and not m.is_background and m.merged_into is None
                    and m.assign_source in ("manual", "partition")):
                out.append(u)
                if len(out) >= n:
                    break
        return out

    def launch_overfit_check(self, *, n: int = 12, epochs: int = 60, config_name: str | None = None) -> dict:
        """GATE 2 — can the training pipeline learn AT ALL? Train on a tiny set of human-verified instances
        and evaluate ON THE SAME IMAGES (train==val==test, circular ON PURPOSE). If segm/AP climbs high the
        loss/LR/label-format/inference path is sound; if it CANNOT memorize its own data the pipeline is
        broken independent of data quality or the synth->real gap. Diagnostic only — never adopted."""
        import time
        binp = self._qseg_train_bin()
        if not Path(binp).exists():
            return {"error": f"qseg-train not found at {binp}"}
        if self._free_gb() < 4.0:
            return {"error": f"only {self._free_gb():.1f} GB free — free space before the overfit check"}
        ius = self.overfit_iuids(int(n))
        if len(ius) < 2:
            return {"error": "need >=2 human-verified (manual/partition) instances for the overfit check — "
                             "assign a few by hand first (classifier-propagated labels don't count)."}
        sub = self.export_coco(out_path=self.store.export_dir / "overfit.json",
                               iuids=ius, class_agnostic=True)
        image_root = self.state.config.get("images", {}).get("root", "")
        mc = self.state.config.get("model", {})
        config_name = config_name or mc.get("config_name", "experiments/curator_loop")
        out_dir = Path(self.store.dir) / "train_runs" / f"overfit_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(binp), "--config-name", str(config_name),
               f"data.json_train={sub}", f"data.json_val={sub}", f"data.json_test={sub}",  # circular ON PURPOSE
               f"data.image_root={image_root}", f"train.output_dir={out_dir}",
               f"train.max_epochs={int(epochs)}", "train.checkpoint_period=100000000",
               "data.repeat_thresh=0.0", "eval.early_stop.enable=false"]
        if mc.get("ckpt"):
            cmd.append(f"train.init_weights={self._warmstart_init(mc['ckpt'], class_agnostic=True)}")
        repo_root = Path(__file__).resolve().parents[2]
        return self._spawn_train(cmd, out_dir, repo_root,
                                 {"export": str(sub), "train_json": str(sub), "overfit": True, "n_overfit": len(ius)})

    def training_status(self) -> dict:
        job = getattr(self, "_train_job", None)
        if not job:
            return {"active": False}
        proc = job.get("proc")
        rc = proc.poll() if proc is not None else None
        tail = ""
        try:
            tail = "".join(open(job["log"]).readlines()[-40:])
        except Exception:
            pass
        od = Path(job["output_dir"])
        best, final = od / "model_best.pth", od / "model_final.pth"
        # LOUD failure: a finished job with a non-zero exit, or a tell-tale error in the log, is FAILED — not
        # silently "done". A disk-full run "completes" with a corrupt checkpoint + no eval; without this the
        # loop adopts garbage. (The recurring root cause we just diagnosed.)
        failed, reason = False, None
        if rc is not None and rc != 0:
            failed, reason = True, f"qseg-train exited with code {rc}"
        for sig, why in (("No space left on device", "DISK FULL (Errno 28) — checkpoint likely truncated/corrupt"),
                         ("Exception during training", "training raised an exception"),
                         ("CUDA out of memory", "CUDA OOM"),
                         ("Traceback (most recent call last)", "uncaught exception")):
            if sig in tail:
                failed, reason = True, why; break
        return {"active": True, "running": rc is None, "returncode": rc, "pid": job["pid"],
                "failed": failed, "reason": reason,
                "output_dir": job["output_dir"], "config_name": job["config_name"], "cmd": job["cmd"],
                "log_tail": tail, "ckpt_best": str(best) if best.exists() else None,
                "ckpt_final": str(final) if final.exists() else None}

    def stop_training(self) -> bool:
        import os
        import signal
        job = getattr(self, "_train_job", None)
        proc = job.get("proc") if job else None
        if proc is None or proc.poll() is not None:
            return False
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
        return True

    def _n_assigned(self) -> int:
        return sum(1 for m in self.state.meta.values()
                   if m.assigned_class is not None and not m.is_background and m.merged_into is None)

    def _read_run_metric(self, output_dir: str | Path):
        """Best-effort (metric_name, value) for a finished qseg-train run, from its summary.csv (the
        single-row tracker qseg-train writes); falls back to (None, None) if not present yet."""
        import csv
        p = Path(output_dir) / "summary.csv"
        if not p.exists():
            return None, None
        try:
            row = next(iter(csv.DictReader(p.read_text().splitlines())), {})
        except Exception:
            return None, None
        name = row.get("best_val_metric_name") or "segm/AP"
        for key in (row.get("best_val_metric_name"), "test/segm/AP", "val/segm/AP", "best_val_metric_value"):
            if key and row.get(key) not in (None, ""):
                try:
                    return name, float(row[key])
                except (ValueError, TypeError):
                    continue
        return name, None

    def adopt_checkpoint(self, ckpt: str = "", *, force: bool = False) -> dict:
        """Point the curator's inference model at a (newly trained) checkpoint; it reloads on next infer.
        GATED (Gate 0/1): refuses a CORRUPT/truncated checkpoint (the disk-full failure), and refuses to
        adopt a checkpoint that REGRESSED below the best prior run on the same metric (catastrophic-forgetting
        guard) unless force=True. On success appends a TRAINING-LOOP LINEAGE record (dataset version ->
        export -> ckpt -> eval metric) to lineage.jsonl — the loop is TRACED, not asserted — then prunes the
        run's heavy eval dumps (the curator re-infers itself; keep only the adopted ckpt + logs)."""
        import os
        import time
        job = getattr(self, "_train_job", None) or {}
        if not ckpt:                                             # prefer best-val, then final
            od = Path(job.get("output_dir", "")) if job.get("output_dir") else None
            for name in ("model_best.pth", "model_final.pth"):
                if od and (od / name).exists():
                    ckpt = str(od / name); break
        if not ckpt or not os.path.exists(ckpt):
            return {"error": "no checkpoint found to adopt"}
        if not self._ckpt_ok(ckpt):                              # Gate 0: never adopt a truncated/garbage ckpt
            return {"error": f"checkpoint is corrupt or unreadable (likely a disk-full truncated write): {ckpt} "
                             f"— do NOT adopt; re-run training with free disk."}
        out_dir = job.get("output_dir") or str(Path(ckpt).parent)
        metric_name, metric = self._read_run_metric(out_dir)
        prior = [e.get("metric") for e in self.store.read_lineage() if e.get("metric") is not None]
        floor = max(prior) if prior else None
        regressed = (metric is not None and floor is not None and metric < floor)
        if regressed and not force:                              # Gate 1: don't silently adopt a worse model
            return {"error": f"REGRESSION: {metric_name}={metric:.3f} is below the best prior run ({floor:.3f}) "
                             f"— fine-tuning made the model worse. Not adopting (pass force=true to override).",
                    "regressed": True, "metric": metric, "floor": floor, "metric_name": metric_name}
        self.state.config.setdefault("model", {})["ckpt"] = ckpt
        self._unload_inference_model()
        self.store.append_lineage_event({
            "ts": time.time(), "coll_version": int(job.get("coll_version", self.state.coll_version)),
            "n_assigned": int(job.get("n_assigned", self._n_assigned())),
            "export_json": job.get("export"), "config_name": job.get("config_name"),
            "output_dir": out_dir, "ckpt": ckpt, "metric_name": metric_name, "metric": metric,
            "regressed": bool(regressed)})
        self.save()
        self._prune_run_dir(out_dir, keep_ckpt=ckpt)             # reclaim the GBs of eval dumps now it's adopted
        return {"ok": True, "ckpt": ckpt, "metric_name": metric_name, "metric": metric,
                "regressed": bool(regressed), "floor": floor}

    def _infer_thresholds(self, score_thresh=None, nms_iou=None):
        """Resolve inference knobs: explicit override -> config default. score_thresh = min prediction
        confidence to keep; nms_iou = mask-IoU dedup threshold (0 disables). Returns (score_thresh, feat_cfg)."""
        st = float(score_thresh) if score_thresh is not None else \
            float(self.state.config["model"].get("score_thresh", 0.3))
        feat_cfg = dict(self.state.config.get("features_runtime", _default_feat_cfg(self.state.config)))
        if nms_iou is not None:
            feat_cfg["nms_iou"] = float(nms_iou)
        return st, feat_cfg

    def ingest_paths(self, file_paths: list[str], *, mode: str = "new", score_thresh=None, nms_iou=None) -> dict:
        """Run the seg model on explicit image paths and ADD their instances to the collection. `mode`:
        - 'new' (default): skip already-processed files (additive discovery on fresh images);
        - 'append': re-run even on processed images and ADD the new model's predictions ALONGSIDE the old
          (compare two checkpoints' outputs on the same images);
        - 'replace': like append, but first HIDE (background) the UN-CURATED instances on those images, so
          the new model re-proposes them while assigned/rejected/merged curation is preserved.
        `score_thresh` / `nms_iou` override the config defaults for THIS run (lower score -> more, weaker
        detections; nms_iou dedups overlapping masks, 0 = off).
        The shared core of random sampling, folder inference, uploaded-image inference, and re-inference."""
        self._set_progress("loading model", 0, 0)
        try:
            model, cfg, d2_cfg = self._ensure_model()
            man = self.store.load_manifest()
            processed = set(man.get("processed_paths", []))
            new_files = [f for f in file_paths if f not in processed] if mode == "new" else list(file_paths)
            if not new_files:
                return {"n_new_images": 0, "n_new_instances": 0, "n_replaced": 0, **self.stats()}
            st, feat_cfg = self._infer_thresholds(score_thresh, nms_iou)
            # CHUNK the file list so /api/progress can report per-chunk progress (collect_instances has no
            # per-image hook; register_images_split is idempotent so re-running per chunk is safe).
            CHUNK = 8
            batch = None
            for i in range(0, len(new_files), CHUNK):
                self._set_progress("segmentation inference", i, len(new_files))
                b = _co.collect_batch(model, cfg, d2_cfg, new_files[i:i + CHUNK], score_thresh=st, feature_cfg=feat_cfg)
                batch = b if batch is None else _co.concat_collections(batch, b)
            self._set_progress("segmentation inference", len(new_files), len(new_files))
        finally:
            self._clear_progress()
        n_new = len(batch["records"])
        n_replaced = 0
        if mode == "replace":                          # hide old UN-CURATED instances on the re-inferred images
            targets = {_co.path_image_id(f) for f in new_files}
            for _u, _m in self.state.meta.items():
                if (int(_m.image_id) in targets and _m.assigned_class is None
                        and not _m.is_background and _m.merged_into is None):
                    _m.is_background = True
                    n_replaced += 1
        self.collection = _co.concat_collections(self.collection, batch)
        # create overlay meta for new instances; rebuild order/rows
        self.state.order = [r["iuid"] for r in self.collection["records"]]
        from .state import InstanceMeta
        for r in batch["records"]:
            self.state.meta[r["iuid"]] = InstanceMeta(
                iuid=r["iuid"], batch_id=r["batch_id"], row=r["row"], image_id=int(r["image_id"]),
                provenance={"file": r.get("abs_path", ""), "src_score": float(r["score"]),
                            "ckpt": self.state.config["model"].get("ckpt", "")})
        self.state.rebuild_rows()
        self.state.assert_aligned(self.collection["feats"][_any_method(self.collection)].shape[0])
        self.state.coll_version += 1
        self.state.collection_dirty = True
        processed.update(new_files)
        man["processed_paths"] = sorted(processed)
        self.store.save_manifest(man)
        self.store.save_collection(self.collection)
        self.history.barrier()                         # additive ingest = undo barrier
        self.save()
        return {"n_new_images": len(new_files), "n_new_instances": n_new, "n_replaced": n_replaced, **self.stats()}

    def sample_more(self, n: int, *, smart: bool = False, seed: int | None = None,
                    score_thresh=None, nms_iou=None) -> dict:
        """Random-sample n not-yet-processed images from the configured root and run inference."""
        self._ensure_model()
        processed = set(self.store.load_manifest().get("processed_paths", []))
        files = _sa.list_images(self.state.config["images"]["root"])
        new_files = _sa.sample_random(files, n, exclude=processed, seed=seed)
        if not new_files:
            return {"n_new_images": 0, "n_new_instances": 0, **self.stats()}
        return self.ingest_paths(new_files, score_thresh=score_thresh, nms_iou=nms_iou)

    def infer_dir(self, directory: str, *, limit: int = 50, mode: str = "new",
                  score_thresh=None, nms_iou=None) -> dict:
        """Run inference on (up to `limit`) images in a server-side folder and add their instances.
        `mode` (new | append | replace) forwarded to ingest_paths (re-inference on already-seen images)."""
        files = _sa.list_images(directory)
        if not files:
            return {"error": f"no images found in {directory}"}
        return self.ingest_paths(files[:int(limit)] if limit else files, mode=mode,
                                 score_thresh=score_thresh, nms_iou=nms_iou)

    @staticmethod
    def _draw_masks(rgb, masks):
        import cv2
        out = rgb.copy()
        for i, m in enumerate(masks):
            if m.shape != out.shape[:2]:
                continue
            col = _color(i)
            out[m] = (0.5 * out[m] + 0.5 * col).astype(np.uint8)
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, tuple(int(v) for v in col), 1)
        return out

    def preview_inference(self, file_paths, *, max_side: int = 640, score_thresh=None, nms_iou=None) -> dict:
        """Run the CURRENT inference model on a few images and render BEFORE (the instances currently in
        the collection on each image) vs AFTER (the model's fresh predictions) — WITHOUT ingesting. A
        non-destructive comparison before committing a re-infer (use it to dial in score_thresh / nms_iou).
        Returns {'items': [{caption, before, after}], 'n_inst': total_after, 'n_before': total_before}."""
        import cv2
        from pycocotools import mask as mu
        model, cfg, d2_cfg = self._ensure_model()
        paths = [p for p in file_paths if p]
        if not paths:
            return {"items": [], "n_inst": 0, "n_before": 0}
        st, feat_cfg = self._infer_thresholds(score_thresh, nms_iou)
        batch = _co.collect_batch(model, cfg, d2_cfg, paths, score_thresh=st, feature_cfg=feat_cfg)
        by_path: dict = {}
        for r in batch["records"]:
            by_path.setdefault(r.get("abs_path") or r["file_name"], []).append(r)
        items, n_before = [], 0
        for p in paths:
            recs = by_path.get(p, [])
            before_iuids = self.image_instance_iuids(_co.path_image_id(p))   # currently-stored instances
            hw = None
            if recs:
                hw = (int(recs[0]["H"]), int(recs[0]["W"]))
            elif before_iuids:
                rr = self.collection["records"][self.state.meta[before_iuids[0]].row]
                hw = (int(rr["H"]), int(rr["W"]))
            if hw:
                rgb = _load_rgb(p, hw)
            else:
                bgr = cv2.imread(p)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((64, 64, 3), np.uint8)
            before = self._draw_masks(rgb, [self._mask(u) for u in before_iuids])
            after = self._draw_masks(rgb, [mu.decode(r["rle"]).astype(bool) for r in recs])
            n_before += len(before_iuids)
            items.append({"caption": f"{Path(p).name}: {len(before_iuids)} → {len(recs)} instances",
                          "before": _downscale(before, max_side), "after": _downscale(after, max_side)})
        return {"items": items, "n_inst": len(batch["records"]), "n_before": n_before}

    def preview_processed(self, n: int = 6, *, score_thresh=None, nms_iou=None) -> dict:
        """Non-destructive preview of the model on a RANDOM sample of n already-processed images — dial in
        score_thresh / nms_iou here before committing a full re-infer."""
        import random
        processed = sorted(self.store.load_manifest().get("processed_paths", []))
        if not processed:
            return {"items": [], "n_inst": 0, "sampled": 0}
        paths = random.sample(processed, min(int(n), len(processed)))
        res = self.preview_inference(paths, score_thresh=score_thresh, nms_iou=nms_iou)
        res["sampled"] = len(paths)
        return res

    def reinfer_processed(self, *, mode: str = "replace", limit: int | None = None,
                          score_thresh=None, nms_iou=None) -> dict:
        """Re-run the (adopted) model on images ALREADY processed — the loop's 're-score the existing pool
        with the new model' step. mode=replace hides old un-curated instances first; append keeps them.
        score_thresh / nms_iou override the detection thresholds for this re-infer."""
        processed = sorted(self.store.load_manifest().get("processed_paths", []))
        if not processed:
            return {"n_new_images": 0, "n_new_instances": 0, "n_replaced": 0, **self.stats()}
        return self.ingest_paths(processed[:int(limit)] if limit else processed, mode=mode,
                                 score_thresh=score_thresh, nms_iou=nms_iou)

    def scaled_pseudolabel(self, *, directory: str | None = None, image_paths=None, out_dir=None,
                           shard_size: int = 2000, method: str = "classifier", thresh: float = 0.5,
                           score_thresh=None, nms_iou=None, pool: str = "bbox",
                           class_agnostic: bool = False, limit: int | None = None) -> dict:
        """TRAIN-ON-SAMPLE, PROPAGATE-AT-SCALE. Run the seg model over many images in SHARDS, propagate labels
        per shard with the already-trained model, write one COCO per shard, then merge — RAM bounded to ONE
        shard (the main collection is never grown). `method`:
        - 'classifier': the classifier trained on the curated sample (self._clf);
        - 'reference':  CSLS top-1 against the loaded reference bank;
        - 'raw':        every detection -> 'object' (no propagation, just the model's predictions).
        This is the scalable path for 64k images / millions of instances."""
        import gc
        import json
        from . import scale as _sc
        from ._bootstrap import get_P
        model, cfg, d2_cfg = self._ensure_model()
        files = (_sa.list_images(directory) if directory else (list(image_paths) if image_paths
                 else sorted(self.store.load_manifest().get("processed_paths", []))))
        if limit:
            files = files[:int(limit)]
        if not files:
            return {"error": "no images to process"}
        if method == "classifier" and getattr(self, "_clf", None) is None:
            return {"error": "train a classifier on a sample first (Classifier tab), or use method=reference/raw"}
        if method == "reference" and getattr(self, "_ref_bank", None) is None:
            return {"error": "load a reference bank first (Reference tab), or use method=classifier/raw"}
        out = Path(out_dir) if out_dir else (self.store.dir / "pseudolabels")
        out.mkdir(parents=True, exist_ok=True)
        st, feat_cfg = self._infer_thresholds(score_thresh, nms_iou)
        need_rad = method == "reference" or (method == "classifier" and "raddino" in (self._clf_spec or {}))
        n = len(files)
        shard_paths, n_inst, n_lab = [], 0, 0
        try:
            for si, s in enumerate(range(0, n, int(shard_size))):
                self._set_progress("scaled pseudolabel", s, n)
                batch = _co.collect_batch(model, cfg, d2_cfg, files[s:s + int(shard_size)],
                                          score_thresh=st, feature_cfg=feat_cfg)
                recs = batch["records"]; n_inst += len(recs)
                if recs:
                    emb = None
                    if need_rad:
                        _co._raddino_by_path(batch, get_P(), pool=pool)
                        emb = batch["feats"].get("raddino")
                    if method == "raw":
                        class_of, scores = ["object"] * len(recs), None
                    elif method == "reference":
                        class_of, scores = _sc.assign_by_reference(emb, self._ref_bank, thresh)
                    else:
                        cids, scores = _sc.assign_by_classifier(batch, self._clf, self._clf_spec, thresh)
                        class_of = [self.state.class_name(c) if c else None for c in cids]
                    n_lab += sum(1 for c in class_of if c)
                    coco = _sc.batch_to_coco(batch, class_of, class_agnostic=class_agnostic, scores=scores)
                    sp = out / f"shard_{si:04d}.json"; sp.write_text(json.dumps(coco)); shard_paths.append(str(sp))
                del batch
                gc.collect()                                   # drop the shard -> RAM bounded to one shard
            self._set_progress("scaled pseudolabel", n, n)
        finally:
            self._clear_progress()
        merged = _sc.merge_cocos(shard_paths, out / "merged.json")
        return {"ok": True, "n_images": n, "n_instances": n_inst, "n_labeled": n_lab,
                "method": method, "shards": len(shard_paths), "merged": merged}

    def compute_raddino(self, *, force: bool = False, pool: str = "mask") -> dict:
        """On-demand RAD-DINO features for the CURRENT collection (no re-detection): soft mask-pool
        each existing instance's mask over the RAD-DINO patch grid (reuses collect._raddino_by_path),
        adding feats['raddino'] aligned to existing rows → 'raddino' becomes selectable. GPU/HF, opt-in.
        `force` recomputes even if present (e.g. after new instances were ingested)."""
        if not self.collection or not self.collection.get("records"):
            return {"error": "no collection — Sample & extract first"}
        if "raddino" in self.collection["feats"] and not force \
                and self.collection["feats"]["raddino"].shape[0] == len(self.collection["records"]):
            return {"ok": True, "msg": "raddino already present", "n": int(self.collection["feats"]["raddino"].shape[0]),
                    "available": self.available_features()}
        from ._bootstrap import get_P
        self._set_progress("loading model", 0, 0)
        try:
            _co._raddino_by_path(self.collection, get_P(), pool=str(pool),
                                 progress=lambda d, t: self._set_progress("RAD-DINO features", d, t))
        finally:
            self._clear_progress()
        if "raddino" not in self.collection["feats"]:
            return {"error": "RAD-DINO extraction produced no features"}
        self.state.assert_aligned(self.collection["feats"]["raddino"].shape[0])
        self.state.coll_version += 1
        self.state.collection_dirty = True
        self.store.save_collection(self.collection)
        self.save()
        return {"ok": True, "n": int(self.collection["feats"]["raddino"].shape[0]),
                "available": self.available_features()}

    # ---- clustering --------------------------------------------------------
    def available_features(self) -> list[str]:
        """Feature methods actually present in the collection (single source of truth for the UI
        selectors). Excludes `_`-prefixed metadata keys (e.g. _shape_cols)."""
        if not self.collection or not self.collection.get("feats"):
            return []
        return sorted(k for k in self.collection["feats"] if not k.startswith("_"))

    def _present_spec(self, spec) -> dict:
        """Spec restricted to feature methods present in the collection (drops absent ones)."""
        avail = set(self.available_features())
        return {m: w for m, w in _cl.normalize_spec(spec).items() if m in avail}

    def _pool_iuids(self) -> list[str]:
        """The curation pool that gets clustered: unassigned, non-background, non-merge-child."""
        return [u for u in self.state.order
                if self.state.meta[u].assigned_class is None
                and not self.state.meta[u].is_background and self.state.meta[u].merged_into is None]

    @_timed
    def cluster(self, spec, *, distance: str = "cosine", per_image: bool = False, level: int | None = None,
                req_clust: int | None = None) -> dict:
        """FINCH-cluster ONLY the unassigned pool — already-assigned instances are not reclustered
        (each class becomes its own standalone pseudo-partition in partition_view)."""
        from ._bootstrap import get_P
        P = get_P()
        pool = self._pool_iuids()
        spec = self._present_spec(spec)
        if not spec:
            raise ValueError(f"none of the selected features are present; available: {self.available_features()}")
        if len(pool) < 2:
            partitions, counts = np.zeros((len(pool), 1), int), [max(1, len(pool))]
        else:
            X = self.fused(spec)[[self.state.meta[u].row for u in pool]]
            if req_clust:
                labels = np.asarray(P.cluster(X, "finch", req_clust=int(req_clust), distance=distance))
                partitions, counts = labels.reshape(-1, 1), [int(len(set(labels.tolist())))]
            else:
                partitions, counts = P.finch_hierarchy(X, distance=distance)
        lvl = level if level is not None else _default_level(counts)
        self._cluster = {"spec": spec, "distance": distance, "partitions": partitions,
                         "counts": counts, "level": lvl, "pool": pool}
        self.state.collection_dirty = False
        self.save()
        return {"counts": counts, "level": lvl, "n_levels": len(counts)}

    def set_level(self, level: int) -> None:
        if self._cluster:
            self._cluster["level"] = max(0, min(level, len(self._cluster["counts"]) - 1))

    def _pool_labels(self) -> np.ndarray:
        return _cl.labels_at_level(self._cluster["partitions"], self._cluster["level"])

    @_timed
    def _pool_groups(self) -> dict[int, list[int]]:
        """{pid -> [pool index, ...]} for the current cluster+level, built ONCE in O(N) and cached.
        Replaces the per-partition np.where (O(P·N) -> ~quadratic at the finest FINCH level, e.g. 4700
        partitions over 25k); the grouping is by FINCH label so it survives mutations (only the
        _is_pool filter changes), recomputed only when the cluster object or level changes."""
        key = (id(self._cluster), self._cluster["level"])
        cached = getattr(self, "_grp_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        groups: dict[int, list[int]] = {}
        for i, lab in enumerate(self._pool_labels().tolist()):
            groups.setdefault(int(lab), []).append(i)
        self._grp_cache = (key, groups)
        return groups

    def _is_pool(self, u: str) -> bool:
        m = self.state.meta[u]
        return m.assigned_class is None and not m.is_background and m.merged_into is None

    def _view_sig(self):
        """O(1) signature of everything partition_view depends on — changes on any mutation (history
        push), undo/redo (depth), (re)cluster (new _cluster object), set_level, or append (coll_version)."""
        u, r = self.history.depths
        return (u, r, self.state.coll_version, id(self._cluster),
                self._cluster["level"] if self._cluster else -1, len(self.state.taxonomy))

    # ---- within-class substructure (self-supervised contrastive + FINCH) ----
    def subcluster(self, target, *, spec, dim: int = 64, epochs: int = 150, temperature: float = 0.2,
                   distance: str = "cosine", device: str = "cpu", cap: int = 6000, seed: int = 0) -> dict:
        """Find SUBSTRUCTURE inside one partition/class: train a feature-space contrastive (SimCLR/NT-Xent)
        encoder on the target's instance features, then FINCH-cluster the learned embeddings. `target` is a
        partition id (finch int as str, or 'class:<cid>'). Result is stored on the engine (ephemeral)."""
        from . import contrastive as _ct
        from ._bootstrap import get_P
        P = get_P()
        iuids = self.partition_iuids(str(target))
        if len(iuids) < 3:
            return {"error": f"need >=3 instances in the target to find substructure (got {len(iuids)})"}
        spec = self._present_spec(spec)
        if not spec:
            return {"error": f"none of the selected features are present; available: {self.available_features()}"}
        capped = len(iuids) > int(cap)
        if capped:                                          # bound training cost on huge classes
            sel = np.random.default_rng(int(seed)).choice(len(iuids), int(cap), replace=False)
            iuids = [iuids[i] for i in sorted(sel.tolist())]
        rows = [self.state.meta[u].row for u in iuids]
        X = self.fused(spec)[rows]
        emb = _ct.train_embeddings(X, dim=int(dim), epochs=int(epochs), temperature=float(temperature),
                                   device=device, seed=int(seed))
        partitions, counts = P.finch_hierarchy(emb, distance=distance)
        self._subcluster = {"target": str(target), "iuids": iuids, "spec": spec,
                            "partitions": np.asarray(partitions), "counts": list(counts),
                            "level": _default_level(counts)}
        return {"ok": True, "target": str(target), "n": len(iuids), "counts": list(counts),
                "level": self._subcluster["level"], "n_levels": len(counts), "capped": capped}

    def subcluster_set_level(self, level: int) -> None:
        if self._subcluster:
            self._subcluster["level"] = max(0, min(int(level), len(self._subcluster["counts"]) - 1))

    def _subcluster_labels(self) -> np.ndarray:
        return _cl.labels_at_level(self._subcluster["partitions"], self._subcluster["level"])

    def subcluster_view(self) -> list[dict]:
        if not self._subcluster:
            return []
        from collections import Counter
        cnt = Counter(int(x) for x in self._subcluster_labels().tolist())
        return [{"subpid": str(p), "size": int(n)} for p, n in sorted(cnt.items(), key=lambda kv: -kv[1])]

    def subcluster_iuids(self, subpid) -> list[str]:
        if not self._subcluster:
            return []
        try:
            target = int(subpid)
        except (ValueError, TypeError):
            return []
        labels = self._subcluster_labels().tolist()
        ius = self._subcluster["iuids"]
        return [ius[i] for i, lab in enumerate(labels)
                if int(lab) == target and ius[i] in self.state.meta]

    @_timed
    def partition_view(self) -> list[dict]:
        """Per-class pseudo-partitions (assigned instances, pid='class:<cid>') first, then the FINCH
        partitions of the still-unassigned pool (pid=str int), filtered to current membership.
        Memoized on _view_sig() — called 2-3x per interaction; recompute only when state actually changes."""
        sig = self._view_sig()
        cached = getattr(self, "_pv_cache", None)
        if cached is not None and cached[0] == sig:
            return cached[1]
        rows = []
        for cid in self.state.taxonomy:
            members = [u for u, m in self.state.meta.items()
                       if m.assigned_class == cid and not m.is_background and m.merged_into is None]
            if members:
                sc = [self.collection["records"][self.state.meta[u].row]["score"] for u in members]
                rows.append({"pid": f"class:{cid}", "size": len(members), "purity": 1.0,
                             "mean_score": round(float(np.mean(sc)), 2), "majority_class": self.state.class_name(cid)})
        if self._cluster:
            groups, pool = self._pool_groups(), self._cluster["pool"]
            for pid in sorted(groups):
                members = [pool[i] for i in groups[pid] if self._is_pool(pool[i])]
                if members:
                    sc = [self.collection["records"][self.state.meta[u].row]["score"] for u in members]
                    rows.append({"pid": str(pid), "size": len(members), "purity": None,
                                 "mean_score": round(float(np.mean(sc)), 2), "majority_class": ""})
        rows.sort(key=lambda r: (not str(r["pid"]).startswith("class:"), -r["size"]))
        self._pv_cache = (sig, rows)
        return rows

    def partition_iuids(self, pid) -> list[str]:
        pid = str(pid)
        if pid.startswith("class:"):
            cid = pid[len("class:"):]
            return [u for u, m in self.state.meta.items()
                    if m.assigned_class == cid and not m.is_background and m.merged_into is None]
        if not self._cluster:
            return []
        try:
            target = int(pid)
        except ValueError:
            return []
        pool = self._cluster["pool"]
        return [pool[i] for i in self._pool_groups().get(target, []) if self._is_pool(pool[i])]

    # ---- rendering ---------------------------------------------------------
    def _eff_rle(self, iuid: str) -> dict:
        # overlay (refine/merge result) is used ONLY when the meta flag is set, so undo/redo of
        # refine/merge — which toggle those flags — actually revert the effective mask.
        m = self.state.meta[iuid]
        if (m.refined or m.merge_members) and iuid in self._overlay_rle:
            return self._overlay_rle[iuid]
        return self.collection["records"][m.row]["rle"]

    def _mask(self, iuid: str) -> np.ndarray:
        from pycocotools import mask as mu
        return mu.decode(self._eff_rle(iuid)).astype(bool)

    def mask_token(self, iuid: str) -> str:
        """Short token that changes iff the instance's effective mask changes (merge/refine/split).
        Used to key crop thumbnails so unchanged ones are served from cache (not recomputed)."""
        import zlib
        counts = self._eff_rle(iuid)["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return f"{zlib.crc32(counts.encode('ascii')) & 0xffffffff:08x}"

    def _rgb(self, iuid: str) -> np.ndarray:
        rec = self.collection["records"][self.state.meta[iuid].row]
        return _load_rgb(rec.get("abs_path") or rec["file_name"], (int(rec["H"]), int(rec["W"])))

    def crop(self, iuid: str, *, mask_overlay: bool = True, pad: int = 10, context: bool = False,
             max_side: int = 512) -> np.ndarray:
        """Thumbnail crop of the instance (default) or the WHOLE source image with the instance
        highlighted (context=True), downscaled to <= max_side. Works on the instance's BBOX sub-region
        only (not a full-image copy/overlay) so cost is independent of the source resolution."""
        import cv2
        ck = (iuid, self.mask_token(iuid), self.state.meta[iuid].row, bool(mask_overlay), int(pad),
              bool(context), int(max_side))
        hit = _CROP_CACHE.get(ck)
        if hit is not None:
            _CROP_CACHE.move_to_end(ck)
            return hit
        rgb = self._rgb(iuid)                               # cached; never mutate in place
        m = self._mask(iuid)
        H, W = m.shape
        c = _color(self.state.meta[iuid].row)
        x, y, w, h = cv2.boundingRect(m.astype(np.uint8))   # (0,0,0,0) when empty
        if context or w == 0:                               # whole image (rare; "in context" view)
            out = rgb.copy()
            if w:
                cv2.rectangle(out, (max(0, x - 2), max(0, y - 2)), (min(W, x + w + 2), min(H, y + h + 2)),
                              tuple(int(v) for v in c), 2)
            return self._cache_crop(ck, _downscale(out, max_side))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
        sub = rgb[y1:y2, x1:x2].copy()                      # SMALL region only
        if mask_overlay:
            subm = m[y1:y2, x1:x2]
            sub[subm] = (0.5 * sub[subm] + 0.5 * c).astype(np.uint8)
            cont, _ = cv2.findContours(subm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sub, cont, -1, tuple(int(v) for v in c), 1)
        return self._cache_crop(ck, _downscale(sub, max_side))   # source name in the UI caption, not pixels

    @staticmethod
    def _cache_crop(ck: tuple, img: np.ndarray) -> np.ndarray:
        _CROP_CACHE[ck] = img
        _CROP_CACHE.move_to_end(ck)
        while len(_CROP_CACHE) > _CROP_CACHE_MAX:
            _CROP_CACHE.popitem(last=False)
        return img

    def _src_name(self, iuid: str) -> str:
        return Path(self.collection["records"][self.state.meta[iuid].row].get("file_name", "")).name

    def _caption(self, iuid: str) -> str:
        m = self.state.meta[iuid]
        cls = f" [{self.state.class_name(m.assigned_class)}]" if m.assigned_class else ""
        return f"{self._src_name(iuid)} · {iuid[:6]} s={self.collection['records'][m.row]['score']:.2f}{cls}"

    def partition_crops(self, pid: int, *, mask_overlay: bool = True, limit: int = 48, context: bool = False):
        iuids = self.partition_iuids(pid)[:limit]
        return [(self.crop(u, mask_overlay=mask_overlay, context=context), self._caption(u)) for u in iuids], iuids

    def image_ids(self) -> list[int]:
        from collections import Counter
        c = Counter(m.image_id for m in self.state.meta.values())
        return [iid for iid, _ in c.most_common()]

    @_timed
    def image_overlay(self, image_id: int, *, color_by: str = "partition", max_side: int = 900,
                      show_masks: bool = True) -> np.ndarray:
        """Whole-image overlay for the In-image tab, computed on a DOWNSCALED canvas (it's shown ~440px),
        so cost is independent of the source resolution (was full-res float ops per instance).
        show_masks=False returns the bare (downscaled) image with no mask fills/contours."""
        import cv2
        iuids = self.image_instance_iuids(image_id)        # excludes merge children (rep shows the union)
        if not iuids:
            return np.zeros((512, 512, 3), np.uint8)
        rgb = self._rgb(iuids[0]); H, W = rgb.shape[:2]
        s = max_side / max(H, W) if max(H, W) > max_side else 1.0
        out = (cv2.resize(rgb, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
               if s < 1.0 else rgb.copy()).astype(np.float32)
        if not show_masks:
            return out.astype(np.uint8)
        h2, w2 = out.shape[:2]
        labels = self._label_for_order(color_by) if (self._cluster and color_by == "partition") else None
        for u in iuids:
            m = self._mask(u)
            if s < 1.0:
                m = cv2.resize(m.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST) > 0
            if color_by == "partition" and labels is not None:
                key = int(labels[self.state.meta[u].row])
            elif color_by == "class":
                key = abs(hash(self.state.meta[u].assigned_class or "")) % 997
            else:
                key = self.state.meta[u].row
            c = _color(key)
            out[m] = 0.5 * out[m] + 0.5 * c
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, tuple(int(v) for v in c), 1)
        return out.astype(np.uint8)

    def keypoints_overlay(self, image_id: int) -> np.ndarray:
        from ._bootstrap import get_P
        P = get_P()
        iuids = [u for u, m in self.state.meta.items() if m.image_id == image_id]
        if not iuids:
            return np.zeros((512, 512, 3), np.uint8)
        img = self._rgb(iuids[0]).copy()                 # cached array is shared; overlay_keypoints may draw in place
        recs = self.collection["records"]
        kp = [recs[self.state.meta[u].row]["keypoints"] for u in iuids
              if "keypoints" in recs[self.state.meta[u].row]]
        vis = [recs[self.state.meta[u].row].get("keypoint_vis") for u in iuids
               if "keypoints" in recs[self.state.meta[u].row]]
        return P.overlay_keypoints(img, kp, vis, vis_thresh=0.3) if kp else img

    # ---- assignment / curation --------------------------------------------
    def new_class(self, name: str) -> str:
        tok = self.history.begin(self.state, [], [])
        cid = self.state.add_class(name)
        tok["class_ids"].append(cid)
        self.history.commit(self.state, tok, "new_class", f"new class {name}")
        self.save()
        return cid

    def assign(self, iuids: list[str], class_name: str, *, source: str = "manual",
               scores: dict | None = None) -> None:
        if not iuids:
            return
        existing = self.state.class_id_by_name(class_name)
        tok = self.history.begin(self.state, iuids, [existing] if existing else [])
        cid = self.state.add_class(class_name)
        if not existing:
            tok["class_ids"].append(cid)
        for u in iuids:
            m = self.state.meta[u]
            m.assigned_class = cid; m.is_background = False
            m.assign_source = source
            m.assign_score = (scores or {}).get(u)
        self.history.commit(self.state, tok, "assign", f"assign {len(iuids)}→{class_name}")
        self._after_mutation()

    def assign_partition(self, pid: int, class_name: str) -> None:
        self.assign(self.partition_iuids(pid), class_name, source="partition")

    def remove_from_class(self, iuids: list[str]) -> None:
        tok = self.history.begin(self.state, iuids, [])
        for u in iuids:
            self.state.meta[u].assigned_class = None
            self.state.meta[u].assign_source = None
            self.state.meta[u].assign_score = None
        self.history.commit(self.state, tok, "remove", f"unassign {len(iuids)}")
        self._after_mutation()

    def set_background(self, iuids: list[str]) -> None:
        tok = self.history.begin(self.state, iuids, [])
        for u in iuids:
            self.state.meta[u].is_background = True
            self.state.meta[u].assigned_class = None
        self.history.commit(self.state, tok, "background", f"reject {len(iuids)}")
        self._after_mutation()

    # ---- nested taxonomy (superclass -> concept -> leaf parts) -------------
    def seed_taxonomy(self, path=None, *, replace: bool = False) -> dict:
        """Load the nested taxonomy seed (superclasses + concepts + part leaves) into state, pinning a stable
        coco_cat_id per leaf. Idempotent: existing leaves keep their id, just gain grouping metadata.
        `replace` first clears superclasses/concepts (leaves are kept — they may carry assignments)."""
        import json
        from .state import Concept, Superclass, TaxonomyClass, _auto_color
        path = str(path) if path else str(Path(__file__).parent / "taxonomy_seed.json")
        d = json.load(open(path))
        if replace:
            self.state.superclasses = {}
            self.state.concepts = {}
        for s in d.get("superclasses", []):
            self.state.superclasses[s["id"]] = Superclass(id=s["id"], name=s["name"],
                color=s.get("color", [150, 150, 150]), description=s.get("description", ""))
        next_id = max([c.coco_cat_id or 0 for c in self.state.taxonomy.values()] + [0]) + 1
        for c in d.get("concepts", []):
            self.state.concepts[c["id"]] = Concept(concept_id=c["id"], name=c["name"], superclass=c.get("superclass"),
                description=c.get("description", ""), structure_type=c.get("structure_type", ""),
                aliases=c.get("aliases", []), part_rules=c.get("part_rules", []), mimic_family=c.get("mimic_family"))
            leaves = c.get("parts") or [{"id": c["id"], "name": c["name"], "structure_type": c.get("structure_type", ""),
                                         "description": c.get("description", "")}]
            for lf in leaves:
                lid = lf["id"]
                t = self.state.taxonomy.get(lid)
                if t is None:
                    self.state.taxonomy[lid] = TaxonomyClass(class_id=lid, name=lf.get("name", lid),
                        color=_auto_color(len(self.state.taxonomy)), coco_cat_id=next_id, concept=c["id"],
                        supercategory=c.get("superclass"), description=lf.get("description", ""),
                        structure_type=lf.get("structure_type", c.get("structure_type", "")), temp=False)
                    next_id += 1
                else:                                            # existing leaf -> attach grouping, keep id/assignments
                    t.concept = c["id"]; t.supercategory = c.get("superclass"); t.temp = False
                    if not t.description:
                        t.description = lf.get("description", "")
        self.save()
        return {"superclasses": len(self.state.superclasses), "concepts": len(self.state.concepts),
                "leaves": len([t for t in self.state.taxonomy.values() if not t.temp])}

    def taxonomy_tree(self) -> dict:
        """Structured taxonomy for the editor + grouped pickers: superclasses -> concepts -> leaves with live
        instance counts, plus a TEMP/scratch bucket (ungrouped or temp leaves, excluded from export)."""
        from collections import Counter
        cnt: Counter = Counter()
        for m in self.state.meta.values():
            if m.assigned_class and not m.is_background and m.merged_into is None:
                cnt[m.assigned_class] += 1
        def leaf(t):
            return {"id": t.class_id, "name": t.name, "color": t.color, "coco_cat_id": t.coco_cat_id,
                    "structure_type": t.structure_type, "n": int(cnt.get(t.class_id, 0))}
        leaves_by_concept: dict = {}
        temp = []
        for t in self.state.taxonomy.values():
            if t.temp or not t.concept:
                temp.append({**leaf(t), "temp": bool(t.temp)})
            else:
                leaves_by_concept.setdefault(t.concept, []).append(leaf(t))
        scs = []
        for sid, s in self.state.superclasses.items():
            cons = []
            for cid, c in self.state.concepts.items():
                if c.superclass != sid:
                    continue
                lv = leaves_by_concept.get(cid, [])
                cons.append({"id": cid, "name": c.name, "description": c.description, "mimic_family": c.mimic_family,
                             "structure_type": c.structure_type, "part_rules": c.part_rules, "leaves": lv,
                             "n": sum(x["n"] for x in lv)})
            scs.append({"id": sid, "name": s.name, "color": s.color, "description": s.description,
                        "concepts": sorted(cons, key=lambda x: x["name"]), "n": sum(x["n"] for x in cons)})
        return {"superclasses": sorted(scs, key=lambda x: x["name"]),
                "temp": sorted(temp, key=lambda x: -x["n"])}

    def set_class_temp(self, class_ids, temp: bool = True) -> dict:
        """Flag classes temp/scratch (excluded from export + taxonomy) or un-flag. Bulk."""
        n = 0
        for c in class_ids:
            if c in self.state.taxonomy:
                self.state.taxonomy[c].temp = bool(temp); n += 1
        self.save()
        return {"ok": True, "n": n}

    def release_qc(self) -> dict:
        """Per-image part-rule completeness check (the release gate): for each concept with part_rules, on each
        image where the rule's `if` leaf is present (assigned, non-bg), all `then` leaves must also be present.
        Returns the violating images so they can be held back from the released annotation set."""
        from collections import defaultdict
        present: dict = defaultdict(set)
        for m in self.state.meta.values():
            if m.assigned_class and not m.is_background and m.merged_into is None:
                present[int(m.image_id)].add(m.assigned_class)
        violations = []
        for cid, c in self.state.concepts.items():
            for r in (c.part_rules or []):
                for iid, leaves in present.items():
                    if r.get("if") in leaves:
                        missing = [t for t in r.get("then", []) if t not in leaves]
                        if missing:
                            violations.append({"image_id": str(iid), "concept": cid, "if": r["if"],
                                               "missing": missing, "desc": r.get("desc", "")})
        bad = {v["image_id"] for v in violations}
        return {"n_images": len(present), "n_violating": len(bad),
                "violating_image_ids": sorted(bad), "violations": violations[:500]}

    def classes_summary(self) -> list[dict]:
        """Every taxonomy class with its live (non-bg, non-merged) instance count, most-populated first."""
        from collections import Counter
        cnt: Counter = Counter()
        for m in self.state.meta.values():
            if m.assigned_class and not m.is_background and m.merged_into is None:
                cnt[m.assigned_class] += 1
        return sorted(({"cls": self.state.class_name(c), "n": int(cnt.get(c, 0))} for c in self.state.taxonomy),
                      key=lambda x: -x["n"])

    def merge_classes(self, sources: list[str], into: str) -> dict:
        """Merge several classes into one. `into` may be an EXISTING class (the others fold into it) or a
        NEW name (all sources fold into it). Every instance of a source class is reassigned to the target;
        the now-empty source classes are removed from the taxonomy. Reversible (one undoable command)."""
        into = (into or "").strip()
        src_cids = list(dict.fromkeys(c for c in (self.state.class_id_by_name(n) for n in (sources or [])) if c))
        if not into or not src_cids:
            return {"error": "pick >=1 source class and a target name"}
        dst_exists = self.state.class_id_by_name(into)
        move = [u for u, m in self.state.meta.items() if m.assigned_class in set(src_cids)]
        class_ids = list(dict.fromkeys(src_cids + ([dst_exists] if dst_exists else [])))
        tok = self.history.begin(self.state, move, class_ids)
        dst = self.state.add_class(into)
        if not dst_exists:
            tok["class_ids"].append(dst)
        for u in move:
            self.state.meta[u].assigned_class = dst
        removed = []
        for c in src_cids:
            if c != dst:
                removed.append(self.state.class_name(c))
                self.state.taxonomy.pop(c, None)
                self.state.class_rules.pop(c, None)
        self.history.commit(self.state, tok, "merge_classes", f"merge {removed}→{into}")
        self._after_mutation()
        return {"ok": True, "into": into, "moved": len(move), "removed": removed}

    # ---- within-image merge ------------------------------------------------
    def merge_preview(self, image_id: int, *, dist_kind: str = "mask_gap", method: str = "decoder",
                      thresh: float = 0.05, max_group_size: int | None = None):
        from ._bootstrap import get_P
        P = get_P()
        groups, _ = P.merge_groups_in_image(self.collection, image_id, dist_kind=dist_kind, method=method,
                                            thresh=thresh, max_group_size=max_group_size)
        before, _ = P.merge_groups_in_image(self.collection, image_id, dist_kind=dist_kind, thresh=0.0)
        img = self._rgb_by_image(image_id)
        return (P.overlay_groups(img, self.collection, before),
                P.overlay_groups(img, self.collection, groups), groups)

    def _rgb_by_image(self, image_id: int) -> np.ndarray:
        iuids = [u for u, m in self.state.meta.items() if m.image_id == image_id]
        return self._rgb(iuids[0]).copy() if iuids else np.zeros((512, 512, 3), np.uint8)  # overlay_groups draws in place

    def _merge_mask(self, iuids: list[str], mode: str):
        """Combine the group's masks per mode (a/b = highest/2nd-highest score):
        union=OR, intersection=AND, pref_a=top instance's mask, pref_b=2nd instance's mask.
        Returns (result_mask, ordered_iuids) with ordered[0] the representative."""
        ordered = sorted(iuids, key=lambda u: -self.collection["records"][self.state.meta[u].row]["score"])
        masks = [self._mask(u) for u in ordered]
        if mode == "intersection":
            res = masks[0].copy()
            for m in masks[1:]:
                res &= m
        elif mode == "pref_a":
            res = masks[0]
        elif mode == "pref_b":
            res = masks[1] if len(masks) > 1 else masks[0]
        else:                                                   # union (default)
            res = masks[0].copy()
            for m in masks[1:]:
                res |= m
        return res, ordered

    def merge_result_preview(self, iuids: list[str], mode: str = "union", *, max_side: int = 256) -> np.ndarray:
        """Render what merging `iuids` with `mode` would look like (green mask on the group's bbox crop),
        WITHOUT committing. For the in-image / merge-rec previews."""
        import cv2
        iuids = [u for u in iuids if u in self.state.meta]
        if len(iuids) < 2:
            return np.zeros((64, 64, 3), np.uint8)
        res, ordered = self._merge_mask(iuids, mode)
        union = None                                            # always crop to the union bbox (stable framing)
        for u in ordered:
            m = self._mask(u); union = m if union is None else (union | m)
        rgb = self._rgb(ordered[0]); H, W = res.shape
        x, y, w, h = cv2.boundingRect(union.astype(np.uint8))
        if w == 0:
            return _downscale(rgb.copy(), max_side)
        x1, y1 = max(0, x - 10), max(0, y - 10); x2, y2 = min(W, x + w + 10), min(H, y + h + 10)
        sub = rgb[y1:y2, x1:x2].copy(); subm = res[y1:y2, x1:x2]
        if subm.any():
            sub[subm] = (0.5 * sub[subm] + 0.5 * np.array([40, 220, 40])).astype(np.uint8)
            cont, _ = cv2.findContours(subm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sub, cont, -1, (40, 220, 40), 1)
        return _downscale(sub, max_side)

    def _merge_group_nohist(self, iuids: list[str], mode: str = "union") -> str:
        """Merge a group into the highest-score representative (no history). Returns rep."""
        from pycocotools import mask as mu
        res, ordered = self._merge_mask(iuids, mode)
        rep = ordered[0]
        rle = mu.encode(np.asfortranarray(res.astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
        for u in iuids:
            self.state.meta[u].merged_into = (None if u == rep else rep)
        self.state.meta[rep].merge_members = [u for u in iuids if u != rep]
        self._overlay_rle[rep] = rle
        self.store.save_refine(rep, {"iuid": rep, "base_rle": self.collection["records"][self.state.meta[rep].row]["rle"],
                                     "ops": [{"name": "merge", "mode": mode, "members": iuids}], "result_rle": rle})
        return rep

    def _commit_merge_groups(self, groups_iuids: list[list[str]], label: str, mode: str = "union",
                             source: str = "manual") -> int:
        groups_iuids = [g for g in groups_iuids if len(g) >= 2]
        if not groups_iuids:
            return 0
        all_iuids = [u for g in groups_iuids for u in g]
        tok = self.history.begin(self.state, all_iuids, [])
        for g in groups_iuids:
            self._merge_group_nohist(g, mode)
            self.store.append_merge_event({"kind": "merge", "iuids": list(g), "mode": mode,   # positive training signal
                                           "image_id": int(self.state.meta[g[0]].image_id),
                                           "source": source, "ts": time.time()})            # manual vs recommended + when
        self.history.commit(self.state, tok, "merge", label)
        self._after_mutation()
        return len(groups_iuids)

    def commit_merge(self, image_id: int, groups: list[list[int]], mode: str = "union") -> None:
        """groups are GLOBAL row indices (from merge_preview)."""
        order = self.state.order
        self._commit_merge_groups([[order[r] for r in g] for g in groups], f"merge img {image_id}", mode)

    def merge_instances(self, iuids: list[str], mode: str = "union", source: str = "manual") -> int:
        """Manual merge of a selected set. Only instances from the SAME base image are merged together
        (cross-image merges are meaningless and their masks have different shapes): the selection is
        grouped by image_id and each same-image group of >=2 is merged into its highest-score
        representative. Returns the number of groups merged (0 if nothing shares an image).
        `source` tags the merge-log event ("manual" or "recommended" when accepted from the recommender)."""
        from collections import defaultdict
        by_img: dict = defaultdict(list)
        for u in iuids:
            if u in self.state.meta:
                by_img[self.state.meta[u].image_id].append(u)
        return self._commit_merge_groups([g for g in by_img.values() if len(g) >= 2],
                                         f"merge {len(iuids)} selected (per image)", mode, source=source)

    def merge_partition_by_image(self, pid: int) -> int:
        """Merge all same-image instances within a partition (small partitions with dup regions)."""
        from collections import defaultdict
        by_img = defaultdict(list)
        for u in self.partition_iuids(pid):
            by_img[self.state.meta[u].image_id].append(u)
        return self._commit_merge_groups([g for g in by_img.values() if len(g) >= 2],
                                         f"merge same-image in partition {pid}")

    def dedup_current(self, iou: float = 0.8) -> int:
        """Mark near-duplicate (mask-IoU >= iou) lower-score instances per image as background
        (reversible). For already-collected sets."""
        from collections import defaultdict
        from pycocotools import mask as mu
        by_img = defaultdict(list)
        for u, m in self.state.meta.items():
            if not m.is_background and m.merged_into is None:
                by_img[m.image_id].append(u)
        to_bg = []
        for iuids in by_img.values():
            order = sorted(iuids, key=lambda u: -self.collection["records"][self.state.meta[u].row]["score"])
            kept = []
            for u in order:
                rle = self._eff_rle(u)
                if kept and float(np.max(mu.iou([rle], kept, [0] * len(kept)))) >= iou:
                    to_bg.append(u)
                else:
                    kept.append(rle)
        if to_bg:
            tok = self.history.begin(self.state, to_bg, [])
            for u in to_bg:
                self.state.meta[u].is_background = True
            self.history.commit(self.state, tok, "dedup", f"dedup→bg {len(to_bg)}")
            self._after_mutation()
        return len(to_bg)

    def instance_at_pixel(self, image_id: int, x: int, y: int) -> str | None:
        """Highest-score instance in the image whose effective mask covers pixel (x,y)."""
        cands = [u for u, m in self.state.meta.items() if m.image_id == image_id]
        cands.sort(key=lambda u: -self.collection["records"][self.state.meta[u].row]["score"])
        for u in cands:
            m = self._mask(u)
            if 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1] and m[int(y), int(x)]:
                return u
        return None

    def image_instance_gallery(self, image_id: int, *, mask_overlay: bool = True):
        iuids = self.image_instance_iuids(image_id)
        return [(self.crop(u, mask_overlay=mask_overlay), self._caption(u)) for u in iuids], iuids

    def image_instance_iuids(self, image_id: int) -> list[str]:
        # hide merge CHILDREN (merged_into set) — a merged group collapses to its representative —
        # AND rejected/background instances, so rejecting in In-image actually removes them from the set
        # (and the overlay) instead of reappearing on reload. Unreject from the Rejected tab to restore.
        return [u for u, m in self.state.meta.items()
                if m.image_id == image_id and m.merged_into is None and not m.is_background]

    def background_iuids(self) -> list[str]:
        return [u for u, m in self.state.meta.items() if m.is_background]

    def unreject(self, iuids: list[str]) -> int:
        """Send rejected (background) instances back to UNASSIGNED. Reversible."""
        bg = [u for u in iuids if u in self.state.meta and self.state.meta[u].is_background]
        if not bg:
            return 0
        tok = self.history.begin(self.state, bg, [])
        for u in bg:
            self.state.meta[u].is_background = False
            self.state.meta[u].assigned_class = None
        self.history.commit(self.state, tok, "unreject", f"unreject {len(bg)}")
        self._after_mutation()
        return len(bg)

    def reset(self, *, keep_config: bool = True) -> None:
        """Drop everything (collection, instances, assignments, classes, overlays, caches,
        processed-image list); keep only the config. Destructive, NOT undoable."""
        cfg = dict(self.state.config) if keep_config else {}
        if self.store.collection_path.exists():
            self.store.collection_path.unlink()
        self.store.clear_cache()
        for f in self.store.refine_dir.glob("*.pkl"):
            f.unlink()
        man = self.store.load_manifest()
        man.update({"processed_paths": [], "coll_version": 0, "n_instances": 0})
        self.store.save_manifest(man)
        self.state = CuratorState(project_dir=str(self.store.dir), config=cfg)
        self.collection = None
        self._overlay_rle = {}
        self._cluster = None
        self._clf = None
        self.history.barrier()
        self.save()

    # ---- refinement --------------------------------------------------------
    def refine_preview(self, iuid: str, ops: list[dict], *, mask_overlay: bool = True,
                       pad: int = 12, max_side: int = 512):
        """Before/after crops on a SHARED, aligned window (the union bbox of base & refined). The 'after'
        panel is a DIFF overlay so even a tiny change is obvious and removals stay visible:
        YELLOW = unchanged, GREEN = added, RED = removed."""
        import cv2
        from pycocotools import mask as mu

        from .refine import apply_ops, to_gray
        img = self._rgb(iuid)
        gray0 = to_gray(img)
        base_m = mu.decode(self._refine_base_rle(iuid)).astype(bool)   # merge union if merged, else original
        refined, work = apply_ops(gray0, base_m, ops, return_image=True)
        # if a `contrast` op changed the working image, show THAT (3-ch) so the user sees what the ops saw
        bg = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB) if work is not gray0 else img
        H, W = base_m.shape
        ys, xs = np.where(base_m | refined)
        if len(xs) == 0:
            z = _downscale(bg.copy(), max_side)
            return z, z
        x1, y1 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
        x2, y2 = min(W, int(xs.max()) + pad + 1), min(H, int(ys.max()) + pad + 1)
        b, a = base_m[y1:y2, x1:x2], refined[y1:y2, x1:x2]

        def _blend(sub, region, col):
            if region.any():
                sub[region] = (0.45 * sub[region] + 0.55 * np.array(col)).astype(np.uint8)

        def _outline(sub, mm, col):
            cont, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sub, cont, -1, col, 1)

        before = img[y1:y2, x1:x2].copy()
        after = img[y1:y2, x1:x2].copy()
        if mask_overlay:
            _blend(before, b, (40, 220, 40)); _outline(before, b, (40, 220, 40))
            _blend(after, b & a, (255, 220, 0))      # unchanged
            _blend(after, a & ~b, (40, 220, 40))     # added
            _blend(after, b & ~a, (235, 50, 40))     # removed
            _outline(after, a, (40, 220, 40))        # the resulting boundary
        return _downscale(before, max_side), _downscale(after, max_side)

    def sam_prompt_preview(self, iuid: str, ops: list[dict], *, n_pos: int = 10, n_neg: int = 12,
                           margin: int = 24):
        """Crop visualising SAM's prompt sampling: green = positive points (along the mask skeleton /
        centerline), red = negatives (ring `margin` px outside), yellow rect = the bbox prompt. Sampled
        on the mask SAM would actually receive — the base mask after any ops PRECEDING the first `sam`
        op in the chain (so it stays faithful when sam is chained after e.g. fill/largest_cc).
        Returns (rgb_image, n_pos, n_neg)."""
        import cv2
        from pycocotools import mask as mu

        from .refine import apply_ops, sam_prompt_points, to_gray
        img = self._rgb(iuid)
        base_m = mu.decode(self._refine_base_rle(iuid)).astype(bool)   # merge union if merged, else original
        pre = []
        for op in (ops or []):
            if op.get("name") == "sam":
                kw = op.get("kw", {})
                n_pos, n_neg = int(kw.get("n_pos", n_pos)), int(kw.get("n_neg", n_neg))
                margin = int(kw.get("margin", margin))
                break
            pre.append(op)
        m = apply_ops(to_gray(img), base_m, pre) if pre else base_m
        pos, neg, box = sam_prompt_points(m, n_pos=int(n_pos), n_neg=int(n_neg), margin=int(margin))
        out = img.copy()
        H, W = m.shape
        if m.any():
            out[m] = (0.7 * out[m] + 0.3 * np.array([40, 220, 40])).astype(np.uint8)   # faint mask tint
        r = 3
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 220, 0), 2)
            r = max(2, int(0.02 * max(x2 - x1, y2 - y1)))
        for (x, y) in pos:
            cv2.circle(out, (int(x), int(y)), r + 1, (0, 0, 0), -1)
            cv2.circle(out, (int(x), int(y)), r, (40, 230, 40), -1)
        for (x, y) in neg:
            cv2.circle(out, (int(x), int(y)), r + 1, (0, 0, 0), -1)
            cv2.circle(out, (int(x), int(y)), r, (235, 50, 40), -1)
        if box is not None:
            p = max(8, r * 2)
            out = out[max(0, y1 - p):min(H, y2 + p), max(0, x1 - p):min(W, x2 + p)]
        return _downscale(out, 512), int(len(pos)), int(len(neg))

    def _refine_base_rle(self, iuid: str) -> dict:
        """Mask the refine ops start FROM: the MERGE UNION for a merged representative (so refining a
        merge edits the union, not the rep's original single mask), else the original record mask."""
        m = self.state.meta[iuid]
        if m.merge_members and iuid in self._overlay_rle:
            return self._overlay_rle[iuid]
        return self.collection["records"][m.row]["rle"]

    def _refine_one_nohist(self, iuid: str, ops: list[dict]) -> None:
        from .refine import apply_ops, to_gray
        from pycocotools import mask as mu
        base = self._refine_base_rle(iuid)
        refined = apply_ops(to_gray(self._rgb(iuid)), mu.decode(base).astype(bool), ops)
        rle = mu.encode(np.asfortranarray(refined.astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
        self.state.meta[iuid].refined = True
        self.state.meta[iuid].rule_ops = list(ops)          # each instance records the chain applied to it
        self._overlay_rle[iuid] = rle
        self.store.save_refine(iuid, {"iuid": iuid, "base_rle": base, "ops": ops, "result_rle": rle})
        if "shapecoord" in self.collection["feats"]:
            self.collection["feats"]["shapecoord"][self.state.meta[iuid].row] = _co.shapecoord_vector(refined)

    def apply_refine(self, iuid: str, ops: list[dict]) -> None:
        tok = self.history.begin(self.state, [iuid], [])
        self._refine_one_nohist(iuid, ops)
        self.history.commit(self.state, tok, "refine", f"refine {iuid[:6]}")
        self._after_mutation()

    def apply_refine_partition(self, pid, ops: list[dict]) -> int:
        """Apply the op stack to EVERY instance in a partition (finch cluster or class: pseudo-partition),
        one undoable command. Each instance records the chain (meta.rule_ops)."""
        iuids = self.partition_iuids(str(pid))
        if not iuids:
            return 0
        tok = self.history.begin(self.state, iuids, [])
        for u in iuids:
            self._refine_one_nohist(u, ops)
        self.history.commit(self.state, tok, "refine_partition", f"refine partition {pid} ({len(iuids)})")
        self._after_mutation()
        return len(iuids)

    # ---- per-class rule chains (the class's stored postprocessing recipe) ----
    def _resolve_cid(self, cls: str) -> str | None:
        return cls if cls in self.state.taxonomy else self.state.class_id_by_name(cls)

    def class_rule_members(self, cid: str) -> list[str]:
        return [u for u, m in self.state.meta.items()
                if m.assigned_class == cid and not m.is_background and m.merged_into is None]

    def set_class_rule(self, cls: str, ops: list[dict]) -> str | None:
        """Store (persist) a refine rule-chain as the recipe for a class. Returns the class_id."""
        cid = self._resolve_cid(cls)
        if cid is None:
            return None
        self.state.class_rules[cid] = list(ops)
        self.save()
        return cid

    def apply_class_rule(self, cls: str, ops: list[dict] | None = None) -> int:
        """Apply a class's rule-chain to ALL its instances (one undoable command). If `ops` is given it is
        stored as the class recipe first; otherwise the stored recipe is used. Each instance records the
        chain (meta.rule_ops), so the class's postprocessing is reproducible and re-applyable."""
        cid = self.set_class_rule(cls, ops) if ops is not None else self._resolve_cid(cls)
        if cid is None:
            return 0
        chain = self.state.class_rules.get(cid)
        if not chain:
            return 0
        return self.apply_refine_many(self.class_rule_members(cid), chain)

    def class_rules_summary(self) -> list[dict]:
        return [{"cls": self.state.class_name(cid), "ops": [o.get("name") for o in ops],
                 "n": len(self.class_rule_members(cid))}
                for cid, ops in self.state.class_rules.items()]

    def class_rule_for(self, cls: str) -> list[dict]:
        """The FULL saved refine rule-chain (ops with their kw) for a class, so the Refine tab can reload it
        into the live chain when the class is reselected (the summary only carries op names). [] if none."""
        cid = self.state.class_id_by_name(cls)
        return list(self.state.class_rules.get(cid, [])) if cid else []

    def apply_refine_many(self, iuids: list[str], ops: list[dict]) -> int:
        """Refine an explicit set of instances in one undoable command."""
        iuids = [u for u in iuids if u in self.state.meta]
        if not iuids:
            return 0
        tok = self.history.begin(self.state, iuids, [])
        for u in iuids:
            self._refine_one_nohist(u, ops)
        self.history.commit(self.state, tok, "refine_many", f"refine {len(iuids)} instances")
        self._after_mutation()
        return len(iuids)

    def refine_partition_preview(self, pid: int, ops: list[dict], n: int = 6):
        """Before/after crops for the first n instances of a partition (no persistence)."""
        befores, afters = [], []
        for u in self.partition_iuids(pid)[:n]:
            o, r = self.refine_preview(u, ops)
            befores.append((o, u[:6])); afters.append((r, u[:6]))
        return befores, afters

    def split_instances(self, iuids: list[str], *, min_area_frac: float = 0.002, connectivity: int = 8) -> int:
        """Split each instance's effective mask into its connected components, appending ONE new
        unassigned instance per component (new iuid/record/feats row) and sending the originals to
        background. New instances copy the parent's model features and recompute geometry/shapecoord;
        appending instances is an undo BARRIER (like sample_more), so this clears the undo stack.
        Re-cluster to see the children in partitions (they show in In-image immediately)."""
        import cv2
        from pycocotools import mask as mu

        from . import ids as _ids
        from .state import InstanceMeta
        feats = self.collection["feats"]
        recs = self.collection["records"]
        methods = [k for k in feats if not k.startswith("_")]
        new_records: list[dict] = []
        new_feats: dict[str, list] = {k: [] for k in methods}
        parents: list[str] = []
        for u in list(iuids):
            if u not in self.state.meta:
                continue
            m = self._mask(u)
            H, W = m.shape
            n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=connectivity)
            comps = [(lab == c) for c in range(1, n)]
            comps = [c for c in comps if int(c.sum()) >= max(1, int(min_area_frac * m.size))]
            if len(comps) < 2:                                    # nothing to split
                continue
            parents.append(u)
            prow = self.state.meta[u].row
            base = recs[prow]
            for c in comps:
                ys, xs = np.where(c)
                rle = mu.encode(np.asfortranarray(c.astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
                nu = _ids.new_uid()
                r = dict(base); r.pop("keypoints", None); r.pop("keypoint_vis", None)
                r.update({"iuid": nu, "rle": rle, "batch_id": f"{base.get('batch_id', 'b')}/split",
                          "cx": float(xs.mean() / W), "cy": float(ys.mean() / H),
                          "bw": float((xs.max() - xs.min() + 1) / W), "bh": float((ys.max() - ys.min() + 1) / H),
                          "box_area": float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) / (W * H)),
                          "mask_area_frac": float(c.mean())})
                new_records.append(r)
                for k in methods:
                    new_feats[k].append(_co.shapecoord_vector(c) if k == "shapecoord" else feats[k][prow].copy())
        if not new_records:
            return 0
        batch = {"records": new_records, "n_images": 0,
                 "feats": {k: np.asarray(v, dtype=feats[k].dtype) for k, v in new_feats.items()}}
        self.collection = _co.concat_collections(self.collection, batch)
        self.state.order = [r["iuid"] for r in self.collection["records"]]
        for r in new_records:
            self.state.meta[r["iuid"]] = InstanceMeta(
                iuid=r["iuid"], batch_id=r["batch_id"], row=r["row"], image_id=int(r["image_id"]),
                provenance={"split_from": "", "file": r.get("abs_path", "")})
        for u in parents:
            self.state.meta[u].is_background = True
            self.state.meta[u].assigned_class = None
        self.state.rebuild_rows()
        self.state.assert_aligned(self.collection["feats"][_any_method(self.collection)].shape[0])
        self.state.coll_version += 1
        self.state.collection_dirty = True
        self.store.save_collection(self.collection)
        self.history.barrier()
        self.save()
        return len(new_records)

    def revert_refine(self, iuid: str) -> None:
        self._overlay_rle.pop(iuid, None)
        self.store.delete_refine(iuid)
        tok = self.history.begin(self.state, [iuid], [])
        self.state.meta[iuid].refined = False
        self.history.commit(self.state, tok, "revert_refine", f"revert {iuid[:6]}")
        self._after_mutation()

    # ---- classifier / similar ---------------------------------------------
    def train_classifier(self, spec, *, algo: str = "logreg", use_unassigned_negatives: bool = True,
                         knn_k: int = 5, knn_metric: str = "cosine", knn_weights: str = "distance") -> dict:
        """algo 'logreg'/'rf' -> factored open-set classifier (P(c vs not-c)*P(c vs others), needs >=2
        per class); algo 'knn' -> distance vote over k nearest assigned (+background as reject neighbours),
        works with >=1 per class. Both expose .classes/.proba so predict/apply are identical downstream."""
        spec = self._present_spec(spec)
        if not spec:
            return {"error": f"none of the selected features are present; available: {self.available_features()}"}
        if algo == "knn":
            clf, report = _clf.train_knn(self.collection, self.state, spec, k=int(knn_k), metric=knn_metric,
                                         weights=knn_weights, use_unassigned_negatives=use_unassigned_negatives)
        else:
            clf, report = _clf.train_factored(self.collection, self.state, spec, algo=algo,
                                              use_unassigned_negatives=use_unassigned_negatives)
        if report.get("skipped_classes"):
            report["skipped_names"] = [self.state.class_name(c) for c in report["skipped_classes"]]
        if clf is None:
            return report
        self._clf = clf
        self._clf_spec = spec
        return report

    @_timed
    def fused(self, spec) -> np.ndarray:
        """Fused feature matrix for the WHOLE collection, cached by (spec, coll_version). The features
        are a pure function of the collection, so predict/apply/train/cluster reuse one build per
        coll_version instead of rebuilding O(total instances) each call (the classifier-Apply hotspot)."""
        spec = _cl.normalize_spec(spec)
        key = (tuple(sorted(spec.items())), int(self.state.coll_version))
        hit = self._fused_cache.get(key)
        if hit is None:
            self._fused_cache.clear()                           # only the current coll_version matters
            hit = self._fused_cache[key] = _cl.fused_matrix(self.collection, spec)
        return hit

    @_timed
    def predict_and_threshold(self, thresh: float, only_class: str | None = None):
        iuids = self.state.unassigned_iuids()                   # only ever scores not-yet-classified instances
        if not iuids or getattr(self, "_clf", None) is None:
            return []
        X = self.fused(self._clf_spec)
        rows = [self.state.meta[u].row for u in iuids]
        proba = self._clf.proba(X[rows])
        return _clf.threshold_assign(iuids, proba, self._clf.classes, float(thresh), only_class=only_class)

    @_timed
    def apply_predictions(self, thresh: float, only_class: str | None = None, exclude=None):
        """Assign the thresholded predictions, then return (n_assigned, refreshed_preview) from a SINGLE
        prediction pass — the assigned iuids are dropped from the returned preview (they leave the
        unassigned pool), so the caller needn't re-predict from scratch."""
        preds = self.predict_and_threshold(thresh, only_class=only_class)
        exclude = set(exclude or [])                            # instances the user removed in the preview
        kept = [(u, c, conf) for u, c, conf in preds if u not in exclude]
        by_class: dict[str, list[str]] = {}
        scores = {}
        for u, cid, conf in kept:
            by_class.setdefault(cid, []).append(u); scores[u] = conf
        for cid, us in by_class.items():
            self.assign(us, self.state.class_name(cid), source="classifier", scores=scores)
        assigned = {u for u, _, _ in kept}                      # excluded instances stay in the preview
        remaining = sorted((t for t in preds if t[0] not in assigned), key=lambda t: -t[2])
        return len(kept), remaining

    @_timed
    def recommend_rejections(self, max_conf: float = 0.3):
        """Unassigned instances the trained classifier matches to NO curated class — max class probability
        < ``max_conf`` => background/reject candidates. Returned ascending by max prob (most clearly-not-a-class
        first), each tagged with its nearest class for context. With open-set negatives the classifier is
        trained to push background-like instances toward low class probabilities, so this surfaces the
        likely-garbage predictions the user should reject (the complement of predict_and_threshold's high
        confidence assign candidates). Empty when nothing is trained or no instance falls below the cutoff."""
        iuids = self.state.unassigned_iuids()
        if not iuids or getattr(self, "_clf", None) is None:
            return []
        X = self.fused(self._clf_spec)
        rows = [self.state.meta[u].row for u in iuids]
        proba = self._clf.proba(X[rows])
        classes = list(self._clf.classes)
        if not classes or not len(proba):
            return []
        out = []
        for u, p in zip(iuids, proba):
            if not len(p):
                continue
            j = int(np.argmax(p)); mx = float(p[j])
            if mx < float(max_conf):
                out.append((u, classes[j], mx))
        out.sort(key=lambda t: t[2])
        return out

    @_timed
    def recommend_interesting(self, n: int = 60, *, metric: str = "entropy"):
        """ACTIVE-LEARNING acquisition: the unassigned instances most INFORMATIVE to label next — where the
        trained classifier is most UNCERTAIN, so a human label there teaches the model the most.
        metric: 'entropy' (default; -Σ p·log p over the normalized class probs), 'margin' (small top1-top2),
        'least_conf' (low max prob). Returns [(iuid, predicted_class|None, uncertainty)] most-uncertain first.
        Fallback when no classifier is trained: the LOWEST-detection-score unassigned instances (the model is
        least sure it even found a real object there) — still the interesting tail to review."""
        iuids = self.state.unassigned_iuids()
        if not iuids:
            return []
        if getattr(self, "_clf", None) is None:                 # no classifier -> uncertain DETECTIONS
            recs = self.collection["records"]
            scored = [(u, None, 1.0 - float(recs[self.state.meta[u].row].get("score", 0.0))) for u in iuids]
            scored.sort(key=lambda t: -t[2])
            return scored[:int(n)]
        X = self.fused(self._clf_spec)
        rows = [self.state.meta[u].row for u in iuids]
        proba = self._clf.proba(X[rows])
        classes = list(self._clf.classes)
        if not classes or not len(proba):
            return []
        out = []
        for u, p in zip(iuids, proba):
            p = np.asarray(p, dtype=np.float64)
            if not len(p):
                continue
            j = int(np.argmax(p))
            s = float(p.sum())
            q = p / s if s > 1e-12 else np.full(len(p), 1.0 / len(p))   # normalize to a distribution
            if metric == "margin":
                top = np.sort(q)[::-1]
                unc = 1.0 - float(top[0] - (top[1] if len(top) > 1 else 0.0))
            elif metric == "least_conf":
                unc = 1.0 - float(q.max())
            else:                                               # entropy (normalized to [0,1] by log C)
                ent = -float((q * np.log(q + 1e-12)).sum())
                unc = ent / float(np.log(len(q))) if len(q) > 1 else 0.0
            out.append((u, classes[j], unc))
        out.sort(key=lambda t: -t[2])
        return out[:int(n)]

    def find_similar(self, iuid: str, *, k: int = 20, spec=None):
        return _sim.find_similar(self.collection, self.state, iuid, k=k,
                                 spec=spec or (self._cluster["spec"] if self._cluster else {"decoder": 1.0}))

    # ---- find-partition-by-uploaded-image (visual NN over a stored feature) ----
    def _iuid_pid_map(self) -> dict[str, str]:
        """{pool iuid -> FINCH pid at current level}, cached by (cluster id, level)."""
        key = (id(self._cluster), self._cluster["level"])
        c = getattr(self, "_pidmap_cache", None)
        if c is not None and c[0] == key:
            return c[1]
        labels, pool = self._pool_labels(), self._cluster["pool"]
        m = {pool[i]: str(int(labels[i])) for i in range(len(pool))}
        self._pidmap_cache = (key, m)
        return m

    def partition_of(self, iuid: str) -> str | None:
        """Which partition currently holds this iuid: class:<cid> if assigned, the FINCH pid if in the
        unassigned pool, else None (rejected / merged-away / not clustered)."""
        m = self.state.meta.get(iuid)
        if m is None:
            return None
        if m.assigned_class:
            return f"class:{m.assigned_class}"
        if m.is_background or m.merged_into is not None or not self._cluster:
            return None
        return self._iuid_pid_map().get(iuid)

    def match_features(self, qvec, *, feature: str = "roialign", k: int = 12,
                       dedup_partition: bool = True) -> dict:
        """Cosine-NN of a query feature vector against ALL instances' `feature` (brute force — one
        matmul, ms at 25k; swap in faiss/hnswlib only at ~1M). With dedup_partition (default), each
        PARTITION appears once — the best-scoring instance per partition, skipping rejected/merged-away
        ones (pid None). Returns the mixed best-first `matches` AND, split out, `matches_class` (assigned
        class:<cid> partitions) + `matches_pool` (unassigned FINCH partitions) each up to k — so the
        reference search surfaces BOTH relevant CLASSES and relevant UNANNOTATED partitions, not only
        classes (which otherwise crowd out the pool once many instances are assigned)."""
        feats = (self.collection or {}).get("feats", {})
        if feature not in feats:
            return {"error": f"feature '{feature}' not in collection; available: {self.available_features()}"}
        X = feats[feature]
        q = np.asarray(qvec, np.float32).ravel()
        if q.shape[0] != X.shape[1]:
            return {"error": f"query dim {q.shape[0]} != index dim {X.shape[1]}"}
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        qn = q / (np.linalg.norm(q) + 1e-9)
        sims = Xn @ qn
        order = self.state.order
        out, cls_out, pool_out, seen = [], [], [], set()
        for i in np.argsort(-sims):                       # all instances, best-first
            pid = self.partition_of(order[i])
            if dedup_partition:
                if pid is None or pid in seen:
                    continue
                seen.add(pid)
            row = {"iuid": order[i], "score": round(float(sims[i]), 4), "pid": pid}
            if len(out) < int(k):
                out.append(row)
            if pid and str(pid).startswith("class:"):
                if len(cls_out) < int(k):
                    cls_out.append({**row, "cls": self.state.class_name(str(pid).split(":", 1)[1])})
            elif pid is not None and len(pool_out) < int(k):
                pool_out.append(row)
            if len(out) >= int(k) and len(cls_out) >= int(k) and len(pool_out) >= int(k):
                break
        return {"matches": out, "matches_class": cls_out, "matches_pool": pool_out}

    def match_image(self, img: np.ndarray, *, feature: str = "roialign", k: int = 12) -> dict:
        """Run the seg model on an uploaded RGB image (reuses collect_batch), take the top-scoring
        detected instance's `feature`, and NN it against the collection — so the match lives in the
        SAME space the instances were extracted/clustered in (roialign = pixel-decoder mask_features)."""
        feats = (self.collection or {}).get("feats", {})
        if feature not in feats:
            return {"error": f"feature '{feature}' not extracted; available: {self.available_features()}"}
        import os
        import tempfile
        import cv2
        model, cfg, d2_cfg = self._ensure_model()
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "query.png")
        cv2.imwrite(fp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        feat_cfg = self.state.config.get("features_runtime", _default_feat_cfg(self.state.config))
        batch = _co.collect_batch(model, cfg, d2_cfg, [fp], score_thresh=0.1, feature_cfg=feat_cfg)
        if not batch["records"] or feature not in batch.get("feats", {}):
            return {"error": "no instance detected in the uploaded image (try a tighter crop of one structure)"}
        scores = [r["score"] for r in batch["records"]]
        qi = int(np.argmax(scores))
        res = self.match_features(batch["feats"][feature][qi], feature=feature, k=k)
        res["query_score"] = round(float(scores[qi]), 3)
        res["n_detected"] = len(scores)
        return res

    # ---- reference exemplar bank (suggest a fine class for unassigned instances) ----
    def _ref_extractor(self):
        ext = getattr(self, "_raddino_ext", None)
        if ext is None:
            import torch
            from ._bootstrap import get_P
            ext = self._raddino_ext = get_P().RadDinoExtractor("cuda" if torch.cuda.is_available() else "cpu")
        return ext

    def _instance_ref_embeddings(self, iuids: list[str]) -> np.ndarray:
        """MASK-gated MAX-pooled RAD-DINO embedding per instance: max over the patch tokens inside the mask
        (max > mean — probed; and the mask gate removes anatomy-background dilution). Falls back to global
        max if the mask is sub-patch. Cached by (iuid, mask_token); RAD-DINO runs once per image."""
        import torch
        import torch.nn.functional as F
        cache = self.__dict__.setdefault("_ref_inst_cache", {})
        need = [u for u in iuids if (u, self.mask_token(u)) not in cache]
        if need:
            ext = self._ref_extractor()
            from collections import defaultdict
            by_img: dict = defaultdict(list)
            for u in need:
                by_img[self.state.meta[u].image_id].append(u)
            for iid, us in by_img.items():
                grid = ext.grid(self._rgb_by_image(iid)); C, g, _ = grid.shape
                gf = grid.reshape(C, -1)                          # (C, P)
                masks = torch.stack([torch.from_numpy(self._mask(u)).float() for u in us])
                soft = F.interpolate(masks.unsqueeze(1), size=(g, g), mode="bilinear",
                                     align_corners=False).squeeze(1).reshape(len(us), -1).to(grid.device)
                for j, u in enumerate(us):
                    gate = soft[j] > 0.1
                    feats = gf[:, gate] if bool(gate.any()) else gf   # sub-patch mask -> global max
                    cache[(u, self.mask_token(u))] = feats.amax(1).detach().cpu().numpy().astype(np.float32)
        return np.stack([cache[(u, self.mask_token(u))] for u in iuids]).astype(np.float32)

    def load_reference_bank(self, coco_path: str, *, rebuild: bool = False) -> dict:
        """Build (or load cached) the RAD-DINO reference bank from a labeled COCO of foreign-object crops and
        BOOTSTRAP the taxonomy with its class names. References are embedded by bbox-crop mean-pool."""
        import json

        import cv2
        from . import reference_bank as _rb
        cache = self.store.dir / "reference_bank"
        bank = None if rebuild else _rb.ReferenceBank.load(cache)
        if bank is not None and getattr(bank, "pool", "mean") != "max":
            bank = None                                          # stale mean-pool cache -> rebuild with max-pool
        if bank is None:
            d = json.load(open(coco_path))
            root = Path(coco_path).parent
            id2n = {c["id"]: c["name"] for c in d["categories"]}
            imgs = {im["id"]: im for im in d["images"]}
            ext = self._ref_extractor()
            embs, labels, exemplars = [], [], []
            for a in d["annotations"]:
                im = imgs.get(a["image_id"])
                if not im:
                    continue
                img = cv2.imread(str(root / im["file_name"]))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x, y, w, h = [int(v) for v in a["bbox"]]
                crop = img[max(0, y):y + h, max(0, x):x + w]
                if crop.size == 0 or min(crop.shape[:2]) < 4:
                    continue
                v = ext.grid(crop).amax(dim=(1, 2)).detach().cpu().numpy().astype(np.float32)  # MAX-pool (> mean: probed)
                name = id2n[a["category_id"]]
                embs.append(v); labels.append(name)
                exemplars.append({"file_name": im["file_name"], "bbox": [x, y, w, h], "cls": name})
            if not embs:
                return {"error": f"no usable reference crops in {coco_path}"}
            bank = _rb.ReferenceBank(np.stack(embs), labels, {n: n for n in id2n.values()}, exemplars)
            bank.save(cache)
        self._ref_bank = bank
        self._ref_coco_root = str(Path(coco_path).parent)
        added = 0
        for name in bank.classes():
            if self.state.class_id_by_name(name) is None:
                self.state.add_class(name); added += 1
        if added:
            self.save()
        return {"classes": len(bank.classes()), "exemplars": bank.n, "added_classes": added}

    def reference_suggest(self, iuids: list[str], *, topk: int = 5, knn: int = 8, use_csls: bool = True) -> dict:
        """Per instance, the top-k reference CLASSES it most resembles (CSLS-de-hubbed kNN class vote over the
        bank). A weak prior to CONFIRM, not auto-apply — surfaced for one-click accept."""
        from . import reference_bank as _rb
        bank = getattr(self, "_ref_bank", None)
        if bank is None or bank.n == 0:
            return {"error": "load a reference bank first (Reference tab)"}
        iuids = [u for u in iuids if u in self.state.meta]
        if not iuids:
            return {"items": []}
        Q = self._instance_ref_embeddings(iuids)
        ranked = _rb.suggest(Q, bank.emb, bank.labels, topk=int(topk), knn=int(knn), use_csls=use_csls)
        return {"items": [{"iuid": u, "suggestions": [{"cls": c, "score": round(float(s), 3)} for c, s in r]}
                          for u, r in zip(iuids, ranked)]}

    def add_to_reference_bank(self, iuids: list[str]) -> dict:
        """Self-improving bank: add CONFIRMED in-domain instances (their mask-pooled embedding + assigned
        class) to the bank, so reference suggestions sharpen toward the real CXR appearance over the session."""
        bank = getattr(self, "_ref_bank", None)
        if bank is None:
            return {"error": "load a reference bank first"}
        use = [u for u in iuids if self.state.meta.get(u) and self.state.meta[u].assigned_class
               and not self.state.meta[u].is_background]
        if not use:
            return {"added": 0, "n": bank.n}
        Q = self._instance_ref_embeddings(use)
        labels = [self.state.class_name(self.state.meta[u].assigned_class) for u in use]
        ex = [{"iuid": u, "cls": labels[i],
               "file_name": self.collection["records"][self.state.meta[u].row].get("abs_path", "")}
              for i, u in enumerate(use)]
        bank.add(Q, labels, ex)
        bank.save(self.store.dir / "reference_bank")
        return {"added": len(use), "n": bank.n}

    def reference_classes(self) -> list[dict]:
        bank = getattr(self, "_ref_bank", None)
        if bank is None:
            return []
        from collections import Counter
        cnt = Counter(bank.class_names.get(l, str(l)) for l in bank.labels)
        return [{"cls": c, "n": n} for c, n in sorted(cnt.items())]

    def reference_exemplar(self, file_name: str, bbox=None, *, max_side: int = 200):
        """Crop of a bank exemplar (for the visual panel). file_name is relative to the loaded ref COCO."""
        import cv2
        root = getattr(self, "_ref_coco_root", None)
        if not root:
            return None
        img = cv2.imread(str(Path(root) / file_name))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if bbox:
            x, y, w, h = [int(v) for v in bbox]
            img = img[max(0, y):y + h, max(0, x):x + w]
        if img.size and max(img.shape[:2]) > max_side:
            s = max_side / max(img.shape[:2])
            img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
        return img

    # ---- merge recommender (learns from past in-image merges) --------------
    def _merge_pos_groups(self) -> list[list[str]]:
        """Positive merge groups: logged merge events (survive undo/unmerge) + current merge_members,
        deduplicated by member-set (a still-active merge appears in BOTH sources)."""
        seen, groups = set(), []
        def _add(g):
            g = [u for u in g if u in self.state.meta]
            key = frozenset(g)
            if len(g) >= 2 and key not in seen:
                seen.add(key); groups.append(g)
        for ev in self.store.read_merge_events():
            if ev.get("kind") == "merge":
                _add(ev.get("iuids", []))
        for u, m in self.state.meta.items():
            if m.merge_members:
                _add([u] + list(m.merge_members))
        return groups

    def _merge_rejected_groups(self) -> list[list[str]]:
        return [[u for u in ev.get("iuids", []) if u in self.state.meta]
                for ev in self.store.read_merge_events() if ev.get("kind") == "reject"]

    def train_merge_recommender(self, spec, *, algo: str = "logreg") -> dict:
        from . import merge_rec as _mr
        spec = self._present_spec(spec)
        pos = self._merge_pos_groups()
        if not pos:
            return {"error": "no merges recorded yet — merge some instances in the In-image tab first"}
        X, y, _, rep = _mr.build_pair_xy(self.collection, self.state, pos, self._merge_rejected_groups(), spec)
        if rep["n_pos"] == 0 or rep["n_neg"] == 0:
            return {"error": f"not enough training pairs (positives={rep['n_pos']}, negatives={rep['n_neg']})"}
        self._merge_clf = _mr.train(X, y, algo=algo)
        self._merge_spec = spec
        rep.update(_mr.pr_youden(X, y, algo=algo))
        rep["n_merge_events"] = len(pos)
        return rep

    def recommend_merges(self, thresh: float, *, max_groups: int = 20) -> list[dict]:
        from . import merge_rec as _mr
        if getattr(self, "_merge_clf", None) is None:
            return []
        return _mr.candidate_groups(self.collection, self.state, self._merge_clf, self._merge_spec,
                                    float(thresh), max_groups=max_groups)

    def recommend_merges_for_image(self, image_id: int, thresh: float, *, max_groups: int = 20) -> list[dict]:
        """In-context suggestions: the trained recommender scored over ONE image's current instances."""
        from . import merge_rec as _mr
        if getattr(self, "_merge_clf", None) is None:
            return []
        return _mr.candidate_groups(self.collection, self.state, self._merge_clf, self._merge_spec,
                                    float(thresh), max_groups=max_groups, only_image=int(image_id))

    def accept_merge(self, iuids: list[str], mode: str = "union") -> None:
        self.merge_instances(list(iuids), mode=mode, source="recommended")   # positive, tagged from the recommender

    def reject_merge(self, iuids: list[str]) -> None:
        iuids = [u for u in iuids if u in self.state.meta]
        if len(iuids) >= 2:
            self.store.append_merge_event({"kind": "reject", "iuids": list(iuids),
                                           "image_id": int(self.state.meta[iuids[0]].image_id),
                                           "source": "recommended", "ts": time.time()})   # the recommender is its only caller

    # ---- export / import ---------------------------------------------------
    def export_coco(self, out_path=None, **kw):
        out_path = Path(out_path) if out_path else (self.store.export_dir / "curated.json")
        return _ex.export(self.collection, self.state, out_path, rle_override=self._overlay_rle, **kw)

    def import_coco(self, path):
        tok_iuids = list(self.state.meta.keys())
        tok = self.history.begin(self.state, tok_iuids, [])
        rep = _ex.import_coco(path, self.collection, self.state)
        for cid in self.state.taxonomy:
            tok["class_ids"].append(cid)
        self.history.commit(self.state, tok, "import", f"import {rep['matched']}")
        self.save()
        return rep

    # ---- undo / redo / stats ----------------------------------------------
    def undo(self):
        op = self.history.undo(self.state); self.save(); return op

    def redo(self):
        op = self.history.redo(self.state); self.save(); return op

    def embed2d(self, *, method: str = "pca", color_by: str = "cluster"):
        from ._bootstrap import get_P
        P = get_P()
        spec = self._cluster["spec"] if self._cluster else {"decoder": 1.0}
        X = self.fused(spec)
        xy = P.embed2d(X, method)
        return xy, self._label_for_order(color_by), list(self.state.order)

    def _label_for_order(self, color_by: str) -> np.ndarray:
        """Per-order integer label for the Map: pool->FINCH label, assigned->1000+classidx,
        background->-1 (color_by='class' colors only by assigned class)."""
        order = self.state.order
        cids = list(self.state.taxonomy)
        if color_by == "class" or not self._cluster:
            return np.array([(1000 + cids.index(self.state.meta[u].assigned_class))
                             if self.state.meta[u].assigned_class in cids else
                             (-1 if self.state.meta[u].is_background else 0) for u in order])
        labels, pool = self._pool_labels(), self._cluster["pool"]
        pool_lab = {pool[i]: int(labels[i]) for i in range(len(pool))}
        out = []
        for u in order:
            m = self.state.meta[u]
            if u in pool_lab and self._is_pool(u):
                out.append(pool_lab[u])
            elif m.is_background:
                out.append(-1)
            elif m.assigned_class in cids:
                out.append(1000 + cids.index(m.assigned_class))
            else:
                out.append(-2)
        return np.array(out)

    def embed_thumbnails(self, *, method: str = "pca", color_by: str = "cluster",
                         max_pts: int = 250, thumb: int = 64):
        """embed2d + a small base64 JPG thumbnail per point (for the Map hover JS) + source names.
        Thumbnails computed for up to max_pts points (others empty) — each thumb is a crop (disk read,
        now cached); the cap keeps 'Compute map' to seconds instead of minutes on big projects."""
        import base64
        import cv2
        xy, labels, order = self.embed2d(method=method, color_by=color_by)
        n = len(order)
        idxs = range(n) if n <= max_pts else set(np.linspace(0, n - 1, max_pts).astype(int).tolist())
        thumbs, names = [""] * n, [""] * n
        for i in range(n):
            u = order[i]
            names[i] = Path(self.collection["records"][self.state.meta[u].row].get("file_name", "")).name
            if i in idxs:
                cr = cv2.resize(self.crop(u, mask_overlay=True), (thumb, thumb))
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(cr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    thumbs[i] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        return xy, labels, order, thumbs, names

    def stats(self) -> dict:
        n_assigned = sum(1 for m in self.state.meta.values() if m.assigned_class and not m.is_background)
        n_bg = sum(1 for m in self.state.meta.values() if m.is_background)
        u, r = self.history.depths
        return {"n_images": len(set(m.image_id for m in self.state.meta.values())),
                "n_instances": len(self.state.order), "n_assigned": n_assigned,
                "n_background": n_bg, "n_unassigned": len(self.state.order) - n_assigned - n_bg,
                "n_classes": len(self.state.taxonomy), "dirty": self.state.collection_dirty,
                "undo": u, "redo": r, "coll_version": self.state.coll_version}

    def statistics(self) -> dict:
        """Comprehensive label-generation stats (O(N), computed on demand): curation progress, per-class
        counts/coverage/score/source, instances-per-image + score + mask-area histograms, assignment
        sources, partition/pool summary, and a class co-occurrence matrix. All JSON-serializable."""
        from collections import Counter, defaultdict
        recs = self.collection["records"] if self.collection else []
        meta = self.state.meta
        n_total = len(self.state.order)
        n_merged = sum(1 for m in meta.values() if m.merged_into is not None)

        per_class = defaultdict(lambda: {"n": 0, "images": set(), "scores": [], "sources": Counter()})
        img_inst, img_assigned = Counter(), Counter()       # live instances / assigned per image
        img_classes = defaultdict(set)                       # image_id -> {assigned class names}
        scores, areas = [], []
        sources = Counter()
        n_assigned = n_bg = 0
        for u, m in meta.items():
            if m.merged_into is not None:                    # merge children collapse into their rep
                continue
            r = recs[m.row] if m.row < len(recs) else {}
            iid = int(m.image_id)
            img_inst[iid] += 1
            scores.append(float(r.get("score", 0.0)))
            areas.append(float(r.get("mask_area_frac", 0.0)))
            if m.is_background:
                n_bg += 1
            elif m.assigned_class:
                n_assigned += 1
                c = self.state.class_name(m.assigned_class)
                d = per_class[c]
                d["n"] += 1; d["images"].add(iid); d["scores"].append(float(r.get("score", 0.0)))
                d["sources"][m.assign_source or "?"] += 1
                sources[m.assign_source or "?"] += 1
                img_assigned[iid] += 1
                img_classes[iid].add(c)
        n_live = n_total - n_merged
        n_unassigned = n_live - n_assigned - n_bg

        def _hist(vals, bins, lo=None, hi=None):
            if not vals:
                return {"edges": [], "counts": []}
            a = np.asarray(vals, float)
            counts, edges = np.histogram(a, bins=bins, range=(lo, hi) if lo is not None else None)
            return {"edges": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts]}

        classes = sorted(({"class": c, "n": d["n"], "images": len(d["images"]),
                           "mean_score": round(float(np.mean(d["scores"])), 3) if d["scores"] else None,
                           "sources": dict(d["sources"])} for c, d in per_class.items()),
                         key=lambda x: -x["n"])
        # class co-occurrence among the top-N assigned classes (bounded matrix)
        top = [c["class"] for c in classes[:12]]
        idx = {c: i for i, c in enumerate(top)}
        cooc = [[0] * len(top) for _ in top]
        for cls_set in img_classes.values():
            present = [idx[c] for c in cls_set if c in idx]
            for i in present:
                for j in present:
                    cooc[i][j] += 1
        ipi = sorted(Counter(img_inst.values()).items())     # (#instances on an image, #such images)

        return {
            "overview": {
                "instances_total": n_total, "instances_live": n_live, "merged_children": n_merged,
                "assigned": n_assigned, "unassigned": n_unassigned, "rejected": n_bg,
                "classes": len(self.state.taxonomy), "images": len(img_inst),
                "images_with_assignment": len(img_assigned),
                "pct_curated": round(100.0 * (n_assigned + n_bg) / n_live, 1) if n_live else 0.0,
            },
            "classes": classes,
            "sources": dict(sources),
            "instances_per_image": [[int(k), int(v)] for k, v in ipi],
            "score_hist": _hist(scores, 20, 0.0, 1.0),
            "area_hist": _hist(areas, 20),
            "cooccurrence": {"classes": top, "matrix": cooc},
            "partitions": ({"clustered": True, "level": self._cluster["level"],
                            "n_partitions": len(set(int(x) for x in self._pool_labels())),
                            "unassigned_pool": len(self._pool_iuids())}
                           if self._cluster else {"clustered": False}),
        }

    def _after_mutation(self):
        self._commits += 1
        self.save(snapshot=(self._commits % _AUTOSNAP_EVERY == 0))


def _any_method(collection: dict) -> str:
    for k in collection["feats"]:
        if not k.startswith("_"):
            return k
    raise ValueError("collection has no feature methods")


def _default_level(counts: list[int]) -> int:
    """Pick a mid-granularity level (closest to ~sqrt(N-ish) clusters)."""
    if len(counts) <= 1:
        return 0
    target = max(counts) ** 0.5
    return int(np.argmin([abs(c - target) for c in counts]))


def _default_feat_cfg(config: dict) -> dict:
    f = config.get("features", {})
    mf = set(f.get("model_features", ["decoder", "maskpool", "roialign"]))
    return {"with_features": bool(mf & {"decoder", "maskpool", "roialign", "backbone"}),
            "with_shape": bool(f.get("handcrafted", {}).get("shape", True)),
            "with_backbone": "backbone" in mf,
            "backbone_level": f.get("backbone_level", "p16"),
            "shapecoord": bool(f.get("handcrafted", {}).get("shape_coords_extra", True)),
            "raddino": bool(f.get("raddino", False)),
            "nms_iou": float(config.get("model", {}).get("nms_iou", 0.8))}
