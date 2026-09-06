"""CuratorEngine — the UI-agnostic facade tying the curator modules together.

Holds the authoritative mutable project (collection + feature matrices + masks + the
overlay state). The Gradio app binds to one server-side instance. All renders return
numpy RGB images; all mutations go through the History for undo/redo + autosave.
"""
from __future__ import annotations

import atexit
import colorsys
import functools
import os
import sys
import threading
import time
import weakref
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
_SAVE_DEBOUNCE = 0.5         # s — a burst of mutations within this window coalesces into ONE state write

# Write-behind state persistence. Every mutation used to re-serialize the WHOLE state.json synchronously in
# the request (O(total instances): asdict×N + json + multi-MB write -> ~0.3s+ at 25k, so assign/merge/etc.
# blocked the UI and got worse as the project grew). Now the hot path (`_after_mutation`) just marks the
# engine dirty and a per-engine background thread coalesces the write off the request thread, so interaction
# latency is decoupled from N. Durable boundaries (ingest finalize, undo/redo, class-rule save, project
# open) still call `save()` synchronously. Crash window = at most _SAVE_DEBOUNCE of un-flushed mutations.
_LIVE_ENGINES: "weakref.WeakSet" = weakref.WeakSet()


@atexit.register
def _flush_live_engines() -> None:
    for e in list(_LIVE_ENGINES):
        try:
            e.flush()
        except Exception:
            pass


def _state_saver_loop(engine_ref: "weakref.ReferenceType") -> None:
    """Per-engine background saver. Holds only a weakref to the engine, so a dropped engine (e.g. a test's)
    is GC'd normally and this thread exits within the wait timeout instead of pinning it alive. The whole
    body is guarded and interpreter-shutdown-aware: a best-effort background thread must NEVER surface an
    exception (e.g. during teardown, when module globals are being torn down — which pytest's thread-exception
    hook would otherwise attribute to a random test)."""
    while True:
        try:
            if sys.is_finalizing():                    # interpreter shutting down -> stop touching globals
                return
            e = engine_ref()
            if e is None:
                return
            dirty, stop = e._save_dirty, e._save_stop
            del e                                      # don't pin the engine while blocked on the event
            woke = dirty.wait(timeout=30.0)
            if sys.is_finalizing() or stop.is_set():
                return
            e = engine_ref()
            if e is None:
                return
            if woke:
                stop.wait(_SAVE_DEBOUNCE)              # coalesce a burst into a single write
                try:
                    with e._mutate_lock:               # read a CONSISTENT snapshot — no torn pickle mid-mutation
                        e._write_state()
                except Exception:
                    pass                               # transient write error -> keep the saver alive
            del e
        except Exception:
            return                                     # deref/wait failure (e.g. at teardown) -> exit quietly

# Bounded LRU of decoded RGB source images keyed by abs path. Source images never change, so no
# invalidation — just eviction. WITHOUT this, every crop re-imread()s the full-res JPEG, and a grid
# render does up to _GRID_CAP disk reads → the app stalls (see plan v5.6). Returned arrays are shared
# (read-only); every mutating caller (crop/_crop_mask/image_overlay) copies before drawing.
# At 1M instances a 60-cell grid commonly spans many source images; a too-small cache thrashes full-res
# JPEG decodes. Default 96 (~200-300 MB of decoded RGB); lower CURATOR_IMG_CACHE on small-RAM hosts.
_IMG_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_IMG_CACHE_MAX = int(os.environ.get("CURATOR_IMG_CACHE", "96"))

# Bounded LRU of finished crop thumbnails keyed by (iuid, mask_token, params). The web grids re-request
# crop() for every visible instance on each reload; caching makes a post-merge reload recompute only the
# crops whose mask actually changed. Keyed by mask_token, so it self-invalidates on merge/refine/split.
_CROP_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_CROP_CACHE_MAX = 128

# Reference-search cosine NN: exact matmul below this many instances (sub-50ms + exact), faiss HNSW above it
# (~O(log N), approximate — the nearest-m, which is what "find similar" wants at scale).
_ANN_MIN = int(os.environ.get("CURATOR_ANN_MIN", "50000"))


def clear_image_caches() -> None:
    """Drop both process-wide LRUs. Called when the active project changes.

    Neither cache can serve a WRONG image across projects (`_IMG_CACHE` is keyed by absolute path and
    `_CROP_CACHE` by uuid4 `iuid`), so this is about capacity, not correctness: the two caches are
    shared by every engine in the process, and leaving a closed project's entries resident would evict
    the incoming project's working set and make its first screens slow.
    """
    _IMG_CACHE.clear()
    _CROP_CACHE.clear()


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


_PSUG_REF_CAP = 40000        # labeled+reject reference vectors kept for the per-partition 1-NN suggestion
_PSUG_QUERY_CAP = 256        # partition members sampled for the suggestion vote (a representative sample)
_PMP_CAP = 20000             # partition members predicted for per-crop markers + subset filter (head slice; truncation surfaced)
_PROJ_CAP = 60000            # instances embedded into the latent-space Map (one DR fit; truncation surfaced)
_WL_QUERY_CAP = 80000        # uncategorized instances scored for the image workload ranking (ONE batched 1-NN pass; truncation surfaced)
_WL_BORDER_FRAC = 0.75       # nearest-dist in [frac*gate, gate] is "borderline" -> 1/2 a manual decision (vs AUTO below, NONE above)


def _nn_build(Xn: np.ndarray):
    """A searchable NN handle over L2-normalized rows `Xn` that returns INDICES (so the caller can map a
    nearest neighbour back to its label). faiss when available (HNSW for big sets), else a brute matmul."""
    Xn = np.ascontiguousarray(np.asarray(Xn, np.float32))
    try:
        import faiss
        d = Xn.shape[1]
        if len(Xn) > 16000:
            ix = faiss.IndexHNSWFlat(d, 32); ix.hnsw.efConstruction = 64; ix.hnsw.efSearch = 64
        else:
            ix = faiss.IndexFlatL2(d)
        ix.add(Xn)
        return ("faiss", ix)
    except Exception:
        return ("np", Xn)


def _nn_search(handle, Qn: np.ndarray, k: int):
    """(cosine_dist[M,k], idx[M,k]) for L2-normalized queries `Qn`. idx is -1 where fewer than k refs exist."""
    kind, ix = handle
    Qn = np.ascontiguousarray(np.asarray(Qn, np.float32))
    if kind == "faiss":
        import faiss
        l2sq, I = ix.search(Qn, int(k))
        return np.maximum(l2sq, 0.0) * 0.5, I            # cosine_dist = L2^2/2 on unit vectors
    sims = Qn @ ix.T                                     # brute cosine
    I = np.argsort(-sims, axis=1)[:, :int(k)]
    return 1.0 - np.take_along_axis(sims, I, axis=1), I


def _downscale(img: np.ndarray, max_side: int = 220) -> np.ndarray:
    """Shrink to <= max_side on the longest side — keeps gallery payloads small (browser RAM)."""
    import cv2
    h, w = img.shape[:2]
    s = max_side / max(h, w) if max(h, w) > max_side else 1.0
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return img


def _mutating(fn):
    """Serialize this engine mutation against other mutations AND the background saver via self._mutate_lock
    (reentrant). The curator is ONE shared in-process engine reachable by CONCURRENT requests (FastAPI sync
    endpoints run in a threadpool), so unguarded mutations race on state.meta / collection / the caches +
    undo_stack. Every PUBLIC state-mutating method MUST wear this."""
    @functools.wraps(fn)
    def _w(self, *a, **k):
        with self._mutate_lock:
            return fn(self, *a, **k)
    return _w


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
        self._proba_cache: dict | None = None           # classifier proba over the unassigned pool (predict/apply hotspot)
        self._clf_version = 0                           # bumped on each (re)train -> invalidates _proba_cache
        self._scope_bids: set[str] | None = None        # view SCOPE: restrict pool/images to these batch_ids
        self._scope_id: str | None = None               # the selected ingest_id (None = all)
        self._scope_token = 0                            # bumped on set_scope/set_source_filter -> busts index/cluster/proj
        self._source_filter: set[str] | None = None      # view FACET: show only these proposal SOURCES (None = all); composes with scope
        self._granularity_filter: set[str] | None = None  # view FACET: instance | sample (None = all)
        self._modality_filter: set[str] | None = None     # view FACET: image | text | video (None = all)
        self._bsrc_cache: dict | None = None             # batch_id -> source(model) map, from the ingest registry
        self._commits = 0
        self._mutation_serial = 0                        # +1 on every state mutation; the live-index validity stamp
        self._index = None                               # incrementally-maintained membership index (see _get_index)
        self._mutate_lock = threading.RLock()            # serialize state MUTATIONS (concurrent requests + saver);
        #                                                  reentrant so nested mutations re-enter on the same thread
        self._save_io_lock = threading.Lock()            # serialize disk writes (background saver vs sync save)
        self._save_dirty = threading.Event()             # set by _after_mutation; consumed by the saver thread
        self._save_stop = threading.Event()
        self._saver = threading.Thread(target=_state_saver_loop, args=(weakref.ref(self),),
                                       name="curator-state-saver", daemon=True)
        self._saver.start()
        _LIVE_ENGINES.add(self)
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
        self._scope_bids = None
        self._source_filter = None; self._bsrc_cache = None
        self._granularity_filter = self._modality_filter = None
        self._scope_id = None
        self._index = None                              # live index belongs to the previous project state
        if self.store.list_collection_shards():        # recover an interrupted incremental ingest
            try:
                n = self._merge_pending_shards()
                if n:
                    print(f"[curator] recovered {n} instances from an interrupted ingest", file=sys.stderr)
            except Exception as e:                     # recovery must never block project open
                print(f"[curator] shard recovery skipped ({e})", file=sys.stderr)

    def init_project(self, config: dict) -> None:
        self.state = CuratorState(project_dir=str(self.store.dir), config=dict(config))
        self.collection = None
        self.store.ensure()
        self.save()

    def _write_state(self) -> None:
        """The actual O(N) state+manifest write. Serialized by _save_io_lock so the background saver and a
        synchronous save() never interleave disk writes. Clears the dirty flag FIRST so a mutation arriving
        during the write re-dirties and is flushed on the next pass (never silently dropped)."""
        with self._save_io_lock:
            self._save_dirty.clear()
            self.store.save_state(self.state)
            man = self.store.load_manifest()
            man.update({"coll_version": self.state.coll_version,
                        "n_instances": len(self.state.order)})
            self.store.save_manifest(man)

    def save(self, *, snapshot: bool = False) -> None:
        """Synchronous full persist (flushes any pending write-behind state). Called at durable boundaries
        (ingest finalize, undo/redo, class-rule save, project init). The hot interactive path goes through
        `_after_mutation` (write-behind) instead, so it does NOT block on this."""
        self._write_state()
        if snapshot:
            self.store.snapshot()

    def flush(self) -> None:
        """Force a synchronous write iff there is un-persisted state (used by atexit + project close)."""
        if self._save_dirty.is_set():
            self._write_state()

    def close(self) -> None:
        """Stop the background saver and flush. The server runs one engine for its lifetime; tests/short-lived
        engines can call this for a deterministic final write (atexit also flushes live engines)."""
        self._save_stop.set()
        self._save_dirty.set()                           # wake the saver so it observes _save_stop and exits
        self._write_state()

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
            from .core.class_head import class_head_fg_count, collapse_checkpoint
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
        repo_root = self._qseg_root()
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

    def _qseg_root(self) -> Path:
        """The qseg checkout the retrain loop runs FROM.

        `qseg-train` needs qseg's own tree: its Hydra configs (the process cwd) and the MaskDINO
        submodule (PYTHONPATH). Before the extraction this was `parents[2]` of this file, which
        happened to be the qseg root when the curator lived at qseg/tools/curator/ — after it, that
        path is whatever directory Chevron was cloned into. It comes from the configured checkout now,
        so Chevron never assumes qseg is its parent.
        """
        from ._bootstrap import qseg_root
        root = qseg_root()
        if root is None or not root.is_dir():
            from ._bootstrap import BackendUnavailable
            raise BackendUnavailable(
                "the retrain loop runs qseg-train from a qseg checkout, but none is configured — "
                "set CHEVRON_QSEG_ROOT (or call chevron._bootstrap.set_qseg_root(...)).")
        return root

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
        repo_root = self._qseg_root()
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

    def _fold_batch(self, batch: dict) -> None:
        """Concat an inference batch into the live collection + create overlay meta for its instances,
        rebuild the order/row alignment, bump coll_version. The in-RAM/in-state half of an ingest (the
        on-disk half is the append-only shards)."""
        from .state import InstanceMeta
        new_records = batch["records"]
        self.collection = _co.concat_collections(self.collection, batch)
        self.state.order = [r["iuid"] for r in self.collection["records"]]
        ck = (self.state.config.get("model") or {}).get("ckpt", "")
        gran, modal = self.state.mode(), self.state.modality()
        for r in new_records:
            self.state.meta[r["iuid"]] = InstanceMeta(
                iuid=r["iuid"], batch_id=r["batch_id"], row=r["row"], image_id=int(r["image_id"]),
                granularity=gran, modality=modal,
                provenance={"file": r.get("abs_path", ""), "src_score": float(r["score"]), "ckpt": ck})
        self.state.rebuild_rows()
        self.state.assert_aligned(self.collection["feats"][_any_method(self.collection)].shape[0])
        self.state.coll_version += 1
        self.state.collection_dirty = True

    # ---- proposal backends (model-free proposers; qseg is one of several) ------------------------
    @staticmethod
    def _align_batch_feats(batch: dict, master: dict | None) -> dict:
        """Make `batch`'s feature methods match `master`'s so they can be concatenated.

        concat_collections refuses a method mismatch, and rightly — the feature space must be fixed
        per project. But a model-free proposer only produces geometry (shapecoord/coords) while a
        qseg-seeded project also has decoder/maskpool/roialign. Missing methods are ZERO-filled (the
        same trick import_proposals_coco uses: finite, so the global NaN check never trips, and the
        cross-source space stays whatever both sides actually have). Methods the master lacks are
        dropped — they cannot be back-filled for instances that already exist.
        """
        if master is None or not master.get("records"):
            return batch
        mf, bf = master["feats"], batch["feats"]
        n = len(batch["records"])
        out = {}
        for k, mv in mf.items():
            if k.startswith("_"):
                continue
            out[k] = bf[k] if k in bf and bf[k].shape[1] == mv.shape[1] else np.zeros((n, mv.shape[1]), np.float32)
        return {**batch, "feats": out}

    @_mutating
    def propose_instances(self, backend: str, *, paths=None, image_root=None, coco_path=None,
                          limit: int | None = None, score_thresh: float = 0.0,
                          nms_iou: float | None = 0.8, source: str | None = None, **cfg) -> dict:
        """Ingest proposals from a backend. Works on an EMPTY project — this is how a project starts
        without qseg. Returns a report; never raises for user-fixable problems."""
        from . import collect as _co
        from .backends import base as _b

        src = source or backend
        batch_id = f"{src}/{len(self.store.read_ingests()):03d}"
        if backend == "coco":
            from .backends.coco_file import build_coco_collection
            if not coco_path:
                return {"error": "a COCO json path is required"}
            col, rep = build_coco_collection(coco_path, image_root=image_root,
                                             batch_id=batch_id, score_thresh=score_thresh)
            if rep.get("error"):
                return rep
        else:
            be = _b.get(backend)
            ok, why = be.available()
            if not ok:
                from ._bootstrap import BackendUnavailable
                raise BackendUnavailable(f"{be.label} is not usable here: {why}. {be.requires}")
            files = self._image_files(paths, image_root, limit)
            if not files:
                return {"error": f"no images found (paths={paths!r} image_root={image_root!r})"}
            self._set_progress(0, len(files), "proposing")
            try:
                col = _b.build_collection(be, files, score_thresh=score_thresh, batch_id=batch_id,
                                          progress=lambda i, n, nm: self._set_progress(i, n, nm), **cfg)
            finally:
                self._clear_progress()

        if not col["records"]:
            return {"error": "the backend returned no usable masks", "n_images": col.get("n_images", 0)}
        if nms_iou:
            col = _co.mask_nms(col, iou_thresh=float(nms_iou))       # class-agnostic, per image
        col = self._align_batch_feats(col, self.collection)
        self._fold_batch(col)
        self._record_ingest(col["records"], context={"mode": "propose", "source": src,
                                                     "backend": backend})
        self.save()
        return {"ok": True, "backend": backend, "source": src,
                "n_instances": len(col["records"]), "n_images": col.get("n_images", 0),
                "features": self.available_features()}

    @staticmethod
    def _image_files(paths, image_root, limit) -> list[str]:
        """Explicit paths, or every image under a root (sorted, so a `limit` is reproducible)."""
        import glob
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        if paths:
            out = [str(p) for p in paths if os.path.isfile(str(p))]
        elif image_root:
            out = sorted(p for p in glob.glob(os.path.join(str(image_root), "**", "*"), recursive=True)
                         if os.path.splitext(p)[1].lower() in exts and os.path.isfile(p))
        else:
            out = []
        return out[:int(limit)] if limit else out

    def _merge_pending_shards(self, *, context: dict | None = None) -> int:
        """Fold any append-only ingest shards into the live collection + state, then save and clear them.
        Used to FINALIZE an incremental ingest AND to RECOVER an interrupted one on project open. Idempotent:
        records already present (by iuid) are skipped, so a crash between save_collection and clear is safe.
        Records an ingest registry event (the run's batch_ids) so a view can later scope to it."""
        pending = self.store.load_collection_shards()
        if pending is None:
            return 0
        existing = {r["iuid"] for r in (self.collection or {}).get("records", [])}
        keep = [i for i, r in enumerate(pending["records"]) if r["iuid"] not in existing]
        keep_recs = [pending["records"][i] for i in keep]
        if keep:
            self._fold_batch(_co.subset_collection(pending, keep))
        del pending
        self.store.save_collection(self.collection)
        self.save()
        self.store.clear_collection_shards()
        if keep_recs:
            self._record_ingest(keep_recs, context=context)
        return len(keep)

    def _record_ingest(self, recs: list[dict], *, context: dict | None = None) -> dict:
        """Append an ingest registry event capturing the batch_ids (the scope key) + counts for a folded run."""
        import time
        bids = sorted({r["batch_id"] for r in recs})
        imgs = {int(r["image_id"]) for r in recs}
        ev = {"ingest_id": f"ing_{len(self.store.read_ingests()):03d}", "ts": time.time(),
              "n_instances": len(recs), "n_images": len(imgs), "batch_ids": bids}
        if context:
            for k in ("mode", "score_thresh", "source"):
                if context.get(k) is not None:
                    ev[k] = context[k]
        self.store.append_ingest_event(ev)
        self._bsrc_cache = None                              # a new source may now exist -> rebuild the batch->source map
        return ev

    def ingest_paths(self, file_paths: list[str], *, mode: str = "new", score_thresh=None, nms_iou=None,
                     with_raddino: bool = False, raddino_pool: str = "mask", source: str | None = None) -> dict:
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
            n_replaced = 0
            if mode == "replace":                      # hide old UN-CURATED instances on the re-inferred
                targets = {_co.path_image_id(f) for f in new_files}   # images FIRST and persist it, so an
                for _u, _m in self.state.meta.items():                # interrupted re-infer stays consistent
                    if (int(_m.image_id) in targets and _m.assigned_class is None
                            and not _m.is_background and _m.merged_into is None):
                        _m.is_background = True
                        n_replaced += 1
                if n_replaced:
                    self.save()
            # INCREMENTAL INGEST: run the model CHUNK by chunk and APPEND each chunk to disk as its own
            # append-only shard (never rewrites collection.pkl), advancing processed_paths per chunk. RAM
            # is bounded to one chunk and an interrupt keeps every finished chunk — recovered on the next
            # open via _merge_pending_shards. The shards are folded into the collection + state once, at
            # the end (or on recovery). register_images_split is idempotent so per-chunk runs are safe.
            CHUNK = 8
            n_new = 0
            for i in range(0, len(new_files), CHUNK):
                self._set_progress("segmentation inference", i, len(new_files))
                b = _co.collect_batch(model, cfg, d2_cfg, new_files[i:i + CHUNK],
                                      score_thresh=st, feature_cfg=feat_cfg)
                if b.get("records"):                   # a 0-detection chunk has no feats methods -> no shard
                    self.store.append_collection_shard(b)
                n_new += len(b.get("records", []))
                processed.update(new_files[i:i + CHUNK])
                man["processed_paths"] = sorted(processed)
                self.store.save_manifest(man)
                del b
            self._set_progress("segmentation inference", len(new_files), len(new_files))
        finally:
            self._clear_progress()
        self._merge_pending_shards(context={"mode": mode, "score_thresh": st,
                                            "source": source or self._default_source()})  # fold shards; record ingest
        self.history.barrier()                         # additive ingest = undo barrier
        self.save()
        out = {"n_new_images": len(new_files), "n_new_instances": n_new, "n_replaced": n_replaced}
        # CHAIN RAD-DINO: extract mask-pooled embeddings for the (now-larger) collection right after the seg
        # run, so 'raddino' stays aligned/selectable without a separate click. compute_raddino self-guards
        # (force=False -> recomputes only when the row count changed, i.e. new instances were added). A
        # raddino failure (no GPU/HF) must NOT discard the seg results, which are already saved -> isolate it.
        if with_raddino and self.collection.get("records"):
            try:
                rad = self.compute_raddino(force=False, pool=raddino_pool)
                out["raddino_n"] = int(rad.get("n", 0)) if rad.get("ok") else 0
                if rad.get("error"):
                    out["raddino_error"] = rad["error"]
            except Exception as e:                     # noqa: BLE001 — surface, don't crash the ingest
                out["raddino_error"] = str(e)
        return {**out, **self.stats()}

    def sample_more(self, n: int, *, smart: bool = False, seed: int | None = None,
                    score_thresh=None, nms_iou=None, with_raddino: bool = False, raddino_pool: str = "mask") -> dict:
        """Random-sample n not-yet-processed images from the configured root and run inference."""
        self._ensure_model()
        processed = set(self.store.load_manifest().get("processed_paths", []))
        files = _sa.list_images(self.state.config["images"]["root"])
        new_files = _sa.sample_random(files, n, exclude=processed, seed=seed)
        if not new_files:
            return {"n_new_images": 0, "n_new_instances": 0, **self.stats()}
        return self.ingest_paths(new_files, score_thresh=score_thresh, nms_iou=nms_iou,
                                 with_raddino=with_raddino, raddino_pool=raddino_pool)

    def infer_dir(self, directory: str, *, limit: int = 50, mode: str = "new",
                  score_thresh=None, nms_iou=None, with_raddino: bool = False, raddino_pool: str = "mask") -> dict:
        """Run inference on (up to `limit`) images in a server-side folder and add their instances.
        `mode` (new | append | replace) forwarded to ingest_paths (re-inference on already-seen images)."""
        files = _sa.list_images(directory)
        if not files:
            return {"error": f"no images found in {directory}"}
        return self.ingest_paths(files[:int(limit)] if limit else files, mode=mode,
                                 score_thresh=score_thresh, nms_iou=nms_iou,
                                 with_raddino=with_raddino, raddino_pool=raddino_pool)

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

    @staticmethod
    def _decode_ann_mask(ann: dict, H: int, W: int):
        """A COCO annotation's mask at (H,W): RLE (dict) | polygon(s) (list) | bbox-only ([x,y,w,h]). None
        if undecodable. Detector-agnostic — any model's COCO proposals decode the same way."""
        from pycocotools import mask as mu
        seg = ann.get("segmentation")
        if isinstance(seg, dict) and "counts" in seg:
            r = dict(seg); r.setdefault("size", [H, W])
            if isinstance(r["counts"], str):
                r = {"size": r["size"], "counts": r["counts"].encode("ascii")}
            return mu.decode(r).astype(bool)
        if isinstance(seg, list) and seg:
            return mu.decode(mu.merge(mu.frPyObjects(seg, H, W))).astype(bool)
        bb = ann.get("bbox")
        if bb and len(bb) == 4:
            x, y, w, h = (int(round(float(v))) for v in bb)
            m = np.zeros((H, W), bool); m[max(0, y):min(H, y + h), max(0, x):min(W, x + w)] = True
            return m
        return None

    def _fill_raddino(self, rows: list, jobs: list, dim: int) -> None:
        """Best-effort crop-forward RAD-DINO for imported proposals (jobs: (row_idx, abs_path, box)). Groups by
        image, one grid_batch per image, MAX-pools the mask-bbox patch grid — same recipe as
        _instance_ref_embeddings. On any failure the rows stay 0-filled (raddino unavailable headless)."""
        from collections import defaultdict
        ext = self._ref_extractor()
        by_img = defaultdict(list)
        for ri, path, box in jobs:
            by_img[path].append((ri, box))
        for path, items in by_img.items():
            img = _load_rgb(path)
            crops = [img[max(0, b[1]):b[3], max(0, b[0]):b[2]] for _ri, b in items]
            B = int(os.environ.get("CURATOR_RADDINO_BATCH", "8"))
            for s in range(0, len(crops), B):
                grids = ext.grid_batch(crops[s:s + B])
                for j in range(int(grids.shape[0])):
                    rows[items[s + j][0]] = grids[j].amax(dim=(1, 2)).detach().cpu().numpy().astype(np.float32)

    @_mutating
    def import_proposals_coco(self, coco_path: str, *, source: str, with_raddino=None) -> dict:
        """Ingest an EXTERNAL model's proposals (a COCO of masks on the project's images) as a tagged SOURCE
        so they join the SAME embedding/cluster/Map space and become filterable by source everywhere. Model-
        AGNOSTIC: detector features (decoder/backbone/...) are unavailable for foreign masks, so they are
        0-filled (finite -> never poisons the global NaN check / native clustering); the cross-source common
        space is `shapecoord` (computed per mask) [+ `raddino` crop-forward, if the collection uses it]. Images
        are matched to the collection by file basename; unmatched proposals are skipped + reported."""
        import json
        from . import ids as _ids
        from .state import InstanceMeta
        if not self.collection or not self.collection.get("feats"):
            return {"error": "no collection loaded"}
        try:
            with open(coco_path) as f:
                coco = json.load(f)
        except Exception as e:
            return {"error": f"could not read COCO: {e}"}
        recs = self.collection["records"]
        by_base = {}
        for r in recs:
            b = os.path.basename(str(r.get("file_name") or r.get("abs_path") or ""))
            if b and b not in by_base:
                by_base[b] = {"image_id": int(r["image_id"]), "abs_path": r.get("abs_path") or r.get("file_name"),
                              "H": int(r["H"]), "W": int(r["W"])}
        cimg = {im.get("id"): im for im in coco.get("images", [])}
        methods = [k for k in self.collection["feats"] if not k.startswith("_")]
        dims = {k: int(self.collection["feats"][k].shape[1]) for k in methods}
        batch_id = f"import/{source}/{len(self.store.read_ingests()):03d}"
        want_rad = ("raddino" in methods) if with_raddino is None else (bool(with_raddino) and "raddino" in methods)
        new_records, new_feats = [], {k: [] for k in methods}
        matched_imgs, unmatched, rad_jobs = set(), set(), []
        for ann in coco.get("annotations", []):
            im = cimg.get(ann.get("image_id"))
            if im is None:
                continue
            base = os.path.basename(str(im.get("file_name") or ""))
            match = by_base.get(base)
            if match is None:
                unmatched.add(base); continue
            H, W = match["H"], match["W"]
            m = self._decode_ann_mask(ann, H, W)
            if m is None or not m.any():
                continue
            ys, xs = np.where(m)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
            from pycocotools import mask as mu
            rle = mu.encode(np.asfortranarray(m.astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
            nu = _ids.new_uid()
            new_records.append({"iuid": nu, "row": 0, "inst_id": 0, "image_id": match["image_id"], "H": H, "W": W,
                                "score": float(ann.get("score", 1.0)), "rle": rle, "file_name": match["abs_path"],
                                "abs_path": match["abs_path"], "batch_id": batch_id,
                                "cx": float(xs.mean() / W), "cy": float(ys.mean() / H),
                                "bw": float((x2 - x1) / W), "bh": float((y2 - y1) / H),
                                "box_area": float((x2 - x1) * (y2 - y1) / (W * H)), "mask_area_frac": float(m.mean())})
            matched_imgs.add(match["image_id"])
            for k in methods:
                new_feats[k].append(_co.shapecoord_vector(m) if k == "shapecoord" else np.zeros(dims[k], np.float32))
            if want_rad:
                rad_jobs.append((len(new_records) - 1, match["abs_path"], (x1, y1, x2, y2)))
        if not new_records:
            return {"error": "no proposals matched the collection's images (matched by file basename)",
                    "unmatched_images": sorted(unmatched)[:20]}
        if want_rad:
            try:
                self._fill_raddino(new_feats["raddino"], rad_jobs, dims["raddino"])
            except Exception:
                pass                                              # raddino unavailable -> stays 0-filled
        batch = {"records": new_records, "n_images": len(matched_imgs),
                 "feats": {k: np.asarray(v, np.float32) for k, v in new_feats.items()}}
        self.collection = _co.concat_collections(self.collection, batch)     # vstacks feats + rewrites rec['row']
        self.state.order = [r["iuid"] for r in self.collection["records"]]
        gran, modal = self.state.mode(), self.state.modality()
        for r in self.collection["records"]:                                 # add meta for the NEW instances only
            if r["iuid"] not in self.state.meta:
                self.state.meta[r["iuid"]] = InstanceMeta(r["iuid"], r.get("batch_id", "b"), int(r["row"]),
                                                          int(r["image_id"]), granularity=gran, modality=modal)
        self.state.assert_aligned(self.collection["feats"][_any_method(self.collection)].shape[0])
        self.state.coll_version += 1
        self._record_ingest(new_records, context={"mode": "import", "source": source})
        self.store.save_collection(self.collection); self.save()
        return {"ok": True, "n_imported": len(new_records), "source": source, "n_images": len(matched_imgs),
                "raddino": bool(want_rad), "unmatched_images": sorted(unmatched)[:20]}

    def reinfer_processed(self, *, mode: str = "replace", limit: int | None = None,
                          score_thresh=None, nms_iou=None, with_raddino: bool = False,
                          raddino_pool: str = "mask") -> dict:
        """Re-run the (adopted) model on images ALREADY processed — the loop's 're-score the existing pool
        with the new model' step. mode=replace hides old un-curated instances first; append keeps them.
        score_thresh / nms_iou override the detection thresholds for this re-infer."""
        processed = sorted(self.store.load_manifest().get("processed_paths", []))
        if not processed:
            return {"n_new_images": 0, "n_new_instances": 0, "n_replaced": 0, **self.stats()}
        return self.ingest_paths(processed[:int(limit)] if limit else processed, mode=mode,
                                 score_thresh=score_thresh, nms_iou=nms_iou,
                                 with_raddino=with_raddino, raddino_pool=raddino_pool)

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

    @_mutating
    def compute_features(self, extractor: str = "raddino", *, force: bool = False,
                         pool: str = "mask") -> dict:
        """Run an extractor over the CURRENT collection and add its per-instance features.

        No re-detection: each existing instance's mask is pooled over the extractor's patch grid, so
        the new column is row-aligned with what is already there and `<name>` simply becomes
        selectable everywhere. `engine.available_features()` is the single source of truth for the
        feature selectors, so nothing else has to know a new extractor exists.

        `force` recomputes even when present (e.g. after new instances were ingested).
        """
        from .extractors import base as _ex

        if not self.collection or not self.collection.get("records"):
            return {"error": "no collection — add proposals first"}
        n_rows = len(self.collection["records"])
        have = self.collection["feats"].get(extractor)
        if have is not None and not force and have.shape[0] == n_rows:
            return {"ok": True, "msg": f"{extractor} already present", "n": int(have.shape[0]),
                    "extractor": extractor, "available": self.available_features()}

        ext = _ex.get(extractor)                      # KeyError -> 400 at the API layer
        ok, why = ext.available()
        if not ok:
            from ._bootstrap import BackendUnavailable
            raise BackendUnavailable(f"{ext.label} is not usable here: {why}. {ext.requires}")

        self._set_progress("loading model", 0, 0)
        try:
            _co.pool_by_path(self.collection, ext, extractor, pool=str(pool),
                             progress=lambda d, t: self._set_progress(f"{extractor} features", d, t))
        finally:
            self._clear_progress()
        if extractor not in self.collection["feats"]:
            return {"error": f"{extractor} extraction produced no features"}
        self.state.assert_aligned(self.collection["feats"][extractor].shape[0])
        self.state.coll_version += 1
        self.state.collection_dirty = True
        self.store.save_collection(self.collection)
        self.save()
        return {"ok": True, "n": int(self.collection["feats"][extractor].shape[0]),
                "extractor": extractor, "available": self.available_features()}

    def compute_raddino(self, *, force: bool = False, pool: str = "mask") -> dict:
        """Back-compat alias — RAD-DINO is one entry in the extractor registry now."""
        return self.compute_features("raddino", force=force, pool=pool)
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

    def recompute_shape_features(self) -> dict:
        """Recompute the MASK-derived shape features (`shape` descriptors + `shapecoord`) for the CURRENT
        collection from each instance's EFFECTIVE mask (so refined/merged masks count), with NaN/inf
        sanitized — degenerate masks used to make cv2.fitEllipse return NaN and that got stored. No
        re-detection; bumps coll_version + persists, so `shape` (previously NaN-flagged and disabled in the
        selectors) is properly stored and selectable again."""
        if not self.collection or not self.collection.get("records"):
            return {"error": "no collection — Sample & extract first"}
        from pycocotools import mask as _mu
        from ._bootstrap import get_P
        P = get_P()
        recs = self.collection["records"]
        n = len(recs)
        self._set_progress("recomputing shape features", 0, n)
        try:
            for i, r in enumerate(recs):
                u = r.get("iuid")
                m = self._mask(u) if (u and u in self.state.meta) else _mu.decode(r["rle"]).astype(bool)
                r["shape"] = P.shape_descriptors(m)
                r["f_shapecoord"] = _co.shapecoord_vector(m)
                if i % 200 == 0:
                    self._set_progress("recomputing shape features", i, n)
        finally:
            self._clear_progress()
        feats = self.collection["feats"]
        cols = list(recs[0]["shape"].keys())
        feats["shape"] = np.nan_to_num(np.array([[r["shape"][c] for c in cols] for r in recs], np.float32),
                                       nan=0.0, posinf=0.0, neginf=0.0)
        feats["_shape_cols"] = cols
        feats["shapecoord"] = np.nan_to_num(np.stack([r["f_shapecoord"] for r in recs]).astype(np.float32),
                                            nan=0.0, posinf=0.0, neginf=0.0)
        feats["_shapecoord_cols"] = list(_co._SHAPECOORD_COLS)
        self.state.assert_aligned(feats["shape"].shape[0])
        self.state.coll_version += 1                        # busts feature_nan_methods cache + fused cache
        self.state.collection_dirty = True
        self.store.save_collection(self.collection)
        self.save()
        return {"ok": True, "n": n, "available": self.available_features()}

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

    def feature_nan_methods(self) -> set[str]:
        """Feature methods whose matrix contains any non-finite value (NaN/inf) — these break sklearn
        (LogReg/RF .fit raises on NaN) and a cosine-kNN, so they must not be used for classification.
        Cached by coll_version (the features are a pure function of the collection)."""
        if not self.collection or not self.collection.get("feats"):
            return set()
        key = int(self.state.coll_version)
        cache = getattr(self, "_nan_methods_cache", None)
        if cache is None or cache[0] != key:
            bad = {m for m in self.available_features()
                   if self.collection["feats"][m].size and not np.isfinite(self.collection["feats"][m]).all()}
            self._nan_methods_cache = (key, bad)
        return self._nan_methods_cache[1]

    def feature_health(self) -> dict:
        """{features: all present, nan: those with NaN/inf} — drives the classifier selector so a feature
        with NaN values isn't selectable."""
        return {"features": self.available_features(), "nan": sorted(self.feature_nan_methods())}

    def _present_spec_nanfree(self, spec) -> tuple[dict, list[str]]:
        """Spec restricted to PRESENT + NaN/inf-free methods (the latter break sklearn/FINCH). Returns
        (clean_spec, dropped_nan). Used by every feature consumer: cluster, classifier, substructure,
        merge-recommender — so a NaN feature can never reach the math."""
        present = self._present_spec(spec)
        bad = self.feature_nan_methods()
        clean = {m: w for m, w in present.items() if m not in bad}
        return clean, sorted(set(present) - set(clean))

    def _in_scope(self, u: str) -> bool:
        """Whether `u` is within the active VIEW = ingest SCOPE (batch_ids) AND the source FACET (proposal
        model) AND the KIND facet (granularity/modality). All default to open. Folded here so the single
        predicate — already threaded through the live index, pool, projection and workload — filters every
        view for free (no per-view code). Adding a filter anywhere else is a design error."""
        m = self.state.meta[u]
        if self._scope_bids is not None and m.batch_id not in self._scope_bids:
            return False
        if self._source_filter is not None and self._source_of(u) not in self._source_filter:
            return False
        if self._granularity_filter is not None and m.granularity not in self._granularity_filter:
            return False
        if self._modality_filter is not None and m.modality not in self._modality_filter:
            return False
        return True

    # ---- item KIND (granularity x modality) — a composable view facet, same shape as sources ----------
    def kinds(self) -> dict:
        """Distinct (granularity, modality) pairs with LIVE counts, plus the active filters. A
        single-mode project reports exactly one row — which is how the UI knows not to show the facet."""
        from collections import Counter
        c = Counter((self.state.meta[u].granularity, self.state.meta[u].modality)
                    for u in self.state.order if self.state.meta[u].merged_into is None)
        return {"kinds": [{"granularity": g, "modality": md, "n": int(n)}
                          for (g, md), n in sorted(c.items(), key=lambda kv: -kv[1])],
                "mode": self.state.mode(), "modality": self.state.modality(),
                "primary_extractor": self.state.primary_extractor(),
                "capabilities": self.state.capabilities(),
                "active_granularity": sorted(self._granularity_filter) if self._granularity_filter is not None else None,
                "active_modality": sorted(self._modality_filter) if self._modality_filter is not None else None}

    @_mutating
    def set_kind_filter(self, *, granularity=None, modality=None) -> dict:
        """Restrict the view to these granularities/modalities (None or 'all' clears each independently).
        Composes with the ingest scope and the source facet; busts the same caches via _scope_token."""
        def _norm(v):
            if not v or v in ("all", ["all"]):
                return None
            return {str(x) for x in ([v] if isinstance(v, str) else v)}

        self._granularity_filter = _norm(granularity)
        self._modality_filter = _norm(modality)
        self._scope_token += 1                    # invalidate index/projection/materialized-partition caches
        self._index = None
        return self.kinds()

    # ---- proposal SOURCE (which model proposed an instance) — a composable, multi-select view facet -------
    def _default_source(self) -> str:
        """Source label for instances not tied to a registered ingest (the initial collection): the project's
        model config name / ckpt basename, else 'model'."""
        mc = self.state.config.get("model", {}) if self.state.config else {}
        import os
        return str(mc.get("config_name") or (os.path.basename(str(mc.get("ckpt"))) if mc.get("ckpt") else "") or "model")

    def _batch_source(self) -> dict:
        """{batch_id -> source} from the ingest registry (each ingest event may carry a `source` = the model
        that proposed it). Cached; invalidated on a new ingest. O(1) lookups in the hot index/scope loops."""
        if self._bsrc_cache is None:
            m = {}
            for ev in self.store.read_ingests():
                src = ev.get("source") or ev.get("ingest_id")
                for bid in ev.get("batch_ids", []):
                    m[bid] = src
            self._bsrc_cache = m
        return self._bsrc_cache

    def _source_of(self, iuid: str) -> str:
        m = self.state.meta.get(iuid)
        if m is None:
            return self._default_source()
        return self._batch_source().get(m.batch_id, self._default_source())

    def sources(self) -> dict:
        """Distinct proposal sources with LIVE counts (over all instances, so the facet can show every source
        to toggle), plus the currently active facet."""
        from collections import Counter
        c = Counter(self._source_of(u) for u in self.state.order
                    if self.state.meta[u].merged_into is None)
        return {"sources": [{"source": s, "n": int(n)} for s, n in sorted(c.items(), key=lambda kv: -kv[1])],
                "active": (sorted(self._source_filter) if self._source_filter is not None else None)}

    @_mutating
    def set_source_filter(self, sources) -> dict:
        """Show only instances proposed by `sources` (a list; None/[]/'all' clears -> all sources). Composable
        with the ingest scope. Busts the index/projection/finch-materialization caches (via _scope_token) but
        does NOT clear the cluster — the FINCH structure stays; membership just re-filters at read time."""
        if not sources or sources in ("all", ["all"]):
            self._source_filter = None
        else:
            self._source_filter = set(str(s) for s in sources)
        self._index = None
        self._scope_token += 1
        return {"ok": True, "active": (sorted(self._source_filter) if self._source_filter is not None else None)}

    def _pool_iuids(self) -> list[str]:
        """The curation pool that gets clustered: unassigned, non-background, non-merge-child, IN SCOPE."""
        return [u for u in self.state.order
                if self.state.meta[u].assigned_class is None
                and not self.state.meta[u].is_background and self.state.meta[u].merged_into is None
                and self._in_scope(u)]

    # ---- ingest registry + view scope (restrict pool/images to one (re)inference run) ----
    def list_ingests(self) -> list[dict]:
        """Recorded ingests (newest first) with LIVE in-pool counts under the current curation
        (assigned/rejected drop out). Drives the 'Scope' selector."""
        out = []
        for ev in self.store.read_ingests():
            bset = set(ev.get("batch_ids", []))
            n_live = sum(1 for u in self.state.order
                         if self.state.meta[u].batch_id in bset and self._is_pool(u))
            out.append({"ingest_id": ev.get("ingest_id"), "ts": ev.get("ts"),
                        "n_instances": ev.get("n_instances"), "n_images": ev.get("n_images"),
                        "mode": ev.get("mode"), "score_thresh": ev.get("score_thresh"), "n_live": n_live})
        out.reverse()
        return out

    @_mutating
    def set_scope(self, ingest_id: str | None) -> dict:
        """Restrict the clustering pool + image picker to ONE ingest's instances. `None`/''/'all' clears it.
        Clears the cluster (it was built on the previous pool) so the next cluster() rebuilds on the scope."""
        if ingest_id in (None, "", "all"):
            self._scope_bids, self._scope_id = None, None
        else:
            ev = next((e for e in self.store.read_ingests() if e.get("ingest_id") == ingest_id), None)
            if ev is None:
                return {"error": f"unknown ingest {ingest_id}"}
            self._scope_bids, self._scope_id = set(ev.get("batch_ids", [])), ingest_id
        self._cluster = self._grp_cache = self._index = None
        self._scope_token += 1
        pool = self._pool_iuids()
        imgs = {int(self.state.meta[u].image_id) for u in pool}
        return {"ok": True, "scope": self._scope_id, "n_pool": len(pool), "n_images": len(imgs)}

    def image_counts(self, query: str = "", limit: int = 100) -> dict:
        """Windowed image-id list (most-populated first) + per-image instance count, respecting the scope.
        Reads the live index's per-image counts (O(#images)) instead of an O(N) Counter scan per tab entry."""
        items = sorted(self._get_index()["img_counts"].items(), key=lambda kv: -kv[1])
        q = (query or "").strip()
        if q:
            items = [(i, n) for i, n in items if q in str(i)]
        return {"total": len(items), "items": [{"image_id": str(i), "n": n} for i, n in items[:limit]]}

    @_timed
    @_mutating
    def cluster(self, spec, *, distance: str = "cosine", per_image: bool = False, level: int | None = None,
                req_clust: int | None = None) -> dict:
        """FINCH-cluster ONLY the unassigned pool — already-assigned instances are not reclustered
        (each class becomes its own standalone pseudo-partition in partition_view)."""
        from ._bootstrap import get_P
        P = get_P()
        pool = self._pool_iuids()
        spec, dropped_nan = self._present_spec_nanfree(spec)
        if not spec:
            ok = [m for m in self.available_features() if m not in self.feature_nan_methods()]
            raise ValueError(f"no usable (present, NaN-free) features selected"
                             + (f"; dropped for NaN: {dropped_nan}" if dropped_nan else "")
                             + f"; NaN-free available: {ok}")
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
                self._cluster["level"] if self._cluster else -1, len(self.state.taxonomy),
                self._scope_token)

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
        spec, dropped_nan = self._present_spec_nanfree(spec)
        if not spec:
            ok = [m for m in self.available_features() if m not in self.feature_nan_methods()]
            return {"error": f"no usable (present, NaN-free) features selected"
                    + (f"; dropped for NaN: {dropped_nan}" if dropped_nan else "") + f"; NaN-free available: {ok}"}
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
    # ---- incrementally-maintained membership index ---------------------------------------------------
    # ONE O(N) pass builds every "set of instances" the UI shows (class buckets, FINCH partition sizes,
    # per-image live lists, the unassigned pool, release composition). It is REBUILT only when the STRUCT
    # (collection/cluster/level/scope/taxonomy) changes; the high-frequency assignment mutations PATCH it in
    # place (O(K)) via `_cache_delta`, so assign/reject don't trigger an O(N) rebuild at 1M. Correctness is
    # self-healing: every state change bumps `_mutation_serial`; any change that did NOT patch the index
    # leaves a serial mismatch → full rebuild on next read (so a missed path is slow, never wrong).
    def _struct_key(self):
        return (self.state.coll_version, id(self._cluster),
                self._cluster["level"] if self._cluster else -1, len(self.state.taxonomy), self._scope_token)

    def _rebuild_index(self) -> dict:
        from collections import defaultdict
        recs = self.collection["records"] if self.collection else []
        score = lambda u: float(recs[self.state.meta[u].row]["score"]) if recs else 0.0
        cluster_pool = set(self._cluster["pool"]) if self._cluster else set()
        pidmap = self._iuid_pid_map() if self._cluster else {}
        idx = {"struct_key": self._struct_key(), "serial": self._mutation_serial,
               "class_members": defaultdict(list), "class_score": defaultdict(float),
               "finch_active": defaultdict(int), "finch_score": defaultdict(float),
               "image_live": defaultdict(list), "unassigned": set(),
               "imgcomp": defaultdict(lambda: [0, 0]), "img_counts": defaultdict(int)}
        for u, m in self.state.meta.items():
            iid = m.image_id
            if self._in_scope(u):
                idx["img_counts"][iid] += 1
            if m.merged_into is not None:
                continue
            if m.assigned_class:
                idx["image_live"][iid].append(u)
                idx["imgcomp"][iid][0] += 1
                if self._in_scope(u):
                    idx["class_members"][m.assigned_class].append(u)
                    idx["class_score"][m.assigned_class] += score(u)
            elif not m.is_background:                      # unassigned, live -> pool
                idx["image_live"][iid].append(u)
                idx["imgcomp"][iid][1] += 1
                idx["unassigned"].add(u)
                if u in cluster_pool:
                    p = pidmap.get(u)                      # _iuid_pid_map values are str; _pool_groups keys are int
                    if p is not None:
                        idx["finch_active"][int(p)] += 1
                        idx["finch_score"][int(p)] += score(u)
        return idx

    def _get_index(self) -> dict:
        idx = self._index
        if idx is None or idx["struct_key"] != self._struct_key() or idx["serial"] != self._mutation_serial:
            self._index = idx = self._rebuild_index()
        return idx

    def _cache_delta(self, before: dict) -> None:
        """Patch the live index for the instances in `before` (iuid -> (old_cid, old_bg, old_merged)), reading
        their NEW state from meta. Called by the simple assignment mutations AFTER `_after_mutation` bumped the
        serial; no-op (→ next read rebuilds) when the index is absent or the STRUCT changed."""
        idx = self._index
        if idx is None or idx["struct_key"] != self._struct_key() or idx["serial"] != self._mutation_serial - 1:
            return                                         # can't safely patch -> leave stale -> rebuild on read
        recs = self.collection["records"] if self.collection else []
        score = lambda u: float(recs[self.state.meta[u].row]["score"]) if recs else 0.0
        cluster_pool = set(self._cluster["pool"]) if self._cluster else set()
        pidmap = self._iuid_pid_map() if self._cluster else {}
        for u, (ocid, obg, omerged) in before.items():
            m = self.state.meta[u]
            ncid, nbg, nmerged = m.assigned_class, m.is_background, m.merged_into
            insc = self._in_scope(u)
            sc = score(u)
            # class buckets (scope-filtered, like _rebuild)
            o_cls = ocid if (ocid and not obg and omerged is None and insc) else None
            n_cls = ncid if (ncid and not nbg and nmerged is None and insc) else None
            if o_cls != n_cls:
                if o_cls:
                    if u in idx["class_members"][o_cls]:
                        idx["class_members"][o_cls].remove(u)
                    idx["class_score"][o_cls] -= sc
                if n_cls:
                    idx["class_members"][n_cls].append(u)
                    idx["class_score"][n_cls] += sc
            # per-image live list (assigned OR unassigned; not bg/merged)
            o_live, n_live = (not obg and omerged is None), (not nbg and nmerged is None)
            if o_live != n_live:
                lst = idx["image_live"][m.image_id]
                if o_live and u in lst:
                    lst.remove(u)
                elif n_live:
                    lst.append(u)
            # unassigned pool + FINCH counts
            o_un = (ocid is None and not obg and omerged is None)
            n_un = (ncid is None and not nbg and nmerged is None)
            if o_un != n_un:
                if n_un:
                    idx["unassigned"].add(u)
                else:
                    idx["unassigned"].discard(u)
                if u in cluster_pool and (p := pidmap.get(u)) is not None:
                    idx["finch_active"][int(p)] += (1 if n_un else -1)
                    idx["finch_score"][int(p)] += (sc if n_un else -sc)
            # release composition (assigned / unassigned counts per image; live only)
            o_a, o_pend = (omerged is None and bool(ocid)), (omerged is None and not ocid and not obg)
            n_a, n_pend = (nmerged is None and bool(ncid)), (nmerged is None and not ncid and not nbg)
            if (o_a, o_pend) != (n_a, n_pend):
                idx["imgcomp"][m.image_id][0] += int(n_a) - int(o_a)
                idx["imgcomp"][m.image_id][1] += int(n_pend) - int(o_pend)
        idx["serial"] = self._mutation_serial

    def partition_view(self) -> list[dict]:
        """Per-class pseudo-partitions (assigned instances, pid='class:<cid>') first, then the FINCH
        partitions of the still-unassigned pool (pid=str int). Sizes/scores come from the live index
        (O(#partitions)), not an O(N) meta scan — so it stays fast after each assign/reject at 1M."""
        idx = self._get_index()

        def _row(pid, n, ssum, purity, cls):
            return {"pid": pid, "size": n, "purity": purity,
                    "mean_score": round(ssum / n, 2) if n else 0.0, "majority_class": cls}
        rows = [_row(f"class:{cid}", len(idx["class_members"][cid]), idx["class_score"][cid], 1.0,
                     self.state.class_name(cid))
                for cid in self.state.taxonomy if idx["class_members"].get(cid)]
        if self._cluster:
            rows += [_row(str(pid), idx["finch_active"][pid], idx["finch_score"][pid], None, "")
                     for pid in sorted(self._pool_groups()) if idx["finch_active"].get(pid)]
        rows.sort(key=lambda r: (not str(r["pid"]).startswith("class:"), -r["size"]))
        return rows

    def partition_iuids(self, pid) -> list[str]:
        pid = str(pid)
        idx = self._get_index()
        if pid.startswith("class:"):
            return idx["class_members"].get(pid[len("class:"):], [])
        if self._cluster and pid.lstrip("-").isdigit() and int(pid) in self._pool_groups():
            pool = self._cluster["pool"]                   # FINCH partition: materialize on demand (O(group)),
            gen = (idx["struct_key"], idx["serial"])       # cached per generation (busts on level/scope change
            cache = getattr(self, "_finch_mat", None)      # AND on any mutation) so paging stays O(1) but fresh
            if cache is None or cache[0] != gen:
                self._finch_mat = cache = (gen, {})
            if pid not in cache[1]:
                cache[1][pid] = [pool[i] for i in self._pool_groups()[int(pid)]
                                 if self._is_pool(pool[i]) and self._in_scope(pool[i])]   # scope + source facet
            return cache[1][pid]
        if pid in self.state.meta:                        # a bare iuid (e.g. an unclustered unlabeled reference
            return [pid]                                  # match) -> the instance itself as a singleton "partition"
        return []

    def _image_members(self, image_id: int) -> list[str]:
        return self._get_index()["image_live"].get(int(image_id), [])

    def _unassigned_iuids(self) -> list[str]:
        """The pool (assignment-only predicate), reused by the classifier predict/recommend paths — O(1) read
        from the live index instead of an O(N) `state.meta` scan per preview."""
        return list(self._get_index()["unassigned"])

    # ---- image-level RELEASE gate (which fully-curated images go to the final dataset) -------------
    def _image_composition(self) -> dict:
        """{image_id -> {'assigned': n, 'unassigned': n}} over LIVE (non-merged) instances, from the live
        index. 'unassigned' = NON-categorized (no class, not background) — i.e. still in the pool."""
        return {iid: {"assigned": a, "unassigned": u}
                for iid, (a, u) in self._get_index()["imgcomp"].items() if a or u}

    def release_candidates(self) -> list[int]:
        """Images that are FINAL: no live instance is still uncategorized (unassigned == 0) AND more than one
        instance was kept (assigned > 1). These are the only images the release gate offers."""
        comp = self._image_composition()
        return sorted(iid for iid, c in comp.items() if c["unassigned"] == 0 and c["assigned"] > 1)

    def release_stats(self) -> dict:
        cands = self.release_candidates()
        rel = self.state.release_gate
        acc = sum(1 for iid in cands if rel.get(str(iid)) == "accepted")
        rej = sum(1 for iid in cands if rel.get(str(iid)) == "rejected")
        return {"fully_categorized": len(cands), "accepted": acc, "rejected": rej,
                "pending": len(cands) - acc - rej}

    def set_release(self, image_ids, status: str) -> int:
        """Set the image-level release decision. status in {'accepted','rejected'} (anything else CLEARS it
        back to pending). Independent of the instance-level reject (is_background) — it gates whole IMAGES for
        the final dataset. Persisted (write-behind); not an undoable history op."""
        status = status if status in ("accepted", "rejected") else ""
        ids = image_ids if isinstance(image_ids, (list, tuple, set)) else [image_ids]
        for iid in ids:
            key = str(int(iid))
            if status:
                self.state.release_gate[key] = status
            else:
                self.state.release_gate.pop(key, None)
        self._save_dirty.set()                            # persist via the background saver (no history push)
        return len(list(ids))

    def release_view(self, *, filter: str = "all", offset: int = 0, limit: int = 24) -> dict:
        """Windowed list of release-candidate images (final = fully categorized, >1 instance) + the gate
        stats. `filter` in {'all','pending','accepted','rejected'}."""
        comp = self._image_composition()
        rel = self.state.release_gate
        cands = self.release_candidates()
        if filter == "pending":
            cands = [i for i in cands if rel.get(str(i), "") == ""]
        elif filter in ("accepted", "rejected"):
            cands = [i for i in cands if rel.get(str(i)) == filter]
        page = cands[int(offset):int(offset) + int(limit)]
        items = [{"image_id": str(int(iid)), "n_assigned": comp[iid]["assigned"],
                  "status": rel.get(str(iid), "")} for iid in page]
        return {"total": len(cands), "items": items, "stats": self.release_stats()}

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

    def _before_states(self, iuids) -> dict:
        """Snapshot (assigned_class, is_background, merged_into) per iuid for the live-index delta hook."""
        return {u: (m.assigned_class, m.is_background, m.merged_into)
                for u in iuids if (m := self.state.meta.get(u)) is not None}

    @_mutating
    def assign(self, iuids: list[str], class_name: str, *, source: str = "manual",
               scores: dict | None = None) -> None:
        if not iuids:
            return
        before = self._before_states(iuids)
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
        self._cache_delta(before)

    def assign_partition(self, pid: int, class_name: str) -> None:
        self.assign(self.partition_iuids(pid), class_name, source="partition")

    def reject_partition(self, pid) -> int:
        """Reject (background) EVERY instance in a partition in one undoable op — the whole-partition analog
        of assign_partition. Returns how many were rejected. Done server-side so a huge partition isn't
        shipped to the browser and back just to reject it."""
        iuids = list(self.partition_iuids(pid))
        if iuids:
            self.set_background(iuids)
        return len(iuids)

    @_mutating
    def remove_from_class(self, iuids: list[str]) -> None:
        before = self._before_states(iuids)
        tok = self.history.begin(self.state, iuids, [])
        for u in iuids:
            self.state.meta[u].assigned_class = None
            self.state.meta[u].assign_source = None
            self.state.meta[u].assign_score = None
        self.history.commit(self.state, tok, "remove", f"unassign {len(iuids)}")
        self._after_mutation()
        self._cache_delta(before)

    @_mutating
    def set_background(self, iuids: list[str]) -> None:
        before = self._before_states(iuids)
        tok = self.history.begin(self.state, iuids, [])
        for u in iuids:
            self.state.meta[u].is_background = True
            self.state.meta[u].assigned_class = None
        self.history.commit(self.state, tok, "background", f"reject {len(iuids)}")
        self._after_mutation()
        self._cache_delta(before)

    # ---- nested taxonomy (superclass -> concept -> leaf parts) -------------
    def seed_taxonomy(self, path=None, *, replace: bool = False, prune: bool = False) -> dict:
        """Load the nested taxonomy seed (superclasses + concepts + part leaves) into state, pinning a stable
        coco_cat_id per leaf. Idempotent: existing leaves keep their id, just gain grouping metadata.
        `replace` first clears superclasses/concepts (leaves are kept — they may carry assignments).
        `prune` (opt-in) drops in-state leaves that are NO LONGER in the JSON AND carry zero assignments — to
        retire restructured/removed leaves (e.g. a concept's old sub-parts). Temp/scratch leaves are never
        pruned (they are user workspace, not seed-derived), and any leaf with >=1 assignment is kept."""
        import json
        from collections import Counter
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
        seeded_lids: set[str] = set()
        for c in d.get("concepts", []):
            self.state.concepts[c["id"]] = Concept(concept_id=c["id"], name=c["name"], superclass=c.get("superclass"),
                description=c.get("description", ""), structure_type=c.get("structure_type", ""),
                aliases=c.get("aliases", []), part_rules=c.get("part_rules", []), mimic_family=c.get("mimic_family"))
            has_parts = bool(c.get("parts"))
            leaves = c.get("parts") or [{"id": c["id"], "name": c["name"], "structure_type": c.get("structure_type", ""),
                                         "description": c.get("description", "")}]
            for lf in leaves:
                lid = lf["id"]
                seeded_lids.add(lid)
                # QUALIFY part-leaf display names with the concept ("Pacemaker — Lead") so leaf names are
                # GLOBALLY UNIQUE (part names like Shaft/Cuff/Tube repeat across concepts) -> assign-by-name
                # is unambiguous. Part-less concepts keep their plain (already-unique) name.
                lname = f"{c['name']} — {lf.get('name', lid)}" if has_parts else c["name"]
                t = self.state.taxonomy.get(lid)
                if t is None:
                    self.state.taxonomy[lid] = TaxonomyClass(class_id=lid, name=lname,
                        color=_auto_color(len(self.state.taxonomy)), coco_cat_id=next_id, concept=c["id"],
                        supercategory=c.get("superclass"), description=lf.get("description", ""),
                        structure_type=lf.get("structure_type", c.get("structure_type", "")), temp=False)
                    next_id += 1
                else:                                            # existing leaf -> attach grouping, refresh name
                    t.concept = c["id"]; t.supercategory = c.get("superclass"); t.temp = False; t.name = lname
                    if not t.description:
                        t.description = lf.get("description", "")
        pruned: list[str] = []
        if prune:
            assigned = Counter(m.assigned_class for m in self.state.meta.values() if m.assigned_class)
            pruned = sorted(lid for lid, t in self.state.taxonomy.items()
                            if not t.temp and lid not in seeded_lids and assigned.get(lid, 0) == 0)
            for lid in pruned:
                del self.state.taxonomy[lid]
        self.save()
        return {"superclasses": len(self.state.superclasses), "concepts": len(self.state.concepts),
                "leaves": len([t for t in self.state.taxonomy.values() if not t.temp]),
                "pruned": pruned}

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

    def assign_leaf(self, class_id: str, concept_id: str | None) -> dict:
        """Place a leaf class under a concept (promote a temp/scratch class into the taxonomy): sets its
        concept + supercategory (from the concept) and clears temp. concept_id=None ungroups it."""
        t = self.state.taxonomy.get(class_id)
        if t is None:
            return {"error": f"unknown class {class_id}"}
        if concept_id is None:
            t.concept = None; t.supercategory = None
        else:
            c = self.state.concepts.get(concept_id)
            if c is None:
                return {"error": f"unknown concept {concept_id}"}
            t.concept = concept_id; t.supercategory = c.superclass; t.temp = False
        self.save()
        return {"ok": True}

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

    def class_samples(self, class_id: str, limit: int = 24) -> list[str]:
        """Representative instances of a class (highest predicted-score first, current scope) for the
        Classes-tab sample preview. Returns iuids the caller renders via `crop`."""
        recs = self.collection["records"]
        members = [u for u, m in self.state.meta.items()
                   if m.assigned_class == class_id and not m.is_background and m.merged_into is None
                   and self._in_scope(u)]
        members.sort(key=lambda u: -float(recs[self.state.meta[u].row].get("score", 0.0)))
        return members[:int(limit)]

    @_mutating
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
        if not iuids:                                           # 1 instance = preview of its own mask (single-select preview)
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

    @_mutating
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

    @_mutating
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
        # O(1) lookup into the live index (was an O(N) meta scan on every image switch / overlay / mask toggle).
        return self._image_members(image_id)

    def background_iuids(self) -> list[str]:
        return [u for u, m in self.state.meta.items() if m.is_background]

    # ---- per-partition "most likely class" suggestion (1-NN over labeled + rejected) -----------------
    def _suggestion_refs(self, spec: dict) -> dict:
        """A combined 1-NN index over ALL labeled instances (label = class name) + rejected instances
        (label = "__reject__") in the fused `spec` space, plus an auto-calibrated distance gate baseline.
        Cached on (_mutation_serial, spec, coll_version) so it rebuilds only when labels change / re-cluster /
        ingest — partition select just queries it. Reference vectors are subsampled to _PSUG_REF_CAP."""
        key = (self._mutation_serial, tuple(sorted(spec.items())), int(self.state.coll_version))
        c = getattr(self, "_psug_cache", None)
        if c is not None and c[0] == key:
            return c[1]
        X = self.fused(spec)
        idx = self._get_index()
        rows, labs, n_classes = [], [], 0
        for cid, ius in idx["class_members"].items():
            nm = self.state.class_name(cid) or cid
            n0 = len(rows)
            rows.extend(self.state.meta[u].row for u in ius)
            labs.extend([nm] * (len(rows) - n0))
            if len(rows) > n0:
                n_classes += 1
        bg = [u for u in self.background_iuids() if u in self.state.meta]
        rows.extend(self.state.meta[u].row for u in bg)
        labs.extend(["__reject__"] * len(bg))
        payload = {"index": None, "rows": None, "labs": None, "margin": None,
                   "n_classes": n_classes, "has_reject": bool(bg)}
        if rows:
            rows = np.asarray(rows, int); labs = np.asarray(labs, object)
            if len(rows) > _PSUG_REF_CAP:
                sel = np.sort(np.random.default_rng(0).choice(len(rows), _PSUG_REF_CAP, replace=False))
                rows, labs = rows[sel], labs[sel]
            Xn = (X[rows] / (np.linalg.norm(X[rows], axis=1, keepdims=True) + 1e-9)).astype(np.float32)
            payload.update(index=_nn_build(Xn), rows=rows, labs=labs, margin=self._calib_margin(Xn, labs))
        self._psug_cache = (key, payload)
        return payload

    @staticmethod
    def _calib_margin(Xn: np.ndarray, labs: np.ndarray):
        """Median inter-class nearest-neighbour cosine distance over a subsample of labeled vectors — the
        'typical gap between two different classes'. The gate defaults to gate_mult× this. None if <2 classes."""
        mask = labs != "__reject__"
        Xc, lc = Xn[mask], labs[mask]
        if len(Xc) < 2 or len(set(lc.tolist())) < 2:
            return None
        if len(Xc) > 2000:
            s = np.random.default_rng(1).choice(len(Xc), 2000, replace=False)
            Xc, lc = Xc[s], lc[s]
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(Xc, metric="cosine")
        nd = np.where(lc[:, None] != lc[None, :], D, np.inf).min(1)
        nd = nd[np.isfinite(nd)]
        return float(np.median(nd)) if len(nd) else None

    def _predict_instances(self, iuids, spec: dict, refs: dict, threshold: float):
        """Per-instance 1-NN over the cached labeled+reject reference `refs`: returns [(iuid, label, dist)]
        where label is a class NAME, '__reject__', or 'none' (nearest beyond `threshold`). An instance is
        NEVER matched to ITSELF: its own reference row is always skipped, and for a query that is itself in
        the reference (an already-LABELED instance — e.g. a class partition's members) a dist≈0 candidate (an
        identical COPY of itself) is skipped too, so it gets a meaningful DISTINCT neighbour rather than a
        trivial 100% self-match. (Unlabeled pool/FINCH queries are NOT in the reference, so their dist≈0 hits
        on a labeled exemplar are kept — there it's a real signal, not a self-match.)"""
        rows = [(u, self.state.meta[u].row) for u in iuids if u in self.state.meta]
        if not rows:
            return []
        X = self.fused(spec)
        qrows = np.asarray([r for _, r in rows], int)
        Qn = (X[qrows] / (np.linalg.norm(X[qrows], axis=1, keepdims=True) + 1e-9)).astype(np.float32)
        rrows, labs = refs["rows"], refs["labs"]
        ref_set = set(int(r) for r in rrows.tolist())         # which query rows ARE labeled references
        k = int(min(len(rrows), 8))                           # headroom to skip self + identical copies + ties
        d, I = _nn_search(refs["index"], Qn, k)
        EPS = 1e-6
        out = []
        for i, (u, _) in enumerate(rows):
            labeled_self = int(qrows[i]) in ref_set            # an already-assigned query (its own row is a ref)
            lab, dd = "none", float("inf")
            for col in range(I.shape[1]):
                j = int(I[i, col])
                if j < 0 or rrows[j] == qrows[i]:              # padding / literal self -> never match it
                    continue
                if labeled_self and float(d[i, col]) <= EPS:   # an identical COPY of a labeled self is not "another" instance
                    continue
                dd = float(d[i, col]); lab = labs[j] if dd <= threshold else "none"
                break
            out.append((u, lab, dd))
        return out

    def image_class_suggestion(self, image_id, *, gate_mult: float = 1.0, thr=None) -> dict:
        """Apply the 1-NN classifier to EVERY instance of an image: per-instance predicted class / 'reject' /
        'none' + a per-label summary (the In-image analog of partition_class_suggestion). Read-only."""
        from collections import Counter
        spec_raw = self._cluster["spec"] if self._cluster else {"decoder": 1.0}
        spec, dropped = self._present_spec_nanfree(spec_raw)
        base = {"image_id": str(int(image_id)), "spec": spec, "dropped_features": dropped,
                "gate_mult": float(gate_mult), "items": [], "summary": {}}
        if not spec:
            return {**base, "error": "no usable (NaN-free) features in the clustering space"}
        refs = self._suggestion_refs(spec)
        if refs["index"] is None:
            return {**base, "note": "no labels yet", "has_reject": False, "n_classes": 0, "threshold": None}
        margin = refs["margin"]
        threshold = float(thr) if thr is not None else (gate_mult * margin if margin is not None else 0.25)
        preds = self._predict_instances(self.image_instance_iuids(int(image_id)), spec, refs, threshold)
        items, summ = [], Counter()
        for u, lab, dd in preds:
            key = "reject" if lab == "__reject__" else lab    # class name | "reject" | "none"
            summ[key] += 1
            m = self.state.meta.get(u)
            acid = m.assigned_class if (m and not m.is_background) else None
            items.append({"iuid": u, "label": key, "pred": (None if lab in ("none", "__reject__") else lab),
                          "score": round(max(0.0, 1.0 - dd), 3) if np.isfinite(dd) else 0.0,
                          "assigned": (self.state.class_name(acid) or str(acid)) if acid is not None else None})
        return {**base, "items": items, "summary": dict(summ), "threshold": round(threshold, 4),
                "margin": round(margin, 4) if margin is not None else None,
                "has_reject": refs["has_reject"], "n_classes": refs["n_classes"]}

    def _instance_nn_dists(self, spec: dict, refs: dict) -> dict:
        """Gate-INDEPENDENT nearest-reference distance AND raw-nearest label for EVERY uncategorized in-scope
        instance, grouped per image, in ONE batched 1-NN pass. Returns {"by_img": {image_id(str): {"d":
        [nearest_dist,...], "lab": [nearest_class_or_'reject',...]}}, "n_inst": {image_id: total_in_scope_count},
        "glob": {class_name: n_labeled} (+ 'reject': n_background), "truncated": int}. Cached on (mutation,
        coll_version, scope, spec) — deliberately NOT on the gate: the gate only re-buckets the distances
        (AUTO/borderline/none) and the labels/glob feed class-variety re-ranking, so the gate slider re-derives
        everything in pure python with NO faiss re-query. Rebuilds only on label / cluster / scope change. The
        single batched search is what makes the whole image-picker ranking cheap."""
        key = (self._mutation_serial, int(self.state.coll_version), self._scope_token, tuple(sorted(spec.items())))
        c = getattr(self, "_iwl_dist_cache", None)
        if c is not None and c[0] == key:
            return c[1]
        idx = self._get_index()
        uncat = sorted(u for u in idx["unassigned"] if self._in_scope(u) and u in self.state.meta)
        truncated = max(0, len(uncat) - _WL_QUERY_CAP)
        preds = self._predict_instances(uncat[:_WL_QUERY_CAP], spec, refs, float("inf"))  # inf -> raw ungated dist
        by_img: dict[str, dict] = {}
        for u, lab, dd in preds:                               # keep the raw-nearest LABEL too (gate-independent)
            b = by_img.setdefault(str(int(self.state.meta[u].image_id)), {"d": [], "lab": []})
            b["d"].append(dd); b["lab"].append("reject" if lab == "__reject__" else lab)
        glob = {(self.state.class_name(cid) or str(cid)): len(ius) for cid, ius in idx["class_members"].items()}
        nbg = sum(1 for u in self.background_iuids() if u in self.state.meta and self._in_scope(u))
        if nbg:
            glob["reject"] = nbg
        out = {"by_img": by_img, "n_inst": {str(int(i)): int(n) for i, n in idx["img_counts"].items()},
               "glob": glob, "truncated": truncated}
        self._iwl_dist_cache = (key, out)
        return out

    @staticmethod
    def _diversify_by_class(nd: list[dict], glob: dict, order: str, diversity: float, limit: int) -> list[dict]:
        """MMR-style re-rank of the non-done images so the HEAD of the picker spans MANY predicted classes
        instead of repeating the over-represented ones — counters the labeling bias where 'easiest to finish'
        keeps surfacing the SAME annotation type (the common, high-confidence classes). Greedy: each step picks
        the image maximizing base_priority (easiness for 'easy', hardness for 'hard') MINUS a redundancy penalty
        = how covered its predicted classes already are. `covered` is SEEDED by the global labeled-class
        frequency (so already-heavily-labeled classes start penalized and rare/under-labeled ones float up) and
        GROWN as picks accumulate (so the same class is not surfaced repeatedly). `diversity` (0..1) scales the
        penalty; 0 leaves the base order untouched. Bounded cost (MMR over the best-base-priority head only)."""
        if diversity <= 0 or len(nd) <= 2:
            return nd
        works = [r["work_est"] for r in nd]
        wmax = max(works) or 1.0
        for r in nd:
            r["_prio"] = (r["work_est"] / wmax) if order == "hard" else (1.0 - r["work_est"] / wmax)
        gmax = max(glob.values()) if glob else 1
        covered = {c: v / gmax for c, v in glob.items()}      # over-labeled classes start "already covered"
        lam = 1.2 * float(diversity)
        pool = nd[:max(int(limit) * 5, 600)]                  # diversify the best-base head; bounded O(pool*limit)
        tail = nd[len(pool):]
        chosen, target = [], min(len(pool), int(limit))
        while pool and len(chosen) < target:
            best_i, best_s = 0, None
            for i, r in enumerate(pool):
                red = 0.0
                for cn, f in r["_frac"].items():
                    red += f * covered.get(cn, 0.0)
                s = r["_prio"] - lam * red
                if best_s is None or s > best_s:
                    best_s, best_i = s, i
            r = pool.pop(best_i)
            for cn, f in r["_frac"].items():
                covered[cn] = covered.get(cn, 0.0) + f
            chosen.append(r)
        return chosen + pool + tail

    def image_workload_ranking(self, *, gate_mult: float = 1.0, order: str = "easy", query: str = "",
                               limit: int = 200, thr=None, diversity: float = 0.0) -> dict:
        """Rank in-scope images by ESTIMATED MANUAL WORK LEFT, using the trained 1-NN classifier. For each
        uncategorized instance the nearest labeled/reject exemplar decides its bucket: AUTO (dist <= border gate
        -> one Accept-all resolves it, ~0 cost), BORDERLINE (just inside the gate -> 1/2 a decision), NONE (beyond
        the gate -> a full manual decision). work_est = n_none + 0.5*n_border. order='easy' -> least work first
        (quick wins / 'Accept-all and done'); 'hard' -> most work first (triage). Fully-categorized images
        (n_uncat==0) are tagged done -> 'ready' and sorted to the END of EITHER order (no action needed).
        `diversity` (0..1) class-variety re-ranks the head so it spans many predicted classes instead of
        repeating the over-represented ones (counters the labeling bias where 'easiest' keeps surfacing the same
        annotation type); 0 = pure work order. Falls back to the most-populated order when there are no labels
        yet. Efficient: ONE batched, gate-independent, cached 1-NN pass (`_instance_nn_dists`) feeds every image,
        gate value AND diversity setting. Read-only; never raises."""
        from collections import Counter
        spec_raw = self._cluster["spec"] if self._cluster else {"decoder": 1.0}
        spec, dropped = self._present_spec_nanfree(spec_raw)
        order = "hard" if str(order).lower().startswith("hard") else "easy"
        diversity = max(0.0, min(1.0, float(diversity)))
        q = (query or "").strip()
        base = {"order": order, "gate_mult": float(gate_mult), "diversity": diversity, "spec": spec,
                "dropped_features": dropped, "items": [], "fallback": False, "threshold": None, "margin": None,
                "truncated": 0, "n_total": 0}

        def _fallback(note):
            items = sorted(self._get_index()["img_counts"].items(), key=lambda kv: -kv[1])
            if q:
                items = [(i, n) for i, n in items if q in str(i)]
            out = [{"image_id": str(int(i)), "n_inst": int(n), "n_uncat": None, "work_est": None, "done": False}
                   for i, n in items[:limit]]
            return {**base, "fallback": True, "note": note, "n_total": len(items), "items": out}

        if not spec:
            return _fallback("no usable (NaN-free) features in the clustering space")
        refs = self._suggestion_refs(spec)
        if refs["index"] is None:
            return _fallback("no labels yet")
        margin = refs["margin"]
        T = float(thr) if thr is not None else (gate_mult * margin if margin is not None else 0.25)
        T_lo = _WL_BORDER_FRAC * T
        dc = self._instance_nn_dists(spec, refs)
        by_img = dc["by_img"]
        rows = []
        for iid, n in dc["n_inst"].items():
            if q and q not in iid:
                continue
            b = by_img.get(iid)
            dists = b["d"] if b else ()
            n_uncat = len(dists)
            n_none = sum(1 for d in dists if d > T)
            n_border = sum(1 for d in dists if T_lo < d <= T)
            n_auto = n_uncat - n_none - n_border
            hist = Counter(b["lab"]) if b else Counter()      # raw-nearest classes (gate-independent)
            rows.append({"image_id": iid, "n_inst": int(n), "n_uncat": n_uncat, "n_auto": n_auto,
                         "n_none": n_none, "n_border": n_border, "work_est": round(n_none + 0.5 * n_border, 2),
                         "auto_frac": round(n_auto / n_uncat, 3) if n_uncat else 1.0, "done": n_uncat == 0,
                         "top_class": hist.most_common(1)[0][0] if hist else None, "n_pred_classes": len(hist),
                         "_frac": {cn: cnt / n_uncat for cn, cnt in hist.items()} if n_uncat else {}})
        if order == "hard":
            rows.sort(key=lambda r: (r["done"], -r["work_est"], -r["n_uncat"], r["image_id"]))
        else:
            rows.sort(key=lambda r: (r["done"], r["work_est"], -r["auto_frac"], r["n_uncat"], r["image_id"]))
        if diversity > 0:                                     # spread the non-done head across predicted classes
            done = [r for r in rows if r["done"]]
            rows = self._diversify_by_class([r for r in rows if not r["done"]],
                                            dc.get("glob", {}), order, diversity, limit) + done
        items = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows[:limit]]
        return {**base, "threshold": round(T, 4), "margin": round(margin, 4) if margin is not None else None,
                "truncated": int(dc["truncated"]), "n_total": len(rows), "items": items}

    def _partition_member_preds(self, pid, *, gate_mult: float = 1.0, thr=None) -> dict:
        """Per-member 1-NN predictions for a partition — the SINGLE source of truth for the Partitions-tab
        markers, the clickable class/reject subset filter, AND the aggregate 'most likely class'. Returns a
        lean items map `{iuid: {label, pred, score, assigned}}` (label = class name | 'reject' | 'none';
        assigned = current category name or None) PLUS the aggregate vote, computed once and cached on
        (pid, gate, thr, mutation, coll_version, spec) so text/markers/filter/apply always agree. Read-only;
        never raises (n/a + error come back as fields). Members capped at _PMP_CAP (truncation surfaced)."""
        from collections import Counter
        spec_raw = self._cluster["spec"] if self._cluster else {"decoder": 1.0}
        spec, dropped = self._present_spec_nanfree(spec_raw)
        key = (str(pid), round(float(gate_mult), 4), thr, self._mutation_serial,
               int(self.state.coll_version), tuple(sorted(spec.items())))
        c = getattr(self, "_pmp_cache", None)
        if c is not None and c[0] == key:
            return c[1]
        base = {"pid": str(pid), "spec": spec, "dropped_features": dropped, "gate_mult": float(gate_mult),
                "items": {}, "n_classes": 0, "has_reject": False, "threshold": None, "margin": None,
                "n_total": 0, "truncated": 0, "top_class": None, "confidence": 0.0, "reject_likelihood": 0.0,
                "none_fraction": 0.0, "n_members": 0, "median_nearest_dist": None, "verdict": "n/a"}
        def _cache(out):
            self._pmp_cache = (key, out)
            return out
        if not spec:
            return _cache({**base, "error": "no usable (NaN-free) features in the clustering space"})
        refs = self._suggestion_refs(spec)
        if refs["index"] is None:
            return _cache({**base, "note": "no labels yet"})
        base["n_classes"], base["has_reject"] = refs["n_classes"], refs["has_reject"]
        members = [u for u in self.partition_iuids(pid) if u in self.state.meta]
        n_total = len(members)
        base["n_total"], base["truncated"] = n_total, max(0, n_total - _PMP_CAP)
        members = members[:_PMP_CAP]
        if not members:
            return _cache({**base, "note": "empty partition"})
        margin = refs["margin"]
        threshold = float(thr) if thr is not None else (gate_mult * margin if margin is not None else 0.25)
        preds = self._predict_instances(members, spec, refs, threshold)
        items, raw_lab, near = {}, [], []
        for u, lab, dd in preds:
            raw_lab.append(lab)
            near.append(dd)
            m = self.state.meta.get(u)
            acid = m.assigned_class if (m and not m.is_background) else None
            items[u] = {"label": ("reject" if lab == "__reject__" else lab),
                        "pred": (None if lab in ("none", "__reject__") else lab),
                        "score": round(max(0.0, 1.0 - dd), 3) if np.isfinite(dd) else 0.0,
                        "assigned": (self.state.class_name(acid) or str(acid)) if acid is not None else None}
        votes = Counter(raw_lab)
        nmem = len(raw_lab)
        cls_votes = {k: v for k, v in votes.items() if k not in ("none", "__reject__")}
        top_class, top_n = (max(cls_votes.items(), key=lambda kv: kv[1]) if cls_votes else (None, 0))
        reject_n, none_n = votes.get("__reject__", 0), votes.get("none", 0)
        top_eff = top_n if top_class is not None else -1
        verdict = "class" if (top_eff >= reject_n and top_eff >= none_n) else ("reject" if reject_n >= none_n else "none")
        finite = [x for x in near if np.isfinite(x)]
        return _cache({**base, "items": items, "threshold": round(threshold, 4),
                       "margin": round(margin, 4) if margin is not None else None,
                       "top_class": top_class, "confidence": round(top_n / nmem, 3),
                       "reject_likelihood": round(reject_n / nmem, 3), "none_fraction": round(none_n / nmem, 3),
                       "n_members": nmem, "verdict": verdict,
                       "median_nearest_dist": round(float(np.median(finite)), 4) if finite else None})

    def partition_class_suggestion(self, pid, *, gate_mult: float = 1.0, thr=None) -> dict:
        """The selected partition's most likely class by 1-NN of its instances to the labeled instances of
        each class (rejected instances are a candidate too → a rejection likelihood). 'no likely class' when
        the nearest distance exceeds the gate. Read-only; the per-member items live behind
        `_partition_member_preds` (this returns the aggregate only)."""
        mp = self._partition_member_preds(pid, gate_mult=gate_mult, thr=thr)
        out = {k: mp[k] for k in ("pid", "spec", "dropped_features", "gate_mult", "verdict", "top_class",
                                  "confidence", "reject_likelihood", "none_fraction", "n_members",
                                  "median_nearest_dist", "threshold", "margin", "n_classes", "has_reject")}
        for k in ("error", "note"):
            if k in mp:
                out[k] = mp[k]
        return out

    def partition_iuids_predicted(self, pid, *, label: str, gate_mult: float = 1.0, thr=None) -> list[str]:
        """The partition's members whose 1-NN predicted label == `label` (a class name, 'reject', or 'none'),
        in display order — backs the clickable class/reject subset filter."""
        items = self._partition_member_preds(pid, gate_mult=gate_mult, thr=thr).get("items", {})
        return [u for u in self.partition_iuids(pid) if items.get(u, {}).get("label") == label]

    def accept_partition_subset(self, pid, *, label: str, gate_mult: float = 1.0, thr=None) -> dict:
        """APPLY one predicted bucket of a partition: assign the members predicted as class `label` to that
        class, or background those predicted 'reject'. Skips already-categorized members (like
        accept_image_predictions). 'none' is a no-op. Undoable."""
        if label == "none":
            return {"action": "none", "n": 0, "skipped_assigned": 0}
        items = self._partition_member_preds(pid, gate_mult=gate_mult, thr=thr).get("items", {})
        matching = [u for u in self.partition_iuids(pid) if items.get(u, {}).get("label") == label]
        fresh = [u for u in matching
                 if (m := self.state.meta.get(u)) is not None and m.assigned_class is None and not m.is_background]
        skipped = len(matching) - len(fresh)
        if label == "reject":
            if fresh:
                self.set_background(fresh)
            return {"action": "reject", "n": len(fresh), "skipped_assigned": skipped}
        if fresh:
            self.assign(fresh, label, source="suggestion")
        return {"action": "assign", "cls": label, "n": len(fresh), "skipped_assigned": skipped}

    def accept_partition_suggestion(self, pid, *, gate_mult: float = 1.0, thr=None) -> dict:
        """One-click APPLY of a partition's recommendation: assign the whole partition to the most-likely
        class, or reject the whole partition, or do nothing ('no likely class'). Undoable."""
        s = self.partition_class_suggestion(pid, gate_mult=gate_mult, thr=thr)
        verdict, cls = s.get("verdict"), s.get("top_class")
        if verdict == "class" and cls:
            iuids = self.partition_iuids(pid)
            self.assign(iuids, cls, source="suggestion")
            return {"action": "assign", "cls": cls, "n": len(iuids), "verdict": verdict}
        if verdict == "reject":
            return {"action": "reject", "n": self.reject_partition(pid), "verdict": verdict}
        return {"action": "none", "n": 0, "verdict": verdict}

    def accept_image_predictions(self, image_id, *, gate_mult: float = 1.0, thr=None) -> dict:
        """One-click APPLY of an image's per-instance recommendations: assign each UNcategorized instance to
        its predicted class, reject those predicted 'reject', and leave 'no likely class' ones untouched.
        Instances that ALREADY have a category (or are rejected) are NOT touched — they fall outside the
        accept gate. Undoable."""
        from collections import defaultdict
        s = self.image_class_suggestion(image_id, gate_mult=gate_mult, thr=thr)
        by_cls, rej, skipped_assigned, skipped_none = defaultdict(list), [], 0, 0
        for it in s.get("items", []):
            u = it["iuid"]; m = self.state.meta.get(u)
            if m is None:
                continue
            if m.assigned_class is not None or m.is_background:   # already categorized -> outside the accept gate
                skipped_assigned += 1
                continue
            if it["label"] == "reject":
                rej.append(u)
            elif it["label"] == "none":
                skipped_none += 1
            else:
                by_cls[it["label"]].append(u)
        assigned = {}
        for cls, ius in by_cls.items():
            self.assign(ius, cls, source="suggestion"); assigned[cls] = len(ius)
        if rej:
            self.set_background(rej)
        return {"assigned": assigned, "rejected": len(rej),
                "skipped": skipped_none, "skipped_assigned": skipped_assigned}

    @_mutating
    def unreject(self, iuids: list[str]) -> int:
        """Send rejected (background) instances back to UNASSIGNED. Reversible."""
        bg = [u for u in iuids if u in self.state.meta and self.state.meta[u].is_background]
        if not bg:
            return 0
        before = self._before_states(bg)
        tok = self.history.begin(self.state, bg, [])
        for u in bg:
            self.state.meta[u].is_background = False
            self.state.meta[u].assigned_class = None
        self.history.commit(self.state, tok, "unreject", f"unreject {len(bg)}")
        self._after_mutation()
        self._cache_delta(before)
        return len(bg)

    def reset(self, *, keep_config: bool = True) -> dict:
        """Drop EVERYTHING (collection, instances, assignments, classes, overlays, caches, ingest registry,
        merge/lineage/history logs, processed-image list); keep only the config. Destructive, NOT undoable."""
        cfg = dict(self.state.config) if keep_config else {}
        if self.store.collection_path.exists():
            self.store.collection_path.unlink()
        self.store.clear_cache()
        self.store.clear_collection_shards()
        for f in self.store.refine_dir.glob("*.pkl"):
            f.unlink()
        for p in (self.store.ingests_path, self.store.merge_log_path,         # logs that index now-deleted
                  self.store.lineage_path, self.store.history_path):           # instances by iuid
            if p.exists():
                p.unlink()
        man = self.store.load_manifest()
        man.update({"processed_paths": [], "coll_version": 0, "n_instances": 0})
        self.store.save_manifest(man)
        self.state = CuratorState(project_dir=str(self.store.dir), config=cfg)
        self.collection = None
        self._overlay_rle = {}
        self._cluster = self._subcluster = None
        self._grp_cache = self._index = None
        self._fused_cache = {}
        self._proba_cache = None
        self._clf = self._merge_clf = self._ref_bank = None
        self._scope_bids = self._scope_id = None
        self._source_filter = None; self._bsrc_cache = None
        self._granularity_filter = self._modality_filter = None
        self._scope_token += 1
        self.history = History(self.store)
        self.save()
        return self.stats()

    # ---- refinement --------------------------------------------------------
    def refine_preview(self, iuid: str, ops: list[dict], *, mask_overlay: bool = True,
                       pad: int = 12, max_side: int = 512, context: bool = False):
        """Before/after crops on a SHARED, aligned window (the union bbox of base & refined; the WHOLE image
        when context=True — the 'c'-toggle in-context view). The 'after' panel is a DIFF overlay so even a
        tiny change is obvious and removals stay visible: YELLOW = unchanged, GREEN = added, RED = removed."""
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
        if context:                                          # in-context view: the whole image, instance highlighted
            x1, y1, x2, y2 = 0, 0, W, H
        else:
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

    def sam_prompt_preview(self, iuid: str, ops: list[dict], *, n_pos: int = 1, n_neg: int = 0,
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

    @_mutating
    def apply_refine(self, iuid: str, ops: list[dict]) -> None:
        tok = self.history.begin(self.state, [iuid], [])
        self._refine_one_nohist(iuid, ops)
        self.history.commit(self.state, tok, "refine", f"refine {iuid[:6]}")
        self._after_mutation()

    @_mutating
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

    @_mutating
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

    # ---- few-shot shape transfer (reference mask -> partition peers, SAM/SAM-HQ within each bbox) -----
    @staticmethod
    def _shape_template(masks: list, size: int = 256):
        """Average bbox-normalized soft template from k reference masks (each cropped to its own bbox and
        resized to size×size). None if no reference has any foreground. This is the k-shot shape prior."""
        import cv2
        acc, n = None, 0
        for m in masks:
            if m is None or not m.any():
                continue
            ys, xs = np.where(m)
            crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
            t = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
            acc = t if acc is None else acc + t
            n += 1
        return None if n == 0 else (acc / n)

    @staticmethod
    def _warp_template_to_box(T: np.ndarray, box_xyxy, H: int, W: int, thresh: float = 0.5) -> np.ndarray:
        """Resize the soft template into the instance's pixel box and threshold -> a full-(H,W) boolean
        'expected mask' E in image coords (the reference shape placed in this instance's bounding box)."""
        import cv2
        x1, y1, x2, y2 = (int(round(float(v))) for v in box_xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, max(x1 + 1, x2)), min(H, max(y1 + 1, y2))
        warped = cv2.resize(T, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        E = np.zeros((H, W), bool)
        E[y1:y2, x1:x2] = warped >= float(thresh)
        return E

    def _set_mask_nohist(self, iuid: str, mask: np.ndarray, *, op: dict) -> None:
        """Write an externally-produced mask (transfer / hand-draw) as the instance's EFFECTIVE mask via the
        same overlay path as _refine_one_nohist — so undo, _eff_rle gating (on meta.refined) and the shape-
        feature recompute all hold. `op` = the provenance op record (e.g. {"name":"draw","kw":{...}}).
        meta.refined is load-bearing (the overlay is ignored without it)."""
        from pycocotools import mask as mu
        base = self._refine_base_rle(iuid)
        m = mask.astype(np.uint8)
        rle = mu.encode(np.asfortranarray(m)); rle["counts"] = rle["counts"].decode("ascii")
        meta = self.state.meta[iuid]
        meta.refined = True
        meta.rule_ops = [dict(op)]
        meta.provenance = {**(meta.provenance or {}), op.get("name", "set_mask"): dict(op.get("kw", {}))}
        self._overlay_rle[iuid] = rle
        self.store.save_refine(iuid, {"iuid": iuid, "base_rle": base, "ops": meta.rule_ops, "result_rle": rle})
        if "shapecoord" in self.collection["feats"]:
            self.collection["feats"]["shapecoord"][meta.row] = _co.shapecoord_vector(mask.astype(bool))

    def edit_view(self, iuid: str, *, context: bool = False, pad: int = 16, max_side: int = 640,
                  zoom_cap: float = 8.0) -> dict:
        """Image + current-mask + mapping for the hand-draw editor. Returns the instance's bbox crop (or the
        WHOLE image when context=True / the mask is empty) as display-resolution arrays, scaled toward max_side
        (UP to zoom_cap× for small crops → precise pixel work, down for big ones). `box` is the full-image
        pixel rect the canvas covers; the client paints at (w,h) and posts that back to /api/set_mask."""
        import cv2
        img = self._rgb(iuid); H, W = img.shape[:2]
        m = self._mask(iuid)
        if context or not m.any():
            x1, y1, x2, y2 = 0, 0, W, H
        else:
            ys, xs = np.where(m)
            x1, y1 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
            x2, y2 = min(W, int(xs.max()) + pad + 1), min(H, int(ys.max()) + pad + 1)
        longest = max(1, max(x2 - x1, y2 - y1))
        scale = min(float(zoom_cap), float(max_side) / longest)
        dw, dh = max(1, int(round((x2 - x1) * scale))), max(1, int(round((y2 - y1) * scale)))
        interp = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
        disp_img = cv2.resize(img[y1:y2, x1:x2], (dw, dh), interpolation=interp)
        disp_m = cv2.resize((m[y1:y2, x1:x2].astype(np.uint8) * 255), (dw, dh), interpolation=cv2.INTER_NEAREST)
        return {"img": disp_img, "mask": disp_m, "box": [x1, y1, x2, y2], "w": dw, "h": dh, "context": bool(context)}

    @_mutating
    def set_mask(self, iuid: str, png_bytes: bytes, box) -> dict:
        """Write a hand-drawn mask (a canvas-resolution binary PNG covering `box` in full-image pixel coords)
        as the instance's EFFECTIVE mask, undoably. Pixels OUTSIDE `box` keep the current mask, so editing the
        zoomed crop never erases structure beyond it (full-image edits pass box = whole image)."""
        import cv2
        if iuid not in self.state.meta:
            return {"error": "unknown instance"}
        arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return {"error": "could not decode mask image"}
        full = self._mask(iuid).copy(); H, W = full.shape
        x1, y1, x2, y2 = (int(round(float(v))) for v in (box or [0, 0, W, H]))
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, max(x1 + 1, x2)), min(H, max(y1 + 1, y2))
        full[y1:y2, x1:x2] = cv2.resize(arr, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) > 127
        tok = self.history.begin(self.state, [iuid], [])
        self._set_mask_nohist(iuid, full, op={"name": "draw", "kw": {"box": [x1, y1, x2, y2]}})
        self.history.commit(self.state, tok, "draw_mask", f"draw {iuid[:6]}")
        self._after_mutation()
        return {"iuid": iuid, "area": int(full.sum())}

    def shape_transfer_members(self, ref_iuids: list, *, pid=None, match_thresh=None):
        """(pid, members, gate_skipped) for a transfer: the partition of the first reference (or `pid`) minus
        the references, optionally RAD-DINO τ-gated to members whose embedding cosine-sim to ANY reference is
        >= match_thresh (so a heterogeneous partition isn't mangled). Mirrors propagate_refinement's gate."""
        refs = [u for u in ref_iuids if u in self.state.meta]
        if not refs:
            return None, [], 0
        pid = str(pid) if pid is not None else self.partition_of(refs[0])
        if pid is None:
            return None, [], 0
        refset = set(refs)
        members = [u for u in self.partition_iuids(pid) if u not in refset]
        skipped = 0
        if match_thresh is not None and members:
            embs = self._instance_ref_embeddings(refs + members)
            R = embs[:len(refs)] / (np.linalg.norm(embs[:len(refs)], axis=1, keepdims=True) + 1e-8)
            M = embs[len(refs):] / (np.linalg.norm(embs[len(refs):], axis=1, keepdims=True) + 1e-8)
            sims = (M @ R.T).max(axis=1)                      # best similarity to ANY reference
            kept = [u for u, s in zip(members, sims.tolist()) if s >= float(match_thresh)]
            skipped = len(members) - len(kept); members = kept
        return pid, members, skipped

    def _transfer_one(self, iuid: str, T, *, sam_model: str = "auto", line_ops=None):
        """COMPACT: warp the template into this instance's bbox, then SAM/SAM-HQ-decode toward it (sam_refine
        derives points + box + mask_input FROM the warped shape). LINE: run the reference-calibrated vessel
        trace (`line_ops`) on the member's OWN seed mask via apply_ops — vesselness re-traces the tube from the
        image, NO SAM (a bbox/template is the wrong prior for a thin curve). Returns (orig_mask, cand_mask)."""
        from .refine import apply_ops, sam_refine, to_gray
        gray = to_gray(self._rgb(iuid))
        orig = self._mask(iuid)
        if line_ops is not None:
            return orig, apply_ops(gray, orig, line_ops).astype(bool)
        H, W = gray.shape[:2]
        nb = self._instance_crop_box_norm(iuid)
        E = self._warp_template_to_box(T, (nb[0] * W, nb[1] * H, nb[2] * W, nb[3] * H), H, W)
        if not E.any():
            return orig, orig
        cand = sam_refine(gray, E, model=sam_model, use_mask_prompt=True, union=False).astype(bool)
        return orig, cand

    def _partition_shape_kind(self, refs, members, sample: int = 24) -> str:
        """'line' or 'blob' for the partition (autorefine.partition_kind, robust to a lying member) — decides
        whether to transfer a SHAPE (compact: template + SAM) or re-trace a TUBE (line: vessel_extend)."""
        from .autorefine import partition_kind
        masks = [self._mask(u) for u in (list(refs) + list(members)[:int(sample)])]
        masks = [m for m in masks if m is not None and m.any()]
        return partition_kind(masks) if masks else "blob"

    def _reference_line_ops(self, refs):
        """Reference-calibrated line chain: vessel_extend (grow/bridge the tube along vesselness) + line_centerline
        (one clean path), tube width = median(area / skeleton-length) over the references. The few-shot signal for
        a line class is its WIDTH/connectivity, not a bbox-normalized shape. Returns (ops, width)."""
        from skimage.morphology import skeletonize
        ws = []
        for u in refs:
            m = self._mask(u)
            if m is None or not m.any():
                continue
            sk = skeletonize(m); L = int(sk.sum())
            ws.append((float(m.sum()) / L) if L else 4.0)
        w = int(max(2, round(float(np.median(ws))))) if ws else 8
        return [{"name": "vessel_extend", "kw": {"max_width": w}},
                {"name": "line_centerline", "kw": {"alpha": 0.7, "width": w}}], w

    def _diff_panels(self, iuid: str, base_m: np.ndarray, refined: np.ndarray, *, pad: int = 12,
                     max_side: int = 384):
        """before/after DIFF overlay arrays for an arbitrary (base, refined) pair — same coloring as
        refine_preview (yellow=unchanged, green=added, red=removed) but for a transferred mask."""
        import cv2
        img = self._rgb(iuid)
        H, W = base_m.shape
        ys, xs = np.where(base_m | refined)
        if len(xs) == 0:
            z = _downscale(img.copy(), max_side); return z, z
        x1, y1 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
        x2, y2 = min(W, int(xs.max()) + pad + 1), min(H, int(ys.max()) + pad + 1)
        b, a = base_m[y1:y2, x1:x2], refined[y1:y2, x1:x2]

        def _blend(sub, region, col):
            if region.any():
                sub[region] = (0.45 * sub[region] + 0.55 * np.array(col)).astype(np.uint8)

        def _outline(sub, mm, col):
            cont, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sub, cont, -1, col, 1)

        before = img[y1:y2, x1:x2].copy(); after = img[y1:y2, x1:x2].copy()
        _blend(before, b, (40, 220, 40)); _outline(before, b, (40, 220, 40))
        _blend(after, b & a, (255, 220, 0)); _blend(after, a & ~b, (40, 220, 40)); _blend(after, b & ~a, (235, 50, 40))
        _outline(after, a, (40, 220, 40))
        return _downscale(before, max_side), _downscale(after, max_side)

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        u = int((a | b).sum())
        return (int((a & b).sum()) / u) if u else 0.0

    def shape_transfer_preview(self, ref_iuids: list, *, pid=None, match_thresh=None, sam_model="auto",
                               agree_iou=None, sample: int = 12) -> dict:
        """DRY-RUN the transfer over up to `sample` partition members: per member warp->SAM-decode->IoU vs the
        original, with before/after panels (numpy arrays — the server base64-encodes them). NO writes. Backs
        the mandatory preview before a bulk commit."""
        refs = [u for u in ref_iuids if u in self.state.meta]
        if not refs:
            return {"error": "no reference instance(s) given"}
        pid, members, gate_skipped = self.shape_transfer_members(refs, pid=pid, match_thresh=match_thresh)
        if pid is None:
            return {"error": "the reference is not in a partition (cluster or assign it first)"}
        kind = self._partition_shape_kind(refs, members)
        T, line_ops, width = None, None, None
        if kind == "line":
            line_ops, width = self._reference_line_ops(refs)
        else:
            T = self._shape_template([self._mask(u) for u in refs])
            if T is None:
                return {"error": "reference mask(s) are empty"}
        shown = members[:int(sample)]
        items = []
        for u in shown:
            orig, cand = self._transfer_one(u, T, sam_model=sam_model, line_ops=line_ops)
            iou = self._mask_iou(orig, cand)
            keep = (agree_iou is None) or (iou >= float(agree_iou))
            before, after = self._diff_panels(u, orig, cand)
            items.append({"iuid": u, "iou": round(iou, 3), "keep": bool(keep),
                          "before": before, "after": after})
        return {"pid": pid, "kind": kind, "width": width, "n_members": len(members),
                "gate_skipped": int(gate_skipped), "shown": len(shown),
                "truncated": max(0, len(members) - len(shown)), "items": items,
                "agree_iou": (None if agree_iou is None else float(agree_iou))}

    @_mutating
    def shape_transfer(self, ref_iuids: list, *, pid=None, match_thresh=None, sam_model="auto",
                       agree_iou=None) -> dict:
        """COMMIT the transfer: per member warp->SAM-decode, drop members below `agree_iou` vs the original,
        write the rest as one undoable command (mirror apply_refine_many). sam_refine raises (-> server 400)
        when SAM/checkpoint is missing."""
        refs = [u for u in ref_iuids if u in self.state.meta]
        if not refs:
            return {"error": "no reference instance(s) given"}
        pid, members, gate_skipped = self.shape_transfer_members(refs, pid=pid, match_thresh=match_thresh)
        if pid is None:
            return {"error": "the reference is not in a partition (cluster or assign it first)"}
        kind = self._partition_shape_kind(refs, members)
        T, line_ops, width = None, None, None
        if kind == "line":
            line_ops, width = self._reference_line_ops(refs)
        else:
            T = self._shape_template([self._mask(u) for u in refs])
            if T is None:
                return {"error": "reference mask(s) are empty"}
        prov = {"refs": refs, "kind": kind, "width": width, "sam": sam_model,
                "match_thresh": match_thresh, "agree_iou": agree_iou}
        masks, gated_out = [], 0
        for u in members:
            orig, cand = self._transfer_one(u, T, sam_model=sam_model, line_ops=line_ops)
            if agree_iou is not None and self._mask_iou(orig, cand) < float(agree_iou):
                gated_out += 1; continue
            masks.append((u, cand))
        if masks:
            ius = [u for u, _ in masks]
            tok = self.history.begin(self.state, ius, [])
            for u, cand in masks:
                self._set_mask_nohist(u, cand, op={"name": "shape_transfer", "kw": prov})
            self.history.commit(self.state, tok, "shape_transfer", f"shape transfer ({kind}) to {len(masks)} in {pid}")
            self._after_mutation()
        return {"applied": len(masks), "skipped": int(gate_skipped), "gated_out": int(gated_out),
                "pid": pid, "kind": kind, "width": width, "refs": refs}

    def propagate_refinement(self, ref_iuid: str, *, pid=None, ops=None, match_thresh=None) -> dict:
        """Within-partition propagation: replay a refine op-chain across a partition so its instances get the
        SAME segmentation treatment as a refined REFERENCE instance (e.g. SAM-HQ for a device, vessel_extend
        for a line). `ops` defaults to the reference's recorded chain (`meta.rule_ops`); `pid` defaults to the
        reference's own partition. When `match_thresh` is given, gate to members whose RAD-DINO embedding
        cosine-similarity to the reference is >= the threshold, so a heterogeneous partition isn't mangled by
        one recipe (skipped members are left untouched). One undoable command via apply_refine_many."""
        if ref_iuid not in self.state.meta:
            return {"error": "unknown reference instance"}
        ops = list(ops) if ops else (self.state.meta[ref_iuid].rule_ops or [])
        if not ops:
            return {"error": "refine the reference instance first — it has no recorded op-chain to propagate"}
        pid = str(pid) if pid is not None else self.partition_of(ref_iuid)
        if pid is None:
            return {"error": "the reference instance is not in a partition (cluster or assign it first)"}
        members = [u for u in self.partition_iuids(pid) if u != ref_iuid]
        skipped = 0
        if match_thresh is not None and members:
            embs = self._instance_ref_embeddings([ref_iuid] + members)   # RAD-DINO (GPU/HF); cached per instance
            r = embs[0] / (np.linalg.norm(embs[0]) + 1e-8)
            sims = (embs[1:] / (np.linalg.norm(embs[1:], axis=1, keepdims=True) + 1e-8)) @ r
            kept = [u for u, s in zip(members, sims.tolist()) if s >= float(match_thresh)]
            skipped = len(members) - len(kept); members = kept
        applied = self.apply_refine_many(members, ops) if members else 0
        return {"applied": int(applied), "skipped": int(skipped), "pid": pid,
                "ops": [o.get("name") for o in ops], "matched": match_thresh is not None}

    # ---- Stage-1 auto-refine (label-free chain search, IN CATEGORY CONTEXT) --
    def _member_masks(self, iuids: list[str], cap: int = 48) -> list:
        from pycocotools import mask as mu
        return [mu.decode(self._refine_base_rle(u)).astype(bool) for u in iuids[:cap] if u in self.state.meta]

    def _class_prior_reward(self, iuids: list[str]):
        """A per-class shape-prior reward when the members share ONE assigned class AND a prior exists for it
        (config `shape_prior.prior_dir` + a `dae_cls<catid>.pth`; catid via `shape_prior.class_to_catid` or a
        numeric class id). Returns (reward_fn, name) — (None, 'geometric') when unavailable (the common case)."""
        import os
        cfg = (self.state.config.get("shape_prior") or {}) if isinstance(self.state.config, dict) else {}
        prior_dir = cfg.get("prior_dir")
        if not prior_dir or not os.path.isdir(str(prior_dir)):
            return None, "geometric"
        cids = {self.state.meta[u].assigned_class for u in iuids if u in self.state.meta}
        cids = {c for c in cids if c is not None}
        if len(cids) != 1:
            return None, "geometric"
        cid = next(iter(cids))
        catid = (cfg.get("class_to_catid") or {}).get(str(cid))
        if catid is None and str(cid).lstrip("-").isdigit():
            catid = int(cid)
        if catid is None:
            return None, "geometric"
        try:
            from .core.shape_prior import load_priors_by_catid

            from .autorefine import shape_prior_reward
            pri = load_priors_by_catid(str(prior_dir), [int(catid)], "cpu")
            if int(catid) not in pri:
                return None, "geometric"
            return shape_prior_reward(pri[int(catid)], device="cpu"), "shape-prior"
        except Exception:
            return None, "geometric"

    def _category_context(self, iuids: list[str]) -> tuple:
        """(kind, reward_fn, reward_name) for a partition/class: kind from the members' AGGREGATE geometry
        (robust to a lying instance) + a per-class shape-prior reward when available. This is the context the
        optimal mask depends on — computed once for the whole category, then reused for every member."""
        from . import autorefine as ar
        masks = self._member_masks(iuids)
        kind = ar.partition_kind(masks) if masks else "auto"
        reward_fn, reward_name = self._class_prior_reward(iuids)
        return kind, reward_fn, reward_name

    def _instance_context(self, iuid: str, kind: str) -> tuple:
        """Context for a SINGLE instance: an explicit `kind` wins; else derive it from the instance's
        category — its assigned class, else its partition — falling back to the lone mask ('auto')."""
        if kind != "auto":
            return kind, None, "geometric"
        cid = self.state.meta[iuid].assigned_class
        if cid is not None:
            return self._category_context(self.class_rule_members(cid))
        pid = self.partition_of(iuid)
        if pid is not None:
            return self._category_context(self.partition_iuids(pid))
        return "auto", None, "geometric"

    def auto_refine_search(self, iuid: str, *, kind: str = "auto", reward_fn=None,
                           reward_name: str = "geometric") -> dict:
        """Pick the chain THIS mask needs by label-free search (no mutation). See autorefine.search."""
        from pycocotools import mask as mu

        from . import autorefine as ar
        from .refine import to_gray
        base = self._refine_base_rle(iuid)
        return ar.search(to_gray(self._rgb(iuid)), mu.decode(base).astype(bool),
                         kind=kind, reward_fn=reward_fn, reward_name=reward_name)

    def auto_refine_preview(self, iuid: str, *, kind: str = "auto"):
        """Before/after crops for the auto-chosen chain (decided in the instance's category context)."""
        k, rf, rn = self._instance_context(iuid, kind)
        res = self.auto_refine_search(iuid, kind=k, reward_fn=rf, reward_name=rn)
        before, after = self.refine_preview(iuid, res["best"]["chain"])
        return before, after, res

    def auto_refine_apply(self, iuid: str, *, kind: str = "auto") -> dict:
        k, rf, rn = self._instance_context(iuid, kind)
        res = self.auto_refine_search(iuid, kind=k, reward_fn=rf, reward_name=rn)
        self.apply_refine(iuid, res["best"]["chain"])
        return res

    @_mutating
    def auto_refine_many(self, iuids: list[str], *, kind: str = "auto") -> dict:
        """Per-instance best chain (each mask gets its OWN argmax) under ONE shared category context, applied
        in one undoable command. Returns the chain histogram — the supervision a Stage-2 policy would imitate."""
        from collections import Counter
        iuids = [u for u in iuids if u in self.state.meta]
        if not iuids:
            return {"n": 0, "kind": kind, "reward": "geometric", "summary": []}
        k, rf, rn = (kind, None, "geometric") if kind != "auto" else self._category_context(iuids)
        tok = self.history.begin(self.state, iuids, [])
        sig = Counter()
        for u in iuids:
            chain = self.auto_refine_search(u, kind=k, reward_fn=rf, reward_name=rn)["best"]["chain"]
            self._refine_one_nohist(u, chain)
            sig[tuple(o.get("name") for o in chain) or ("(none)",)] += 1
        self.history.commit(self.state, tok, "auto_refine", f"auto-refine {len(iuids)} instances")
        self._after_mutation()
        return {"n": len(iuids), "kind": k, "reward": rn,
                "summary": [{"chain": list(key), "n": c} for key, c in sig.most_common()]}

    def auto_refine_partition(self, pid, *, kind: str = "auto") -> dict:
        return self.auto_refine_many(self.partition_iuids(str(pid)), kind=kind)

    def auto_refine_class(self, cls: str, *, kind: str = "auto") -> dict:
        cid = self._resolve_cid((cls or "").strip())
        return self.auto_refine_many(self.class_rule_members(cid), kind=kind) if cid else {"n": 0, "summary": []}

    def auto_refine_consensus(self, iuids: list[str], *, kind: str = "auto") -> dict:
        """Borrow strength across the category: search every member, return the MODAL full chain (the class's
        dominant recipe) + the vote tally. No mutation — for previewing a class-wide rule before committing."""
        import json
        from collections import Counter
        iuids = [u for u in iuids if u in self.state.meta]
        if not iuids:
            return {"n": 0, "kind": kind, "reward": "geometric", "chain": [], "votes": 0, "summary": []}
        k, rf, rn = (kind, None, "geometric") if kind != "auto" else self._category_context(iuids)
        votes, rep = Counter(), {}
        for u in iuids:
            chain = self.auto_refine_search(u, kind=k, reward_fn=rf, reward_name=rn)["best"]["chain"]
            key = json.dumps(chain, sort_keys=True)
            votes[key] += 1; rep[key] = chain
        best_key, n = votes.most_common(1)[0]
        return {"n": len(iuids), "kind": k, "reward": rn, "chain": rep[best_key], "votes": n,
                "summary": [{"chain": [o.get("name") for o in rep[key]] or ["(none)"], "n": c}
                            for key, c in votes.most_common()]}

    def auto_refine_class_consensus(self, cls: str, *, kind: str = "auto") -> dict:
        """Apply the class's MODAL chain uniformly to all its instances and SAVE it as the class rule."""
        cid = self._resolve_cid((cls or "").strip())
        if cid is None:
            return {"n": 0, "chain": [], "summary": [], "applied": 0}
        members = self.class_rule_members(cid)
        con = self.auto_refine_consensus(members, kind=kind)
        self.state.class_rules[cid] = list(con["chain"])
        applied = self.apply_refine_many(members, con["chain"])
        self.save()
        return {**con, "applied": int(applied), "saved_rule": True}

    def refine_partition_preview(self, pid: int, ops: list[dict], n: int = 6):
        """Before/after crops for the first n instances of a partition (no persistence)."""
        befores, afters = [], []
        for u in self.partition_iuids(pid)[:n]:
            o, r = self.refine_preview(u, ops)
            befores.append((o, u[:6])); afters.append((r, u[:6]))
        return befores, afters

    @_mutating
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
                          "split_from": u,                    # parent iuid: real provenance + lets a child inherit its kind
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
            parent = self.state.meta.get(r.get("split_from") or "")
            self.state.meta[r["iuid"]] = InstanceMeta(
                iuid=r["iuid"], batch_id=r["batch_id"], row=r["row"], image_id=int(r["image_id"]),
                granularity=parent.granularity if parent else self.state.mode(),
                modality=parent.modality if parent else self.state.modality(),
                provenance={"split_from": r.get("split_from", ""), "file": r.get("abs_path", "")})
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

    @_mutating
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
        spec, dropped_nan = self._present_spec_nanfree(spec)
        if not spec:
            ok = [m for m in self.available_features() if m not in self.feature_nan_methods()]
            return {"error": (f"no usable (present, NaN-free) features selected. "
                              f"dropped for NaN: {dropped_nan or '—'}; NaN-free available: {ok}")}
        if algo == "knn":
            clf, report = _clf.train_knn(self.collection, self.state, spec, k=int(knn_k), metric=knn_metric,
                                         weights=knn_weights, use_unassigned_negatives=use_unassigned_negatives)
        else:
            clf, report = _clf.train_factored(self.collection, self.state, spec, algo=algo,
                                              use_unassigned_negatives=use_unassigned_negatives)
        if report.get("skipped_classes"):
            report["skipped_names"] = [self.state.class_name(c) for c in report["skipped_classes"]]
        if dropped_nan:
            report["dropped_nan_features"] = dropped_nan      # excluded so NaN can't break the fit/predict
        if clf is None:
            return report
        self._clf = clf
        self._clf_spec = spec                                 # already NaN-clean -> predict/apply stay clean
        self._clf_version += 1                                # invalidate the cached unassigned-pool proba
        self._proba_cache = None
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

    # ---- latent-space Map: 2D/3D projection of instances (Spacewalker-style lens over the SAME embeddings) --
    def project(self, spec=None, *, method: str = "hnne", dims: int = 2) -> dict:
        """2D/3D embedding of ALL in-scope LIVE instances from the fused `spec` space, for the latent Map.
        Label-INDEPENDENT (features only) -> cached on (coll_version, scope, spec, method, dims); recomputes only
        on ingest / re-cluster / scope change, never on a label. h-NNE preferred, falls back UMAP -> PCA."""
        spec_raw = spec if spec is not None else (self._cluster["spec"] if self._cluster else {"decoder": 1.0})
        spec, dropped = self._present_spec_nanfree(spec_raw)
        if not spec:
            return {"error": "no usable (NaN-free) features in the requested space", "dropped": dropped}
        iuids = sorted(u for u, m in self.state.meta.items() if m.merged_into is None and self._in_scope(u))
        truncated = max(0, len(iuids) - _PROJ_CAP)
        iuids = iuids[:_PROJ_CAP]
        key = (int(self.state.coll_version), self._scope_token, tuple(sorted(spec.items())),
               str(method), int(dims), len(iuids))
        c = getattr(self, "_proj_cache", None)
        if c is not None and c[0] == key:
            return c[1]
        if not iuids:
            out = {"iuids": [], "coords": np.zeros((0, int(dims)), np.float32), "method": method,
                   "dims": int(dims), "spec": spec, "dropped": dropped, "truncated": 0}
            self._proj_cache = (key, out); return out
        rows = [self.state.meta[u].row for u in iuids]
        X = self.fused(spec)[rows]
        Xn = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
        Y, used, reducer = self._project_embed(Xn, str(method), int(dims))
        Y = np.asarray(Y, np.float32)

        # Keep what a query needs: the reducer, the coordinate frame AS FITTED, and the per-block
        # fusion statistics. Recomputing any of the three at query time puts the point in the wrong
        # place — see chevron/projection.py.
        from .projection import FittedProjection, block_stats
        fit = FittedProjection(
            reducer=reducer, method=used, dims=int(dims), spec=dict(spec),
            block_stats=block_stats({m: self.collection["feats"][m][rows] for m in spec}, spec),
            coord_min=Y.min(0), coord_max=Y.max(0), n_fit=len(iuids), truncated=int(truncated))
        self._fit = fit
        try:
            fit.save(self.store.dir / "dr" / f"{used}_{int(dims)}d.joblib")
        except Exception:
            pass                                  # persistence is a convenience; never fail the map on it

        out = {"iuids": iuids, "coords": Y, "method": used, "dims": int(dims), "spec": spec,
               "dropped": dropped, "truncated": int(truncated), "queryable": fit.queryable}
        self._proj_cache = (key, out)
        return out

    @staticmethod
    def _project_embed(X: np.ndarray, method: str, dims: int):
        """Embed X (already L2-normalized) to `dims`. h-NNE (if installed) -> UMAP -> PCA.

        Returns (coords, method_used, reducer). The REDUCER is kept, not discarded: it is what lets a
        new image or phrase be placed on the map afterwards. `reducer` is None when the embedding was
        trivial (degenerate N), which simply means the map cannot take queries.
        """
        n = X.shape[0]
        if n <= dims + 1:
            Y = np.zeros((n, dims), np.float32); Y[:, :min(dims, X.shape[1])] = X[:, :dims]
            return Y, "trivial", None
        if method == "hnne":
            try:
                from hnne import HNNE
                r = HNNE(dim=dims)
                return np.asarray(r.fit_transform(X)), "hnne", r
            except Exception:
                method = "umap"
        if method == "umap":
            try:
                import umap
                nn = int(min(15, max(2, n - 1)))
                r = umap.UMAP(n_components=dims, metric="cosine", n_neighbors=nn,
                              min_dist=0.1, random_state=0)
                return np.asarray(r.fit_transform(X)), "umap", r
            except Exception:
                method = "pca"
        from sklearn.decomposition import PCA
        r = PCA(n_components=int(dims), random_state=0)
        return np.asarray(r.fit_transform(X)), "pca", r

    def project_query(self, spec=None, *, iuid: str | None = None, text: str | None = None,
                      image_rgb=None, extractor: str | None = None, k: int = 12,
                      method: str = "hnne", dims: int = 2) -> dict:
        """Place a NEW point on the current map: an existing instance, a phrase, or an image.

        Returns its coordinates in the SAME normalised frame the map is drawn in, plus its nearest
        instances. Read-only — a query never changes the projection.
        """
        # Must project through the SAME spec the map was drawn with; defaulting here would refit a
        # different space and place the query on a map the user is not looking at.
        p = self.project(spec, method=method, dims=dims)
        if p.get("error"):
            return p
        fit = getattr(self, "_fit", None)
        if fit is None:
            return {"error": "no fitted projection"}
        if not fit.queryable:
            return {"error": f"'{fit.method}' cannot place new points (no .transform); "
                             f"re-project with umap or pca to enable queries"}

        # ---- build the query in each feature block the projection space uses
        by_method: dict = {}
        if iuid is not None:
            m = self.state.meta.get(iuid)
            if m is None:
                return {"error": f"unknown instance {iuid!r}"}
            for name in fit.spec:
                by_method[name] = self.collection["feats"][name][m.row]
        else:
            from .extractors import base as _ex
            name = extractor or self.state.primary_extractor() or next(iter(fit.spec))
            if list(fit.spec) != [name]:
                return {"error": f"a text/image query can only be built for a single-feature map; "
                                 f"this one is fused over {sorted(fit.spec)}. Re-project on "
                                 f"'{name}' alone, or query by instance."}
            try:
                ext = _ex.get(name)
            except KeyError:
                return {"error": f"'{name}' is a feature column, not an embedding model — "
                                 f"a text/image query needs one of {sorted(_ex.list_extractors.__globals__['_REGISTRY'])}"}
            ok, why = ext.available()
            if not ok:
                from ._bootstrap import BackendUnavailable
                raise BackendUnavailable(f"{ext.label} is not usable here: {why}. {ext.requires}")
            if text is not None:
                if not hasattr(ext, "embed_text"):
                    return {"error": f"'{name}' has no text encoder — use CLIP or SigLIP for text queries"}
                by_method[name] = np.asarray(ext.embed_text([text])[0], np.float32)
            elif image_rgb is not None:
                import torch
                grid = ext.grid_batch([image_rgb])                    # (1, C, g, g)
                pooled = grid.float().mean(dim=(2, 3))                # whole image = an all-ones mask
                proj = getattr(ext, "project_pooled", None)
                if proj is not None:
                    pooled = proj(pooled)
                by_method[name] = pooled.detach().cpu().numpy()[0].astype(np.float32)
            else:
                return {"error": "give one of iuid, text or image"}

        try:
            q = fit.fuse_query(by_method)
        except ValueError as ex:
            return {"error": str(ex)}
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        Y = np.asarray(fit.reducer.transform(qn), np.float32)
        xy = fit.normalise(Y)[0]

        # nearest instances in MAP space — what the user is actually looking at
        d = np.linalg.norm(fit.normalise(p["coords"]) - xy[None, :], axis=1)
        order = np.argsort(d)[:max(1, int(k))]
        pt = {"x": float(xy[0]), "y": float(xy[1])}
        if int(dims) > 2:
            pt["z"] = float(xy[2])
        return {"ok": True, "point": pt, "method": fit.method, "dims": int(dims),
                "spec": fit.spec, "truncated": fit.truncated,
                "neighbors": [{"iuid": p["iuids"][i], "dist": float(d[i])} for i in order]}

    def projection_points(self, spec=None, *, method: str = "hnne", dims: int = 2) -> dict:
        """Map payload: per in-scope instance {iuid, x, y[, z], state, cls, pid, score, image_id}, coords
        min-max normalized to [0,1] (colored client-side by state/class/partition/score). Read-only."""
        p = self.project(spec, method=method, dims=dims)
        if p.get("error"):
            return p
        Y, iuids = p["coords"], p["iuids"]
        # normalise through the FITTED frame, not a fresh min/max of the current points: a query
        # projected later must land in the same coordinates the points are drawn in
        fit = getattr(self, "_fit", None)
        Yn = (fit.normalise(Y) if (fit is not None and len(iuids)) else
              (((Y - Y.min(0)) / np.maximum(Y.max(0) - Y.min(0), 1e-9)) if len(iuids) else Y))
        pidmap = self._iuid_pid_map() if self._cluster else {}
        recs = self.collection["records"] if self.collection else None
        pts = []
        for u, row in zip(iuids, Yn.tolist()):
            m = self.state.meta[u]
            if m.assigned_class:
                state, pid, cls = "class", f"class:{m.assigned_class}", self.state.class_name(m.assigned_class)
            elif m.is_background:
                state, pid, cls = "reject", None, None
            else:
                pp = pidmap.get(u)
                state, pid, cls = "pool", (str(pp) if pp is not None else None), None
            pt = {"iuid": u, "x": round(row[0], 4), "y": round(row[1], 4), "state": state, "cls": cls,
                  "pid": pid, "source": self._source_of(u), "image_id": str(int(m.image_id)),
                  "score": round(float(recs[m.row]["score"]), 3) if recs else 0.0}
            if int(p["dims"]) >= 3:
                pt["z"] = round(row[2], 4)
            pts.append(pt)
        return {"n": len(pts), "method": p["method"], "dims": p["dims"], "truncated": p.get("truncated", 0),
                "spec": p["spec"], "dropped": p["dropped"], "points": pts}

    def _normed_feats(self, feature: str) -> np.ndarray:
        """L2-normalized `feature` matrix for the WHOLE collection, cached by (feature, coll_version) — the
        brute-force reference-search path. Feature rows are a pure function of the collection (independent of
        assignment / scope / mask edits), so coll_version is the tight, correct key. ~N·D·4 bytes."""
        key = (feature, int(self.state.coll_version))
        c = getattr(self, "_normfeat_cache", None)
        if c is None or c[0] != key:
            X = self.collection["feats"][feature]
            Xn = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
            self._normfeat_cache = (key, Xn)
        return self._normfeat_cache[1]

    def _ann_index(self, feature: str):
        """A faiss HNSW index over the L2-normalized `feature` matrix for sub-linear cosine NN, cached by
        (feature, coll_version). Returns (index, n) or None — None when faiss is unavailable, the build fails,
        OR the collection is SMALL (< _ANN_MIN): below that, an exact matmul is already sub-50ms and exact (and
        the approximate top-m would drop far-but-present classes that the exhaustive scan surfaces). The caller
        falls back to the brute-force normalized matmul. Built lazily; faiss copies the vectors after add()."""
        n = int(self.collection["feats"][feature].shape[0]) if (self.collection or {}).get("feats", {}).get(feature) is not None else 0
        if n < _ANN_MIN:
            return None
        key = (feature, int(self.state.coll_version))
        c = getattr(self, "_ann_cache", None)
        if c is not None and c[0] == key:
            return c[1]
        idx_n = None
        try:
            import faiss
            Xb = np.ascontiguousarray(self.collection["feats"][feature].astype(np.float32))
            faiss.normalize_L2(Xb)                              # cosine == L2 on unit vectors
            n, d = Xb.shape
            idx = faiss.IndexHNSWFlat(d, 32)
            idx.hnsw.efConstruction = 64
            idx.hnsw.efSearch = 64
            idx.add(Xb)
            idx_n = (idx, int(n))
        except Exception:
            idx_n = None                                        # faiss missing or build error -> brute fallback
        self._ann_cache = (key, idx_n)
        return idx_n

    def _nn_candidates(self, feature: str, qn: np.ndarray, want: int):
        """Best-first [(row_index, cosine_score), ...] of length ≤ want for a unit query `qn`. Uses the faiss
        HNSW index (approximate, ~O(log N)) when available, else an exact normalized matmul + argsort."""
        ann = self._ann_index(feature)
        if ann is not None:
            idx, n = ann
            D, I = idx.search(np.ascontiguousarray(qn[None].astype(np.float32)), int(min(want, n)))
            return [(int(i), 1.0 - 0.5 * float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
        Xn = self._normed_feats(feature)                        # exact brute fallback
        sims = Xn @ qn
        return [(int(i), float(sims[i])) for i in np.argsort(-sims)]

    def _unassigned_proba(self):
        """Classifier proba over the current unassigned pool, CACHED by (clf_version, spec, coll_version).
        The three preview paths (predict / recommend_rejections / recommend_interesting) and Apply all reuse
        ONE O(M·C) pass instead of recomputing it per click — the classifier-tab hotspot. Within a fixed key
        the unassigned set only SHRINKS (assignment removes instances; any growth bumps coll_version), so a
        cache hit just slices the stored proba for the still-unassigned rows. Lock-free like _fused_cache
        (single uvicorn worker). Returns (iuids, proba, classes)."""
        iuids = self._unassigned_iuids()
        if not iuids or getattr(self, "_clf", None) is None:
            return [], np.zeros((0, 0), np.float32), []
        key = (int(self._clf_version), tuple(sorted(self._clf_spec.items())), int(self.state.coll_version))
        c = self._proba_cache
        if c is not None and c["key"] == key:
            idx = c["index"]
            if all(u in idx for u in iuids):                # current pool ⊆ cached pool -> slice, no recompute
                return iuids, c["proba"][[idx[u] for u in iuids]], c["classes"]
        X = self.fused(self._clf_spec)
        proba = self._clf.proba(X[[self.state.meta[u].row for u in iuids]])
        classes = list(self._clf.classes)
        self._proba_cache = {"key": key, "index": {u: i for i, u in enumerate(iuids)},
                             "proba": proba, "classes": classes}
        return iuids, proba, classes

    @_timed
    def predict_and_threshold(self, thresh: float, only_class: str | None = None):
        iuids, proba, classes = self._unassigned_proba()   # cached O(M·C) pass; thresh/only_class are post-filters
        if not iuids:
            return []
        return _clf.threshold_assign(iuids, proba, classes, float(thresh), only_class=only_class)

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
        iuids, proba, classes = self._unassigned_proba()        # shared cached pass (see _unassigned_proba)
        if not iuids or not classes or not len(proba):
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
        iuids = self._unassigned_iuids()
        if not iuids:
            return []
        if getattr(self, "_clf", None) is None:                 # no classifier -> uncertain DETECTIONS
            recs = self.collection["records"]
            scored = [(u, None, 1.0 - float(recs[self.state.meta[u].row].get("score", 0.0))) for u in iuids]
            scored.sort(key=lambda t: -t[2])
            return scored[:int(n)]
        iuids, proba, classes = self._unassigned_proba()        # shared cached pass (see _unassigned_proba)
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
        if spec is None:                                          # default to the domain-matched RAD-DINO
            if self._cluster:                                     # space (best for intuitive matching), as
                spec = self._cluster["spec"]                      # match_image / the Reference tab do; fall
            elif "raddino" in (self.collection or {}).get("feats", {}):  # back to decoder when it's absent
                spec = {"raddino": 1.0}
            else:
                spec = {"decoder": 1.0}
        return _sim.find_similar(self.collection, self.state, iuid, k=k, spec=spec)

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
        """Cosine-NN of a query feature vector against ALL instances' `feature` (faiss HNSW index when
        available — ~O(log N) at 1M — else an exact normalized matmul). With dedup_partition (default), each
        PARTITION appears once — the best-scoring instance per partition, skipping rejected/merged-away
        ones (pid None). Returns the mixed best-first `matches` AND, split out, `matches_class` (assigned
        class:<cid> partitions) + `matches_pool` (UNLABELED matches) each up to k — so the reference search
        surfaces BOTH relevant CLASSES and relevant UNANNOTATED instances, not only classes (which otherwise
        crowd out the pool once many instances are assigned). The pool group is keyed by FINCH partition when
        the instance is in the current cluster, else by the instance itself — so it stays populated even when
        the (frozen) cluster pool is stale or absent, the cause of an empty unlabeled group."""
        feats = (self.collection or {}).get("feats", {})
        if feature not in feats:
            return {"error": f"feature '{feature}' not in collection; available: {self.available_features()}"}
        X = feats[feature]
        q = np.asarray(qvec, np.float32).ravel()
        if q.shape[0] != X.shape[1]:
            return {"error": f"query dim {q.shape[0]} != index dim {X.shape[1]}"}
        qn = q / (np.linalg.norm(q) + 1e-9)
        order = self.state.order
        pidmap = self._iuid_pid_map() if self._cluster else {}   # unlabeled -> FINCH pid (only the frozen pool)
        need = int(k)
        # best-first candidates (faiss HNSW when available, else exact). Pull plenty so the per-class /
        # per-partition dedup below can still fill both groups to k from the nearest neighbours.
        cand = self._nn_candidates(feature, qn, max(2000, need * 200))
        out, cls_out, pool_out, seen_cls, seen_pool = [], [], [], set(), set()
        for i, s in cand:                                 # already best-first
            u = order[i]
            m = self.state.meta.get(u)
            if m is None or m.is_background or m.merged_into is not None:
                continue                                  # rejected / merged-away are never a navigable target
            s = round(float(s), 4)
            if m.assigned_class:                          # LABELED -> class group, deduped by class
                pid = f"class:{m.assigned_class}"
                if dedup_partition and pid in seen_cls:
                    continue
                seen_cls.add(pid)
                if len(cls_out) < need:
                    cls_out.append({"iuid": u, "score": s, "pid": pid,
                                    "cls": self.state.class_name(m.assigned_class)})
            else:                                         # UNLABELED -> pool group; group by FINCH partition if it
                fpid = pidmap.get(u)                      # has one, else surface the instance itself (pid=iuid), so
                gkey = fpid if fpid is not None else u    # the group is populated even when the cluster pool is stale
                if dedup_partition and gkey in seen_pool:  # / absent (the cause of the "only labeled" regression)
                    continue
                seen_pool.add(gkey)
                if len(pool_out) < need:
                    pool_out.append({"iuid": u, "score": s, "pid": (fpid if fpid is not None else u)})
            if len(out) < need:                           # mixed best-first (deduped), kept for back-compat
                out.append({"iuid": u, "score": s,
                            "pid": (f"class:{m.assigned_class}" if m.assigned_class else (pidmap.get(u) or u))})
            if len(cls_out) >= need and len(pool_out) >= need and len(out) >= need:
                break
        return {"matches": out, "matches_class": cls_out, "matches_pool": pool_out}

    def match_image(self, img: np.ndarray, *, feature: str = "raddino", k: int = 12) -> dict:
        """Run the seg model on an uploaded RGB image (reuses collect_batch), take the top-scoring detected
        instance, embed it in the SAME space the index uses, and cosine-NN it against the collection.
        Defaults to RAD-DINO — the domain-matched feature the Reference tab ranks in, and the best feature
        for intuitive visual matching (verified on the real bank: in-domain instance↔instance retrieval lands
        the right class far more often than roialign/decoder). So this 'find partition by image' and the
        Reference tab now share ONE retrieval mechanism (RAD-DINO mask-pool + cosine NN, deduped by
        partition). Falls back to roialign/decoder when RAD-DINO hasn't been extracted (Config → Compute
        RAD-DINO)."""
        feats = (self.collection or {}).get("feats", {})
        if feature == "raddino" and "raddino" not in feats:
            feature = next((f for f in ("roialign", "decoder") if f in feats), "")
        if feature not in feats:
            return {"error": f"feature '{feature or 'raddino'}' not extracted; available: {self.available_features()}"}
        import os
        import tempfile
        import cv2
        model, cfg, d2_cfg = self._ensure_model()
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "query.png")
        cv2.imwrite(fp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        feat_cfg = self.state.config.get("features_runtime", _default_feat_cfg(self.state.config))
        batch = _co.collect_batch(model, cfg, d2_cfg, [fp], score_thresh=0.1, feature_cfg=feat_cfg)
        if not batch["records"]:
            return {"error": "no instance detected in the uploaded image (try a tighter crop of one structure)"}
        scores = [r["score"] for r in batch["records"]]
        qi = int(np.argmax(scores))
        if feature == "raddino":
            qvec = self._query_raddino(fp, batch["records"][qi])   # mask-pool RAD-DINO on the upload → index space
        elif feature in batch.get("feats", {}):
            qvec = batch["feats"][feature][qi]
        else:
            return {"error": f"feature '{feature}' not produced for the uploaded image"}
        res = self.match_features(qvec, feature=feature, k=k)
        res["query_score"] = round(float(scores[qi]), 3)
        res["n_detected"] = len(scores)
        res["feature"] = feature
        return res

    def _query_raddino(self, path: str, rec: dict) -> np.ndarray:
        """RAD-DINO embedding of ONE detected instance on an uploaded image, in the SAME space as the cached
        `raddino` collection feature (full-image grid, soft-mask MEAN-pool — exactly collect._raddino_by_path),
        reusing the engine's already-loaded extractor so match_features can cosine-NN it with no model reload."""
        import cv2
        import torch
        import torch.nn.functional as F
        from ._bootstrap import get_P
        ext = self._ref_extractor()
        grid = ext.grid(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)); C, g, _ = grid.shape
        gf = grid.reshape(C, -1)
        m = torch.from_numpy(get_P().decode_mask(rec)).float()
        soft = F.interpolate(m[None, None], size=(g, g), mode="bilinear", align_corners=False).reshape(-1).to(grid.device)
        return ((gf * soft).sum(1) / soft.sum().clamp_min(1e-6)).detach().cpu().numpy().astype(np.float32)

    # ---- reference exemplar bank (suggest a fine class for unassigned instances) ----
    def _ref_extractor(self):
        ext = getattr(self, "_raddino_ext", None)
        if ext is None:
            import torch
            from ._bootstrap import get_P
            ext = self._raddino_ext = get_P().RadDinoExtractor("cuda" if torch.cuda.is_available() else "cpu")
        return ext

    def _instance_crop_box_norm(self, u: str) -> tuple[float, float, float, float]:
        """Normalized (x1,y1,x2,y2) in [0,1] of the instance's CURRENT mask bbox (refine-aware), falling
        back to the record detection box when the mask is empty. Normalized so it maps onto the full image
        regardless of any mask-vs-image resolution difference (mask is at record H/W, image is native)."""
        m = self._mask(u)
        if m is not None and m.any():
            mh, mw = m.shape
            ys, xs = np.where(m)
            return (xs.min() / mw, ys.min() / mh, (xs.max() + 1) / mw, (ys.max() + 1) / mh)
        rec = self.collection["records"][self.state.meta[u].row]
        bx = rec.get("box_xyxy")
        if bx is not None:
            return (bx[0] / float(rec["W"]), bx[1] / float(rec["H"]),
                    bx[2] / float(rec["W"]), bx[3] / float(rec["H"]))
        cx, cy, bw, bh = rec["cx"], rec["cy"], rec["bw"], rec["bh"]
        return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)

    def _instance_ref_embeddings(self, iuids: list[str]) -> np.ndarray:
        """Crop-forward RAD-DINO embedding per instance — SYMMETRIC with the reference bank
        (`load_reference_bank` embeds each reference by bbox-crop → grid → MAX-pool). Here we crop each
        instance's mask bbox out of the full image and run the SAME forward + MAX-pool, so the same object
        embeds the same way on both sides. This removes the scale / receptive-field mismatch that wrecked
        cross-domain retrieval: the old path embedded the WHOLE CXR and mask-gated a handful of
        globally-contextualized tokens, so an instance and a bbox-cropped, frame-filling reference of the
        same device landed in very different regions of RAD-DINO space (it looked like different features
        were used — functionally they were: full-image-forward vs crop-forward). Crops are batched through
        `grid_batch`; cached by (iuid, mask_token) so a refine (which moves the bbox) re-embeds."""
        cache = self.__dict__.setdefault("_ref_inst_cache", {})
        need = [u for u in iuids if (u, self.mask_token(u)) not in cache]
        if need:
            ext = self._ref_extractor()
            from collections import defaultdict
            by_img: dict = defaultdict(list)
            for u in need:
                by_img[self.state.meta[u].image_id].append(u)
            crops, keys = [], []
            for iid, us in by_img.items():
                img = self._rgb_by_image(iid); H, W = img.shape[:2]
                for u in us:
                    nx1, ny1, nx2, ny2 = self._instance_crop_box_norm(u)
                    x1, y1 = max(0, int(nx1 * W)), max(0, int(ny1 * H))
                    x2 = min(W, max(x1 + 1, int(np.ceil(nx2 * W))))
                    y2 = min(H, max(y1 + 1, int(np.ceil(ny2 * H))))
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0 or min(crop.shape[:2]) < 4:
                        crop = img                               # degenerate bbox → whole image (rare)
                    crops.append(crop); keys.append((u, self.mask_token(u)))
            B = int(os.environ.get("CURATOR_RADDINO_BATCH", "8"))
            for s in range(0, len(crops), B):
                grids = ext.grid_batch(crops[s:s + B])           # (b, C, g, g) in one forward
                for j in range(int(grids.shape[0])):
                    cache[keys[s + j]] = grids[j].amax(dim=(1, 2)).detach().cpu().numpy().astype(np.float32)
        return np.stack([cache[(u, self.mask_token(u))] for u in iuids]).astype(np.float32)

    @staticmethod
    def _resolve_ref_root(start, rel: str):
        """Find the root directory under which the relative exemplar path `rel` actually exists, tolerating a
        moved/renamed dataset: try `start`, walk UP its parents, then a bounded DOWN-search for rel's leading
        component (e.g. a dataset moved into a 'foo 2/' sibling). Returns the working root, or `start`."""
        start = Path(start); rel = str(rel)
        if (start / rel).exists():
            return start
        for base in start.parents:                              # dataset moved UP a level
            if (base / rel).exists():
                return base
        lead = Path(rel).parts[0] if Path(rel).parts else ""    # dataset moved DOWN into a sibling/child dir
        if lead:
            for pat in (lead, f"*/{lead}", f"*/*/{lead}"):
                for hit in start.glob(pat):
                    if hit.is_dir() and (hit.parent / rel).exists():
                        return hit.parent
        return start

    def load_reference_bank(self, coco_path: str, *, rebuild: bool = False, image_root: str | None = None) -> dict:
        """Build (or load cached) the RAD-DINO reference bank from a labeled COCO of foreign-object crops and
        BOOTSTRAP the taxonomy with its class names. References are embedded by bbox-crop MAX-pool — the SAME
        forward `_instance_ref_embeddings` runs on the curator's own instances, so retrieval is symmetric."""
        import json
        import os

        import cv2
        from . import reference_bank as _rb
        cache = self.store.dir / "reference_bank"
        bank = None if rebuild else _rb.ReferenceBank.load(cache)
        if bank is not None and getattr(bank, "pool", "mean") != "max":
            bank = None                                          # stale mean-pool cache -> rebuild with max-pool
        if bank is None:
            d = json.load(open(coco_path))
            id2n = {c["id"]: c["name"] for c in d["categories"]}
            imgs = {im["id"]: im for im in d["images"]}
            start = Path(image_root) if image_root else Path(coco_path).parent
            sample = next((im["file_name"] for im in d["images"] if not os.path.isabs(im["file_name"])), None)
            root = self._resolve_ref_root(start, sample) if sample else start   # tolerate a moved dataset
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
        sample_rel = next((e["file_name"] for e in bank.exemplars
                           if e.get("file_name") and not os.path.isabs(e["file_name"])), None)
        start = Path(image_root) if image_root else Path(coco_path).parent
        self._ref_img_root = str(self._resolve_ref_root(start, sample_rel) if sample_rel else start)
        exemplars_ok = bool(sample_rel) and (Path(self._ref_img_root) / sample_rel).exists()
        self.state.config["last_reference_coco"] = str(coco_path)         # remembered for the path autocomplete
        self.state.config["last_reference_root"] = self._ref_img_root
        added = 0
        for name in bank.classes():
            if self.state.class_id_by_name(name) is None:
                self.state.add_class(name); added += 1
        self.save()
        return {"classes": len(bank.classes()), "exemplars": bank.n, "added_classes": added,
                "image_root": self._ref_img_root, "exemplars_ok": exemplars_ok}

    def reference_suggest(self, iuids: list[str], *, topk: int = 5, knn: int = 8, use_csls: bool = False) -> dict:
        """Per instance, the top-k reference CLASSES it most resembles (kNN class vote over the bank). A weak
        prior to CONFIRM, not auto-apply — surfaced for one-click accept. Defaults to PLAIN COSINE: with the
        symmetric crop-forward embedding, cosine beats CSLS de-hubbing on the real bank (CSLS was a band-aid
        for the old non-discriminative full-image embeddings); pass use_csls=True to re-enable de-hubbing."""
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

    def reference_find_instances(self, cls: str, *, k: int = 24, knn: int = 8, use_csls: bool = False,
                                 cap: int = 4000, dedup_partition: bool = True) -> dict:
        """Reverse reference retrieval: given a reference CLASS (the presented reference sample), rank the
        curator's OWN instances by how strongly they resemble it — same CSLS-de-hubbed space as
        reference_suggest, just inverted (fix the class, rank the instances). Lets you see WHICH PARTITIONS
        are nearest a reference sample across ALL present instances, with no partition preselected. With
        dedup_partition (default), each partition appears once via its best instance, skipping
        rejected/merged/un-clustered (pid None) ones."""
        from . import reference_bank as _rb
        bank = getattr(self, "_ref_bank", None)
        if bank is None or bank.n == 0:
            return {"error": "load a reference bank first (Reference tab)"}
        names = {bank.class_names.get(l, str(l)) for l in bank.labels}
        if cls not in names:
            return {"error": f"class '{cls}' is not in the reference bank"}
        iuids = [u for u in self.state.order
                 if self._in_scope(u) and not self.state.meta[u].is_background
                 and self.state.meta[u].merged_into is None]
        truncated = len(iuids) > int(cap)
        iuids = iuids[:int(cap)]
        if not iuids:
            return {"items": [], "truncated": False, "cls": cls}
        self._set_progress("embedding instances (RAD-DINO)", 0, len(iuids))
        try:
            Q = self._instance_ref_embeddings(iuids)
        finally:
            self._clear_progress()
        ref_labels = [bank.class_names.get(l, str(l)) for l in bank.labels]
        order, scores = _rb.rank_instances_for_class(Q, bank.emb, ref_labels, cls, knn=int(knn),
                                                     use_csls=bool(use_csls))
        out, seen = [], set()
        for i in order:
            u = iuids[int(i)]; pid = self.partition_of(u)
            if dedup_partition:
                if pid is None or pid in seen:
                    continue
                seen.add(pid)
            cur = self.state.meta[u].assigned_class
            out.append({"iuid": u, "score": round(float(scores[int(i)]), 3), "pid": pid,
                        "cls": self.state.class_name(cur) if cur else None})
            if len(out) >= int(k):
                break
        return {"items": out, "truncated": truncated, "cls": cls}

    def add_to_reference_bank(self, iuids: list[str]) -> dict:
        """Self-improving bank: add CONFIRMED in-domain instances (their crop-forward embedding + assigned
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
        """Crop of a bank exemplar (for the visual panel). file_name is relative to the loaded ref COCO's
        image root; if that root went stale (dataset moved), re-resolve it once against the filesystem."""
        import os

        import cv2
        root = getattr(self, "_ref_img_root", None) or getattr(self, "_ref_coco_root", None)
        if not root:
            return None
        p = Path(root) / file_name
        if not os.path.isabs(file_name) and not p.exists():           # dataset moved since load -> re-resolve
            nr = self._resolve_ref_root(root, file_name)
            if (Path(nr) / file_name).exists():
                self._ref_img_root = str(nr); p = Path(nr) / file_name
        img = cv2.imread(str(p))
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
        spec, dropped_nan = self._present_spec_nanfree(spec)
        if not spec:
            ok = [m for m in self.available_features() if m not in self.feature_nan_methods()]
            return {"error": f"no usable (present, NaN-free) features selected"
                    + (f"; dropped for NaN: {dropped_nan}" if dropped_nan else "") + f"; NaN-free available: {ok}"}
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
        if dropped_nan:
            rep["dropped_nan_features"] = dropped_nan
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
    @_mutating
    def undo(self):
        op = self.history.undo(self.state); self._mutation_serial += 1  # bump (no delta) -> live index rebuilds
        self.save(); return op

    @_mutating
    def redo(self):
        op = self.history.redo(self.state); self._mutation_serial += 1
        self.save(); return op

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
                "undo": u, "redo": r, "coll_version": self.state.coll_version,
                "scope": self._scope_id, "serial": self._mutation_serial}

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

    def activity_summary(self, bins: int = 48, session_gap_s: float = 1800.0) -> dict:
        """Read-only curation-activity timeline assembled from the four append-only logs (history /
        ingests / merge_log / lineage). Pure provenance: no collection scan, no mutation, all from disk.
        Drives the Activity tab — op breakdown, activity-over-time, merge-decision split, ingest runs, the
        retrain metric trajectory, and session segmentation (idle gap > session_gap_s starts a new
        session). Fully JSON-serializable; degrades to zeros/empties on a project with no logs yet."""
        import os.path as _osp
        from collections import Counter

        hist = self.store.read_history()
        ingests = self.store.read_ingests()
        merges = self.store.read_merge_events()
        lineage = self.store.read_lineage()

        CAT = {                                          # raw history `op` -> coarse category
            "assign": "assign", "import": "assign", "new_class": "class",
            "remove": "unassign", "background": "reject", "unreject": "unreject",
            "merge": "merge", "merge_classes": "merge", "dedup": "dedup", "split": "split",
            "refine": "refine", "refine_partition": "refine", "refine_many": "refine",
            "auto_refine": "refine", "revert_refine": "refine",
            "undo": "undo/redo", "redo": "undo/redo",
        }
        op_counts, op_insts = Counter(), Counter()
        cat_counts, cat_insts = Counter(), Counter()
        mut_events, all_ts = [], []                      # mut_events excludes undo/redo
        for h in hist:
            ts = float(h.get("ts", 0.0)); op = str(h.get("op", "?")); n = int(h.get("n_instances", 0) or 0)
            op_counts[op] += 1; op_insts[op] += n
            cat = CAT.get(op, op); cat_counts[cat] += 1; cat_insts[cat] += n
            all_ts.append(ts)
            if op not in ("undo", "redo"):
                mut_events.append((ts, n))
        for ev in ingests + merges + lineage:
            all_ts.append(float(ev.get("ts", 0.0)))
        all_ts = sorted(t for t in all_ts if t > 0)
        t_min, t_max = (all_ts[0], all_ts[-1]) if all_ts else (0.0, 0.0)

        nb = max(1, int(bins))
        counts, insts = [0] * nb, [0] * nb
        span = max(1e-9, t_max - t_min)
        for ts, n in mut_events:
            b = min(nb - 1, int((ts - t_min) / span * nb))
            counts[b] += 1; insts[b] += n
        edges = [round(t_min + span * i / nb, 3) for i in range(nb + 1)]

        m_kind = Counter(str(ev.get("kind", "?")) for ev in merges)
        m_source = Counter(str(ev.get("source", "?")) for ev in merges if ev.get("kind") == "merge")
        merge_insts = sum(len(ev.get("iuids", [])) for ev in merges if ev.get("kind") == "merge")

        ingest_rows = [{"ingest_id": ev.get("ingest_id", ""), "ts": float(ev.get("ts", 0.0)),
                        "n_images": int(ev.get("n_images", 0)), "n_instances": int(ev.get("n_instances", 0)),
                        "mode": ev.get("mode"), "score_thresh": ev.get("score_thresh")} for ev in ingests]
        lineage_rows = [{"ts": float(ev.get("ts", 0.0)), "metric_name": ev.get("metric_name"),
                         "metric": ev.get("metric"), "ckpt": _osp.basename(str(ev.get("ckpt", "") or "")),
                         "config_name": ev.get("config_name"), "n_assigned": ev.get("n_assigned"),
                         "regressed": bool(ev.get("regressed", False))} for ev in lineage]

        stream = [(float(h.get("ts", 0.0)), CAT.get(str(h.get("op", "?")), str(h.get("op", "?")))) for h in hist]
        stream += [(float(ev.get("ts", 0.0)), "ingest") for ev in ingests]
        stream += [(float(ev.get("ts", 0.0)), "retrain") for ev in lineage]
        stream = sorted((t, c) for t, c in stream if t > 0)
        sessions, cur = [], None
        for ts, c in stream:
            if cur is None or ts - cur["end"] > float(session_gap_s):
                cur = {"start": ts, "end": ts, "n_ops": 0, "ops": Counter()}; sessions.append(cur)
            cur["end"] = ts; cur["n_ops"] += 1; cur["ops"][c] += 1
        sessions = [{"start": s["start"], "end": s["end"], "dur_s": round(s["end"] - s["start"], 1),
                     "n_ops": s["n_ops"], "ops": dict(s["ops"])} for s in sessions]

        total_cmd = sum(v for o, v in op_counts.items() if o not in ("undo", "redo"))
        return {
            "span": {"t_min": t_min, "t_max": t_max, "n_events": len(all_ts), "n_sessions": len(sessions)},
            "totals": {"commands": total_cmd, "instances_touched": sum(n for _, n in mut_events),
                       "undos": op_counts.get("undo", 0), "redos": op_counts.get("redo", 0),
                       "ingests": len(ingests), "merges": m_kind.get("merge", 0),
                       "merge_rejects": m_kind.get("reject", 0), "retrains": len(lineage)},
            "op_counts": dict(op_counts), "op_insts": dict(op_insts),
            "cat_counts": dict(cat_counts), "cat_insts": dict(cat_insts),
            "timeline": {"edges": edges, "counts": counts, "insts": insts},
            "merge": {"by_kind": dict(m_kind), "by_source": dict(m_source), "instances": merge_insts},
            "ingests": ingest_rows, "lineage": lineage_rows, "sessions": sessions,
        }

    def _after_mutation(self):
        """Hot interactive path (assign / merge / refine / split / reject / …). Write-behind: mark the engine
        dirty and let the background saver coalesce the O(N) state write off the request thread, so a click
        returns immediately regardless of project size. Every _AUTOSNAP_EVERY-th commit takes a synchronous
        snapshot (a cheap durable checkpoint) — that one also flushes the pending state."""
        self._commits += 1
        self._mutation_serial += 1                        # live-index validity stamp (mutations that don't
        #                                                   call _cache_delta thus force an index rebuild)
        if self._commits % _AUTOSNAP_EVERY == 0:
            self.save(snapshot=True)
        else:
            self._save_dirty.set()


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
