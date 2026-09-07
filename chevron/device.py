"""The one place Chevron decides which torch device to run on.

Before this module there were nine copies of `"cuda" if torch.cuda.is_available() else "cpu"`
scattered across the extractors, the proposal backends and the engine. Every one of them was a
silent `else cpu` on Apple silicon: an M-series Mac has a perfectly good GPU, got none of it, and
nothing anywhere said so. The copies also disagreed — `RadDinoExtractor` defaulted to a bare
`"cuda"`, which is not a fallback but a crash on any machine without it.

So the choice is made here, once, and everything else asks.

Three targets, three genuinely different sets of constraints:

* **cuda** — autocast pays for itself (~1.5-2x on a ViT forward). bf16 where the card supports it
  natively (Ampere+), fp16 otherwise: bf16 on pre-Ampere silicon is emulated and can be slower than
  plain fp32, so "bf16 on CUDA" was itself a hidden assumption about which CUDA.
* **mps** — a real GPU, but an incomplete one. torch's Metal backend still has op gaps, and hitting
  one raises mid-batch. `run_or_fallback` catches exactly that and reruns on the CPU, so a missing
  operator costs speed rather than the run. Autocast is off by default here — see `amp_dtype`.
* **cpu** — always available, always correct, and the only device the test suite needs.

Nothing here imports torch at module scope. Chevron's core is deliberately torch-free (that is what
lets the base install and the whole suite run with no model stack), so every torch import is inside
a function and `resolve_device()` answers `"cpu"` when torch is absent entirely.

Overrides:

    CHEVRON_DEVICE=cpu|mps|cuda|cuda:1|auto    force a device. An escape hatch, not a scheduler: it
                                               overrides automatic selection, but will not move a
                                               caller that explicitly asked for the CPU onto a GPU.
    CHEVRON_AMP=0|1                            force mixed precision off / on, including on MPS
"""
from __future__ import annotations

import os
import sys
import warnings
from contextlib import nullcontext
from typing import Any, Callable

ENV_DEVICE = "CHEVRON_DEVICE"
ENV_AMP = "CHEVRON_AMP"

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    """A device fallback is worth saying, but once per process — these sit inside per-image loops."""
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def _cuda_ok() -> bool:
    t = _torch()
    try:
        return bool(t is not None and t.cuda.is_available())
    except Exception:
        return False


def _mps_ok() -> bool:
    t = _torch()
    try:
        b = getattr(t.backends, "mps", None) if t is not None else None
        return bool(b is not None and b.is_available())
    except Exception:
        return False


def available_devices() -> list[str]:
    """Everything usable here, best first. `[0]` is what `auto` picks."""
    out = []
    if _cuda_ok():
        out.append("cuda")
    if _mps_ok():
        out.append("mps")
    out.append("cpu")
    return out


def device_type(device: str) -> str:
    """`"cuda:1"` -> `"cuda"`. autocast and the availability probes want the family, not the index."""
    return str(device or "cpu").split(":", 1)[0].strip().lower()


def resolve_device(pref: str | None = None) -> str:
    """The device to actually use, given a preference that may be stale, absent or impossible.

    Never raises and never returns something unusable: an explicit `"cuda"` on a Mac, or a project
    configured on a CUDA box and reopened on a laptop, degrades to the best device that IS here and
    says so once. That is the whole reason call sites do not do this themselves.
    """
    want = (pref or "auto").strip().lower()
    env = (os.environ.get(ENV_DEVICE) or "").strip().lower()
    # CHEVRON_DEVICE is an escape hatch, not a scheduler. It can move work OFF a misbehaving
    # accelerator or choose between several, but it must NOT drag a caller that deliberately asked
    # for the CPU onto a GPU: `subcluster` and the shape priors pass "cpu" because for a tiny net
    # the host<->device copy costs more than the arithmetic.
    if env and not (device_type(want) == "cpu" and device_type(env) != "cpu"):
        want = env
    if want in ("", "auto", "default"):
        return available_devices()[0]

    kind = device_type(want)
    if kind == "cpu":
        return "cpu"
    if kind == "cuda":
        if _cuda_ok():
            return want                              # keep any ":N" index — multi-GPU still works
        best = available_devices()[0]
        _warn_once("cuda-missing", f"CHEVRON: {want!r} was requested but CUDA is not available here; "
                                   f"using {best!r}.")
        return best
    if kind == "mps":
        if _mps_ok():
            return "mps"
        best = available_devices()[0]
        _warn_once("mps-missing", f"CHEVRON: {want!r} was requested but Apple MPS is not available "
                                  f"here; using {best!r}.")
        return best

    _warn_once(f"unknown-{kind}", f"CHEVRON: unknown device {want!r}; using automatic selection.")
    return available_devices()[0]


def amp_dtype(device: str) -> Any | None:
    """The autocast dtype for this device, or None for "run in fp32".

    CUDA gets bf16 on Ampere and later, fp16 before that. MPS gets nothing unless asked: fp16 on
    Metal has shipped real numerical differences, and these tensors are not throwaway — a pooled
    embedding goes straight into `collection["feats"]`, where a bad block silently poisons
    clustering, kNN and the map for the whole project. Opt in with `CHEVRON_AMP=1`.
    """
    env = os.environ.get(ENV_AMP)
    if env is not None and env.strip().lower() in ("0", "false", "no", "off"):
        return None
    forced = env is not None and env.strip().lower() in ("1", "true", "yes", "on")

    t = _torch()
    if t is None:
        return None
    kind = device_type(device)
    if kind == "cuda":
        try:
            return t.bfloat16 if t.cuda.is_bf16_supported() else t.float16
        except Exception:
            return t.float16
    if kind == "mps":
        return t.float16 if forced else None
    return t.bfloat16 if (kind == "cpu" and forced) else None


def autocast_ctx(device: str):
    """Autocast for `device`, or a no-op context when this device should stay in fp32.

    Deliberately a real `nullcontext` rather than `torch.autocast(..., enabled=False)`: the latter
    hardcodes a device_type, and passing `"cuda"` on a CPU-only box makes torch emit a disabling
    warning on every single batch.
    """
    dt = amp_dtype(device)
    if dt is None:
        return nullcontext()
    t = _torch()
    return t.autocast(device_type=device_type(device), dtype=dt)


def prefers_channels_last(device: str) -> bool:
    """channels_last is a CUDA convolution optimisation; on CPU and MPS it is at best neutral."""
    return device_type(device) == "cuda"


def empty_cache(device: str | None = None) -> None:
    """Release cached GPU memory on whichever backend holds it.

    Only touches torch if it is ALREADY imported — this is called from request threads while
    shutting a model down, and triggering a first torch import there can half-initialise it.
    """
    t = sys.modules.get("torch")
    if t is None:
        return
    kind = device_type(device) if device else None
    try:
        if kind in (None, "cuda") and t.cuda.is_available() and t.cuda.is_initialized():
            t.cuda.empty_cache()
    except Exception:
        pass
    try:
        mps = getattr(t, "mps", None)
        if kind in (None, "mps") and mps is not None and _mps_ok():
            mps.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- MPS op gaps
_MPS_GAP_MARKERS = (
    "not currently implemented for the mps",
    "not implemented for the mps",
    "not supported on mps",
    "mps backend",
    "could not run",
    # not a missing kernel but the same shape of problem: Metal has no float64 AT ALL, so a library
    # that hands torch a numpy float64 array dies with a TypeError instead of an op-gap error.
    "doesn't support float64",
    "does not support float64",
)


def is_unsupported_op_error(exc: BaseException) -> bool:
    """True for "torch's Metal backend cannot run this" — a missing kernel, or a dtype it lacks.

    Matched on the message because torch raises it as a plain `NotImplementedError`/`RuntimeError`
    (or, for float64, a `TypeError`) with no distinguishing type. The point is to be narrow: a
    genuine bug in our own code must keep propagating, so anything that does not name MPS is not
    caught.
    """
    s = str(exc).lower()
    return any(m in s for m in _MPS_GAP_MARKERS) and "mps" in s


def run_or_fallback(fn: Callable[[], Any], *, device: str, demote: Callable[[], None] | None = None,
                    what: str = "this model") -> Any:
    """Run `fn()`; if MPS turns out to lack an operator it needs, run it on the CPU instead.

    `demote` is called before the retry so the caller can move its weights and update its own
    `.device` — without it every later call would pay the same doomed GPU attempt first.
    """
    try:
        return fn()
    except Exception as e:
        if device_type(device) != "mps" or not is_unsupported_op_error(e):
            raise
        _warn_once(f"mps-gap-{what}", f"CHEVRON: {what} hit an operator Apple MPS does not implement "
                                      f"({type(e).__name__}: {e}); falling back to the CPU for it. "
                                      f"Set PYTORCH_ENABLE_MPS_FALLBACK=1 for a faster per-op "
                                      f"fallback, or CHEVRON_DEVICE=cpu to stop trying.")
        if demote is not None:
            demote()
        return fn()


def move_to(model, device: str):
    """`model.to(device)`, degrading to the CPU rather than failing if the move itself is rejected."""
    try:
        return model.to(device)
    except Exception as e:
        if device_type(device) == "cpu":
            raise
        _warn_once(f"move-{device}", f"CHEVRON: could not place a model on {device!r} "
                                     f"({type(e).__name__}: {e}); using the CPU.")
        return model.to("cpu")


_described: dict | None = None


def describe(refresh: bool = False) -> dict:
    """What the UI shows, so "which device am I actually on" stops being a guess.

    Memoised, and `create_app` warms it on the main thread: probing means importing torch, and a
    FIRST torch import inside a request worker thread is the hazard `_unload_inference_model` has
    always guarded against. Availability cannot change within a process, so one probe is enough.
    """
    global _described
    if _described is None or refresh:
        t = _torch()
        dev = resolve_device()
        dt = amp_dtype(dev)
        _described = {"device": dev, "type": device_type(dev), "available": available_devices(),
                      "amp": (str(dt).replace("torch.", "") if dt is not None else None),
                      "torch": (getattr(t, "__version__", None) if t is not None else None),
                      "override": os.environ.get(ENV_DEVICE)}
    return _described
