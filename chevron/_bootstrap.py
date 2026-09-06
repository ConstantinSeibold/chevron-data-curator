"""Locate an external qseg checkout and import its playground module (`P`).

Chevron is model-agnostic; qseg is one OPTIONAL proposal backend. Nothing here is
imported unless a qseg-backed code path actually runs (`get_P()` is called lazily by
the model/feature-touching modules), so the package imports cleanly — and the test
suite runs — with no qseg, torch or detectron2 present.

The checkout is located via, in order:
  1. `set_qseg_root()`   — set from a project's config by the caller
  2. `$CHEVRON_QSEG_ROOT` environment variable
  3. `$QSEG_ROOT` environment variable
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

class BackendUnavailable(RuntimeError):
    """A proposal backend was asked for something it cannot do here — e.g. the qseg backend with no
    qseg checkout configured. A CONFIGURATION problem, not a crash, so the API answers 400 with the
    message rather than a bare 500."""


_QSEG_ROOT: Path | None = None


def set_qseg_root(path: str | Path | None) -> None:
    """Point Chevron at a qseg checkout (typically from `state.config['qseg_root']`)."""
    global _QSEG_ROOT
    _QSEG_ROOT = Path(path).expanduser().resolve() if path else None


def qseg_root() -> Path | None:
    if _QSEG_ROOT is not None:
        return _QSEG_ROOT
    for var in ("CHEVRON_QSEG_ROOT", "QSEG_ROOT"):
        v = os.environ.get(var)
        if v:
            return Path(v).expanduser().resolve()
    return None


def ensure_playground() -> None:
    """Put the qseg checkout (+ its notebooks/ and MaskDINO submodule) on sys.path."""
    root = qseg_root()
    if root is None:
        raise BackendUnavailable(
            "qseg backend requested but no qseg checkout is configured. Set CHEVRON_QSEG_ROOT "
            "to a qseg checkout, or call chevron._bootstrap.set_qseg_root(...). Chevron's other "
            "proposal backends (SAM, COCO import, ...) need none of this."
        )
    if not root.is_dir():
        raise BackendUnavailable(f"configured qseg root does not exist: {root}")
    for p in (root / "notebooks", root, root / "third_party" / "MaskDINO"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def real_playground():
    """Import qseg's own `qseg_playground` module. Only the three genuinely qseg-coupled entry
    points (`setup_env`, `load_model`, `collect_instances`) need this."""
    ensure_playground()
    import qseg_playground  # noqa
    return qseg_playground


def get_P():
    """The `P` surface the engine calls. Generic helpers resolve inside Chevron; qseg-only ones
    delegate to `real_playground()` on first use, so no qseg checkout is needed to import, test,
    or run any non-qseg backend."""
    from .core import playground
    return playground
