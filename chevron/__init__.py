"""Chevron — local dataset curation from segmentation proposals.

Turn a segmentation model's raw, class-agnostic proposals into a curated, labelled dataset:
cluster the unassigned pool, assign / reject / merge / refine instances, and export COCO.

Layout:
  engine.py      `CuratorEngine` — the UI-agnostic facade the server binds to
  server.py      thin FastAPI layer (windowed JSON + lazy per-instance crops) + `web/` frontend
  core/          generic, model-agnostic machinery (collection, clustering, morphology, priors)
  extractors/    embedding models — one encoder pass per image, pooled per item
  backends/      proposal sources; `qseg` is one OPTIONAL backend among several

Nothing here imports torch, detectron2 or any model stack at module load: every heavy import is
lazy, which is why the full test suite runs CPU-only with none of them installed. Keep it that way.
"""
from __future__ import annotations

import os
import sys

# faiss, torch, scikit-learn and scikit-image each ship their own copy of the OpenMP runtime
# (libomp.dylib), and on macOS one Chevron process ends up loading several of them. faiss's copy
# cannot be deduplicated away: its extension links libomp through `@loader_path`, so it always
# resolves to the one inside `faiss/.dylibs/` regardless of what was loaded first. When a second
# runtime initializes in the same process, Intel's OpenMP aborts the whole server with "OMP: Error
# #15: Initializing libomp.dylib, but found libomp.dylib already initialized"
# (`__kmp_abort_process`). Setting this variable before any of those packages is imported tells the
# runtime to tolerate the duplicate. It lives here because this `__init__` is the first Chevron code
# to run on both entry paths (`python -m chevron.server` and the `chevron` console script), ahead of
# every lazy torch/faiss import. `setdefault` rather than assignment, so a user who set the variable
# on purpose keeps their value. `OMP_NUM_THREADS` is deliberately NOT set here: it would throttle
# torch's CPU inference across the board. The segfault variant of the clash
# (`__kmp_suspend_initialize_thread` inside faiss's worker threads) is handled by capping faiss
# alone to one thread in `chevron/_faiss.py`.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
