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

__version__ = "0.1.0"
