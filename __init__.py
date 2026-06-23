"""qseg instance-curation tool.

Backend engine (UI-agnostic) wrapping notebooks/qseg_playground.py, served by a
FastAPI app (`server.py`) over a single-page web frontend (`web/`). `CuratorEngine`
is the facade the server binds to; import it lazily to avoid pulling
torch/detectron2 for pure-python unit tests.
"""
from __future__ import annotations
