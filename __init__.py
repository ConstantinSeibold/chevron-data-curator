"""qseg instance-curation tool.

Backend engine (UI-agnostic) wrapping notebooks/qseg_playground.py, plus a Gradio
app. `CuratorEngine` is the facade the app binds to; import it lazily to avoid
pulling torch/detectron2 for pure-python unit tests.
"""
from __future__ import annotations
