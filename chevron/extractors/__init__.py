"""Embedding extractors — one encoder pass per image, pooled per item.

Importing this package registers the line-up. Registration is cheap: every heavy import is deferred
to first use, so listing what is *available* never requires torch to be installed.
"""
from __future__ import annotations

from .base import Extractor, HFPatchGridExtractor, get, list_extractors, register  # noqa: F401
from . import registry  # noqa: F401,E402  — importing registers raddino / dinov2 / clip / siglip2

__all__ = ["Extractor", "HFPatchGridExtractor", "get", "list_extractors", "register"]
