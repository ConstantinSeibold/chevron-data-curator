"""Proposal backends: sources of class-agnostic instance masks.

Chevron curates proposals; which model made them is a detail. Importing this package registers every
backend — registration is cheap because each one's heavy imports are deferred to first use, so
listing what is *available* never requires torch to be installed.
"""
from __future__ import annotations

from .base import Proposal, ProposalBackend, build_collection, get, list_backends, register  # noqa: F401

# Importing a module registers its backend. Order is display order in the UI: the ones that need
# nothing come first.
from . import coco_file          # noqa: F401,E402  — no ML stack at all
from . import sam_auto           # noqa: F401,E402  — no trained model needed
from . import torchvision_maskrcnn  # noqa: F401,E402
from . import hf_seg             # noqa: F401,E402

__all__ = ["Proposal", "ProposalBackend", "build_collection", "get", "list_backends", "register"]
