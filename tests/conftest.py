"""Test configuration.

Chevron's base install is deliberately CPU-only and torch-free — that is what keeps the core honest
about not depending on any model stack. A handful of tests do exercise paths that need an optional
extra (torch for checkpoint surgery and the contrastive encoder, faiss for the ANN index, SAM-HQ for
the refine op). On a base install those must SKIP, not fail: `git clone && pip install -e ".[dev]" &&
pytest` should come back green for a newcomer.

The modules listed here are all declared as optional extras in pyproject. Adding a name that is a
*core* dependency would hide a genuine packaging bug, so keep this list in step with the extras.
"""
from __future__ import annotations

import pytest

OPTIONAL_MODULES = {
    "torch",                   # chevron[embed] — checkpoint surgery, contrastive encoder, shape prior
    "transformers",            # chevron[embed] — RAD-DINO / DINOv2 / CLIP extractors
    "faiss",                   # chevron[faiss] — approximate NN above ~50k instances
    "segment_anything",        # chevron[sam]
    "segment_anything_hq",     # chevron[sam]
    "hnne", "umap", "openTSNE",  # chevron[viz] — the projection stack
    "hdbscan",                 # chevron[cluster]
    "matplotlib",              # chevron[viz] — paper stats only
}


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Turn 'this optional extra is not installed' into a skip rather than a failure.

    The import happens deep inside the engine rather than in the test body, so `importorskip` at the
    top of each test would not catch it — the exception surfaces mid-call.
    """
    try:
        return (yield)
    except ModuleNotFoundError as e:
        if e.name in OPTIONAL_MODULES:
            pytest.skip(f"optional dependency {e.name!r} is not installed (see pyproject extras)")
        raise
