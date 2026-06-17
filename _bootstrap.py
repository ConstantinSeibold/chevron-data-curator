"""Put notebooks/ (qseg_playground) + qseg root + MaskDINO on sys.path and import P.
Lazy: only the model/feature-touching modules call get_P()."""
from __future__ import annotations

import sys
from pathlib import Path

QSEG_ROOT = Path(__file__).resolve().parents[2]   # /home/cms/workspace/qseg


def ensure_playground() -> None:
    for p in (str(QSEG_ROOT / "notebooks"), str(QSEG_ROOT), str(QSEG_ROOT / "third_party" / "MaskDINO")):
        if p not in sys.path:
            sys.path.insert(0, p)


def get_P():
    ensure_playground()
    import qseg_playground as P  # noqa
    return P
