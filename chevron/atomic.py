"""os.replace after the data is on disk: renaming first can leave an EMPTY file after a power cut."""
from __future__ import annotations

import os


def replace(tmp, path) -> None:
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
