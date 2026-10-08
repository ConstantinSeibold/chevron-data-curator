"""macOS: torch's OpenMP runtime must load before faiss's, or a later SAM preview segfaults the server."""
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="the libomp clash is macOS-only")
@pytest.mark.parametrize("entry", ["from chevron._faiss import load_faiss; load_faiss()",
                                   "from chevron.core.collection import _import_finch; _import_finch()"])
def test_torch_loads_before_faiss(entry):
    pytest.importorskip("torch")
    pytest.importorskip("faiss")
    if entry.endswith("_import_finch()"):
        pytest.importorskip("finch")
    code = ("import sys; " + entry + "; m=list(sys.modules); "
            "print(m.index('torch') < m.index('faiss'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().endswith("True"), out.stderr[-2000:]
