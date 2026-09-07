"""The one place Chevron imports faiss, so its OpenMP thread pool is capped before first use.

On macOS the server has repeatedly died inside faiss's OpenMP worker threads, with the signature

    Segmentation fault ... libomp.dylib  __kmp_suspend_initialize_thread

and, less often, an outright abort from `__kmp_abort_process` ("OMP: Error #15 ... libomp.dylib
already initialized"). Both come from more than one OpenMP runtime being loaded into the process:
faiss, torch, scikit-learn and scikit-image each bundle their own libomp, and faiss's is linked via
`@loader_path`, so it cannot be pointed at any of the others. The abort is tolerated by
`KMP_DUPLICATE_LIB_OK`, set in `chevron/__init__.py`. The segfault happens in the worker threads
faiss spawns for a parallel search, and a faiss pinned to one thread never spawns any, so there is
nothing left to crash.

One thread costs nothing measurable here. Chevron's faiss use is small-N approximate
nearest-neighbour search (kNN reference sets are subsampled to tens of thousands of rows, and the
map's neighbour queries are a few thousand points), well below the point where OpenMP fan-out pays
for itself. `OMP_NUM_THREADS` would give the same cap but also throttle torch's CPU inference, which
is why the cap is applied through faiss's own API instead.

Every `import faiss` in the codebase goes through `load_faiss()`, so no call site can reach an
uncapped module.
"""
from __future__ import annotations


def load_faiss():
    """Import faiss, pin it to one OpenMP thread, and return the module.

    Raises whatever `import faiss` raises when the `faiss` extra is not installed, so callers keep
    their existing fallback (an `except Exception` around the import, then sklearn).
    """
    import faiss
    try:
        faiss.omp_set_num_threads(1)
    except AttributeError:
        # A build without OpenMP support has no thread pool to cap, and also none to crash in.
        pass
    return faiss
