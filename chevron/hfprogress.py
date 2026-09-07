"""Byte-level progress for Hugging Face weight downloads.

`from_pretrained` downloads on first use and reports nothing back to the caller, so the only honest
thing the UI could show was an indeterminate bar labelled "loading model" — the thing a user stares
at for the minutes (or, on a bad link, hours) a 600 MB ViT takes to arrive, learning nothing about
whether it is moving at all.

huggingface_hub already counts those bytes. Both transfer backends — plain HTTPS and the Rust
`hf_xet` one — drive a tqdm built by `huggingface_hub.utils.tqdm._get_progress_bar_context`, so
swapping the class that function instantiates is the single hook that catches both. That is why this
reads the transfer's own counter rather than watching the cache directory: the HTTPS backend flushes
in 10 MB chunks and the xet backend writes through a content cache, so file sizes on disk sit at 0
for minutes at a time and would make a *worse* progress bar than none.

Two details the obvious version gets wrong:

* the counting is our own, not the bar's `n`. When progress bars are disabled (no TTY, or
  `HF_HUB_DISABLE_PROGRESS_BARS`) tqdm's `update()` returns before incrementing `n`, so a hook that
  trusted `n` would report a frozen 0 exactly in the headless case this exists for.
* `unit`/`total` are read from the constructor kwargs, not off the instance, for the same reason —
  a disabled tqdm never assigns those attributes.

Best-effort by construction: if the hook cannot be installed the download still runs, just
unreported, and the class is restored on the way out.
"""
from __future__ import annotations

import contextlib
import importlib
import threading

_LOCK = threading.Lock()


class _Tally:
    """Sum of every live byte-counting bar, so N files downloading at once read as one transfer.

    A closed bar is folded into `_retired` rather than dropped — otherwise finishing a file would
    make the total jump backwards mid-download.
    """

    def __init__(self, on_change):
        self._on_change = on_change
        self._live: dict[int, tuple[float, int]] = {}
        self._retired = [0.0, 0]

    def _emit(self) -> None:
        with _LOCK:
            done = self._retired[0] + sum(d for d, _ in self._live.values())
            total = self._retired[1] + sum(t for _, t in self._live.values())
        try:
            self._on_change(int(done), int(total))
        except Exception:
            pass                                    # a reporting sink must never break a download

    def note(self, key: int, done: float, total: int) -> None:
        with _LOCK:
            self._live[key] = (float(done), int(total))
        self._emit()

    def retire(self, key: int) -> None:
        with _LOCK:
            done, total = self._live.pop(key, (0.0, 0))
            self._retired[0] += done
            self._retired[1] += total
        self._emit()


@contextlib.contextmanager
def report(on_bytes):
    """Call `on_bytes(done_bytes, total_bytes)` while HF downloads run inside this block.

    `total` is 0 until a transfer actually starts (the size is only known once the file's metadata
    has been fetched), which the caller should render as "not started yet" rather than as 0%.
    """
    try:
        mod = importlib.import_module("huggingface_hub.utils.tqdm")
        base = mod.tqdm
    except Exception:
        yield                                       # no huggingface_hub, or it moved — run unreported
        return

    tally = _Tally(on_bytes)

    class _Reporting(base):                         # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            self._chev = kw.get("unit") == "B"      # byte bars only; the "Fetching N files" bar counts items
            self._chev_n = float(kw.get("initial") or 0)
            self._chev_total = int(kw.get("total") or 0)
            super().__init__(*a, **kw)
            if self._chev:
                tally.note(id(self), self._chev_n, self._chev_total)

        def update(self, n=1):
            if getattr(self, "_chev", False):
                self._chev_n += float(n or 0)
                tally.note(id(self), self._chev_n, self._chev_total)
            return super().update(n)

        def close(self):
            if getattr(self, "_chev", False):
                tally.retire(id(self))
                self._chev = False                  # close() is allowed to be called twice
            return super().close()

    mod.tqdm = _Reporting
    try:
        yield
    finally:
        mod.tqdm = base
