"""What /api/progress actually tells the user while a long job runs.

The bar used to be indeterminate for the whole model download — an animation that claimed motion
without measuring any. These cover the two halves of the replacement: the engine's tick carries a
UNIT plus rate/ETA/stall, and `chevron.hfprogress` picks the byte counts out of huggingface_hub's own
transfer bars. Run: pytest tests/test_progress.py -q  (from repo root)
"""
from __future__ import annotations

import importlib
import time

from chevron import hfprogress
from chevron.engine import CuratorEngine


def _bare_engine():
    """The progress helpers are pure state — no project, no disk."""
    eng = CuratorEngine.__new__(CuratorEngine)
    eng._clear_progress()
    return eng


# --------------------------------------------------------------------------- #
# engine tick
# --------------------------------------------------------------------------- #
def test_idle_progress_is_inactive_and_fully_shaped():
    p = _bare_engine().progress()
    assert p["active"] is False
    for k in ("phase", "done", "total", "unit", "detail", "note", "have", "target"):
        assert k in p, k


def test_tick_carries_unit_and_what_is_already_computed():
    eng = _bare_engine()
    eng._set_progress("downloading clip weights", 4_000_000, 605_247_071, unit="bytes",
                      detail="openai/clip-vit-base-patch32", have=["coords", "shapecoord"],
                      target="clip")
    p = eng.progress()
    assert p["unit"] == "bytes" and p["total"] == 605_247_071
    assert p["detail"] == "openai/clip-vit-base-patch32"
    # the point of `have`/`target`: "which features exist, which one is being filled in"
    assert p["have"] == ["coords", "shapecoord"] and p["target"] == "clip"


def test_rate_and_eta_are_derived_once_the_counter_moves():
    eng = _bare_engine()
    eng._set_progress("clip features", 0, 1000, unit="images")
    time.sleep(0.05)
    eng._set_progress("clip features", 100, 1000, unit="images")
    p = eng.progress()
    assert p["rate"] > 0
    assert p["eta"] > 0                       # 900 images left at a measured rate
    assert p["elapsed"] >= 0.05


def test_no_eta_before_anything_has_moved():
    """A total alone must not manufacture an ETA — that is the fake-progress failure again."""
    eng = _bare_engine()
    eng._set_progress("downloading clip weights", 0, 605_247_071, unit="bytes")
    p = eng.progress()
    assert p["rate"] == 0.0 and p["eta"] == 0.0


def test_stall_is_measured_at_read_time_not_at_tick_time():
    """A dead download stops TICKING, so a stored duration would freeze with it. `stalled` has to
    grow between two reads of the same unchanged tick — that is what lets the UI call it out."""
    eng = _bare_engine()
    eng._set_progress("downloading clip weights", 0, 605_247_071, unit="bytes")
    first = eng.progress()["stalled"]
    time.sleep(0.15)
    assert eng.progress()["stalled"] > first


def test_advancing_clears_the_stall_clock():
    eng = _bare_engine()
    eng._set_progress("downloading clip weights", 1, 100, unit="bytes")
    time.sleep(0.15)
    eng._set_progress("downloading clip weights", 2, 100, unit="bytes")
    assert eng.progress()["stalled"] < 0.1


def test_phase_change_reanchors_the_rate():
    """Download bytes must not bleed into the encode phase's images/s."""
    eng = _bare_engine()
    eng._set_progress("downloading clip weights", 600_000_000, 605_247_071, unit="bytes")
    time.sleep(0.05)
    eng._set_progress("clip features", 0, 900, unit="images")
    p = eng.progress()
    assert p["done"] == 0 and p["rate"] == 0.0 and p["eta"] == 0.0


def test_clear_progress_goes_inactive():
    eng = _bare_engine()
    eng._set_progress("clip features", 5, 9, unit="images")
    eng._clear_progress()
    assert eng.progress()["active"] is False


# --------------------------------------------------------------------------- #
# huggingface_hub byte reporter
# --------------------------------------------------------------------------- #
def _hub_tqdm_module():
    return importlib.import_module("huggingface_hub.utils.tqdm")


def test_report_counts_bytes_and_restores_the_class():
    mod = _hub_tqdm_module()
    before = mod.tqdm
    ticks = []
    with hfprogress.report(lambda d, t: ticks.append((d, t))):
        assert mod.tqdm is not before                       # hook installed
        bar = mod.tqdm(unit="B", unit_scale=True, total=1000, initial=0, disable=True)
        bar.update(400)
        bar.update(600)
        bar.close()
    assert mod.tqdm is before                               # and removed again
    assert (400, 1000) in ticks and ticks[-1] == (1000, 1000)


def test_disabled_bars_still_count():
    """tqdm's `update()` returns before touching `n` when the bar is disabled — which is the normal
    state for a server with no TTY. Counting off `n` would report a frozen 0 exactly there."""
    mod = _hub_tqdm_module()
    ticks = []
    with hfprogress.report(lambda d, t: ticks.append((d, t))):
        bar = mod.tqdm(unit="B", unit_scale=True, total=500, initial=0, disable=True)
        bar.update(250)
        assert bar.n == 0                                   # the bar itself counted nothing
    assert ticks[-1] == (250, 500)                          # we did


def test_item_bars_are_ignored():
    """`snapshot_download`'s outer bar counts FILES; folding it into the byte total would corrupt it."""
    mod = _hub_tqdm_module()
    ticks = []
    with hfprogress.report(lambda d, t: ticks.append((d, t))):
        bar = mod.tqdm(unit="it", total=7, initial=0, disable=True)
        bar.update(3)
    assert ticks == []


def test_concurrent_files_sum_and_a_finished_one_does_not_go_backwards():
    mod = _hub_tqdm_module()
    ticks = []
    with hfprogress.report(lambda d, t: ticks.append((d, t))):
        a = mod.tqdm(unit="B", total=100, initial=0, disable=True)
        b = mod.tqdm(unit="B", total=900, initial=0, disable=True)
        a.update(100)
        b.update(300)
        assert ticks[-1] == (400, 1000)                     # both files in one total
        a.close()                                           # retired, not dropped
        assert ticks[-1] == (400, 1000)
        b.update(600)
    assert ticks[-1] == (1000, 1000)


def test_a_raising_sink_cannot_break_a_download():
    mod = _hub_tqdm_module()

    def boom(done, total):
        raise RuntimeError("sink is broken")

    with hfprogress.report(boom):
        bar = mod.tqdm(unit="B", total=10, initial=0, disable=True)
        bar.update(10)                                      # must not propagate
        bar.close()
    assert mod.tqdm.__name__ == "tqdm"


# --------------------------------------------------------------------------- #
# compute_features staging (no weights, no network)
# --------------------------------------------------------------------------- #
def test_compute_features_reports_download_load_and_encode_separately(tmp_path, monkeypatch):
    """The download used to happen lazily inside the first forward pass, so a first run on a slow
    link showed "0/900 images" for however long 600 MB took. The weights are fetched as their own
    reported phase now, and every tick names the features that already exist."""
    import pytest
    pytest.importorskip("torch")
    from test_extractors import _StubEncoder, _project

    from chevron.extractors import base as E

    eng = _project(tmp_path, n=4)
    seen = []

    class _Staged(_StubEncoder):
        hf_id = "stub/encoder"
        name = "stub"

        def _load(self):
            seen.append(("load", dict(eng.progress())))

        def grid_batch(self, images_rgb):
            seen.append(("encode", dict(eng.progress())))
            return super().grid_batch(images_rgb)

    monkeypatch.setattr(E, "get", lambda name: _Staged())
    try:
        out = eng.compute_features("stub")
        assert out.get("ok"), out

        phases = [p["phase"] for _, p in seen]
        assert phases[0].startswith("downloading"), phases   # weights first, named as such
        assert seen[0][1]["unit"] == "bytes"
        assert seen[0][1]["detail"] == "stub/encoder"        # the UI can say WHAT it is fetching

        load_tick = seen[0][1]
        assert load_tick["target"] == "stub"
        assert "shapecoord" in load_tick["have"] and "stub" not in load_tick["have"]

        encode = [p for k, p in seen if k == "encode"]
        assert encode and encode[0]["phase"] == "loading model"   # download is behind us by then
    finally:
        eng.close()

    assert eng.progress()["active"] is False                 # always cleared
