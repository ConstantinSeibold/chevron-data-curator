"""What the progress line claims about a running job.

A job that is downloading 400 MB, or spending a minute per image in SAM, is HEALTHY — and the UI
used to call it "stalled" because its one threshold was written for download chunks. That warning
is only worth having if it is rare, so the rule is tested here against fabricated ticks: the real
block from `app.js` runs under node (same approach as the ingest and router probes).

Run: pytest tests/test_progress_ui.py -q
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests" / "js" / "progress_probe.js"
APP_JS = ROOT / "chevron" / "web" / "app.js"


@pytest.fixture(scope="module")
def probe() -> dict:
    if not shutil.which("node"):
        pytest.skip("node not available")
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["PATH"] = "/usr/bin:/bin:" + env.get("PATH", "")
    out = subprocess.run(["node", str(PROBE), str(APP_JS)], capture_output=True, text=True,
                         env=env, timeout=60, check=True)
    got = json.loads(out.stdout)
    assert "error" not in got, got
    return got


def test_a_dead_download_is_still_called_out_fast(probe):
    """The case the warning was written for keeps its 20s: a stopped transfer is broken, not slow."""
    assert probe["deadDownload"]["at10"] is False
    assert probe["deadDownload"]["at30"] is True
    assert "network" in probe["deadDownload"]["why"]


def test_a_slow_first_image_is_not_called_dead(probe):
    """SAM's automatic generator is a minute per image on a CPU; the phase says so with stall_after."""
    assert probe["firstImage"]["at60"] is False
    assert probe["firstImage"]["at400"] is True
    assert "network" not in probe["firstImage"]["why"], "a stuck image loop is not a download problem"


def test_a_phases_own_pace_sets_the_bar_once_it_has_ticked(probe):
    """100s per image, quiet for 90s: healthy. Quiet for 400s: past 3x its own pace, so say so."""
    assert probe["slowPace"]["at90"] is False
    assert probe["slowPace"]["at400"] is True


def test_a_silent_model_load_does_not_blame_the_network(probe):
    assert "network" not in probe["loading"] and "elapsed" in probe["loading"]


def test_the_running_line_shows_pace_and_the_current_file(probe):
    line = probe["healthy"]
    assert "frame_100_endo.png" in line and "8" in line and "80" in line
    assert "left" in line and "stalled" not in line
