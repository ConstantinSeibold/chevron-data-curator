"""The 3D view must survive a machine with no usable WebGL device.

The Python side of Chevron now picks CUDA / MPS / CPU deliberately (see `test_device.py`); the
browser has the same class of problem and it was unguarded. Not every display can give a page a
WebGL context — software or remote GL, a driver blocklist, WebGL switched off — and three.js throws
when the renderer is constructed.

The original `mapSet3D` flipped the view state *before* building the renderer, so that throw left
the 2D canvas hidden, an empty canvas shown and the toggle reading "2D": a map that looks broken,
with nothing said and an unhandled promise rejection in the console.

Executed the same way as the router test — the REAL block out of `app.js` under a DOM shim in node.
No browser, no bundler.

Run: pytest tests/test_map3d_fallback.py -q
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests" / "js" / "map3d_probe.js"
APP_JS = ROOT / "chevron" / "web" / "app.js"


def _node_env() -> dict:
    """An activated conda env can shadow the system libsqlite3 and break node."""
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["PATH"] = "/usr/bin:/bin:" + env.get("PATH", "")
    return env


@pytest.fixture(scope="module")
def probe() -> dict:
    if not shutil.which("node"):
        pytest.skip("node not available")
    try:
        out = subprocess.run(["node", str(PROBE), str(APP_JS)], capture_output=True, text=True,
                             env=_node_env(), timeout=60, check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        stderr = getattr(e, "stderr", "") or ""
        if "symbol lookup error" in stderr or isinstance(e, OSError):
            pytest.skip(f"node is broken in this environment: {stderr[:120]}")
        raise
    r = json.loads(out.stdout)
    assert "error" not in r, r["error"]
    return r


def test_the_toggle_does_not_reject(probe):
    """Pre-fix this was an unhandled promise rejection out of an onclick handler."""
    assert probe["noWebglThrew"] is None, f"mapSet3D rejected: {probe['noWebglThrew']}"


def test_no_webgl_leaves_a_working_2d_map_not_a_blank_canvas(probe):
    """The core guard. Every one of these was wrong before the fix."""
    s = probe["noWebgl"]
    assert s["canvas2d"] == "block", "the 2D canvas was hidden and never restored"
    assert s["canvas3d"] == "none", "an empty 3D canvas was left showing"
    assert s["is3d"] is False and s["viewer"] is False


def test_the_toggle_does_not_lie_about_which_view_is_active(probe):
    s = probe["noWebgl"]
    assert s["label"] == "3D", "the button still offered '2D', so the map read as 3D when it was not"
    assert s["primary"] is False


def test_the_user_is_told_why(probe):
    """A silent fallback is the failure mode this whole change is about."""
    said = " ".join(probe["noWebglSaid"]).lower()
    assert said, "no message at all"
    assert "webgl" in said and "2d" in said
    assert probe["noWebglRedrew"], "the 2D map was not redrawn, so the canvas would sit stale"


def test_the_guard_did_not_just_disable_3d(probe):
    """Falling back is only correct if the healthy path still works."""
    s = probe["ok"]
    assert s["canvas3d"] == "block" and s["canvas2d"] == "none"
    assert s["is3d"] is True and s["label"] == "2D" and s["primary"] is True
    assert probe["built"] == [1], "the 3D cloud was not handed its points"


def test_toggling_back_restores_the_2d_canvas(probe):
    s = probe["back"]
    assert s["canvas2d"] == "block" and s["canvas3d"] == "none" and s["is3d"] is False
