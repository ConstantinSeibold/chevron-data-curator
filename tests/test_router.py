"""Behavioural test of the area/pane router — the one piece of genuinely new frontend logic in P3.1.

The static contract test proves selectors resolve; it cannot prove the router routes. This executes
the REAL router block out of `app.js` against a small DOM shim (`tests/js/router_probe.js`) under
node, and asserts what it actually did. No browser, no bundler, no new Python dependency — node is
already required for the `--check` syntax pass.

Skipped when node is unavailable or broken.

Run: pytest tests/test_router.py -q
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests" / "js" / "router_probe.js"
APP_JS = ROOT / "chevron" / "web" / "app.js"


def _node_env() -> dict:
    """An activated conda env can shadow the system libsqlite3 and break node with
    `undefined symbol: sqlite3session_attach`. Run node with a clean library path."""
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["PATH"] = "/usr/bin:/bin:" + env.get("PATH", "")
    return env


@pytest.fixture(scope="module")
def routed() -> dict:
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
    return json.loads(out.stdout)


def test_opens_on_curate_partitions(routed):
    init = routed["initial"]
    assert init["area"] == "curate"
    assert init["body"] == ["tab-partitions"]
    assert init["panes"] == ["partitions", "map", "inimage", "refine"], \
        "Curate must offer exactly its own panes — no leakage from other areas"


def test_clicking_a_pane_in_another_area_switches_area(routed):
    """The five existing cross-view jumps click a pane button directly
    (`$('nav button[data-tab=refine]').click()`); they must keep working without knowing about areas."""
    c = routed["classifier"]
    assert c["area"] == "assist" and c["body"] == ["tab-classifier"]
    assert c["panes"] == ["classifier", "mergerec", "reference", "substructure"]
    assert c["hash"] == "#/assist/classifier"


def test_on_show_hook_fires_for_the_pane(routed):
    """The old per-tab if-chain became a table; the same call must still happen on show."""
    assert routed["classifier"]["hooks"] == ["syncClfFeats"]


def test_export_pane_lives_in_ship(routed):
    """Export moved out of the global header into its own Ship pane."""
    e = routed["export"]
    assert e["area"] == "ship" and e["body"] == ["tab-export"]
    assert "export" in e["panes"] and e["hash"] == "#/ship/export"


def test_deep_link_restores_area_and_pane(routed):
    """A refresh keeps your place — the app had no URL state at all before."""
    d = routed["deeplink"]
    assert d["area"] == "insights" and d["body"] == ["tab-stats"]


def test_unknown_route_falls_back_instead_of_blanking(routed):
    """A bad hash must never leave the user staring at an empty shell."""
    b = routed["bogus"]
    assert b["area"] == "curate" and b["body"] == ["tab-partitions"]


def test_navigation_survives_an_unwritable_url(routed):
    """`history.replaceState` throws SecurityError on an opaque origin — a sandboxed iframe or a
    file:// embed. It used to throw BEFORE the on-show hook ran, so the pane switched but never
    loaded its data: the UI looked present but dead. The URL is a convenience, not a precondition."""
    s = routed["sandboxed"]
    assert s["threw"] is None, f"showRoute propagated {s['threw']} instead of navigating"
    assert s["body"] == ["tab-map"]
