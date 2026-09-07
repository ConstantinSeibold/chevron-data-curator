"""Set up: the four steps that turn a folder of images into something curatable.

Chevron curates instances, and a new project has none. Getting masks and computing an embedding are
therefore not settings — they are the whole of step 1, and every other area is inert until they have
run. They used to sit at the bottom of the Settings pane, behind the last button of the last area,
which made the app look as though it offered no choice of proposal model or embedding at all.

The status ticks are the app's only answer to "what do I do next", and a wrong tick is invisible —
four boxes are drawn either way — so `setupSync` is executed here against a DOM shim (the approach
the router, 3D-fallback and ingest tests use) rather than checked for structure.

Run: pytest tests/test_setup_flow.py -q
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests" / "js" / "setup_probe.js"
APP_JS = ROOT / "chevron" / "web" / "app.js"
INDEX = ROOT / "chevron" / "web" / "index.html"


def _node_env() -> dict:
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


# --------------------------------------------------------------------------- the pane exists
def test_every_step_is_in_the_served_page():
    html = INDEX.read_text()
    for cid in ("stepImages", "stepMasks", "stepFeats", "stepCluster",
                "okImages", "okMasks", "okFeats", "okCluster",
                "setupRoot", "setupCluster", "setupClusterNote"):
        assert f'id="{cid}"' in html, f"#{cid} missing from index.html"


def test_both_model_choices_are_offered_here():
    """The complaint this pane answers: neither the proposal model nor the embedding model could be
    chosen anywhere obvious. Both dropdowns must be in the FIRST area, in step order."""
    html = INDEX.read_text()
    setup = html.index('id="tab-setup"')
    assert setup < html.index('id="ingBackend"') < html.index('id="cfgExtractor"'), \
        "masks must be choosable before features — they are what the features are computed on"


def test_the_image_root_is_shown_so_an_empty_project_is_diagnosable():
    """A root that resolves to nothing yields a successful-looking run and no instances. Showing the
    path the project actually reads is what makes that a five-second diagnosis."""
    assert '"image_root"' in (ROOT / "chevron" / "server.py").read_text(), \
        "/api/state does not report the image root"
    assert 'id="setupRoot"' in INDEX.read_text()


# --------------------------------------------------------------------------- what the steps say
def test_a_fresh_project_points_at_getting_masks(probe):
    s = probe["fresh"]
    assert s["done"] == ["Images"], "only the image root is settled on a fresh project"
    assert s["next"] == ["Masks"], "a fresh project's next move is to get masks"
    assert s["root"] == "/data/imgs"


def test_a_project_with_no_image_root_says_so(probe):
    s = probe["noRoot"]
    assert s["done"] == []
    assert s["next"] == ["Images"]
    assert "no image root" in s["root"]


def test_geometry_features_alone_do_not_count_as_an_embedding(probe):
    """shape/shapecoord/coords are derived from the mask at ingest and need no model. Counting them
    would tick step 3 for a project that has never run an encoder — and then the clustering the user
    is sent on to do would group by outline only, with no idea what anything LOOKS like."""
    s = probe["geomOnly"]
    assert s["done"] == ["Images", "Masks"], f"geometry ticked the embedding step: {s['done']}"
    assert s["next"] == ["Feats"]
    assert "no embedding" in s["notes"]["Feats"]


def test_a_real_embedding_ticks_the_step_and_is_named(probe):
    s = probe["embedded"]
    assert "Feats" in s["done"]
    assert s["next"] == ["Cluster"]
    assert "dinov3" in s["notes"]["Feats"], "the user is not told WHICH embedding is present"


def test_exactly_one_step_is_ever_the_next_one(probe):
    """Two 'you are here' markers is no marker. Every state, including the finished one."""
    for name in ("fresh", "noRoot", "geomOnly", "embedded", "ready"):
        nxt = probe[name]["next"]
        assert len(nxt) <= 1, f"{name} highlights more than one next step: {nxt}"
    assert probe["ready"]["next"] == [], "a finished project should have no next step"
    assert probe["ready"]["done"] == ["Images", "Masks", "Feats", "Cluster"]


def test_instance_count_is_reported_once_masks_are_in(probe):
    assert "300" in probe["geomOnly"]["notes"]["Masks"]


def test_sample_mode_is_not_told_to_go_and_get_masks(probe):
    """In sample mode the item is the whole image and there are no masks. This pane is the front door
    now, so using the wrong noun for a project's own contents is the first thing the user reads."""
    assert probe["maskTitleDefault"] == "Get masks"
    assert probe["maskTitleSampleMode"] == "Get items"


# --------------------------------------------------------------------------- the hand-off to Curate
def test_clustering_an_empty_project_is_refused_locally(probe):
    """Nothing to cluster is a question the frontend can answer without a round trip, and the server's
    error for it is far less clear than saying which step is missing."""
    r = probe["clusterWhenEmpty"]
    assert r["posts"] == 0, "an empty project still posted a cluster request"
    assert r["routed"] == [], "and it navigated away from the step that still needs doing"
    assert "step 2" in r["note"]


def test_a_successful_cluster_hands_over_to_curate(probe):
    """The point of the pane is to end somewhere useful — leaving the user on a finished checklist
    makes them hunt for the workspace they were being set up for."""
    r = probe["clusterOk"]
    assert r["url"] == "/api/cluster"
    assert r["routed"] == ["partitions"]


def test_a_failed_cluster_keeps_the_user_here(probe):
    r = probe["clusterError"]
    assert r["routed"] == [], "navigated to Curate despite the cluster failing"
    assert "no usable features" in r["note"]


# --------------------------------------------------------------------------- wiring
def test_computing_features_updates_the_step_without_a_reload():
    """The tick is the feedback that the step worked; requiring F5 to see it reads as a failure."""
    js = APP_JS.read_text()
    handler = re.search(r'\$\("#cfgRaddino"\)\.onclick\s*=\s*async\s*\(\)\s*=>\s*\{(.*?)loadExtractors\(\);', js, re.S)
    assert handler, "the compute-features handler is gone"
    assert "setupSync()" in handler.group(1), "step 3 will not tick over until the page is reloaded"


def test_the_setup_globals_are_declared_above_the_router():
    """`routeFromHash()` runs during app.js's own top-level pass and can call the setup on-show hook
    straight away. A `let` declared further down the file would still be in its temporal dead zone,
    so a deep link to #/setup/setup would throw before the pane ever rendered."""
    js = APP_JS.read_text()
    assert js.index("let SETUP = ") < js.index("function routeFromHash()"), \
        "SETUP is declared after the router that can read it"
    assert js.index("const GEOM_FEATURES") < js.index("function routeFromHash()")
