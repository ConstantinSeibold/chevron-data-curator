"""The ingest panel: getting masks into a project without a terminal.

Until this existed, `/api/propose` had no UI at all — starting a project meant a curl, which is a
strange first step for a labelling tool. The panel is only useful if it sends the RIGHT request, and
that is what structure tests cannot see, so the real block from `app.js` runs under a DOM shim here
(same approach as the router and 3D-fallback tests) and the outgoing body is asserted.

Run: pytest tests/test_ingest_ui.py -q
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
PROBE = ROOT / "tests" / "js" / "ingest_probe.js"
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


# --------------------------------------------------------------------------- the control exists
def test_the_panel_is_in_the_served_page():
    html = INDEX.read_text()
    for cid in ("ingBackend", "ingRun", "ingRoot", "ingCoco", "ingMsg", "ingBar"):
        assert f'id="{cid}"' in html, f"#{cid} missing from index.html"


def test_it_is_the_first_thing_in_settings():
    """Ingest is step 1 of a project; it should not be below the model and feature sections."""
    html = INDEX.read_text()
    cfg = html.index('id="tab-config"')
    assert html.index('id="ingBackend"', cfg) < html.index('id="cfgCkpt"', cfg)


def test_the_backend_dropdown_is_loaded_when_settings_opens():
    """A dropdown nobody populates is an empty dropdown."""
    js = APP_JS.read_text()
    route = re.search(r"config:\s*\(\)\s*=>\s*\{([^}]*)\}", js)
    assert route and "loadBackends()" in route.group(1), "loadBackends is not on the config route"


# --------------------------------------------------------------------------- what it offers
def test_unavailable_backends_are_shown_disabled_with_their_install_hint(probe):
    """Hiding them would leave the user wondering why SAM is not an option."""
    by = {o["name"]: o for o in probe["options"]}
    assert by["sam_auto"]["disabled"] is False
    assert by["hf_seg"]["disabled"] is True and "not installed" in by["hf_seg"]["text"]
    assert probe["noteForUnavailable"] == "pip install 'chevron-curator[embed]'"


def test_it_preselects_something_usable(probe):
    assert probe["selected"] == "coco"
    assert probe["noteForSelected"] == "reads a COCO json; no model or GPU"


def test_the_coco_field_appears_only_for_the_coco_backend(probe):
    assert probe["cocoRowForCoco"] == "flex"
    assert probe["cocoRowForSam"] == "none"


# --------------------------------------------------------------------------- what it sends
def test_it_posts_to_propose_with_the_fields_filled_in(probe):
    assert probe["samUrl"] == "/api/propose"
    assert probe["samBody"] == {"backend": "sam_auto", "image_root": "/data/images",
                                "limit": 50, "score_thresh": 0.3, "source": "sam_run1"}, \
        "paths must be trimmed and numbers sent as numbers"


def test_blank_fields_are_omitted_rather_than_sent_empty(probe):
    """The server has its own defaults; sending "" or NaN would override them with nonsense."""
    assert probe["minimalBody"] == {"backend": "sam_auto"}


def test_coco_without_a_path_never_reaches_the_server(probe):
    assert probe["cocoNoPath"]["posts"] == 0
    assert "required" in probe["cocoNoPath"]["msg"]


def test_coco_with_a_path_sends_it(probe):
    assert probe["cocoBody"] == {"backend": "coco", "coco_path": "/data/masks.json"}


# --------------------------------------------------------------------------- what it says after
def test_success_refreshes_and_reports_the_counts(probe):
    assert probe["refreshedAfterSuccess"] == 1, "the rest of the UI would still show an empty project"
    assert "42" in probe["successMsg"] and "7" in probe["successMsg"]
    assert "Compute features" in probe["successMsg"], "the user needs the next step, not just a count"


def test_a_server_error_is_shown_and_not_mistaken_for_success(probe):
    assert "no images found" in probe["errorMsg"]
    assert probe["refreshedAfterError"] == 0


def test_every_backend_is_distinguishable_in_the_dropdown():
    """`sam_auto` and `samhq_auto` shared a class-level label, so the picker showed two identical
    rows. Harmless while nothing rendered the list; a coin-flip once it does."""
    from chevron.backends import list_backends
    labels = [b["label"] for b in list_backends()]
    dupes = {x for x in labels if labels.count(x) > 1}
    assert not dupes, f"backends indistinguishable in the picker: {dupes}"


def test_an_empty_project_points_at_the_panel():
    """The moment the user needs ingest is when Curate is empty — say it there, not only in Settings."""
    js = APP_JS.read_text()
    assert "emptyGetMasks" in js and "#/settings/config" in js
