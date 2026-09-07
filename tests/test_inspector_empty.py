"""With nothing selected, the inspector must not read as a panel about the selection.

The rail opened on a big "0" over "0 selected", then a hint, then live "Assign every instance" /
"Reject every instance" buttons. Everything above those buttons said *selection*, so the buttons read
as acting on the zero selected instances — and "Assign every instance" was worse than misleading: the
class field it reads lived inside the selection-only block, so with nothing selected it was off screen
and the enabled button silently did nothing.

Now the count disappears when there is no selection and a scope line takes over the header, naming what
the whole-scope verbs would actually hit; the class field sits outside the selection block so it is
there for both assigns; and Assign is gated on having a class name at all. The whole-scope group hides
itself in the rejected bin, where neither of its verbs applies.

The behaviour is executed, not read: the REAL renderInspector / syncScopeUI / gate registrations out of
`app.js` run under a DOM shim in node, the same way the router, 3D and map-scope tests work.

Run: pytest tests/test_inspector_empty.py -q
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
PROBE = ROOT / "tests" / "js" / "inspector_probe.js"
APP_JS = ROOT / "chevron" / "web" / "app.js"
INDEX = ROOT / "chevron" / "web" / "index.html"


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
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert "error" not in data, data["error"]
    return data


def test_no_selection_hides_the_selection_count(probe):
    for state in ("cold", "scopeNoSel", "clearedAgain"):
        s = probe[state]
        assert not s["count"], f"{state}: the big selection numeral is still shown with nothing selected"
        assert not s["countLine"], f"{state}: '0 selected' is still shown with nothing selected"
        assert not s["selActs"], f"{state}: the selection verbs are still shown with nothing selected"
        assert s["hint"], f"{state}: the 'select instances' hint should be what fills the rail instead"


def test_the_count_comes_back_with_a_selection(probe):
    s = probe["withSel"]
    assert s["count"] and s["countLine"], "the selection count must return once instances are selected"
    assert s["selActs"], "the selection verbs must return once instances are selected"
    assert not s["hint"], "the empty hint must give way to the verbs"


def test_the_header_names_the_scope_the_whole_scope_verbs_hit(probe):
    assert probe["cold"]["scopeText"] == "no scope selected"
    # the rail's own label for the row, so the rail and the inspector cannot disagree about the name
    assert probe["scopeNoSel"]["scopeText"] == "scope: 3 [duct]"
    assert probe["rejectedBin"]["scopeText"] == "scope: Rejected"


def test_assign_is_dead_until_there_is_a_class_name(probe):
    """An enabled Assign that no-ops on an empty class field is the same lie as verbs over no selection."""
    assert not probe["scopeNoSel"]["assignAll"], "'Assign every instance' must be disabled with no class"
    assert probe["scopeNoSel"]["rejectAll"], "Reject needs no class, so it stays live"
    assert probe["scopeNoSelWithClass"]["assignAll"], "typing a class must enable the whole-scope assign"
    assert probe["classInputListens"], "the class field must re-gate the buttons as it is typed into"
    assert not probe["cold"]["assignAll"] and not probe["cold"]["rejectAll"], \
        "with no scope at all, neither whole-scope verb has a target"


def test_whole_scope_verbs_stay_available_without_a_selection(probe):
    """They act on the scope, not the selection — hiding the whole rail would strand them."""
    assert probe["scopeNoSel"]["scopeBlock"], "the whole-scope group must survive an empty selection"
    assert not probe["rejectedBin"]["scopeBlock"], \
        "in the rejected bin neither whole-scope verb applies, so the group must not linger empty"


def test_the_class_field_is_not_inside_the_selection_only_block():
    """Markup-level: whichever way the block is hidden, the field the whole-scope assign reads must not
    go with it."""
    html = INDEX.read_text()
    acts = html.index('<div id="inspActs"')
    end = html.index('<div id="inspScopeBlock"', acts)
    assert 'id="classInput"' not in html[acts:end], \
        "#classInput is inside #inspActs again — the whole-scope assign loses its class field"
    assert html.index('id="classInput"') < acts, "#classInput should sit above the selection verbs"
