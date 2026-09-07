"""Picking a scope in the rail must show up on the Map.

The rail and the Map are two views of the same instances, but clicking a class or a partition only
ever reloaded the grid — the canvas kept painting every point identically, so the question the Map
exists to answer ("where does this partition sit in the latent space?") could not be asked at all.

Selecting a scope now lights its points and mutes the rest, for every kind of scope the rail lists:
a class, a FINCH partition, a sub-cluster and the rejected bin. Two properties matter beyond "it
colours something": the muting must keep each point's hue (a muted point still reads as the class it
belongs to), and a scope the current projection has no points for must NOT mute everything — an
all-grey canvas reads as a broken map.

Executed the same way as the router and 3D tests — the REAL block out of `app.js` under a DOM shim
in node. No browser, no bundler.

Run: pytest tests/test_map_scope_highlight.py -q
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
PROBE = ROOT / "tests" / "js" / "map_scope_probe.js"
APP_JS = ROOT / "chevron" / "web" / "app.js"

HSL = re.compile(r"^hsl\((\d+),(\d+)%,(\d+)%\)$")


def _node_env() -> dict:
    """An activated conda env can shadow the system libsqlite3 and break node."""
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["PATH"] = "/usr/bin:/bin:" + env.get("PATH", "")
    return env


def _hsl(color: str) -> tuple[int, int, int]:
    m = HSL.match(color)
    assert m, f"expected an hsl() colour so that muting can keep the hue, got {color!r}"
    return int(m[1]), int(m[2]), int(m[3])


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


def test_nothing_is_muted_until_a_scope_is_picked(probe):
    """The map with no scope selected is exactly the map as it was before this change."""
    s = probe["noScope"]
    assert s["scope"] is None and s["scopePid"] is None and s["missing"] is False
    assert len(set(s["widths"])) == 1, "points were sized differently with no scope to highlight"
    assert "highlighting" not in s["info"]


def test_a_partition_lights_its_own_points_and_mutes_the_rest(probe):
    """The core of it: clicking partition 3 in the rail marks a, b on the canvas."""
    s, base = probe["partition"], probe["noScope"]["colors"]
    assert s["scope"] == ["a", "b"]
    for u in ("a", "b"):
        assert s["colors"][u] == base[u], f"{u} is in the scope but was not left at full colour"
    for u in ("c", "d", "e"):
        _, sat, light = _hsl(s["colors"][u])
        _, bsat, blight = _hsl(base[u])
        assert sat < bsat and light < blight, f"{u} is outside the scope but was not muted"


def test_muting_keeps_the_hue_so_a_dimmed_point_still_reads_as_its_class(probe):
    """A single flat grey for everything out of scope would throw away the colour-by axis."""
    base = probe["noScope"]["colors"]
    dimmed = probe["partition"]["colors"]
    for u in ("c", "d", "e"):
        assert _hsl(dimmed[u])[0] == _hsl(base[u])[0], f"{u} changed hue when it was muted"
    # 'd' (a class) and 'e' (rejected) must still be distinguishable from each other while muted
    assert dimmed["d"] != dimmed["e"]


def test_the_highlighted_points_are_painted_last_and_bigger(probe):
    """At 1-2 px, colour alone does not pick a cluster out of a crowd, and paint order decides
    whether the highlight is visible at all where the cloud is dense."""
    s = probe["partition"]
    assert sorted(s["order"]) == ["a", "b", "c", "d", "e"], "a point was dropped or painted twice"
    assert s["order"][-2:] == ["a", "b"], "an out-of-scope point was painted over the highlighted ones"
    big, small = max(s["widths"]), min(s["widths"])
    assert big > small, "highlighted points were not drawn any larger"
    lit = {u: w for u, w in zip(s["order"], s["widths"])}
    assert lit["a"] == lit["b"] == big and lit["c"] == small


def test_it_works_for_every_kind_of_scope_the_rail_lists(probe):
    """A class, the rejected bin and a sub-cluster are scopes too — the rail shows all four."""
    assert probe["klass"]["scope"] == ["d"]
    assert probe["rejected"]["scope"] == ["e"]
    assert probe["sub"]["scope"] == ["b", "c"]


def test_only_a_sub_cluster_costs_a_request(probe):
    """A class or a partition is already in the projection payload — asking the server for what the
    map is holding would put a round-trip on every click in the rail."""
    assert probe["subCalls"] == 1, "the sub-cluster's members were not fetched"
    for key in ("partition", "klass", "rejected"):
        assert probe[key]["scope"], f"{key} did not resolve from the projection payload"


def test_the_scope_is_named_by_what_the_rail_calls_it(probe):
    """'class:1' is an internal pid; the rail says 'duct' and so must the map."""
    assert "highlighting duct — 1 of 5" in probe["klass"]["info"]
    assert "highlighting partition 3 — 2 of 5" in probe["partition"]["info"]
    assert "highlighting Rejected" in probe["rejected"]["info"]
    assert "highlighting sub 2" in probe["sub"]["info"]


def test_a_scope_missing_from_the_projection_says_so_instead_of_greying_everything(probe):
    """Coords are cached, so a scope can exist in the rail and not on the map (a stale projection, a
    capped one). Muting all 5 points would read as a broken map rather than an out-of-date one."""
    s = probe["absent"]
    assert s["scope"] is None and s["missing"] is True
    assert s["colors"] == probe["noScope"]["colors"], "every point was muted for an empty scope"
    assert "no points on this map" in s["info"] and "reload" in s["info"]


def test_clearing_the_scope_restores_the_plain_map(probe):
    s = probe["cleared"]
    assert s["scope"] is None and s["missing"] is False
    assert s["colors"] == probe["noScope"]["colors"]
    assert "highlighting" not in s["info"]


def test_a_slow_sub_cluster_fetch_cannot_repaint_over_a_newer_scope(probe):
    """The only async path here. Without the generation check, clicking a sub-cluster and then a
    partition would land the sub-cluster's highlight seconds later, over the wrong scope."""
    assert probe["freshBeforeRace"] == ["a", "b"]
    assert probe["raced"]["scope"] == ["a", "b"], "a stale fetch overwrote the current scope"
    assert probe["raced"]["scopePid"] == "3"


def test_the_highlight_survives_a_change_of_colour_axis(probe):
    """Colouring by partition uses a hashed hue rather than the state palette; the muting is one
    expression over both, so it must dim there too."""
    s = probe["byPartition"]
    assert s["scope"] == ["a", "b"]
    assert _hsl(s["colors"]["a"]) == _hsl(s["colors"]["b"]), "same partition, different colour"
    for u in ("c", "d", "e"):
        assert _hsl(s["colors"][u])[1] < 20, f"{u} was not muted when colouring by partition"


def test_the_scope_line_carries_the_way_out_of_the_highlight(probe):
    """The scope is picked in the rail, but the muting is seen on the canvas — so the map itself has
    to offer the undo. Without it a highlight is a one-way door: 22 of 1411 points lit and no control
    anywhere near them that puts the other 1389 back."""
    for key in ("partition", "klass", "rejected", "sub"):
        assert 'id="mapScopeClear"' in probe[key]["info"], f"{key} highlight offers no way to clear it"
    # a scope that missed the projection mutes nothing, but it is still a picked scope to let go of
    assert 'id="mapScopeClear"' in probe["absent"]["info"]
    assert 'id="mapScopeClear"' not in probe["noScope"]["info"], "offered a clear with nothing to clear"


def test_clicking_that_control_puts_every_point_back(probe):
    s = probe["clearedByLink"]
    assert probe["beforeLinkClear"]["scope"] == ["a", "b"]
    assert s["scope"] is None and s["scopePid"] is None and s["missing"] is False
    assert s["colors"] == probe["noScope"]["colors"], "a point was left muted after clearing"
    assert len(set(s["widths"])) == 1, "the in-scope points kept their bigger square"
    assert "highlighting" not in s["info"] and 'id="mapScopeClear"' not in s["info"]
