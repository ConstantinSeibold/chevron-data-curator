"""Static contract between the frontend JS and the HTML it drives.

There are no browser tests here (deliberate: no bundler, no headless-Chrome dependency), so the DOM
coupling is guarded statically instead. A restructure's characteristic failure is a *dangling
selector* — JS reaching for an element that was renamed or removed — which fails silently in the
browser and loudly here.

Baseline when this was written: 330 selector literals in app.js, 351 ids in index.html, ZERO
dangling. That is the line to hold.

Run: pytest tests/test_frontend_contract.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "chevron" / "web"

# A "#xxx" literal is a CSS colour, not a selector.
HEX_COLOUR = re.compile(r"^(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")

# Elements created at runtime rather than declared in the HTML. Anything added here must be a
# genuine `document.createElement` / innerHTML product — not a typo being papered over.
DYNAMIC_IDS: set[str] = set()

SELECTOR_LITERAL = re.compile(r"""["'`]#([A-Za-z][A-Za-z0-9_-]*)["'`]""")
ELEMENT_ID = re.compile(r'\bid="([^"]+)"')


def _js_files() -> list[Path]:
    return sorted(p for p in WEB.rglob("*.js") if "vendor" not in p.parts)


def _pages() -> dict[str, str]:
    return {p.name: p.read_text() for p in WEB.glob("*.html")}


def _declared_ids() -> set[str]:
    """Every id declared across the served pages, plus ids the JS injects into the DOM.

    The union is deliberate: a module may live on either page, and both are same-origin siblings.
    Ids created by innerHTML are picked up by scanning the JS for `id="..."` too.
    """
    ids: set[str] = set()
    for text in _pages().values():
        ids |= set(ELEMENT_ID.findall(text))
    for p in _js_files():
        ids |= set(ELEMENT_ID.findall(p.read_text()))
    return ids | DYNAMIC_IDS


def _selectors(text: str) -> set[str]:
    return {s for s in SELECTOR_LITERAL.findall(text) if not HEX_COLOUR.match(s)}


def test_every_selector_resolves_to_an_element():
    """The core guard: no JS may reach for an id that nothing declares."""
    ids = _declared_ids()
    dangling: dict[str, set[str]] = {}
    for p in _js_files():
        missing = _selectors(p.read_text()) - ids
        if missing:
            dangling[p.name] = missing
    assert not dangling, (
        "JS selectors with no matching element id (renamed or deleted during a refactor?):\n"
        + "\n".join(f"  {f}: {sorted(v)}" for f, v in dangling.items()))


def test_gate_targets_exist():
    """`gate("#id", pred)` disables a button when its precondition is unmet. A gate pointing at a
    nonexistent id silently never fires, so the button stays enabled and acts on nothing."""
    ids = _declared_ids()
    bad: set[str] = set()
    for p in _js_files():
        bad |= set(re.findall(r'gate\("#([A-Za-z0-9_-]+)"', p.read_text())) - ids
    assert not bad, f"gate() targets that do not exist: {sorted(bad)}"


def test_no_duplicate_ids_within_a_page():
    """Duplicate ids make `querySelector` silently pick the first match — a classic source of
    'the button does nothing' after markup is copied between panes."""
    dupes = {}
    for name, text in _pages().items():
        seen, dup = set(), set()
        for i in ELEMENT_ID.findall(text):
            (dup if i in seen else seen).add(i)
        if dup:
            dupes[name] = sorted(dup)
    assert not dupes, f"duplicate element ids: {dupes}"


def test_api_paths_used_by_the_frontend_exist_on_the_server():
    """Every /api/... the UI calls must be a route the server actually serves."""
    server = (Path(__file__).resolve().parents[1] / "chevron" / "server.py").read_text()
    defined = {re.sub(r"\{[^}]+\}", "*", m)
               for m in re.findall(r'@app\.(?:get|post|delete)\("(/api/[^"]*)"', server)}
    # `doUndo` builds its path as `/api/${which}` — the one dynamic construction in the frontend.
    dynamic = {"/api/undo", "/api/redo"}
    called: set[str] = set()
    for p in _js_files():
        called |= {re.sub(r"\{[^}]+\}", "*", m)
                   for m in re.findall(r"/api/[a-z_0-9]+(?:/[a-z_0-9{}*]+)*", p.read_text())}
    missing = called - defined - dynamic
    assert not missing, f"frontend calls endpoints the server does not define: {sorted(missing)}"


@pytest.mark.parametrize("page", ["index.html", "launcher.html"])
def test_pages_reference_only_scripts_that_exist(page):
    text = (WEB / page).read_text()
    for src in re.findall(r'<script[^>]+src="([^"]+)"', text):
        assert (WEB / src.lstrip("/")).is_file(), f"{page} references a missing script: {src}"


# --------------------------------------------------------------------------- served shell
AREAS = ["curate", "assist", "classes", "ship", "insights", "settings"]
PANES = ["partitions", "map", "inimage", "refine", "classifier",
         "mergerec", "reference", "substructure", "classes", "release", "export", "loop", "stats", "activity", "config"]


def _served_app_page() -> str:
    from fastapi.testclient import TestClient
    from chevron.engine import CuratorEngine
    from chevron.server import create_app
    import tempfile
    eng = CuratorEngine(tempfile.mkdtemp())
    eng.init_project({"model": {}})
    return TestClient(create_app(engine=eng)).get("/").text


def test_served_shell_has_every_area_and_pane():
    """The shell the browser actually receives — not just the file on disk."""
    page = _served_app_page()
    for a in AREAS:
        assert f'data-area="{a}"' in page, f"area button missing from the served shell: {a}"
    for p in PANES:
        assert f'data-tab="{p}"' in page, f"pane button missing from the served shell: {p}"
        assert f'id="tab-{p}"' in page, f"pane body missing from the served shell: {p}"


# --------------------------------------------------------------------------- shared selection
# "Grid, Map and Image are views of ONE selection" is the property that makes them views rather than
# tabs. It is invisible when broken — a grid that quietly allocates its own Set still works, it just
# stops sharing — so the wiring is asserted here rather than left to be noticed.
def _app_js() -> str:
    return (WEB / "app.js").read_text()


def test_curate_views_share_one_selection():
    js = _app_js()
    assert re.search(r"^const SEL = new Set\(\);", js, re.M), "the shared selection store is gone"
    for grid in ("pGrid", "iiGrid", "clfGrid", "clfRejGrid", "clfIntGrid", "refSugGrid"):
        # the constructor is one line; an arrow callback in the args contains ';' so scan the LINE
        line = next((l for l in js.splitlines() if f"const {grid} = makeGrid(" in l), None)
        assert line, f"{grid} is no longer built with makeGrid"
        assert re.search(r",\s*SEL\s*\)", line), \
            f"{grid} does not share SEL — it would allocate a private Set and stop sharing"
    assert re.search(r"sel:\s*SEL\b", js), "MAP.sel is not the shared selection"


def test_grid_reset_does_not_wipe_the_whole_shared_selection():
    """Every grid shares one Set now, so `reset()` clearing it wholesale would discard a selection
    made in another view each time ANY grid reloaded. It must deselect only its own cells."""
    js = _app_js()
    body = re.search(r"    reset\(\)\{(.*?)\},", js, re.S)
    assert body, "makeGrid.reset is gone"
    assert "sel.clear()" not in body.group(1), \
        "reset() clears the shared selection instead of only the cells it is removing"
    assert "sel.delete" in body.group(1), "reset() no longer deselects the cells it removes"


def test_map_load_does_not_wipe_the_shared_selection():
    """mapOnShow() auto-loads the map on the first Grid->Map switch. Clearing there would discard
    what was just selected in the Grid — the exact behaviour sharing a selection exists to provide."""
    js = _app_js()
    body = re.search(r"async function mapLoad\(\)\{(.*?)\n\}", js, re.S)
    assert body, "mapLoad is gone"
    assert ".sel.clear()" not in body.group(1), \
        "mapLoad clears the shared selection; a Grid->Map switch would silently lose it"


def test_every_scope_kind_can_resolve_its_instances():
    """The whole-scope actions must work for every scope the rail can select, not just partitions."""
    js = _app_js()
    body = re.search(r"async function scopeIuids\(\)\{(.*?)\n\}", js, re.S)
    assert body, "scopeIuids is gone"
    for endpoint in ("/api/subcluster_instances", "/api/rejected", "/api/instances"):
        assert endpoint in body.group(1), f"scopeIuids cannot resolve {endpoint}"


def test_every_pane_button_declares_an_area():
    """A pane with no area would be unreachable: the router only ever shows one area's buttons.
    Scoped to `data-tab` buttons — the nav also hosts the Curate tool bar, whose buttons are not panes."""
    page = (WEB / "index.html").read_text()
    nav = re.search(r'<nav id="nav">(.*?)</nav>', page, re.S)
    assert nav, "the pane nav is gone"
    for btn in re.findall(r"<button[^>]*data-tab=[^>]*>", nav.group(1)):
        assert "data-area=" in btn, f"pane button with no data-area (unreachable): {btn}"
