#!/usr/bin/env python
"""Regenerate the README tutorial's screenshots and GIFs.

Builds demo projects in a scratch folder, starts a Chevron server over them, drives a headless
Chromium through the real UI with Playwright, and writes PNGs and GIFs into docs/tutorial/. The
README embeds those files, so rerun this after a UI change and commit what changed.

    python docs/tutorial/capture.py \
        --pedestrians /data/PennFudanPed/PNGImages \
        --cholec /data/cholecseg8k/video01/video01_00080/images \
        --curated /path/to/an/already-curated/project

Three projects, three jobs:
  * "Pedestrians" is built THROUGH THE UI and recorded — launcher, Set up, Get masks, Compute
    features, Cluster — so the tutorial shows the flow a new user actually goes through.
  * "Cholec" is built silently over the API; it is a second card on the launcher and the In-image
    example (several instruments and organs per frame).
  * The curated project is COPIED into the scratch folder and linked; the copy is what gets opened,
    so the original is never written to. It provides the at-scale views: a map coloured by class,
    the 1-NN suggestion bar, the classifier, the taxonomy tree.

Playwright and ffmpeg are dev-machine tools, not package dependencies: `pip install playwright &&
playwright install chromium`, and ffmpeg from your package manager. Nothing under tests/ runs this.

Dev-machine timing: the first run is dominated by SAM-HQ on ~45 images (about 15 minutes on an
Apple-silicon MPS device). `--reuse` restores the built projects from a pristine copy and skips the
build scenes, so iterating on the other shots takes a couple of minutes.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import pickle
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

REPO = Path(__file__).resolve().parents[2]
VIEWPORT = {"width": 1280, "height": 800}
GEOM = {"shape", "shapecoord", "coords"}
GIF_MAX_BYTES = 2 * 1024 * 1024

# ---------------------------------------------------------------- tiny HTTP client
class Api:
    def __init__(self, base: str):
        self.base = base

    def _req(self, method: str, path: str, body=None, timeout=3600):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode(errors='replace')}") from None

    def get(self, path, **kw):
        return self._req("GET", path, **kw)

    def post(self, path, body=None, **kw):
        return self._req("POST", path, body or {}, **kw)

    def wait_ready(self, seconds=60):
        t0 = time.time()
        while time.time() - t0 < seconds:
            try:
                self.get("/api/session", timeout=2)
                return
            except Exception:
                time.sleep(0.3)
        raise RuntimeError("server did not come up")

    def wait_idle(self, seconds=3600):
        """Block until no background job reports progress (a propose/compute_features started from
        a page we already closed keeps running server-side)."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            p = self.get("/api/progress")
            if not p.get("active"):
                return
            time.sleep(1.0)
        raise RuntimeError("background job did not finish")

    def open_project(self, pid: str):
        return self.post(f"/api/projects/{pid}/open")

    def state(self):
        return self.get("/api/state")

    def cluster(self, prefer=("dinov3", "raddino", "dinov2")):
        st = self.state()
        feats = [f for f in st["features"] if f not in GEOM and f not in st.get("feature_nan", [])]
        pick = next((p for p in prefer if p in feats), None)
        use = [pick] if pick else (feats[:1] or st["features"][:1])
        self.post("/api/cluster", {"features": use})
        return use


# ---------------------------------------------------------------- server subprocess
class Server:
    def __init__(self, root: Path, port: int):
        self.root, self.port, self.proc = root, port, None
        self.log = open(root.parent / "server.log", "w")

    def start(self):
        env = dict(os.environ)
        env.setdefault("HF_HUB_OFFLINE", "1")           # every model this needs is already cached
        # faiss and torch each bring an OpenMP runtime; on macOS the pair segfaults in an OpenMP
        # worker now and then (libomp __kmp_suspend_initialize_thread). One thread is plenty here
        # and takes the worker threads out of the picture.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "chevron.server", "--root", str(self.root), "--port", str(self.port)],
            cwd=str(REPO), env=env, stdout=self.log, stderr=subprocess.STDOUT)
        atexit.register(self.stop)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ---------------------------------------------------------------- browser helpers
# Playwright's video never shows the pointer, so a fake one follows the real mouse events and
# flashes a ring on every press. Injected into every page before its own scripts run.
CURSOR_JS = r"""
(() => {
  const mk = () => {
    if (document.getElementById('__cur') || !document.documentElement) return;
    const c = document.createElement('div'); c.id = '__cur';
    c.style.cssText = 'position:fixed;left:-50px;top:-50px;width:20px;height:28px;z-index:2147483647;'
      + 'pointer-events:none;';
    c.innerHTML = '<svg viewBox="0 0 24 32" width="20" height="28"><path d="M2 2 L2 24 L8 18.5 L12 29 '
      + 'L16 27 L12 17 L20 17 Z" fill="#fff" stroke="#111" stroke-width="1.7" stroke-linejoin="round"/></svg>';
    const r = document.createElement('div'); r.id = '__ring';
    r.style.cssText = 'position:fixed;width:12px;height:12px;border:3px solid #4f8cff;border-radius:50%;'
      + 'z-index:2147483646;pointer-events:none;opacity:0;transform:translate(-50%,-50%);';
    document.documentElement.appendChild(c); document.documentElement.appendChild(r);
  };
  const at = (x, y) => { mk(); const c = document.getElementById('__cur');
    if (c) { c.style.left = x + 'px'; c.style.top = y + 'px'; } };
  const ring = (x, y) => { mk(); const r = document.getElementById('__ring'); if (!r) return;
    r.style.transition = 'none'; r.style.left = x + 'px'; r.style.top = y + 'px';
    r.style.width = '12px'; r.style.height = '12px'; r.style.opacity = '1';
    requestAnimationFrame(() => requestAnimationFrame(() => {
      r.style.transition = 'width .4s ease-out, height .4s ease-out, opacity .4s ease-out';
      r.style.width = '40px'; r.style.height = '40px'; r.style.opacity = '0'; })); };
  document.addEventListener('mousemove', e => at(e.clientX, e.clientY), true);
  document.addEventListener('pointermove', e => at(e.clientX, e.clientY), true);
  document.addEventListener('mousedown', e => ring(e.clientX, e.clientY), true);
  document.addEventListener('DOMContentLoaded', mk);
})();
"""


class Driver:
    """One browser; a fresh context per scene. PNG scenes render at 2x, GIF scenes record at 1x."""

    def __init__(self, pw, base: str, out: Path, work: Path):
        self.pw, self.base, self.out, self.work = pw, base, out, work
        self.browser = pw.chromium.launch(headless=True)
        self.ctx = self.page = None
        self.t_page = None

    def open(self, route: str | None, *, record=False):
        self.close()
        kw = dict(viewport=VIEWPORT, device_scale_factor=1 if record else 2)
        if record:
            kw.update(record_video_dir=str(self.work / "video"), record_video_size=VIEWPORT)
        self.ctx = self.browser.new_context(**kw)
        self.ctx.add_init_script(CURSOR_JS)
        self.page = self.ctx.new_page()
        self.page.on("dialog", lambda d: d.accept())      # every confirm() in the app says yes
        self.t_page = time.time()
        url = self.base + ("/" if route is None else f"/app#/{route}")
        self.page.goto(url, wait_until="load")
        self.page.wait_for_timeout(400)
        return self.page

    def close(self):
        if self.ctx:
            self.ctx.close()
            self.ctx = self.page = None

    def png(self, name: str):
        path = self.out / f"{name}.png"
        self.page.screenshot(path=str(path))
        print(f"  wrote {path.relative_to(REPO)}")

    def gif(self, name: str, t_start: float, t_end: float):
        """Close the recording context and cut [t_start, t_end] (wall-clock) out of the video."""
        page = self.page
        self.close()
        webm = Path(page.video.path())
        ss = max(0.0, t_start - self.t_page)
        dur = max(1.0, t_end - t_start)
        path = self.out / f"{name}.gif"
        for fps, width in ((10, 960), (8, 880), (7, 800)):
            vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
                  f"[a]palettegen=max_colors=160:stats_mode=diff[p];"
                  f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.2f}", "-i", str(webm),
                            "-t", f"{dur:.2f}", "-vf", vf, str(path)], check=True)
            if path.stat().st_size <= GIF_MAX_BYTES:
                break
        print(f"  wrote {path.relative_to(REPO)} ({path.stat().st_size // 1024} kB, {dur:.0f}s)")


# ---- human-paced actions: move the pointer there first so the recording shows what is clicked
def box_center(loc):
    b = loc.bounding_box()
    if not b:
        raise RuntimeError("element has no box")
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def hmove(page, loc, steps=18):
    x, y = box_center(loc)
    page.mouse.move(x, y, steps=steps)
    page.wait_for_timeout(120)


def hclick(page, sel_or_loc, pause=500):
    loc = (page.locator(sel_or_loc) if isinstance(sel_or_loc, str) else sel_or_loc).first
    loc.scroll_into_view_if_needed()
    hmove(page, loc)
    try:
        loc.click(timeout=8_000)
    except PlaywrightTimeoutError:
        # Playwright refuses a click when another element covers the hit point (a caption row can
        # overlap a control by a pixel). The pointer is already there for the recording; fire the
        # handler directly rather than fail the scene over it.
        loc.dispatch_event("click")
    page.wait_for_timeout(pause)


def htype(page, sel: str, text: str):
    hclick(page, sel, pause=150)
    page.locator(sel).fill("")
    page.locator(sel).press_sequentially(text, delay=55)
    page.wait_for_timeout(300)


def hselect(page, sel: str, value: str, pause=600):
    # No physical click: headless Chromium would open a native popup that the recording cannot
    # show anyway, and select_option fires the same change event the app listens for.
    loc = page.locator(sel)
    loc.scroll_into_view_if_needed()
    hmove(page, loc)
    page.wait_for_timeout(250)
    loc.select_option(value)
    page.wait_for_timeout(pause)


def drag_cells(page, cells, n=6):
    """Press on the first cell and sweep across the next ones — the grid's paint-select."""
    n = min(n, cells.count())
    x0, y0 = box_center(cells.nth(0))
    page.mouse.move(x0, y0, steps=12)
    page.mouse.down()
    for i in range(1, n):
        x, y = box_center(cells.nth(i))
        page.mouse.move(x, y, steps=8)
        page.wait_for_timeout(90)
    page.mouse.up()
    page.wait_for_timeout(400)


def first_finch_partition(page, skip=0):
    rows = page.locator('#plist .prow:not([data-pid^="class:"]):not(#scopeRejected)')
    rows.first.wait_for(timeout=60_000)
    return rows.nth(skip)


def wait_grid(page, sel="#pgrid"):
    page.wait_for_selector(f"{sel} .cell img[src]", timeout=60_000)
    page.wait_for_timeout(900)          # let the lazy crops paint


# ---------------------------------------------------------------- scene registry
SCENES = []


def scene(name):
    def deco(fn):
        SCENES.append((name, fn))
        return fn
    return deco


class Ctx:
    """What every scene shares."""
    api: Api
    drv: Driver
    args: argparse.Namespace
    root: Path
    pristine: Path
    pids: dict


def ensure_open(c: Ctx, key: str, prefer=("dinov3", "raddino", "dinov2"), *, cluster=True):
    """Make `key` the active (and, by default, clustered) project. Opening resets the engine
    (clustering lives in memory only), so this only reopens when the active project differs —
    scenes that run in sequence keep whatever state the previous one left, and `--only` still
    works on its own."""
    pid = c.pids[key]
    if c.api.get("/api/session")["active"] != pid:
        c.api.open_project(pid)
    if cluster and not c.api.state()["clustered"]:
        c.api.cluster(prefer)


CXR_FEATS = ("raddino", "dinov3", "dinov2")


# ---- 1. set up a project ---------------------------------------------------------------
@scene("launcher")
def s_launcher(c: Ctx):
    page = c.drv.open(None)
    page.wait_for_selector("#grid .card", timeout=30_000)
    page.wait_for_timeout(600)
    c.drv.png("launcher")


@scene("new-project")
def s_new_project(c: Ctx):
    page = c.drv.open(None)
    page.wait_for_selector("#grid .card", timeout=30_000)
    hclick(page, "#newBtn")
    htype(page, "#pName", "Pedestrians")
    htype(page, "#pRoot", c.args.pedestrians)
    page.wait_for_timeout(400)
    c.drv.png("new-project")
    if c.args.reuse:
        page.keyboard.press("Escape")               # the project already exists; do not create twice
        c.api.open_project(c.pids["pedestrians"])
        return
    hclick(page, "#createBtn")
    page.wait_for_url(re.compile(r"/app"), timeout=30_000)
    page.wait_for_selector("#tab-setup.active #stepImages.done", timeout=30_000)
    page.wait_for_timeout(700)
    c.drv.png("setup")
    c.pids["pedestrians"] = c.api.get("/api/session")["active"]


@scene("get-masks")
def s_get_masks(c: Ctx):
    if c.args.reuse:
        return
    page = c.drv.open("setup/setup", record=True)
    page.wait_for_selector("#ingBackend option", state="attached", timeout=30_000)
    page.wait_for_timeout(800)
    t0 = time.time()
    hselect(page, "#ingBackend", "samhq_auto")
    htype(page, "#ingLimit", str(c.args.limit))
    hclick(page, "#ingRun", pause=200)
    page.wait_for_selector("#ingBar", state="visible", timeout=30_000)
    page.wait_for_timeout(22_000)                   # long enough for the counter to tick a few times
    c.drv.gif("get-masks", t0, time.time())
    c.api.wait_idle()                               # the job keeps running after the page is gone
    st = c.api.state()
    assert st["stats"]["n_instances"] > 0, "SAM produced no instances"
    print(f"  {st['stats']['n_instances']} instances")


@scene("setup-done")
def s_setup_done(c: Ctx):
    ensure_open(c, "pedestrians", cluster=False)    # step 4 is still the user's to press
    page = c.drv.open("setup/setup")
    page.wait_for_selector("#cfgExtractor option", state="attached", timeout=30_000)
    hselect(page, "#cfgExtractor", "dinov3")
    if not c.args.reuse:                            # with --reuse the features are already there
        hclick(page, "#cfgRaddino", pause=200)
        page.wait_for_selector("#stepFeats.done", timeout=1_800_000)
        c.api.wait_idle()
        # the list of models reloads once the job is done and lands on its first entry, so put the
        # one that was just computed back in view
        hselect(page, "#cfgExtractor", "dinov3")
    # all four steps in one frame: the pane scrolls inside the tab, so grow the viewport instead
    page.set_viewport_size({"width": VIEWPORT["width"], "height": 1360})
    page.evaluate("document.querySelector('#tab-setup .pane').scrollTop = 0")
    page.wait_for_timeout(800)
    c.drv.png("setup-done")
    if not c.args.reuse:
        snapshot_pristine(c, "pedestrians")


# ---- 2. clusters -----------------------------------------------------------------------
@scene("partitions")
def s_partitions(c: Ctx):
    c.api.open_project(c.pids["pedestrians"])
    page = c.drv.open("setup/setup")
    page.wait_for_selector("#stepFeats.done", timeout=30_000)
    hclick(page, "#setupCluster", pause=300)
    page.wait_for_selector("#tab-partitions.active", timeout=120_000)
    hclick(page, first_finch_partition(page))
    wait_grid(page)
    c.drv.png("partitions")


@scene("partitions-gif")
def s_partitions_gif(c: Ctx):
    ensure_open(c, "pedestrians")
    page = c.drv.open("curate/partitions", record=True)
    first_finch_partition(page).wait_for(timeout=60_000)
    page.wait_for_timeout(800)
    t0 = time.time()
    for i in range(3):
        hclick(page, first_finch_partition(page, i), pause=200)
        wait_grid(page)
        page.wait_for_timeout(700)
    hselect(page, "#levelSel", page.locator("#levelSel option").last.get_attribute("value"))
    page.wait_for_timeout(400)
    hclick(page, first_finch_partition(page, 0), pause=200)
    wait_grid(page)
    page.mouse.move(640, 500, steps=10)
    page.keyboard.press("m")                        # masks off/on
    page.wait_for_timeout(1_400)
    page.keyboard.press("m")
    page.wait_for_timeout(900)
    page.keyboard.press("c")                        # crop -> in-context
    page.wait_for_timeout(1_800)
    page.keyboard.press("c")
    page.wait_for_timeout(900)
    c.drv.gif("partitions", t0, time.time())


# ---- 3. map ----------------------------------------------------------------------------
MAP_POINT_JS = """
(k) => {
  const cv = document.getElementById('mapCanvas'), r = cv.getBoundingClientRect(), v = MAP.view;
  const pts = MAP.pts; if (!pts.length) return null;
  const xs = pts.map(p => p.x).sort((a, b) => a - b), ys = pts.map(p => p.y).sort((a, b) => a - b);
  const mx = xs[(xs.length * k) | 0], my = ys[(ys.length * k) | 0];
  let best = pts[0], bd = 1e9;
  for (const p of pts) { const d = (p.x - mx) ** 2 + (p.y - my) ** 2; if (d < bd) { bd = d; best = p; } }
  return [r.left + (best.x * v.s + v.ox) / MAP.dpr, r.top + (best.y * v.s + v.oy) / MAP.dpr];
}
"""


def wait_map(page):
    page.wait_for_function("typeof MAP !== 'undefined' && MAP.loaded", timeout=300_000)
    page.wait_for_timeout(700)


def open_pane(c: Ctx, area: str, tab: str, *, record=False):
    """Reach a pane the way a user does: land on Set up, click the area in the rail, then the tab.
    A deep link to #/ship/release renders the pane but its on-show hook runs before the pane's own
    state is declared, so the grid never loads; clicking after the page has settled is fine."""
    page = c.drv.open("setup/setup", record=record)
    page.wait_for_selector("#stepImages", timeout=60_000)
    hclick(page, f'#areas button[data-area="{area}"]', pause=400)
    hclick(page, f'nav#nav button[data-tab="{tab}"]', pause=400)
    return page


def open_curate_view(c: Ctx, tab: str, *, record=False):
    """Land on Partitions and switch to `tab` by clicking, the way a user does. A deep link straight
    to #/curate/map runs the router before the map's own state exists, so the pane shows but never
    loads; the tab click after the page has settled does."""
    page = c.drv.open("curate/partitions", record=record)
    page.wait_for_selector("#plist .prow", timeout=60_000)
    page.wait_for_timeout(500)
    hclick(page, f'nav#nav button[data-tab="{tab}"]', pause=300)
    return page


@scene("map-gif")
def s_map_gif(c: Ctx):
    ensure_open(c, "pedestrians")
    page = c.drv.open("curate/partitions", record=True)
    first_finch_partition(page).wait_for(timeout=60_000)
    page.wait_for_timeout(600)
    t0 = time.time()
    hclick(page, 'nav#nav button[data-tab="map"]', pause=200)
    wait_map(page)
    page.wait_for_timeout(600)
    x, y = page.evaluate(MAP_POINT_JS, 0.5)
    page.mouse.move(x, y, steps=20)                 # hover -> crop tooltip
    page.wait_for_timeout(1_300)
    page.mouse.wheel(0, -420)                       # zoom in about the pointer
    page.wait_for_timeout(700)
    page.mouse.wheel(0, -300)
    page.wait_for_timeout(700)
    page.mouse.down()                               # drag = pan
    page.mouse.move(x - 160, y - 90, steps=18)
    page.mouse.up()
    page.wait_for_timeout(600)
    hclick(page, "#mapReset", pause=500)
    hselect(page, "#mapColor", "partition", pause=800)
    hclick(page, "#mapMode", pause=300)             # paint-select
    x, y = page.evaluate(MAP_POINT_JS, 0.35)
    page.mouse.move(x, y, steps=12)
    page.mouse.down()
    page.mouse.move(x + 70, y + 40, steps=16)
    page.mouse.move(x + 20, y + 90, steps=16)
    page.mouse.up()
    page.wait_for_timeout(1_000)
    hclick(page, "#mapMode", pause=300)             # back to pan
    hclick(page, first_finch_partition(page, 1), pause=200)   # scope highlight from the rail
    page.wait_for_timeout(1_800)
    c.drv.gif("map", t0, time.time())


@scene("map-by-class")
def s_map_by_class(c: Ctx):
    ensure_open(c, "cxr", CXR_FEATS)
    page = open_curate_view(c, "map")
    wait_map(page)
    page.locator("#mapColor").select_option("class")
    # UMAP packs the labelled bulk into the middle and scatters outliers around it; zoom in on the
    # median point so the classes are legible rather than a coloured speck.
    x, y = page.evaluate(MAP_POINT_JS, 0.5)
    page.mouse.move(x, y, steps=8)
    for _ in range(3):
        page.mouse.wheel(0, -400)
        page.wait_for_timeout(250)
    page.mouse.move(x - 40, y + 60, steps=6)
    page.wait_for_timeout(900)
    c.drv.png("map-by-class")


# ---- 4. labelling ----------------------------------------------------------------------
def people_partitions(c: Ctx) -> list[str]:
    """Rank the Pedestrians partitions by how person-shaped their members are (tall boxes, a sane
    area) so the assign GIF labels something that actually looks like a person. Reads the scratch
    project's collection directly; nothing about the app depends on this."""
    coll = pickle.load(open(c.root / "pedestrians" / "collection.pkl", "rb"))
    geo = {r["iuid"]: (r["bh"] / max(r["bw"], 1e-6), r["mask_area_frac"]) for r in coll["records"]}
    rows = c.api.get("/api/partitions?offset=0&limit=500&kind=part")["rows"]
    scored = []
    for p in rows:
        if p["size"] < 6:
            continue
        iu = [i["iuid"] for i in c.api.get(f"/api/instances?pid={p['pid']}&offset=0&limit=100000")["items"]]
        g = [geo[u] for u in iu if u in geo]
        if not g:
            continue
        ar, area = statistics.median(x[0] for x in g), statistics.median(x[1] for x in g)
        if 0.01 < area < 0.6:
            scored.append((ar, p["pid"]))
    scored.sort(reverse=True)
    return [pid for _, pid in scored]


def partition_iuids(c: Ctx, pid: str) -> list[str]:
    return [i["iuid"] for i in c.api.get(f"/api/instances?pid={pid}&offset=0&limit=100000")["items"]]


@scene("assign-gif")
def s_assign_gif(c: Ctx):
    ensure_open(c, "pedestrians")
    pids = people_partitions(c)
    if len(pids) < 2:
        raise RuntimeError("could not find two person-shaped partitions")
    page = c.drv.open("curate/partitions", record=True)
    first_finch_partition(page).wait_for(timeout=60_000)
    page.wait_for_timeout(600)
    t0 = time.time()
    hclick(page, page.locator(f'#plist .prow[data-pid="{pids[0]}"]'), pause=200)
    wait_grid(page)
    drag_cells(page, page.locator("#pgrid .cell"), n=6)
    htype(page, "#classInput", "person")
    hclick(page, "#assignBtn", pause=1_400)
    hclick(page, page.locator(f'#plist .prow[data-pid="{pids[1]}"]'), pause=200)
    wait_grid(page)
    hclick(page, page.locator("#pgrid .cell").first, pause=300)   # the inspector needs a selection to show the input
    htype(page, "#classInput", "pedestrian")
    hclick(page, "#assignAllBtn", pause=1_800)
    c.drv.gif("assign", t0, time.time())


@scene("suggestion")
def s_suggestion(c: Ctx):
    ensure_open(c, "cxr", CXR_FEATS)
    page = c.drv.open("curate/partitions")
    first_finch_partition(page).wait_for(timeout=60_000)
    # the first partitions whose 1-NN verdict is a class (Accept enabled)
    for i in range(12):
        hclick(page, first_finch_partition(page, i), pause=200)
        wait_grid(page)
        page.wait_for_function("!document.querySelector('#psugText .spin')", timeout=60_000)
        page.wait_for_timeout(600)
        if page.locator("#psugAccept:not([disabled])").count():
            break
    try:
        page.wait_for_selector("#pgrid .cell.willAccept", timeout=20_000)
    except Exception:
        print("  (no crop inside the gate on this partition — shooting it anyway)")
    page.wait_for_timeout(500)
    c.pids["cxr_sug_index"] = i
    c.drv.png("suggestion")


@scene("accept-gif")
def s_accept_gif(c: Ctx):
    ensure_open(c, "cxr", CXR_FEATS)
    page = c.drv.open("curate/partitions", record=True)
    first_finch_partition(page).wait_for(timeout=60_000)
    page.wait_for_timeout(600)
    t0 = time.time()
    hclick(page, first_finch_partition(page, c.pids.get("cxr_sug_index", 0)), pause=200)
    wait_grid(page)
    page.wait_for_function("!document.querySelector('#psugText .spin')", timeout=60_000)
    page.wait_for_selector("#psugAccept:not([disabled])", timeout=30_000)
    page.wait_for_timeout(1_200)
    hclick(page, "#psugAccept", pause=200)
    page.wait_for_function("document.querySelector('#plist .grp') && !document.querySelector('#pgrid .cell')",
                           timeout=60_000)
    page.wait_for_timeout(1_600)
    c.drv.gif("accept", t0, time.time())


@scene("in-image")
def s_in_image(c: Ctx):
    ensure_open(c, "cholec")
    # Twenty frames of one clip cluster into three coarse groups at the default level, which colours
    # the whole frame in one hue; pick the level nearest a dozen partitions so the overlay reads.
    levels = c.api.state()["levels"]
    c.api.post("/api/level", {"level": min(levels, key=lambda l: abs(l["n"] - 12))["i"]})
    page = open_curate_view(c, "inimage")
    page.wait_for_selector("#imgSelect option", state="attached", timeout=60_000)
    page.locator("#imgSelect").select_option(index=0)
    hclick(page, "#ovLoad", pause=200)
    page.wait_for_function("document.getElementById('ovImg').naturalWidth > 0", timeout=60_000)
    wait_grid(page, "#iigrid")
    c.drv.png("in-image")


@scene("classifier")
def s_classifier(c: Ctx):
    ensure_open(c, "cxr", CXR_FEATS)
    page = c.drv.open("assist/classifier")
    page.wait_for_selector("#clfFeats input", state="attached", timeout=30_000)
    page.wait_for_timeout(500)
    hclick(page, "#clfTrain", pause=300)
    page.wait_for_function("document.getElementById('clfReport').textContent.startsWith('trained')", timeout=600_000)
    hclick(page, "#clfPredict", pause=300)
    wait_grid(page, "#clfgrid")
    c.drv.png("classifier")


# ---- 5. taxonomy -----------------------------------------------------------------------
@scene("taxonomy")
def s_taxonomy(c: Ctx):
    ensure_open(c, "cxr", CXR_FEATS)
    page = c.drv.open("classes/classes")
    page.wait_for_selector("#txTree .txsc", timeout=60_000)
    hclick(page, "#txQc", pause=300)
    page.wait_for_selector("#txReport", state="visible", timeout=120_000)
    page.wait_for_timeout(600)
    c.drv.png("taxonomy")


@scene("merge-classes-gif")
def s_merge_classes_gif(c: Ctx):
    ensure_open(c, "pedestrians")
    page = c.drv.open("classes/classes", record=True)
    page.wait_for_selector("#txTree .txtemp .mccls", timeout=60_000)
    page.wait_for_timeout(800)
    t0 = time.time()
    hclick(page, page.locator('#txTree .mccls[value="pedestrian"]'), pause=500)
    hclick(page, "#tab-classes details.section summary", pause=500)
    htype(page, "#mcInto", "person")
    hclick(page, "#mcMerge", pause=300)
    page.wait_for_function("document.getElementById('mcMsg').textContent.startsWith('moved')", timeout=60_000)
    page.wait_for_timeout(2_000)
    c.drv.gif("merge-classes", t0, time.time())


# ---- 6. export -------------------------------------------------------------------------
@scene("export")
def s_export(c: Ctx):
    ensure_open(c, "pedestrians")
    # Finish the demo project the blunt way: the next two person-shaped partitions are people, and
    # everything still in the pool is background. That is what makes images "final" for the
    # release gate, which is what the export reads.
    for pid in people_partitions(c)[:2]:
        c.api.post("/api/assign", {"iuids": partition_iuids(c, pid), "cls": "person"})
    for p in c.api.get("/api/partitions?offset=0&limit=100000&kind=part")["rows"]:
        iu = partition_iuids(c, p["pid"])
        if iu:
            c.api.post("/api/reject", {"iuids": iu})
    page = open_pane(c, "ship", "release")
    page.wait_for_selector("#relGrid .rcell", timeout=60_000)
    page.wait_for_timeout(800)
    for _ in range(3):
        hclick(page, page.locator("#relGrid .rcell .rAcc").first, pause=500)
    page.wait_for_timeout(500)
    c.drv.png("release")
    hclick(page, 'nav#nav button[data-tab="export"]', pause=400)
    hclick(page, "#exportBtn", pause=300)
    page.wait_for_function("document.getElementById('exportMsg').textContent.includes('Exported')", timeout=120_000)
    page.wait_for_timeout(500)
    c.drv.png("export")


# ---------------------------------------------------------------- project building
def snapshot_pristine(c: Ctx, name: str):
    src, dst = c.root / name, c.pristine / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  saved pristine copy of {name}")


def build_cholec(c: Ctx):
    """Silently, over the API — the launcher card and the In-image example."""
    r = c.api.post("/api/projects", {"name": "Cholec", "config": {"images": {"root": c.args.cholec}}, "open": True})
    pid = r["project"]["id"]
    c.api.post("/api/propose", {"backend": "samhq_auto", "limit": min(20, c.args.limit)})
    c.api.wait_idle()
    c.api.post("/api/compute_features", {"extractor": "dinov3"})
    c.api.wait_idle()
    st = c.api.state()
    print(f"  cholec: {st['stats']['n_instances']} instances, features {st['features']}")
    c.api.post("/api/projects/close")
    return pid


def prepare(args) -> tuple[Path, Path, Path]:
    scratch = Path(args.scratch).expanduser()
    root, pristine, work = scratch / "root", scratch / "pristine", scratch / "work"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    if (work / "video").exists():
        shutil.rmtree(work / "video")
    (work / "video").mkdir(parents=True)
    # The curated project is copied, never opened in place: the server writes state.json, history
    # and refine overlays into whatever it opens, and this run assigns and rejects things.
    cxr = scratch / "cxr"
    if cxr.exists():
        shutil.rmtree(cxr)
    shutil.copytree(args.curated, cxr, ignore=shutil.ignore_patterns("snapshots", "cluster_cache", "exports"))
    # Cholec is never on screen while it is built, so a pristine copy is reused whenever there is
    # one; Pedestrians is only restored with --reuse, since building it IS the recorded flow.
    if args.fresh and pristine.exists():
        shutil.rmtree(pristine)
    for name in ("pedestrians", "cholec"):
        src = pristine / name
        if src.exists() and (args.reuse or name == "cholec"):
            shutil.copytree(src, root / name)
        elif args.reuse:
            sys.exit(f"--reuse: no pristine copy at {src}; run once without it")
    return root, pristine, work


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pedestrians", help="folder of images built through the UI")
    ap.add_argument("--cholec", help="folder of images for the In-image example")
    ap.add_argument("--curated", help="an existing, partly curated project (copied, not touched)")
    ap.add_argument("--out", default=str(REPO / "docs" / "tutorial"))
    ap.add_argument("--scratch", default="~/.cache/chevron/tutorial")
    ap.add_argument("--port", type=int, default=7891)
    ap.add_argument("--limit", type=int, default=25, help="images to run SAM on")
    ap.add_argument("--reuse", action="store_true", help="restore the built projects; skip the build scenes")
    ap.add_argument("--fresh", action="store_true", help="discard the pristine copies and rebuild everything")
    ap.add_argument("--only", default="", help="comma-separated scene names (see --list)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(n for n, _ in SCENES))
        return
    missing = [k for k in ("pedestrians", "cholec", "curated") if not getattr(args, k)]
    if missing:
        ap.error("required: " + ", ".join(f"--{k}" for k in missing))
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    unknown = only - {n for n, _ in SCENES}
    if unknown:
        sys.exit(f"unknown scene(s): {sorted(unknown)}")

    root, pristine, work = prepare(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    srv = Server(root, args.port)
    srv.start()
    api = Api(f"http://127.0.0.1:{args.port}")
    api.wait_ready()

    c = Ctx()
    c.api, c.args, c.root, c.pristine, c.pids = api, args, root, pristine, {}
    linked = api.post("/api/projects/link", {"path": str(root.parent / "cxr"), "name": "CXR foreign objects"})
    c.pids["cxr"] = linked["project"]["id"]
    ids = {p["name"].lower(): p["id"] for p in api.get("/api/projects")["projects"]}
    if "cholec" in ids:
        c.pids["cholec"] = ids["cholec"]
        api.post(f"/api/projects/{c.pids['cholec']}/rename", {"name": "Cholec"})
    else:
        print("building Cholec over the API…")
        c.pids["cholec"] = build_cholec(c)
        snapshot_pristine(c, "cholec")
    if args.reuse:
        c.pids["pedestrians"] = ids["pedestrians"]
        api.post(f"/api/projects/{c.pids['pedestrians']}/rename", {"name": "Pedestrians"})

    with sync_playwright() as pw:
        c.drv = Driver(pw, api.base, out, work)
        try:
            for name, fn in SCENES:
                if only and name not in only:
                    continue
                print(f"scene {name}")
                fn(c)
        finally:
            c.drv.close()
            c.drv.browser.close()
    srv.stop()
    print("done")


if __name__ == "__main__":
    main()
