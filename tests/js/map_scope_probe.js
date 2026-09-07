// Executes the REAL map block out of app.js against a DOM shim: picking a scope in the rail must
// light that scope's points on the Map and mute the rest.
//
// Same harness as map3d_probe.js — no browser, no bundler, the shipped source is the thing tested.
const fs = require("fs");

class El {
  constructor(id) {
    this.id = id;
    this.style = {};
    this.textContent = "";
    this.innerHTML = "";
    this.value = "";
    this.clientWidth = 800;
    this.clientHeight = 600;
    this.width = 800;
    this.height = 600;
    this._classes = new Set();
    this.classList = {
      toggle: (c, on) => { on ? this._classes.add(c) : this._classes.delete(c); },
      add: c => this._classes.add(c), remove: c => this._classes.delete(c),
      contains: c => this._classes.has(c),
    };
    this._on = {};                                      // one handler per type is all this block installs
    this.addEventListener = (type, fn) => { this._on[type] = fn; };
  }
}

// what mapDraw painted, in order: one entry per fillRect with the fillStyle in force
const drawn = [];
const ctx = {
  fillStyle: null,
  clearRect: () => { drawn.length = 0; },
  fillRect: (x, y, w, h) => drawn.push({ color: ctx.fillStyle, w, cx: x + w / 2 }),
  beginPath: () => {}, arc: () => {}, stroke: () => {}, strokeStyle: null, lineWidth: 0,
};

const els = {};
for (const id of ["mapCanvas", "mapCanvas3d", "mapStage", "mapInfo", "mapPtSize", "mapBrush", "mapLoad", "mapTip"])
  els[id] = new El(id);
els.mapCanvas.getContext = () => ctx;
els.mapPtSize.value = "2.5";
els.mapBrush.value = "26";

const $ = s => els[s.slice(1)] || null;
const $$ = () => [];
globalThis.addEventListener = () => {};                 // the block installs a resize handler at eval time
globalThis.window = { devicePixelRatio: 1 };
globalThis.document = { querySelector: () => null };

// ---- the app-side collaborators the map block reads ------------------------------------------
const SEL = new Set();
const SPIN = "";
const REJECTED_SCOPE = "__rejected__";
const SUB_PREFIX = "sub:";
const INST = { pid: null };
const isRejectedScope = () => INST.pid === REJECTED_SCOPE;
const isSubScope = () => String(INST.pid || "").startsWith(SUB_PREFIX);
const withBusy = (_sel, fn) => fn();
const escAttr = v => String(v).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const selectPartition = pid => { INST.pid = pid; return globalThis.__sync(); };   // what the "✕ clear" link calls
const renderInspector = () => {};
const mapRenderSel = () => {};
let MAP3D = null;

// 5 instances: two in FINCH partition "3", one in partition "7", one assigned to class 1, one rejected.
const POINTS = [
  { iuid: "a", x: 0.10, y: 0.10, state: "pool",   cls: null,   pid: "3",       source: "sam", score: 0.9 },
  { iuid: "b", x: 0.20, y: 0.20, state: "pool",   cls: null,   pid: "3",       source: "sam", score: 0.8 },
  { iuid: "c", x: 0.30, y: 0.30, state: "pool",   cls: null,   pid: "7",       source: "sam", score: 0.7 },
  { iuid: "d", x: 0.40, y: 0.40, state: "class",  cls: "duct", pid: "class:1", source: "sam", score: 0.6 },
  { iuid: "e", x: 0.50, y: 0.50, state: "reject", cls: null,   pid: null,      source: "sam", score: 0.5 },
];
const api = async () => ({ n: POINTS.length, method: "umap", spec: { dinov3: 1 }, points: POINTS.map(p => ({ ...p })) });

// the ONE server round-trip the highlight is allowed to make: a sub-cluster is not in the projection
let SCOPE_CALLS = 0, SCOPE_GATE = null;
const scopeIuids = async () => {
  SCOPE_CALLS++;
  if (SCOPE_GATE) await SCOPE_GATE;
  return ["b", "c"];
};

const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("const MAP = { pts:[]");
const end = src.indexOf("function mapEvtPos(");
if (start < 0 || end < 0) { console.log(JSON.stringify({ error: "could not locate the map block" })); process.exit(0); }
eval(src.slice(start, end) +
  "\nglobalThis.__peek = () => MAP;" +
  "\nglobalThis.__load = mapLoad; globalThis.__sync = mapSyncScope; globalThis.__color = mapColorOf;");

const snapshot = () => {
  const M = globalThis.__peek();
  return {
    scope: M.scope ? [...M.scope].sort() : null,
    scopePid: M.scopePid,
    missing: M.scopeMissing,
    info: $("#mapInfo").innerHTML,
    // which point each fillRect was, in paint order (recovered from the rect's centre through the
    // same view transform mapDraw used) — the highlight is only visible if it is painted LAST
    order: drawn.map(d => {
      const v = M.view, hit = POINTS.find(p => Math.abs(p.x * v.s + v.ox - d.cx) < 0.01);
      return hit ? hit.iuid : "?";
    }),
    widths: drawn.map(d => d.w),
    colors: Object.fromEntries(POINTS.map(p => [p.iuid, globalThis.__color(p)])),
  };
};
const select = async pid => { INST.pid = pid; await globalThis.__sync(); return snapshot(); };

// Clicking the "✕ clear" the scope line renders, the way the delegated handler on #mapInfo sees it.
const clickMapClear = async () => {
  const h = els.mapInfo._on.click;
  if (!h) throw new Error("#mapInfo has no click handler, so the ✕ in the scope line does nothing");
  h({ target: { closest: sel => (sel === "#mapScopeClear" ? {} : null) } });
  await new Promise(r => setTimeout(r, 0));
  return snapshot();
};

(async () => {
  const out = {};
  await globalThis.__load();
  out.noScope = snapshot();

  out.partition = await select("3");
  out.klass = await select("class:1");
  out.rejected = await select(REJECTED_SCOPE);

  SCOPE_CALLS = 0;
  out.sub = await select("sub:2");
  out.subCalls = SCOPE_CALLS;

  // a scope that this projection has no points for: dimming everything would read as a broken map
  out.absent = await select("999");

  out.cleared = await select(null);

  // a slow sub-cluster fetch must not repaint after a newer scope has been picked
  let release; SCOPE_GATE = new Promise(r => { release = r; });
  const slow = (INST.pid = "sub:2", globalThis.__sync());
  const fresh = await select("3");
  release(); await slow;
  out.raced = snapshot();
  out.freshBeforeRace = fresh.scope;

  // the way OUT of a highlight, from the view that shows it: the ✕ the scope line renders
  out.beforeLinkClear = await select("3");
  out.clearedByLink = await clickMapClear();

  // colouring by partition (a hashed hue, not the state palette) must dim the same way
  SCOPE_GATE = null;
  globalThis.__peek().colorBy = "partition";
  out.byPartition = await select("3");

  console.log(JSON.stringify(out, null, 1));
})().catch(e => console.log(JSON.stringify({ error: String(e && e.stack || e) })));
