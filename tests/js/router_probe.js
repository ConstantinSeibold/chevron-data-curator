// Minimal DOM shim: enough to execute the real router block from app.js and observe what it does.
const fs = require("fs");

class El {
  constructor(tag, attrs = {}) {
    this.tagName = tag.toUpperCase();
    this.attrs = { ...attrs };
    this.children = [];
    this.hidden = false;
    this._classes = new Set((attrs.class || "").split(/\s+/).filter(Boolean));
    this.dataset = {};
    for (const [k, v] of Object.entries(attrs)) {
      if (k.startsWith("data-")) this.dataset[k.slice(5).replace(/-(.)/g, (_, c) => c.toUpperCase())] = v;
    }
    this.id = attrs.id || "";
    this.classList = {
      toggle: (c, on) => { on ? this._classes.add(c) : this._classes.delete(c); },
      add: c => this._classes.add(c), remove: c => this._classes.delete(c),
      contains: c => this._classes.has(c),
    };
    this.style = {};
  }
  get className() { return [...this._classes].join(" "); }
  matches(sel) {
    // supports: tag, #id, .cls, [data-x], [data-x="v"], and simple conjunctions
    const m = sel.match(/^([a-z]+)?(#[\w-]+)?(\.[\w-]+)?(\[[^\]]+\])?$/i);
    if (!m) return false;
    const [, tag, id, cls, attr] = m;
    if (tag && this.tagName !== tag.toUpperCase()) return false;
    if (id && this.id !== id.slice(1)) return false;
    if (cls && !this._classes.has(cls.slice(1))) return false;
    if (attr) {
      const a = attr.slice(1, -1);
      const eq = a.match(/^([\w-]+)="([^"]*)"$/);
      if (eq) { if (this.attrs[eq[1]] !== eq[2]) return false; }
      else if (!(a in this.attrs)) return false;
    }
    return true;
  }
}

const root = new El("body");
const all = [];
function add(parent, el) { parent.children.push(el); all.push(el); return el; }

const nav = add(root, new El("nav", { id: "nav" }));
const areas = add(root, new El("nav", { id: "areas", class: "areas" }));
const PANES = [
  ["partitions", "curate"], ["map", "curate"], ["inimage", "curate"], ["substructure", "curate"],
  ["refine", "curate"], ["classifier", "assist"], ["mergerec", "assist"],
  ["reference", "assist"], ["classes", "classes"], ["release", "ship"], ["export", "ship"], ["loop", "ship"],
  ["stats", "insights"], ["activity", "insights"], ["config", "settings"],
];
for (const [p, a] of PANES) {
  add(nav, new El("button", { "data-tab": p, "data-area": a, class: p === "partitions" ? "active" : "" }));
  add(root, new El("div", { id: `tab-${p}`, class: "tab" + (p === "partitions" ? " active" : "") }));
}
for (const a of ["curate", "assist", "classes", "ship", "insights", "settings"])
  add(areas, new El("button", { "data-area": a }));

// descendant-combinator query: "A B [C]" -> last part matched, ancestors checked loosely
function queryAll(sel) {
  const parts = sel.trim().split(/\s+/);
  const last = parts[parts.length - 1];
  const anc = parts.slice(0, -1);
  return all.filter(el => {
    if (!el.matches(last)) return false;
    if (!anc.length) return true;
    // walk up: we only have one nesting level, so check the parent chain
    const parent = all.concat([root]).find(p => p.children.includes(el));
    return anc.every(a => parent && parent.matches(a));
  });
}
global.document = {
  querySelector: s => queryAll(s)[0] || null,
  querySelectorAll: s => queryAll(s),
  addEventListener() {},
};
global.location = { hash: "" };
global.history = { replaceState(_a, _b, h) { global.location.hash = h; } };
global.addEventListener = () => {};

// --- pull the real router block out of app.js and run it -------------------
const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("// ---------- areas + panes + hash routing ----------");
const end = src.indexOf("// Header project chip");
const block = src.slice(start, end);

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const api = async () => ({});
// The panes' on-show hooks belong to app.js proper; stub them so the ROUTER can be exercised alone.
const _calls = [];
for (const f of ["syncClfFeats","syncMrFeats","syncSubFeats","loadSubLevels","loadSubList","loadClasses",
                 "refLoadClasses","trDefaults","trRefresh","showCkpt","populateImages","loadStats",
                 "loadActivity","loadClassRules","rfLoadPeers","loadRelease","mapOnShow"])
  global[f] = (...a) => _calls.push(f);
global.INST = { pid: null };
global.pGrid = { syncSel: () => _calls.push("pGrid.syncSel") };
global.renderInspector = () => _calls.push("renderInspector");

eval(block);

const _ck = b => b.classList.contains("offarea");
const vis = () => queryAll('nav#nav button[data-tab]').filter(b => !_ck(b)).map(b => b.dataset.tab);
const act = () => queryAll('#areas button[data-area]').filter(b => b.classList.contains("active")).map(b => b.dataset.area)[0];
const body = () => queryAll('.tab').filter(t => t.classList.contains("active")).map(t => t.id);

const out = { initial: { panes: vis(), area: act(), body: body() } };
_calls.length = 0;
showRoute("classifier");                       // switching AREA via a pane in another area
out.classifier = { panes: vis(), area: act(), body: body(), hash: global.location.hash, hooks: [..._calls] };
_calls.length = 0;
showRoute("export");                           // the new Ship/Export pane
out.export = { panes: vis(), area: act(), body: body(), hash: global.location.hash };
global.location.hash = "#/insights/stats";     // deep link
routeFromHash();
out.deeplink = { panes: vis(), area: act(), body: body() };
global.location.hash = "#/nope/nope";          // unknown route must fall back, not blank the UI
routeFromHash();
out.bogus = { area: act(), body: body() };

// A sandboxed iframe / opaque origin makes history.replaceState throw SecurityError. Navigation must
// survive that: the URL is a convenience, not a precondition.
global.history.replaceState = () => { const e = new Error("SecurityError"); e.name = "SecurityError"; throw e; };
Object.defineProperty(global.location, "hash", {
  get: () => "", set: () => { throw new Error("SecurityError"); }, configurable: true });
let threw = null;
try { showRoute("map"); } catch (e) { threw = String(e && e.message); }
out.sandboxed = { threw, area: act(), body: body() };
console.log(JSON.stringify(out));
