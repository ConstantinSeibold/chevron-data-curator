// Executes the REAL inspector block out of app.js against a DOM shim: with nothing selected the rail
// must stop presenting itself as a selection panel, and the verbs it still shows must be honest.
//
// Same harness as map_scope_probe.js / router_probe.js — no browser, no bundler, the shipped source
// is the thing tested.
const fs = require("fs");

class El {
  constructor(id) {
    this.id = id;
    this.style = {};
    this.textContent = "";
    this.innerHTML = "";
    this.value = "";
    this.disabled = false;
    this._on = {};
    this.addEventListener = (type, fn) => { this._on[type] = fn; };
    this.querySelector = () => null;
  }
}

const IDS = ["inspN", "inspSelLine", "inspEmpty", "inspActs", "inspStrip", "inspScope", "inspScopeBlock",
             "classInput", "assignBtn", "assignAllBtn", "rejectBtn", "rejectAllBtn", "unassignBtn",
             "subTarget", "psugReport"];
const els = {};
for (const id of IDS) els[id] = new El(id);

// The rail row for the current scope, the way the rail renders it: first <span> carries the name.
let scopeRow = null;
const $ = s => {
  if (s === ".prow.sel") return scopeRow;
  return els[s.slice(1)] || null;
};
const $$ = () => [];
globalThis.window = {};
globalThis.document = { querySelector: () => null, getElementById: id => els[id] || null };

const SEL = new Set();
const REJECTED_SCOPE = "__rejected__";
const SUB_PREFIX = "sub:";
const INST = { pid: null };
const isRejectedScope = () => INST.pid === REJECTED_SCOPE;
const isSubScope = () => String(INST.pid || "").startsWith(SUB_PREFIX);
const escAttr = v => String(v).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const pGrid = { sel: SEL };

const src = fs.readFileSync(process.argv[2], "utf8");
const slice = (from, to, what) => {
  const a = src.indexOf(from), b = src.indexOf(to, a + 1);
  if (a < 0 || b < 0) { console.log(JSON.stringify({ error: `could not locate ${what}` })); process.exit(0); }
  return src.slice(a, b);
};
// the gate registry, the two functions that render the rail, and the gate registrations for the
// buttons the rail shows — each lifted verbatim out of app.js
const gates = slice("const _GATES = [];", "// The shared #classList datalist", "the gate registry");
const render = slice("function renderInspector(){", "\nconst SUBS = {", "renderInspector");
const scope = slice("// The rail's header line for the scope", "\n// Most-likely-class", "syncScopeUI");
const gateLines = src.split("\n")
  .filter(l => /^gate\("#assign(All)?Btn"|^const hasCls =|^\$\("#classInput"\)\.addEventListener/.test(l.trim()))
  .join("\n");
if (!/assignAllBtn/.test(gateLines) || !/#assignBtn/.test(gateLines)) {
  console.log(JSON.stringify({ error: "could not locate the assign gates" })); process.exit(0);
}
eval([gates, render, scope, gateLines].join("\n"));

const shown = el => el.style.display !== "none";
const snapshot = () => {
  refreshGates();
  return {
    count: shown(els.inspN),                 // the big numeral
    countLine: shown(els.inspSelLine),       // "N selected"
    hint: shown(els.inspEmpty),
    selActs: shown(els.inspActs),
    scopeBlock: shown(els.inspScopeBlock),
    scopeText: els.inspScope.textContent,
    assign: !els.assignBtn.disabled,
    assignAll: !els.assignAllBtn.disabled,
    rejectAll: !els.rejectAllBtn.disabled,
  };
};

const out = {};

// no scope at all, nothing selected — the state the app opens in
renderInspector();
out.cold = snapshot();

// a scope is picked in the rail; still nothing selected
INST.pid = "3";
scopeRow = { querySelector: () => ({ textContent: " 3 [duct] " }) };
syncScopeUI();
renderInspector();
out.scopeNoSel = snapshot();

// a class name is typed: the whole-scope assign becomes real
els.classInput.value = "duct";
out.scopeNoSelWithClass = snapshot();
out.classInputListens = typeof els.classInput._on.input === "function";

// two instances selected
SEL.add("a"); SEL.add("b");
renderInspector();
out.withSel = snapshot();

// the selection is dropped again
SEL.clear();
renderInspector();
out.clearedAgain = snapshot();

// the rejected bin: neither whole-scope verb applies there, so the group must not linger empty
INST.pid = REJECTED_SCOPE;
scopeRow = { querySelector: () => ({ textContent: "Rejected" }) };
syncScopeUI();
renderInspector();
out.rejectedBin = snapshot();

console.log(JSON.stringify(out));
