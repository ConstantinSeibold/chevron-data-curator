// Executes the REAL mapSet3D block out of app.js against a DOM shim, on a machine with no WebGL.
//
// The failure being guarded: not every display has a usable WebGL device (software/remote GL, a
// driver blocklist, a locked-down browser), and there three.js throws when the renderer is built.
// The view state must not have flipped by then, or the user is left with a hidden 2D canvas, an
// empty 3D one, and a toggle reading "2D" — a map that looks broken, silently.
const fs = require("fs");

class El {
  constructor(id) {
    this.id = id;
    this.style = {};
    this.textContent = "";
    this._classes = new Set();
    this.classList = {
      toggle: (c, on) => { on ? this._classes.add(c) : this._classes.delete(c); },
      add: c => this._classes.add(c), remove: c => this._classes.delete(c),
      contains: c => this._classes.has(c),
    };
  }
}

const els = {};
for (const id of ["mapCanvas", "mapCanvas3d", "map3d", "status", "mapBrush", "mapPtSize"]) els[id] = new El(id);
const $ = s => els[s.slice(1)] || null;
const $$ = () => [];

const log = [];
const setStatus = s => log.push(["status", s]);
const mapCanvasSize = () => log.push(["mapCanvasSize"]);
const mapDraw = () => log.push(["mapDraw"]);
const withBusy = (_sel, fn) => fn();
const api = async () => ({ points: [{ iuid: "a", x: 0.1, y: 0.2, z: 0.3 }] });
const mapColorOf = () => 0;
const renderInspector = () => {};
const refreshGates = () => {};
const SEL = new Set();
console.error = () => {};                      // the fallback logs the real error; keep stdout clean

const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("let MAP3D = null");
const end = src.indexOf('$("#map3d").onclick');
if (start < 0 || end < 0) { console.log(JSON.stringify({ error: "could not locate the mapSet3D block" })); process.exit(0); }
// `let` inside a direct eval stays eval-scoped, so the block has to hand its state out itself.
// (Function declarations DO reach this scope, which is how map3dEnsure gets stubbed below.)
eval(src.slice(start, end) + "\nglobalThis.__peek = () => ({ is3d: MAP_IS_3D, viewer: !!MAP3D });");

const snapshot = () => ({
  canvas2d: els.mapCanvas.style.display,
  canvas3d: els.mapCanvas3d.style.display,
  label: els.map3d.textContent,
  primary: els.map3d._classes.has("primary"),
  ...globalThis.__peek(),
});

(async () => {
  const out = {};

  // 1. no usable WebGL device: three.js throws while the renderer is constructed
  map3dEnsure = async () => { throw new Error("Error creating WebGL context."); };
  log.length = 0;
  // caught here so the probe can REPORT a rejection rather than dying on it — an unhandled
  // rejection is precisely the pre-fix symptom, so it has to be observable, not fatal
  out.noWebglThrew = null;
  try { await mapSet3D(true); } catch (e) { out.noWebglThrew = String(e && e.message); }
  out.noWebgl = snapshot();
  out.noWebglSaid = log.filter(l => l[0] === "status").map(l => l[1]);
  out.noWebglRedrew = log.some(l => l[0] === "mapDraw");

  // 2. the healthy path still flips, so the guard did not disable 3D outright
  const built = [];
  map3dEnsure = async () => ({
    build: p => built.push(p.length), resize() {}, frame() {}, recolor() {}, pin() {},
  });
  await mapSet3D(true);
  out.ok = snapshot();
  out.built = built;

  // 3. and toggling back restores the 2D canvas
  await mapSet3D(false);
  out.back = snapshot();

  console.log(JSON.stringify(out));
})();
