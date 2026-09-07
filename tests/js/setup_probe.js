// Runs the REAL Set-up block out of app.js against a DOM shim and records what the four steps say.
//
// The steps are the app's only account of "what do I do next", and getting them wrong is invisible:
// a project whose masks never landed still shows four boxes, so the tick is the entire signal. The
// interesting cases are the ones that look done and are not — geometry features are computed at
// ingest and would otherwise mark step 3 complete with no embedding in the project at all.
const fs = require("fs");

class El {
  constructor(id) {
    this.id = id; this.value = ""; this.textContent = ""; this.innerHTML = "";
    this.style = {}; this.disabled = false; this.onclick = null;
    this._classes = new Set();
    this.classList = {
      add: c => this._classes.add(c), remove: c => this._classes.delete(c),
      toggle: (c, on) => { if (on === undefined) on = !this._classes.has(c);
                           if (on) this._classes.add(c); else this._classes.delete(c); return on; },
      contains: c => this._classes.has(c),
    };
  }
  get cls() { return [...this._classes].sort(); }
}
const els = {};
for (const id of ["stepImages", "stepMasks", "stepFeats", "stepCluster",
                  "okImages", "okMasks", "okFeats", "okCluster", "stepMasksTitle",
                  "setupRoot", "setupCluster", "setupClusterNote"])
  els[id] = new El(id);
const $ = s => els[s.slice(1)] || null;
const $$ = () => [];                       // no feature checkboxes in the shim
const window = { _caps: null };            // capabilities arrive with /api/state

const escAttr = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const GEOM_FEATURES = ["shape", "shapecoord", "coords"];
let SETUP = { n_instances: 0, features: [], clustered: false, image_root: "" };

const posted = [], routed = [];
let NEXT_RESPONSE = { ok: true };
const post = async (url, body) => { posted.push({ url, body }); return NEXT_RESPONSE; };
const withBusy = (_sel, fn) => fn();
const refreshState = async () => {};
const showRoute = p => routed.push(p);

const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("// ---- Set up: images -> masks -> features -> clusters");
const end = src.indexOf("// ---- ingest: where a project's masks come from");
if (start < 0 || end < 0) { console.log(JSON.stringify({ error: "could not locate the Set-up block" })); process.exit(0); }
eval(src.slice(start, end));

// What the pane shows for a given project state: which steps are ticked, and which one is "you are
// here". Exactly one step should ever be `next`, and it must be the FIRST unfinished one.
function snapshot(state) {
  SETUP = state;
  setupSync();
  const names = ["Images", "Masks", "Feats", "Cluster"];
  return {
    done: names.filter(n => els["step" + n].cls.includes("done")),
    next: names.filter(n => els["step" + n].cls.includes("next")),
    notes: Object.fromEntries(names.map(n => [n, els["ok" + n].textContent])),
    root: els.setupRoot.textContent,
    clusterNote: els.setupClusterNote.textContent,
  };
}

(async () => {
  const out = {};

  out.fresh = snapshot({ n_instances: 0, features: [], clustered: false, image_root: "/data/imgs" });
  out.noRoot = snapshot({ n_instances: 0, features: [], clustered: false, image_root: "" });
  // masks in, but the only features are the geometry ones ingest derives for free
  out.geomOnly = snapshot({ n_instances: 300, features: ["shape", "shapecoord", "coords"],
                            clustered: false, image_root: "/data/imgs" });
  out.embedded = snapshot({ n_instances: 300, features: ["shape", "shapecoord", "dinov3"],
                            clustered: false, image_root: "/data/imgs" });
  out.ready = snapshot({ n_instances: 300, features: ["shapecoord", "dinov3"],
                         clustered: true, image_root: "/data/imgs" });

  // an instance-mode project calls its items masks; a sample-mode one has none
  out.maskTitleDefault = els.stepMasksTitle.textContent;
  window._caps = { masks: false };
  snapshot({ n_instances: 0, features: [], clustered: false, image_root: "/data/imgs" });
  out.maskTitleSampleMode = els.stepMasksTitle.textContent;
  window._caps = null;

  // clustering an empty project would fail server-side; say so instead of sending it
  SETUP = { n_instances: 0, features: [], clustered: false, image_root: "/data/imgs" };
  setupSync();
  posted.length = 0; routed.length = 0;
  await els.setupCluster.onclick();
  out.clusterWhenEmpty = { posts: posted.length, routed: routed.slice(), note: els.setupClusterNote.textContent };

  // the happy path clusters and moves the user into Curate
  SETUP = { n_instances: 300, features: ["dinov3"], clustered: false, image_root: "/data/imgs" };
  setupSync();
  posted.length = 0; routed.length = 0;
  await els.setupCluster.onclick();
  out.clusterOk = { url: posted[0] && posted[0].url, routed: routed.slice() };

  // a server-side refusal must not silently look like success
  NEXT_RESPONSE = { detail: "no usable features" };
  routed.length = 0;
  await els.setupCluster.onclick();
  out.clusterError = { routed: routed.slice(), note: els.setupClusterNote.innerHTML };

  console.log(JSON.stringify(out));
})();
