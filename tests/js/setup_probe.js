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
                  "setupRoot", "setupRootEdit", "setupRootSave", "setupRootMsg",
                  "setupCluster", "setupClusterNote"])
  els[id] = new El(id);
const $ = s => els[s.slice(1)] || null;
const $$ = () => [];                       // no feature checkboxes in the shim
const window = { _caps: null };            // capabilities arrive with /api/state
const document = { activeElement: null };  // the root editor is not overwritten while it has focus

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
// The Cluster button falls back to the selector's own default check-set (its checkboxes live on a pane
// the user cannot see from here), so the REAL defaultFeatSet is pulled in rather than stubbed.
const dfStart = src.indexOf("function defaultFeatSet()");
const dfEnd = src.indexOf("function featBoxes(");
if (dfStart < 0 || dfEnd < 0) { console.log(JSON.stringify({ error: "could not locate defaultFeatSet" })); process.exit(0); }
eval(src.slice(dfStart, dfEnd));
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
  window._features = ["dinov3"]; window._featureNan = [];
  SETUP = { n_instances: 300, features: ["dinov3"], clustered: false, image_root: "/data/imgs" };
  setupSync();
  posted.length = 0; routed.length = 0;
  await els.setupCluster.onclick();
  out.clusterOk = { url: posted[0] && posted[0].url, routed: routed.slice() };

  // With no checkbox reachable from this pane ($$ returns none, as in the real DOM before Curate has
  // been opened), the button must still send the features the project actually has — clustering on the
  // embedding when there is one, on geometry when there is not, and refusing only when there is nothing.
  const clusterPost = async (features) => {
    window._features = features; window._featureNan = [];
    SETUP = { n_instances: 300, features, clustered: false, image_root: "/data/imgs" };
    setupSync();
    posted.length = 0; routed.length = 0;
    NEXT_RESPONSE = { ok: true };
    await els.setupCluster.onclick();
    return { features: posted[0] && posted[0].body.features, posts: posted.length,
             note: els.setupClusterNote.textContent };
  };
  out.defaultEmbedding = await clusterPost(["shapecoord", "coords", "dinov3"]);
  out.defaultDecoder = await clusterPost(["decoder", "shape", "coords", "dinov3"]);
  out.defaultGeomOnly = await clusterPost(["shapecoord", "coords"]);
  out.defaultNothing = await clusterPost([]);
  window._features = []; window._featureNan = [];

  // a server-side refusal must not silently look like success
  window._features = ["dinov3"];
  SETUP = { n_instances: 300, features: ["dinov3"], clustered: false, image_root: "/data/imgs" };
  setupSync();
  NEXT_RESPONSE = { detail: "no usable features" };
  routed.length = 0;
  await els.setupCluster.onclick();
  out.clusterError = { routed: routed.slice(), note: els.setupClusterNote.innerHTML };

  // ---- step 1 is editable: the root can be corrected without rebuilding the project ----
  SETUP = { n_instances: 0, features: [], clustered: false, image_root: "/data/imgs" };
  setupSync();
  out.rootEditorPrefilled = els.setupRootEdit.value;

  // a path being typed survives the background state refresh
  document.activeElement = els.setupRootEdit;
  els.setupRootEdit.value = "/half/typed/pa";
  setupSync();
  out.rootEditorWhileTyping = els.setupRootEdit.value;
  document.activeElement = null;

  posted.length = 0;
  NEXT_RESPONSE = { ok: true, image_root: "/data/other", n_images: 80, capped: false };
  els.setupRootEdit.value = "  /data/other  ";
  await els.setupRootSave.onclick();
  out.rootSaved = { url: posted[0] && posted[0].url, body: posted[0] && posted[0].body,
                    msg: els.setupRootMsg.textContent };

  NEXT_RESPONSE = { ok: true, image_root: "", n_images: 0, capped: false };
  els.setupRootEdit.value = "";
  await els.setupRootSave.onclick();
  out.rootCleared = { body: posted[1] && posted[1].body, msg: els.setupRootMsg.textContent };

  NEXT_RESPONSE = { detail: "not a folder on this machine: /nope" };
  els.setupRootEdit.value = "/nope";
  await els.setupRootSave.onclick();
  out.rootRejected = { msg: els.setupRootMsg.innerHTML };

  console.log(JSON.stringify(out));
})();
