// Executes the REAL ingest block out of app.js against a DOM shim and records what it would POST.
//
// Structure tests prove the button exists; they cannot prove it sends the right thing. Ingest is
// step 1 of every project and its body is fiddly — blank fields must be omitted (the server has its
// own defaults), `coco` needs a path the other backends do not, and numbers arrive as strings.
const fs = require("fs");

class El {
  constructor(id) {
    this.id = id; this.value = ""; this.textContent = ""; this.innerHTML = "";
    this.style = {}; this.disabled = false; this.onclick = null; this.onchange = null;
    this._classes = new Set();
    this.classList = { add: c => this._classes.add(c), remove: c => this._classes.delete(c),
                       toggle: () => {}, contains: c => this._classes.has(c) };
  }
}
const els = {};
for (const id of ["ingBackend", "ingRun", "ingRoot", "ingLimit", "ingScore", "ingSource",
                  "ingCoco", "ingCocoRow", "ingImagesRow", "ingNote", "ingMsg", "ingBar"])
  els[id] = new El(id);
const $ = s => els[s.slice(1)] || null;

const escAttr = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const posted = [];
const refreshed = [];
let NEXT_RESPONSE = { ok: true, n_instances: 42, n_images: 7 };

const BACKEND_LIST = { backends: [
  { name: "coco", label: "COCO file", available: true, detail: "reads a COCO json; no model or GPU", requires: "" },
  { name: "sam_auto", label: "SAM — automatic masks", available: true, detail: "checkpoint ready", requires: "pip install segment-anything" },
  { name: "hf_seg", label: "HF Mask2Former", available: false, detail: "", requires: "pip install 'chevron-curator[embed]'" },
]};
const api = async (u) => (u === "/api/backends" ? BACKEND_LIST : {});
const post = async (u, b) => { posted.push({ url: u, body: b }); return NEXT_RESPONSE; };
const withProgress = (_bar, _status, fn) => fn();
const refreshState = () => refreshed.push(1);

const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("// ---- ingest: where a project's masks come from");
const end = src.indexOf("// The embedding-model dropdown.");
if (start < 0 || end < 0) { console.log(JSON.stringify({ error: "could not locate the ingest block" })); process.exit(0); }
eval(src.slice(start, end));

const opts = () => [...els.ingBackend.innerHTML.matchAll(/value="([^"]+)"([^>]*)>([^<]*)</g)]
  .map(m => ({ name: m[1], disabled: /disabled/.test(m[2]), text: m[3] }));

(async () => {
  const out = {};

  await loadBackends();
  out.options = opts();
  out.selected = els.ingBackend.value;
  out.noteForSelected = els.ingNote.textContent;

  // the COCO field belongs to the coco backend alone
  els.ingBackend.value = "sam_auto"; ingSyncForm();
  out.cocoRowForSam = els.ingCocoRow.style.display;

  // an unavailable backend still offers its install hint rather than vanishing
  els.ingBackend.value = "hf_seg"; ingSyncForm();
  out.noteForUnavailable = els.ingNote.textContent;

  // coco reveals its own field
  els.ingBackend.value = "coco"; ingSyncForm();
  out.cocoRowForCoco = els.ingCocoRow.style.display;

  // coco WITHOUT a path must not reach the server
  posted.length = 0;
  await els.ingRun.onclick();
  out.cocoNoPath = { posts: posted.length, msg: els.ingMsg.innerHTML };

  // coco WITH a path
  els.ingCoco.value = "/data/masks.json";
  posted.length = 0;
  await els.ingRun.onclick();
  out.cocoBody = posted[0] && posted[0].body;

  // sam with every optional field filled
  els.ingBackend.value = "sam_auto"; ingSyncForm();
  els.ingRoot.value = "  /data/images  "; els.ingLimit.value = "50";
  els.ingScore.value = "0.3"; els.ingSource.value = " sam_run1 ";
  posted.length = 0; refreshed.length = 0;
  await els.ingRun.onclick();
  out.samBody = posted[0] && posted[0].body;
  out.samUrl = posted[0] && posted[0].url;
  out.refreshedAfterSuccess = refreshed.length;
  out.successMsg = els.ingMsg.innerHTML;

  // blank optional fields must be OMITTED, not sent as empty/NaN
  els.ingRoot.value = ""; els.ingLimit.value = ""; els.ingScore.value = ""; els.ingSource.value = "";
  posted.length = 0;
  await els.ingRun.onclick();
  out.minimalBody = posted[0] && posted[0].body;

  // a server error is surfaced, and must not be mistaken for success
  NEXT_RESPONSE = { detail: "no images found (image_root='/nope')" };
  refreshed.length = 0;
  await els.ingRun.onclick();
  out.errorMsg = els.ingMsg.innerHTML;
  out.refreshedAfterError = refreshed.length;

  console.log(JSON.stringify(out));
})();
