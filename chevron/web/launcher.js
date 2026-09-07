"use strict";
// Chevron launcher: pick, create or adopt a project, then hand off to the curator shell at /app.
// "Add existing" scans a path the user types and LINKS what it finds — the folder stays where it is,
// which is how projects that predate the launcher (or sit next to their images on another volume)
// get in without being moved.
// Deliberately dependency-free and self-contained, like the rest of the frontend.

const $ = (id) => document.getElementById(id);
let PROJECTS = [];
let ACTIVE = null;

// ---------------------------------------------------------------- utilities
async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || body.error || `${r.status} ${r.statusText}`);
  return body;
}
const post = (path, body) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}),
});

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Mirrors chevron.projects.slugify so the folder-name preview matches what the server will do.
const slugify = (s) =>
  (String(s || "").trim().replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-+|-+$/g, "").toLowerCase()) || "project";

function ago(ts) {
  if (!ts) return "never";
  const s = Date.now() / 1000 - ts;
  if (s < 90) return "just now";
  const units = [[60, "min"], [3600, "hour"], [86400, "day"], [604800, "week"]];
  for (const [div, name] of units) {
    const next = div * (name === "min" ? 60 : name === "hour" ? 24 : name === "day" ? 7 : 1e9);
    if (s < next) { const n = Math.round(s / div); return `${n} ${name}${n === 1 ? "" : "s"} ago`; }
  }
  return new Date(ts * 1000).toLocaleDateString();
}
const num = (n) => (n >= 10000 ? `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k` : String(n ?? 0));

function showError(msg) {
  const e = $("err");
  if (!msg) { e.style.display = "none"; return; }
  e.textContent = msg;
  e.style.display = "";
}

// ------------------------------------------------------------------ render
function cardHTML(p) {
  const live = (p.n_assigned || 0) + (p.n_rejected || 0) + (p.n_unassigned || 0);
  const donePct = live ? (100 * p.n_assigned) / live : 0;
  const rejPct = live ? (100 * p.n_rejected) / live : 0;
  const isActive = p.id === ACTIVE;

  // A linked project lives outside the root, so dropping it from the launcher must forget the link
  // rather than delete a folder the user keeps somewhere of their own choosing.
  const removeBtn = p.linked
    ? `<button class="btn ghost" data-act="unlink" data-id="${esc(p.id)}" title="Forget this project — the folder is left untouched">Remove</button>`
    : `<button class="btn ghost danger" data-act="delete" data-id="${esc(p.id)}">Delete</button>`;

  if (p.error) {
    return `<div class="card" data-id="${esc(p.id)}">
      <h3>${esc(p.name)}</h3>
      <div class="sub"><span class="pill" style="border-color:var(--warn);color:var(--warn)">unreadable</span></div>
      <div class="sub">${esc(p.error)}</div>
      ${p.linked ? `<div class="path" title="${esc(p.path)}">${esc(p.path)}</div>` : ""}
      <footer><span class="grow"></span>${removeBtn}</footer></div>`;
  }

  const pills = [
    isActive ? `<span class="pill live">open</span>` : "",
    p.linked ? `<span class="pill link" title="${esc(p.path)}">linked</span>` : "",
    p.mode !== "instance" ? `<span class="pill">${esc(p.mode)}</span>` : "",
    p.modality !== "image" ? `<span class="pill">${esc(p.modality)}</span>` : "",
    ...(p.sources || []).slice(0, 2).map((s) => `<span class="pill">${esc(s)}</span>`),
  ].filter(Boolean).join("");

  return `<div class="card ${isActive ? "active" : ""}" data-id="${esc(p.id)}">
    <div>
      <h3 title="${esc(p.name)}">${esc(p.name)}</h3>
      <div class="sub">${pills}<span>${esc(ago(p.modified))}</span></div>
    </div>
    <div class="bar" title="${p.n_assigned} assigned · ${p.n_rejected} rejected · ${p.n_unassigned} unreviewed">
      <i class="done" style="width:${donePct}%"></i><i class="rej" style="width:${rejPct}%"></i>
    </div>
    <div class="stats">
      <div class="stat"><b>${num(p.n_instances)}</b><span>instances</span></div>
      <div class="stat"><b>${num(p.n_classes)}</b><span>classes</span></div>
      <div class="stat"><b>${(p.pct_curated ?? 0).toFixed(0)}%</b><span>curated</span></div>
    </div>
    <footer>
      <button class="btn primary" data-act="open" data-id="${esc(p.id)}">Open</button>
      <span class="grow"></span>
      <button class="btn ghost" data-act="rename" data-id="${esc(p.id)}">Rename</button>
      ${removeBtn}
    </footer>
  </div>`;
}

function render() {
  const q = $("search").value.trim().toLowerCase();
  const shown = q ? PROJECTS.filter((p) => p.name.toLowerCase().includes(q) || p.id.includes(q)) : PROJECTS;

  $("count").textContent = PROJECTS.length
    ? `${shown.length}${shown.length !== PROJECTS.length ? ` of ${PROJECTS.length}` : ""}`
    : "";

  const noneAtAll = PROJECTS.length === 0;
  $("empty").style.display = noneAtAll ? "" : "none";
  $("grid").style.display = noneAtAll ? "none" : "";
  if (noneAtAll) { $("grid").innerHTML = ""; return; }

  $("grid").innerHTML =
    shown.map(cardHTML).join("") +
    `<button class="newcard" data-act="new"><span class="plus">+</span>New project</button>`;
}

async function load() {
  try {
    const r = await api("/api/projects");
    PROJECTS = r.projects || [];
    ACTIVE = r.active;
    $("rootPath").textContent = r.root;
    showError("");
  } catch (e) {
    showError(`Could not list projects: ${e.message}`);
  }
  render();
}

// ----------------------------------------------------------------- actions
async function openProject(id) {
  showError("");
  try {
    await post(`/api/projects/${encodeURIComponent(id)}/open`);
    location.href = "/app";
  } catch (e) {
    showError(`Could not open ${id}: ${e.message}`);
  }
}

async function renameProject(id) {
  const cur = PROJECTS.find((p) => p.id === id);
  const name = prompt("Project name", cur ? cur.name : id);
  if (name === null || !name.trim()) return;
  try {
    await post(`/api/projects/${encodeURIComponent(id)}/rename`, { name: name.trim() });
    await load();
  } catch (e) { showError(e.message); }
}

async function deleteProject(id) {
  const p = PROJECTS.find((x) => x.id === id);
  const label = p ? p.name : id;
  // Irreversible and it removes labelling work — require typing the name, not just an OK.
  const typed = prompt(
    `Permanently delete "${label}" and everything in it (labels, refinements, exports)?\n` +
    `This cannot be undone.\n\nType the project name to confirm:`);
  if (typed === null) return;
  if (typed.trim() !== label) { showError("Name did not match — nothing was deleted."); return; }
  try {
    await api(`/api/projects/${encodeURIComponent(id)}`, { method: "DELETE" });
    await load();
  } catch (e) { showError(e.message); }
}

async function unlinkProject(id) {
  const p = PROJECTS.find((x) => x.id === id);
  const label = p ? p.name : id;
  // Reversible — the folder is untouched and can be added again — so one confirm is enough.
  if (!confirm(`Remove "${label}" from the launcher?\n\nThe folder stays on disk; you can add it again later.`)) return;
  try {
    await post(`/api/projects/${encodeURIComponent(id)}/unlink`);
    await load();
  } catch (e) { showError(e.message); }
}

// ------------------------------------------------------------------ dialog
function openDialog() {
  $("pName").value = "";
  $("pRoot").value = "";
  $("pThresh").value = "0.5";
  $("slugPreview").textContent = "—";
  $("scrim").classList.add("on");
  $("pName").focus();
}
function closeDialog() { $("scrim").classList.remove("on"); }

async function createProject() {
  const name = $("pName").value.trim();
  if (!name) { $("pName").focus(); return; }
  const btn = $("createBtn");
  btn.disabled = true;
  btn.textContent = "Creating…";
  const config = { score_thresh: parseFloat($("pThresh").value) || 0.5, model: {} };
  const imageRoot = $("pRoot").value.trim();
  if (imageRoot) config.images = { root: imageRoot };   // `images.root` is what the project reads
  try {
    await post("/api/projects", { name, config, open: true });
    location.href = "/app";
  } catch (e) {
    showError(`Could not create project: ${e.message}`);
    closeDialog();
    await load();
  } finally {
    btn.disabled = false;
    btn.textContent = "Create & open";
  }
}

// ------------------------------------------------- add-existing (link) dialog
let FOUND = [];                        // last scan result, in the order it is rendered

function openLinkDialog() {
  FOUND = [];
  $("lPath").value = "";
  $("found").innerHTML = "";
  $("scanNote").textContent = "";
  $("linkBtn").disabled = true;
  $("scrim2").classList.add("on");
  $("lPath").focus();
}
function closeLinkDialog() { $("scrim2").classList.remove("on"); }

function renderFound() {
  // `known_as` entries are already in the launcher: shown so the user can see the scan worked, but
  // not selectable — adding one again would be a no-op.
  $("found").innerHTML = FOUND.map((f, i) => `
    <label class="${f.known_as ? "taken" : ""}">
      <input type="checkbox" data-i="${i}" ${f.known_as ? "disabled" : "checked"}>
      <span class="who">
        <b>${esc(f.name)}</b>
        <span class="path" title="${esc(f.path)}">${esc(f.path)}</span>
      </span>
      ${f.known_as ? `<span class="pill">already added</span>` : ""}
    </label>`).join("");
  syncLinkBtn();
}

const selected = () =>
  [...$("found").querySelectorAll("input:checked:not(:disabled)")].map((c) => FOUND[+c.dataset.i]);

function syncLinkBtn() {
  const n = selected().length;
  $("linkBtn").disabled = n === 0;
  $("linkBtn").textContent = n > 1 ? `Add ${n} projects` : "Add";
}

async function scanPath() {
  const path = $("lPath").value.trim();
  if (!path) { $("lPath").focus(); return; }
  const btn = $("scanBtn");
  btn.disabled = true;
  $("scanNote").textContent = "Scanning…";
  try {
    const r = await post("/api/projects/scan", { path });
    FOUND = r.found || [];
    renderFound();
    $("scanNote").textContent = FOUND.length
      ? `${FOUND.length} project${FOUND.length === 1 ? "" : "s"} in ${r.path}`
      : `No projects in ${r.path}. A project folder is one holding state.json.`;
  } catch (e) {
    FOUND = [];
    renderFound();
    $("scanNote").textContent = e.message;
  } finally {
    btn.disabled = false;
  }
}

async function linkSelected() {
  const picks = selected();
  if (!picks.length) return;
  const btn = $("linkBtn");
  btn.disabled = true;
  btn.textContent = "Adding…";
  const failed = [];
  for (const f of picks) {
    try {
      await post("/api/projects/link", { path: f.path });
    } catch (e) { failed.push(`${f.name}: ${e.message}`); }
  }
  closeLinkDialog();
  await load();
  showError(failed.length ? `Could not add ${failed.join("; ")}` : "");
}

// ------------------------------------------------------------------- wiring
document.addEventListener("click", (ev) => {
  const t = ev.target.closest("[data-act]");
  if (!t) return;
  const { act, id } = t.dataset;
  if (act === "open") openProject(id);
  else if (act === "rename") renameProject(id);
  else if (act === "delete") deleteProject(id);
  else if (act === "unlink") unlinkProject(id);
  else if (act === "new") openDialog();
});

$("newBtn").addEventListener("click", openDialog);
$("emptyNew").addEventListener("click", openDialog);
$("cancelBtn").addEventListener("click", closeDialog);
$("createBtn").addEventListener("click", createProject);
$("addBtn").addEventListener("click", openLinkDialog);
$("emptyAdd").addEventListener("click", openLinkDialog);
$("lCancelBtn").addEventListener("click", closeLinkDialog);
$("scanBtn").addEventListener("click", scanPath);
$("linkBtn").addEventListener("click", linkSelected);
$("found").addEventListener("change", syncLinkBtn);
$("scrim2").addEventListener("click", (ev) => { if (ev.target === $("scrim2")) closeLinkDialog(); });
$("search").addEventListener("input", render);
$("pName").addEventListener("input", () => {
  const v = $("pName").value.trim();
  $("slugPreview").textContent = v ? slugify(v) : "—";
});
$("scrim").addEventListener("click", (ev) => { if (ev.target === $("scrim")) closeDialog(); });
document.addEventListener("keydown", (ev) => {
  if ($("scrim2").classList.contains("on")) {
    if (ev.key === "Escape") closeLinkDialog();
    // Enter in the path box means "scan", not "add" — the user has not seen the results yet.
    if (ev.key === "Enter" && ev.target === $("lPath")) scanPath();
    return;
  }
  if (!$("scrim").classList.contains("on")) return;
  if (ev.key === "Escape") closeDialog();
  if (ev.key === "Enter" && ev.target.tagName === "INPUT") createProject();
});

load();
