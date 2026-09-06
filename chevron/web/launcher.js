"use strict";
// Chevron launcher: pick or create a project, then hand off to the curator shell at /app.
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

  if (p.error) {
    return `<div class="card" data-id="${esc(p.id)}">
      <h3>${esc(p.name)}</h3>
      <div class="sub"><span class="pill" style="border-color:var(--warn);color:var(--warn)">unreadable</span></div>
      <div class="sub">${esc(p.error)}</div>
      <footer><span class="grow"></span>
        <button class="btn ghost danger" data-act="delete" data-id="${esc(p.id)}">Delete</button>
      </footer></div>`;
  }

  const pills = [
    isActive ? `<span class="pill live">open</span>` : "",
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
      <button class="btn ghost danger" data-act="delete" data-id="${esc(p.id)}">Delete</button>
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
  if (imageRoot) config.image_root = imageRoot;
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

// ------------------------------------------------------------------- wiring
document.addEventListener("click", (ev) => {
  const t = ev.target.closest("[data-act]");
  if (!t) return;
  const { act, id } = t.dataset;
  if (act === "open") openProject(id);
  else if (act === "rename") renameProject(id);
  else if (act === "delete") deleteProject(id);
  else if (act === "new") openDialog();
});

$("newBtn").addEventListener("click", openDialog);
$("emptyNew").addEventListener("click", openDialog);
$("cancelBtn").addEventListener("click", closeDialog);
$("createBtn").addEventListener("click", createProject);
$("search").addEventListener("input", render);
$("pName").addEventListener("input", () => {
  const v = $("pName").value.trim();
  $("slugPreview").textContent = v ? slugify(v) : "—";
});
$("scrim").addEventListener("click", (ev) => { if (ev.target === $("scrim")) closeDialog(); });
document.addEventListener("keydown", (ev) => {
  if (!$("scrim").classList.contains("on")) return;
  if (ev.key === "Escape") closeDialog();
  if (ev.key === "Enter" && ev.target.tagName === "INPUT") createProject();
});

load();
