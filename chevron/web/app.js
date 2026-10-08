// Chevron — custom frontend logic. Loads only windowed JSON + lazy per-instance crops, so
// responsiveness is independent of instance/partition count.
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const api = async (u, o) => (await fetch(u, o)).json();
const post = (u, b) => api(u, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(b||{})});
const enc = encodeURIComponent;

function setStatus(s){ if(!s) return; $("#status").textContent =
  `${s.n_instances} inst · ${s.n_assigned} assigned · ${s.n_unassigned} unassigned · ${s.n_background} rejected · ${s.n_classes} classes`
  + (s.n_mask_unreviewed ? ` · ${s.n_mask_unreviewed} masks unreviewed` : "");
  window._undoN = s.undo|0; window._redoN = s.redo|0; window._nBg = s.n_background|0; refreshGates();
  if(s.serial!=null){ LAST_SEEN_SERIAL = s.serial; const lr=$("#liveRefresh"); if(lr) lr.style.display="none"; } }  // our own actions advance the seen-serial

// ---- multi-session live-refresh: poll the shared state's mutation serial; another session's change -> banner ----
let LAST_SEEN_SERIAL = -1;
async function pollVersion(){
  if(document.hidden) return;
  try{ const v = await api("/api/version"); if(!v || v.serial==null) return;
    if(LAST_SEEN_SERIAL < 0) LAST_SEEN_SERIAL = v.serial;
    else if(v.serial > LAST_SEEN_SERIAL) $("#liveRefresh").style.display="";     // someone else changed the data
  }catch(e){} }
setInterval(pollVersion, 4000);
function liveRefreshActive(){ const t=document.querySelector(".tab.active")?.id;
  if(t==="tab-map"){ MAP.loaded=false; if(typeof mapLoad==="function") mapLoad(); }
  else if(t==="tab-inimage"){ if(IIMG.id) loadImage(true); else if($("#imgSelect").options.length) populateImages(""); }
  else if(t==="tab-classes"){ if(typeof loadClasses==="function") loadClasses(); }
  else if(t==="tab-release"){ if(typeof loadRelease==="function") loadRelease(true); }
  else { loadPartitions(true); if(typeof INST!=="undefined" && INST.pid) selectPartition(INST.pid); } }
$("#liveRefresh").onclick = async ()=>{ $("#liveRefresh").style.display="none";
  await refreshState(); if(typeof loadSources==="function") loadSources(); liveRefreshActive(); };  // fresh stats (advances LAST_SEEN via setStatus) + reload the active view
const escAttr = s => String(s).replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;");
const SPIN = '<span class="spin"></span>';                  // small inline spinner (reuses @keyframes spin)
const loadingBox = (t="loading…") => `<div class="loading">${SPIN}<span>${t}</span></div>`;   // centered block placeholder
// ---------- button gating: disable a button (greyed via `button:disabled`, no hover) when its
// precondition is unmet ("nothing to act on"). Each gate is [selector, ()=>enabled?]; refreshGates()
// re-evaluates them all and is called from every grid's selection onChange + on partition change. ----------
const _GATES = [];
function gate(sel, pred){ _GATES.push([sel, pred]); }
function refreshGates(){ for(const [sel, pred] of _GATES){ const b=$(sel); if(b) b.disabled = !pred(); } }
// The shared #classList datalist is the assign-from-taxonomy picker for EVERY class input (Partitions,
// In-image, Classifier, Reference, Substructure, merge target). Options are the taxonomy LEAVES grouped
// by "superclass ▸ concept" (leaf names are globally unique via concept-qualification), with the temp
// /scratch bucket flagged "not exported"; any ad-hoc class not in the taxonomy is appended bare. Free
// text still works (type a new name to create a class) — datalist is a suggestion list, not a constraint.
function buildClassList(){
  const dl = $("#classList"); if(!dl) return;
  const tx = window._txtree, seen = new Set(), opts = [];
  if(tx && tx.superclasses){
    for(const s of tx.superclasses) for(const c of (s.concepts||[])) for(const l of (c.leaves||[])){
      seen.add(l.name);
      opts.push(`<option value="${escAttr(l.name)}">${escAttr(s.name)} ▸ ${escAttr(c.name)}</option>`); }
    for(const t of (tx.temp||[])){ seen.add(t.name);
      opts.push(`<option value="${escAttr(t.name)}">scratch · not exported</option>`); }
  }
  for(const n of (window._classnames||[])) if(!seen.has(n)) opts.push(`<option value="${escAttr(n)}">`);
  dl.innerHTML = opts.join("");
  const popt = txPickOptions(); $$(".txpick").forEach(sel=>{ sel.innerHTML = popt; sel.value = ""; });
}
function setClasses(cls){ if(cls) window._classnames = cls; buildClassList(); }
async function loadTxLeaves(){ try{ window._txtree = await api("/api/taxonomy"); buildClassList(); }catch(e){} }
// Grouped <select> companion for the main assign inputs (Partitions/In-image/Classifier): visible
// "superclass ▸ concept" optgroups whose options are the concept's part LEAVES. Picking one fills the
// paired text input (data-target), so free-text + new-class creation still works.
function txPickOptions(){
  const tx = window._txtree;
  if(!tx || !tx.superclasses) return `<option value="">— seed taxonomy —</option>`;
  let h = `<option value="">▾ from taxonomy…</option>`;
  for(const s of tx.superclasses) for(const c of (s.concepts||[])){
    if(!(c.leaves||[]).length) continue;
    h += `<optgroup label="${escAttr(s.name)} ▸ ${escAttr(c.name)}">`+
         c.leaves.map(l=>`<option value="${escAttr(l.name)}">${escAttr(l.name)}${l.n?` (${l.n})`:""}</option>`).join("")+`</optgroup>`;
  }
  if((tx.temp||[]).length) h += `<optgroup label="scratch · not exported">`+
    tx.temp.map(t=>`<option value="${escAttr(t.name)}">${escAttr(t.name)}</option>`).join("")+`</optgroup>`;
  return h;
}
document.addEventListener("change", e=>{
  const s = e.target.closest && e.target.closest(".txpick"); if(!s || !s.value) return;
  const tgt = document.getElementById(s.dataset.target); if(tgt) tgt.value = s.value;
  s.value = "";
  refreshGates();                                   // picking a class from the taxonomy enables Assign
});

// ---------- global view state: masks on/off ('m' shortcut) + crop vs in-context ----------
let MASKS = true, VIEW = "crop";                    // VIEW: "crop" (bbox) | "context" (whole image)
function cropUrl(iuid){ return `/api/crop?iuid=${enc(iuid)}&max_side=256&mask=${MASKS?1:0}&context=${VIEW==='context'?1:0}`; }

// Batched lazy crop loading: instead of one GET /api/crop per visible cell (60+ round-trips per grid,
// throttled by the browser's ~6-connections-per-host cap), an IntersectionObserver collects the cells
// entering the viewport and fetches them in ONE POST /api/crops. Cells that aren't instance crops (bank
// exemplars — no cell data-iuid) are left alone. Falls back to the per-cell URL if the batch misses one.
const _cropObs = new IntersectionObserver(ents=>{
  const ready=[]; for(const e of ents){ if(e.isIntersecting){ _cropObs.unobserve(e.target); ready.push(e.target); } }
  if(ready.length) _queueCrops(ready);
}, {rootMargin:"300px"});
let _cropQ=new Set(), _cropT=null;
function _queueCrops(imgs){ imgs.forEach(im=>_cropQ.add(im)); clearTimeout(_cropT); _cropT=setTimeout(_flushCrops, 30); }
async function _flushCrops(){
  const imgs=[..._cropQ].filter(im=>im.isConnected); _cropQ=new Set();
  const byU={}; for(const im of imgs){ const u=im.closest(".cell")?.dataset.iuid; if(u)(byU[u]??=[]).push(im); }
  const iuids=Object.keys(byU); if(!iuids.length) return;
  let crops={}; try{ const r=await post("/api/crops",{iuids, mask:MASKS?1:0, context:VIEW==='context'?1:0, max_side:256}); crops=(r&&r.crops)||{}; }catch(e){}
  for(const u of iuids){ const uri=crops[u]||cropUrl(u); for(const im of byU[u]) if(im.isConnected) im.src=uri; }
}
function observeCrops(root){ (root||document).querySelectorAll(".cell img").forEach(im=>{
  if(im.closest(".cell")?.dataset.iuid && !im.getAttribute("src")) _cropObs.observe(im); }); }
// Clear the shimmer placeholder once a server image finishes (load/error don't bubble → capture phase).
document.addEventListener("load",  e=>{ if(e.target.tagName==="IMG") e.target.classList.remove("imgld"); }, true);
document.addEventListener("error", e=>{ if(e.target.tagName==="IMG") e.target.classList.remove("imgld"); }, true);
function refreshVisibleCrops(){                     // mask/view toggle: drop srcs in the ACTIVE tab + re-batch
  const tab = document.querySelector(".tab.active"); if(!tab) return;
  tab.querySelectorAll(".cell").forEach(c=>{ const u=c.dataset.iuid, im=c.querySelector("img");
    if(u && im){ im.removeAttribute("src"); im.classList.add("imgld"); _cropObs.observe(im); } });   // exemplars (no cell iuid) untouched
  if(tab.id==="tab-inimage") reloadOverlay();
  if(tab.id==="tab-refine") rfDoPreview();          // before/after are not .cell imgs → re-render with the mask flag
}
function syncViewButtons(){ $$(".viewToggle").forEach(b=> b.textContent = `view: ${VIEW} (c)`); }

// ---------- reusable selectable image grid ----------
function cell(it, cap){
  return `<div class="cell${it.mr?" mrev":""}" data-iuid="${it.iuid}" data-img="${it.image_id??''}">`+
    `<img loading="lazy" class="imgld">`+            // src set by the batched crop loader (observeCrops on append); imgld = shimmer until loaded
    `<button class="cellEdit" title="edit this mask by hand">✏️</button>`+   // hover-revealed; delegated click below
    `<span class="mrevB" title="mask reviewed by a human">✓</span>`+
    `<div class="cap" title="${cap}">${cap}</div></div>`;
}
// `shared` lets several views drive ONE selection Set. Curate's Grid, Map and Image views all pass
// SEL, so switching view keeps what you had selected — the point of making them views, not tabs.
function makeGrid(gridSel, countSel, noun="selected", onChange, shared){
  const el = $(gridSel), sel = shared || new Set();
  const upd = ()=>{ if(countSel) $(countSel).textContent = `${sel.size} ${noun}`; if(onChange) onChange(sel); };
  // Selection: click to toggle · press-and-DRAG to paint a run (first cell sets direction:
  // press an UNselected cell to sweep-select, a selected one to sweep-deselect) · SHIFT+click
  // to select the whole run between the last click (anchor) and the shift-clicked cell.
  let dragging = false, paintSel = true, anchor = null;
  const cells = ()=>[...el.querySelectorAll(".cell")];
  const setSel = (c, on)=>{ const u = c.dataset.iuid;
    if(on){ if(!sel.has(u)){ sel.add(u); c.classList.add("sel"); } }
    else  { if(sel.has(u)){ sel.delete(u); c.classList.remove("sel"); } } };
  el.addEventListener("mousedown", e=>{ if(e.target.closest("button")) return;   // in-cell action buttons (e.g. ✓/✗) click without painting a selection
    const c = e.target.closest(".cell"); if(!c) return; e.preventDefault();
    if(e.shiftKey && anchor){                                  // range-select anchor..c (DOM order) -> selected
      const cs = cells(), ai = cs.findIndex(x=>x.dataset.iuid===anchor), ti = cs.indexOf(c);
      if(ai>=0 && ti>=0){ for(let i=Math.min(ai,ti); i<=Math.max(ai,ti); i++) setSel(cs[i], true); upd(); }
      anchor = c.dataset.iuid; return;
    }
    dragging = true; paintSel = !sel.has(c.dataset.iuid); setSel(c, paintSel); upd(); anchor = c.dataset.iuid; });
  el.addEventListener("mouseover", e=>{ if(!dragging) return; const c = e.target.closest(".cell"); if(c){ setSel(c, paintSel); upd(); } });
  document.addEventListener("mouseup", ()=>{ dragging = false; });
  return {
    sel, el,
    // Clearing a grid deselects what WAS IN IT — not the whole shared selection. With `shared` in
    // play, sel.clear() here would silently discard a selection made in another view every time any
    // grid reloaded (this is the bug that bit mapLoad).
    reset(){ el.querySelectorAll(".cell").forEach(c=>sel.delete(c.dataset.iuid));
             el.innerHTML=""; anchor=null; upd(); },
    append(items, capFn){ el.insertAdjacentHTML("beforeend", items.map(it=>cell(it, capFn?capFn(it):it.caption)).join("")); observeCrops(el); },
    drop(iuids){ const s=new Set(iuids); el.querySelectorAll(".cell").forEach(c=>{ if(s.has(c.dataset.iuid)) c.remove(); }); iuids.forEach(u=>sel.delete(u)); upd(); },
    selectPage(){ el.querySelectorAll(".cell").forEach(c=>{ sel.add(c.dataset.iuid); c.classList.add("sel"); }); upd(); },
    clearSel(){ sel.clear(); el.querySelectorAll(".cell").forEach(c=>c.classList.remove("sel")); upd(); },
    // Re-apply the highlight from the shared set. Needed because another VIEW (the Map's paint-select)
    // can change the selection while this grid is off-screen; without this you would come back to a
    // count of N and no cells highlighted.
    syncSel(){ el.querySelectorAll(".cell").forEach(c=>c.classList.toggle("sel", sel.has(c.dataset.iuid))); upd(); },
    firstSelImg(){ for(const c of el.querySelectorAll(".cell.sel")) return c.dataset.img; return null; },
    msg(h){ el.innerHTML=`<div class="muted">${h}</div>`; }
  };
}

// Set-up progress. Declared HERE, above the router, and not next to the rest of the Set-up code:
// the boot `routeFromHash()` once ran during this script's own top-level pass and could call
// ON_SHOW.setup -> setupSync() immediately, which reads this. A `let` further down the file would
// still have been in its temporal dead zone at that point, so a deep link to #/setup/setup threw.
// That dispatch now runs at the end of the file, but the ordering is kept so nothing that runs
// above the router can ever reach this too early again.
let SETUP = {n_instances:0, features:[], clustered:false, image_root:""};
// Mask geometry is derived at ingest and needs no model, so it must not count as "an embedding has
// been computed" — a project holding only these has nothing that knows what its instances LOOK like.
const GEOM_FEATURES = ["shape", "shapecoord", "coords"];

// ---------- areas + panes + hash routing ----------
// The 15 flat tabs are grouped into 6 AREAS. Every pane keeps its <button data-tab> in the DOM —
// cross-view jumps (`$('nav button[data-tab="refine"]').click()`) and the UI-parity guard both rely
// on that — and the router just shows the active area's buttons and hides the rest.
//
// What a pane needs doing when it becomes visible. Same calls the old if-chain made, in one table.
const ON_SHOW = {
  classifier:  ()=> syncClfFeats(),
  mergerec:    ()=> syncMrFeats(),
  substructure:()=>{ syncSubFeats(); $("#subTarget").textContent=INST.pid||"none"; loadSubLevels(); loadSubList(); },
  classes:     ()=> loadClasses(),
  reference:   ()=> refLoadClasses(),
  loop:        ()=>{ trDefaults(); trRefresh(); },
  setup:       ()=>{ loadBackends(); loadExtractors(); loadDevice(); setupSync(); },
  config:      ()=>{ showCkpt(); },
  inimage:     ()=>{ populateImages($("#imgFilter").value); iiGrid.syncSel(); renderInspector(); },   // re-list: ✓ marks move
  stats:       ()=> loadStats(),
  activity:    ()=> loadActivity(),
  refine:      ()=>{ rqEnter(); },
  release:     ()=> loadRelease(true),
  // The Export pane states whether the gate is holding anything back — it has to be right even for
  // someone who never opens Release, so it reads the stats itself rather than waiting for that tab.
  export:      ()=> expSyncGateNote(),
  map:         ()=> mapOnShow(),
  // coming back to the Grid, re-apply highlights for anything selected in another view
  partitions:  ()=>{ pGrid.syncSel(); renderInspector(); },
};
const VIEW_PANES = ["partitions", "map", "inimage"];   // the Curate workspace views
const paneBtn   = p => $(`nav#nav button[data-tab="${p}"]`);
const areaOf    = p => paneBtn(p)?.dataset.area || "curate";
const firstPane = a => $(`nav#nav button[data-area="${a}"]`)?.dataset.tab;
let AREA = "curate", PANE = "partitions";

function showRoute(pane, {push=true}={}){
  const btn = paneBtn(pane); if(!btn) return;
  AREA = btn.dataset.area; PANE = pane;
  $$("#areas button[data-area]").forEach(b => b.classList.toggle("active", b.dataset.area===AREA));
  $$("nav#nav button[data-tab]").forEach(b => {
    // an explicit class, not the `hidden` attribute: `hidden` is only display:none via the UA
    // stylesheet, which any author rule setting `display` on a button silently defeats
    b.classList.toggle("offarea", b.dataset.area !== AREA);
    b.classList.toggle("active", b.dataset.tab===pane);
  });
  $$(".tab").forEach(t => t.classList.toggle("active", t.id===`tab-${pane}`));
  // Curate-wide tools (cluster/level/scope/source) belong to the area, not the global header
  const ct = $("#curateTools"); if(ct) ct.style.display = AREA==="curate" ? "" : "none";
  // Grid / Map / Image are VIEWS inside one workspace: they share the scope rail and the inspector,
  // so the wrapper shows for those three and hides for every other pane.
  const cw = $("#curatewrap"); if(cw) cw.classList.toggle("on", VIEW_PANES.includes(pane));
  // The URL is a convenience (deep links, refresh keeps your place) — never a precondition for
  // navigating. history.replaceState throws a SecurityError on an opaque origin (a sandboxed iframe,
  // a file:// embed), which would otherwise abort showRoute mid-way and freeze the whole nav.
  const hash = `#/${AREA}/${pane}`;
  if(push && location.hash !== hash){
    try { history.replaceState(null, "", hash); }
    catch(e){ try { location.hash = hash; } catch(e2){ /* URL is not writable here; navigation still works */ } }
  }
  try { ON_SHOW[pane]?.(); }
  catch(e){ console.error(`[chevron] on-show hook failed for "${pane}"`, e); }   // one bad pane must not wedge the router
}
// Clicking any pane button routes — including the ones hidden in another area, which is how the
// existing cross-view jumps keep working without knowing about areas.
$("#nav").onclick = e => { const b = e.target.closest("button[data-tab]"); if(b) showRoute(b.dataset.tab); };
$("#areas").onclick = e => { const b = e.target.closest("button[data-area]"); if(!b) return;
  const p = firstPane(b.dataset.area); if(p) showRoute(p); };

function routeFromHash(){
  const m = /^#\/([a-z]+)\/([a-z]+)$/.exec(location.hash || "");
  const pane = m && paneBtn(m[2]) ? m[2] : "partitions";
  showRoute(pane, {push:false});
}
addEventListener("hashchange", routeFromHash);          // deep links; refresh keeps your place
// The boot route is dispatched at the end of the file, once every pane's state exists.

// Header project chip: multi-project installs get a way back to the launcher (there was none).
(async ()=>{ try{
  const s = await api("/api/session");
  if(!s || !s.multi_project) return;
  const b = $("#projBtn"); if(!b) return;
  b.textContent = (s.project && s.project.name) ? `${s.project.name} ▾` : "Projects ▾";
  b.style.display = "";
  b.onclick = ()=>{ location.href = "/"; };
}catch(e){} })();

// ---------- Statistics ----------
async function loadStats(){
  $("#statsBody").innerHTML = loadingBox("computing…");
  const s = await withBusy("#statsRefresh", ()=>api("/api/statistics")), o = s.overview;
  const card = (v,l)=>`<div class="card"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  const hist = h => !h.counts || !h.counts.length ? "<i class='muted'>none</i>" :
    `<div class="hist">${h.counts.map(c=>`<div style="height:${Math.round(100*c/Math.max(1,...h.counts))}%" title="${c}"></div>`).join("")}</div>`+
    `<div class="muted" style="font-size:10px">${h.edges[0]} … ${h.edges[h.edges.length-1]}</div>`;
  const maxN = Math.max(1, ...s.classes.map(c=>c.n));
  const clsRows = s.classes.length ? s.classes.map(c=>
    `<tr><td>${c.class}</td><td><div class="bar" style="width:120px"><span style="width:${Math.round(100*c.n/maxN)}%"></span></div></td>`+
    `<td>${c.n}</td><td>${c.images}</td><td>${c.mean_score??'—'}</td><td class="muted">${Object.entries(c.sources).map(([k,v])=>`${k}:${v}`).join(" ")}</td></tr>`).join("")
    : "<tr><td class='muted' colspan='6'>no classes assigned yet</td></tr>";
  const ipi = s.instances_per_image, ipiMax = Math.max(1, ...ipi.map(([,v])=>v));
  const ipiBars = ipi.length ? `<div class="hist">${ipi.map(([k,v])=>`<div style="height:${Math.round(100*v/ipiMax)}%" title="${k} inst → ${v} imgs"></div>`).join("")}</div>`+
    `<div class="muted" style="font-size:10px">x = #instances per image (${ipi[0][0]}…${ipi[ipi.length-1][0]})</div>` : "<i class='muted'>—</i>";
  const co = s.cooccurrence;
  const coTable = co.classes.length ? `<table class="st cooc"><tr><th></th>${co.classes.map(c=>`<th>${c.slice(0,8)}</th>`).join("")}</tr>`+
    co.classes.map((c,i)=>`<tr><td>${c}</td>${co.matrix[i].map((v,j)=>`<td style="color:${i===j?'var(--mut)':'var(--fg)'}">${v||''}</td>`).join("")}</tr>`).join("")+
    `</table><div class="muted" style="font-size:10px">cell = #images where both classes appear</div>` : "<i class='muted'>assign ≥2 classes to see co-occurrence</i>";
  const p = s.partitions;
  $("#statsBody").innerHTML = `
    <div class="cards">
      ${card(o.instances_live,"live instances")}${card(o.assigned,"assigned")}${card(o.unassigned,"unassigned")}
      ${card(o.rejected,"rejected")}${card(o.merged_children,"merged")}${card(o.classes,"classes")}
      ${card(o.images,"images")}${card(o.pct_curated+"%","curated")}</div>
    <div class="statsec"><h4>Per class</h4><table class="st"><tr><th>class</th><th></th><th>#inst</th><th>#images</th><th>mean score</th><th>sources</th></tr>${clsRows}</table></div>
    <div class="statsec"><h4>Assignment sources</h4>${Object.keys(s.sources).length?Object.entries(s.sources).map(([k,v])=>`${k}: <b>${v}</b>`).join(" · "):"<i class='muted'>none</i>"}</div>
    <div class="statsec"><h4>Instances per image</h4>${ipiBars}</div>
    <div class="statsec"><h4>Confidence (score) distribution</h4>${hist(s.score_hist)}</div>
    <div class="statsec"><h4>Mask area distribution (fraction of image)</h4>${hist(s.area_hist)}</div>
    <div class="statsec"><h4>Class co-occurrence (top classes)</h4>${coTable}</div>
    <div class="statsec"><h4>Clustering</h4>${p.clustered?`level ${p.level} · <b>${p.n_partitions}</b> partitions · <b>${p.unassigned_pool}</b> unassigned in pool`:"<i class='muted'>not clustered</i>"}</div>`;
  $("#statsNote").textContent = `${o.instances_total} total instances`;
}
$("#statsRefresh").onclick = loadStats;

// ---------- Activity (read-only interaction timeline) ----------
const _actTs = ts => !ts ? "—" : new Date(ts*1000).toLocaleString([], {month:"short", day:"numeric", hour:"2-digit", minute:"2-digit"});
function _actDur(s){ s = Math.max(0, Math.round(s)); if(s<60) return s+"s";
  const m=Math.floor(s/60), h=Math.floor(m/60); return h ? `${h}h ${m%60}m` : `${m}m ${s%60}s`; }
async function loadActivity(){
  $("#actBody").innerHTML = loadingBox("loading…");
  const a = await withBusy("#actRefresh", ()=>api("/api/activity")), t = a.totals, sp = a.span;
  if(!sp.n_events){ $("#actBody").innerHTML = "<div class='muted'>No activity logged yet — assign/merge/refine/ingest some instances and they'll show up here.</div>"; $("#actNote").textContent=""; return; }
  const card = (v,l)=>`<div class="card"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  const spanDays = (sp.t_max-sp.t_min)/86400;
  // op breakdown (coarse category) as horizontal bars
  const cats = Object.entries(a.cat_counts).sort((x,y)=>y[1]-x[1]);
  const maxCat = Math.max(1, ...cats.map(([,n])=>n));
  const catRows = cats.map(([c,n])=>
    `<tr><td>${escAttr(c)}</td><td><div class="bar" style="width:160px"><span style="width:${Math.round(100*n/maxCat)}%"></span></div></td>`+
    `<td>${n}</td><td class="muted">${a.cat_insts[c]||0}</td></tr>`).join("");
  // activity over time — commands per time-bin (height %), tooltip carries both counts
  const tl = a.timeline, maxC = Math.max(1, ...tl.counts);
  const tlBars = `<div class="hist">${tl.counts.map((c,i)=>
    `<div style="height:${Math.round(100*c/maxC)}%" title="${_actTs(tl.edges[i])} · ${c} cmds · ${tl.insts[i]} inst touched"></div>`).join("")}</div>`
    + `<div class="muted" style="font-size:10px;padding:0">${_actTs(sp.t_min)} … ${_actTs(sp.t_max)} · bar height = commands per time-bin (hover for instances)</div>`;
  // merge decisions
  const mg = a.merge, acc = mg.by_kind.merge||0, rej = mg.by_kind.reject||0, mgTot = acc+rej;
  const mgBar = mgTot ? `<div class="bar" style="width:240px;height:18px"><span style="width:${Math.round(100*acc/mgTot)}%"></span></div>`+
    `<div class="muted" style="font-size:11px;padding:4px 0">accepted <b>${acc}</b> (${mg.instances} inst) · rejected <b>${rej}</b> · sources: ${Object.entries(mg.by_source).map(([k,v])=>`${escAttr(k)} ${v}`).join(" · ")||"—"}</div>`
    : "<i class='muted'>no merge decisions yet</i>";
  // ingest runs
  const igRows = a.ingests.length ? a.ingests.slice().reverse().map(g=>
    `<tr><td>${escAttr(g.ingest_id)}</td><td>${_actTs(g.ts)}</td><td>${g.n_images}</td><td>${g.n_instances}</td>`+
    `<td>${g.score_thresh??'—'}</td><td class="muted">${escAttr(g.mode||'—')}</td></tr>`).join("")
    : "<tr><td class='muted' colspan='6'>no inference runs logged</td></tr>";
  // retrain lineage + metric trajectory
  const ln = a.lineage;
  const lnRows = ln.length ? ln.map(e=>
    `<tr><td>${_actTs(e.ts)}</td><td>${escAttr(e.metric_name||'—')}</td><td>${e.metric!=null?(+e.metric).toFixed(3):'—'}</td>`+
    `<td class="muted">${escAttr(e.ckpt||'—')}</td><td>${e.n_assigned??'—'}</td>`+
    `<td>${e.regressed?'<span class="badge warn">regressed</span>':'<span class="badge ok">adopted</span>'}</td></tr>`).join("")
    : "<tr><td class='muted' colspan='6'>no retrain/adopt events</td></tr>";
  const mvals = ln.filter(e=>e.metric!=null).map(e=>+e.metric);
  const lnTraj = mvals.length>1 ? `<div class="hist" style="height:60px;max-width:${Math.max(120,mvals.length*16)}px">${mvals.map(v=>`<div style="height:${Math.round(100*v/Math.max(...mvals))}%" title="${v.toFixed(3)}"></div>`).join("")}</div><div class="muted" style="font-size:10px;padding:0">metric per adopted checkpoint (oldest→newest)</div>` : "";
  // sessions
  const ss = a.sessions.slice().reverse();
  const ssRows = ss.length ? ss.map(s=>{
    const top = Object.entries(s.ops).sort((x,y)=>y[1]-x[1]).slice(0,4).map(([k,v])=>`${escAttr(k)}:${v}`).join(" ");
    return `<tr><td>${_actTs(s.start)}</td><td>${_actDur(s.dur_s)}</td><td>${s.n_ops}</td><td class="muted">${top}</td></tr>`;
  }).join("") : "<tr><td class='muted' colspan='4'>—</td></tr>";

  $("#actBody").innerHTML = `
    <div class="cards">
      ${card(t.commands,"commands")}${card(t.instances_touched,"instances touched")}
      ${card(t.merges+"/"+t.merge_rejects,"merges ✓/✗")}${card(t.ingests,"inference runs")}
      ${card(t.retrains,"retrains")}${card(sp.n_sessions,"sessions")}
      ${card(spanDays>=1?spanDays.toFixed(1)+"d":_actDur(sp.t_max-sp.t_min),"time span")}
      ${card(t.undos+"/"+t.redos,"undo/redo")}</div>
    <div class="statsec"><h4>Activity over time</h4>${tlBars}</div>
    <div class="statsec"><h4>Operations</h4><table class="st"><tr><th>operation</th><th></th><th>#cmds</th><th>#inst</th></tr>${catRows}</table></div>
    <div class="statsec"><h4>Merge decisions</h4>${mgBar}</div>
    <div class="statsec"><h4>Inference runs (ingests)</h4><table class="st"><tr><th>id</th><th>when</th><th>#imgs</th><th>#inst</th><th>score≥</th><th>mode</th></tr>${igRows}</table></div>
    <div class="statsec"><h4>Retrain lineage</h4><table class="st"><tr><th>when</th><th>metric</th><th>value</th><th>ckpt</th><th>#assigned</th><th>status</th></tr>${lnRows}</table>${lnTraj}</div>
    <div class="statsec"><h4>Sessions <span class="muted" style="font-size:10px">(idle gap &gt; 30 min = new session)</span></h4><table class="st"><tr><th>start</th><th>duration</th><th>#ops</th><th>top ops</th></tr>${ssRows}</table></div>`;
  $("#actNote").textContent = `${sp.n_events} events · ${_actTs(sp.t_min)} → ${_actTs(sp.t_max)}`;
}
$("#actRefresh").onclick = loadActivity;

// ---------- state / cluster / undo ----------
// Sample mode has no masks, so the mask-only tools are not merely useless — offering them is a lie
// about what the project can do. The server decides (state.capabilities); the UI never re-derives it.
function applyCapabilities(caps){
  if(!caps) return;
  window._caps = caps;
  const hide = (sel, on) => { const el=$(sel); if(el) el.style.display = on ? "" : "none"; };
  // mask-only PANES drop out of the nav entirely
  for(const [pane, ok] of [["refine", caps.refine], ["mergerec", caps.merge],
                           ["substructure", caps.substructure]]){
    const b = $(`nav#nav button[data-tab="${pane}"]`);
    if(b) b.style.display = ok ? "" : "none";
  }
  hide("#mergeBtn", caps.merge);
  hide("#iiMergeSection", caps.merge);
  hide("#toRefineBtn", caps.refine);
  const ex = $("#exportBtn");
  if(ex) ex.textContent = caps.coco_export ? "Export COCO" : "Export manifest (CSV + JSON)";
}

async function refreshState(){
  const st = await api("/api/state");
  applyCapabilities(st.capabilities);
  setStatus(st.stats); setClasses(st.classes); loadTxLeaves();
  window._modelcfg = st.model_config; window._modelckpt = st.model_ckpt;
  const qc=$("#ingQsegCkpt"); if(qc && !qc.value && st.model_ckpt) qc.value = st.model_ckpt;
  const qo=$("#ingQsegOvr"); if(qo && !qo.value && st.model_overrides) qo.value = st.model_overrides.join(" ");
  const qj=$("#ingQsegJson"); if(qj && !qj.value && st.model_train_json) qj.value = st.model_train_json;
  $("#levelSel").innerHTML = st.levels.map(l=>`<option value="${l.i}" ${l.i===st.level?'selected':''}>L${l.i} (${l.n})</option>`).join("");
  window._featureNan = st.feature_nan || [];       // features with NaN/inf -> non-selectable in the classifier
  refreshFeatures(st.features);                    // builds #feats + all selectors + the Set-up readout
  loadIngests(); loadSources();
  SETUP = {n_instances: st.stats?.n_instances|0, features: st.features||[],
           clustered: !!st.clustered, image_root: st.image_root||""};
  setupSync();
  // An empty project has nothing to cluster, and the fix lives in another area — say so where the
  // user is looking rather than leaving them at "Cluster, then pick a scope on the left".
  if(SETUP.n_instances === 0){
    $("#pgrid").innerHTML = `<div class="muted">No instances yet. `
      + `<a href="#/setup/setup" id="emptyGetMasks">Set up this project</a> — get masks for your images, `
      + `then compute features.</div>`;
    // and land there rather than only pointing at it: a project with nothing in it cannot use any
    // other area, and the user arriving at an empty grid is exactly who needs step 1. Only on a bare
    // entry — an explicit deep link (including a reload) still wins.
    if(!location.hash || location.hash === "#/curate/partitions") showRoute("setup");
  }
  // Classes are scopes whether or not anything is clustered — and a cluster lives only in memory, so a
  // freshly opened project never is. Gating the rail on `clustered` left it empty on every open.
  if(SETUP.n_instances !== 0) loadPartitions(true);
}
// SCOPE: each (re)inference run is recorded as an "ingest"; scoping to one restricts the cluster pool +
// image picker to ITS instances (e.g. "show only the latest, lower-threshold preds"). 'all' clears it.
async function loadIngests(){
  try{
    const r = await api("/api/ingests"), cur = r.scope || "all";
    const opts = [`<option value="all" ${cur==="all"?"selected":""}>all instances</option>`];
    for(const g of (r.ingests||[])){
      const lab = `${g.ingest_id} · ${g.n_live}/${g.n_instances} live · ${g.n_images} imgs`
        + (g.mode?` · ${g.mode}`:"") + (g.score_thresh!=null?` @${g.score_thresh}`:"");
      opts.push(`<option value="${g.ingest_id}" ${cur===g.ingest_id?"selected":""}>${lab}</option>`);
    }
    $("#scopeSel").innerHTML = opts.join("");
  }catch(e){}
}
// ---- proposal-source facet (which model proposed an instance) — multi-select, respected in every tab ----
let SRC = { all:[], active:null, mall:[], mactive:null };   // active / mactive: null = all sources / methods
// One facet strip, two dimensions: which model PROPOSED an instance (source) and which generator made its
// CURRENT mask (method — re-mask backend, a per-box pick, hand-drawn). Each part shows only when it has
// more than one value, so a single-source, single-method project sees no strip at all.
async function loadSources(){
  let r; try{ r = await api("/api/sources"); }catch(e){ return; }
  SRC.all=(r.sources||[]).map(s=>s.source); SRC.active=r.active;
  SRC.mall=(r.methods||[]).map(s=>s.method); SRC.mactive=r.methods_active ?? null;
  const bar=$("#srcFacet");
  const chips=(label, rows, key, active, dim)=>{ const on=s=> active===null || active.includes(s);
    return `${label}: `+rows.map(s=>`<a class="srcChip${on(s[key])?" on":""}" data-dim="${dim}" data-src="${escAttr(s[key])}">${escAttr(s[key])}<span class="muted"> ${s.n}</span></a>`).join(" ")
      + (active!==null?` <a class="srcChip" data-dim="${dim}" data-src="__all__">all</a>`:""); };
  const parts=[];
  if(SRC.all.length>1) parts.push(chips("source", r.sources, "source", SRC.active, "src"));
  if(SRC.mall.length>1) parts.push(chips("mask", r.methods, "method", SRC.mactive, "method"));
  bar.style.display = parts.length ? "" : "none";
  // folded into one toolbar control: the chips only take room while the dropdown is open
  const off = (SRC.active!==null) + (SRC.mactive!==null);
  const open = !!bar.querySelector("details[open]");
  bar.innerHTML = `<details class="facetDd"${open?" open":""}><summary${off?' class="on"':""}>filter${off?` · ${off} active`:""} ▾</summary>`+
                  `<div class="facetPanel">${parts.join("<br>")}</div></details>`;
  // the toolbar (#nav) clips overflow, so the panel is FIXED and placed under its button when it opens
  const d=bar.querySelector("details"), place=()=>{ if(!d.open) return;
    const r=d.querySelector("summary").getBoundingClientRect(), p=d.querySelector(".facetPanel");
    p.style.top=`${r.bottom+4}px`; p.style.right=`${Math.max(8, innerWidth-r.right)}px`; };
  d.addEventListener("toggle", place); place();
}
addEventListener("click", e=>{ const d=$("#srcFacet details[open]"); if(d && !d.contains(e.target)) d.open=false; });
$("#srcFacet").onclick=async e=>{ const a=e.target.closest("[data-src]"); if(!a)return;
  const m = a.dataset.dim==="method", all = m ? SRC.mall : SRC.all, cur = m ? SRC.mactive : SRC.active;
  let active = cur===null ? all.slice() : cur.slice();
  if(a.dataset.src==="__all__"){ active=null; }
  else { const s=a.dataset.src; active = active.includes(s) ? active.filter(x=>x!==s) : active.concat([s]);
    if(active.length===0 || active.length===all.length) active=null; }   // none / all -> clear facet
  const r=await post("/api/source_filter", m ? {methods:active} : {sources:active}); setStatus(r.stats);
  await loadSources(); srcReloadActive(); };
function srcReloadActive(){ const t=document.querySelector(".tab.active")?.id;
  if(t==="tab-map"){ MAP.loaded=false; mapLoad(); }
  else if(t==="tab-inimage"){ if($("#imgSelect").options.length) populateImages(""); }
  else { loadPartitions(true); if(typeof INST!=="undefined" && INST.pid) selectPartition(INST.pid); } }
$("#scopeSel").onchange = async e=>{
  const r = await post("/api/scope",{ingest_id:e.target.value});
  if(r.error){ alert(r.error); return; }
  $("#levelSel").innerHTML = "";                   // the cluster was cleared (it was built on the old pool)
  PART.query=""; if($("#search")) $("#search").value="";
  loadPartitions(true);                            // in-scope class partitions + "Cluster, then pick…" hint
  if($("#imgSelect").options.length) populateImages("");   // image picker is now scoped
  loadIngests();
  $("#status").textContent = (e.target.value==="all"?"scope: all instances":`scope: ${e.target.value}`)
    + ` · ${r.n_pool} in pool / ${r.n_images} imgs — click Cluster`;
};
// DANGER: full reset — drop every instance + all curation + classes + logs (config kept). Double-gated.
$("#resetBtn").onclick = async ()=>{
  const st = await api("/api/state"); const n = (st.stats && st.stats.n_instances) || 0;
  if(!confirm(`Drop EVERYTHING?\n\nPermanently deletes all ${n} instances, every assignment/class, the ingest registry and caches. The project config (model + paths) is kept.\n\nThis CANNOT be undone.`)) return;
  if(prompt('Type DROP to confirm the full reset:') !== 'DROP'){ $("#resetMsg").textContent="cancelled."; return; }
  $("#resetMsg").innerHTML=SPIN+"resetting…";
  const r = await post("/api/reset",{confirm:true});
  if(!r.ok){ $("#resetMsg").innerHTML=`<span style="color:var(--warn)">${r.detail||'reset failed'}</span>`; return; }
  await refreshState(); loadPartitions(true); if(typeof pGrid!=='undefined') pGrid.reset();
  $("#resetMsg").innerHTML='<b>done</b> — project emptied (config kept). Sample &amp; extract to start again.';
};
// Feature checkboxes shared by EVERY selector (cluster / classifier / merge-rec / substructure). A feature
// whose matrix has NaN/inf is DISABLED (greyed, "⚠NaN") — it would break sklearn/FINCH; the engine also
// drops it server-side as a safety net. `isDefault(name)` decides the initial check (skipped for NaN ones).
// What a selector should tick when it is first built. `decoder` is the seg-model head's own features and
// stays the default WHERE IT EXISTS, but a model-free project (whole-image / qseg proposals + a computed
// embedding) never has one — hardcoding it left every selector empty, and an empty selector is a hard
// server error ("no usable (present, NaN-free) features selected"), not a nudge. Falls back to whatever
// embedding was computed, then to geometry, so there is always something checked.
function defaultFeatSet(){
  const nan = new Set(window._featureNan||[]);
  const fs = (window._features||[]).filter(f=>!nan.has(f));
  if(fs.includes("decoder")) return new Set(["decoder"]);
  const emb = fs.filter(f=>!GEOM_FEATURES.includes(f));
  return new Set(emb.length ? emb : fs);
}
// What the user reads next to each checkbox. The stored feature keys stay as they are (specs, caches and
// saved configs use them); only the label and the hover text say what the numbers describe.
const FEAT_INFO = {
  shape:      ["shape",    "mask geometry: area, perimeter, solidity, extent, elongation, pieces, skeleton length, tortuosity, 7 Hu moments"],
  shapecoord: ["outline",  "outline only — axis lengths/ratio/orientation, 16-direction radial profile, 8 Fourier harmonics; independent of where the instance sits and how big it is"],
  coords:     ["position", "where and how big: box centre, box width/height, box area, mask area fraction"],
  raddino:    ["raddino",  "RAD-DINO appearance embedding of the instance crop"],
};
const featLabel = f => (FEAT_INFO[f]||[f])[0];
function featBoxes(cls, isDefault){
  const nan = new Set(window._featureNan||[]);
  return (window._features||[]).map(f=>{ const bad=nan.has(f);
    const checked = (!bad && isDefault(f)) ? 'checked' : '';
    const tip = bad ? 'contains NaN/inf — not usable' : `${f}: ${(FEAT_INFO[f]||[])[1]||"model features"}`;
    return `<label title="${escAttr(tip)}" style="${bad?'opacity:.45':''}">`
      + `<input type=checkbox class=${cls} value="${f}" ${checked} ${bad?'disabled':''}>${featLabel(f)}${bad?' ⚠NaN':''}</label>`;
  }).join("");
}
// single source of truth for the feature selectors: rebuild #feats (cluster) + classifier/sub/merge-rec
// from window._features, and show what's available (so computed embeddings like raddino are visible).
function refreshFeatures(list){
  if(list) window._features = list;
  const fs = window._features || [];
  $("#feats").innerHTML = (D=>featBoxes("feat", f=>D.has(f)))(defaultFeatSet())
    // a project that came in without the mask-geometry descriptors (COCO / box-guided / older ingests)
    // gets them one click away, right where they would be used — not behind Config → maintenance
    + (fs.length && !fs.includes("shape") ? ` <button id="featAddShape" title="compute the 'shape' descriptors (area, solidity, Hu moments…) from every instance's mask — CPU, no re-detection">＋ shape</button>` : "");
  const add=$("#featAddShape");
  if(add) add.onclick=async()=>{ const old=add.textContent; add.disabled=true; add.textContent="computing shape…";
    const r=await post("/api/recompute_shape",{});
    if(r.detail||r.error){ add.disabled=false; add.textContent=old; alert(r.detail||r.error); return; }
    window._featureNan=(window._featureNan||[]).filter(f=>f!=="shape"); refreshFeatures(r.available); };
  if($("#cfgFeatList")) $("#cfgFeatList").innerHTML = "available features: "+(fs.length?fs.map(f=>`<code>${f}</code>`).join(" · "):"— (Sample &amp; extract first)");
  // once RAD-DINO is in the collection, default the "chain after infer" box ON so it stays in sync — but
  // never override a manual choice (the change handler stamps data-touched).
  const chain=$("#cfgChainRaddino"); if(chain && !chain.dataset.touched) chain.checked = fs.includes("raddino");
  syncClfFeats(); syncMrFeats(); syncSubFeats();
}
$("#plRun").onclick=async()=>{
  const body={dir:$("#plDir").value.trim(), shard_size:+$("#plShard").value, method:$("#plMethod").value,
    thresh:+$("#plThresh").value, pool:$("#plPool").value, class_agnostic:$("#plAgnostic").checked,
    limit:($("#plLimit").value.trim()?+$("#plLimit").value:null), ...inferThr()};
  $("#plStatus").textContent="sharded pseudo-labeling (loading model)…";
  const r=await withProgress("#plBar","#plStatus",()=>post("/api/scaled_pseudolabel",body), "#plRun");
  if(r.error||r.detail){ $("#plStatus").innerHTML=`<span style="color:var(--warn)">${r.error||r.detail}</span>`; return; }
  $("#plStatus").innerHTML=`done: <b>${r.n_images}</b> imgs · <b>${r.n_instances}</b> instances · <b>${r.n_labeled}</b> labeled (${r.method}) · ${r.shards} shards → <code>${r.merged.path}</code> (${r.merged.annotations} anns, ${r.merged.categories} classes)`; };
// ---- Set up: images -> masks -> features -> clusters ------------------------
// The four steps are a strict chain — each one is unusable until the one before it has run — so the
// pane shows which link you are on rather than presenting four equal buttons. `next` is the first
// step not yet done; everything after it stays neutral rather than being disabled, because a step
// can legitimately be re-run (more images, a second proposal source, a different embedding).
function setupSync(){
  if(!$("#stepImages")) return;
  const emb = (SETUP.features||[]).filter(f=>!GEOM_FEATURES.includes(f));
  const root = SETUP.image_root;
  $("#setupRoot").textContent = root || "— no image root set for this project";
  // Keep the editor in step with the served value, but never overwrite a path being typed: setupSync
  // also runs on the background state refresh.
  const box = $("#setupRootEdit");
  if(box && document.activeElement !== box) box.value = root || "";
  // In sample mode the item IS the whole image, so there is nothing to call a mask. This pane is the
  // front door now, and greeting a sample-mode project with the wrong noun for its own contents reads
  // as the app not knowing what kind of project it opened.
  const t = $("#stepMasksTitle");
  if(t) t.textContent = (window._caps && window._caps.masks === false) ? "Get items" : "Get masks";
  const steps = [
    ["#stepImages",  "#okImages",  !!root,               root ? "" : "no folder configured"],
    ["#stepMasks",   "#okMasks",   SETUP.n_instances>0,  SETUP.n_instances>0 ? `${SETUP.n_instances} instances` : "nothing ingested yet"],
    ["#stepFeats",   "#okFeats",   emb.length>0,         emb.length ? emb.join(", ") : "no embedding computed yet"],
    ["#stepCluster", "#okCluster", SETUP.clustered,      SETUP.clustered ? "clustered" : "not clustered yet"],
  ];
  let next = true;
  for(const [sel, okSel, done, note] of steps){
    const el=$(sel); if(!el) continue;
    el.classList.toggle("done", done);
    el.classList.toggle("next", !done && next);
    if(!done) next = false;
    $(okSel).textContent = (done ? "✓ " : "") + note;
  }
  $("#setupClusterNote").textContent = SETUP.n_instances===0 ? "nothing to cluster yet — do step 2 first"
    : !emb.length ? "you can cluster on mask geometry alone, but an embedding groups far better"
    : "";
}
$("#setupRootSave").onclick = async ()=>{
  const root = $("#setupRootEdit").value.trim();
  const msg = $("#setupRootMsg");
  msg.textContent = "";
  const r = await withBusy("#setupRootSave", ()=>post("/api/image_root",{root}));
  if(!r || r.error || r.detail){
    msg.innerHTML = `<span style="color:var(--warn)">${escAttr((r&&(r.error||r.detail))||"could not set the image folder")}</span>`;
    return;
  }
  await refreshState();
  // The count is the point of the round trip: it says whether the folder holds what the user thinks
  // it does, which a path alone never does.
  msg.textContent = !r.image_root ? "image folder cleared"
    : `${r.n_images}${r.capped ? "+" : ""} image(s) here`;
};
$("#setupCluster").onclick = async ()=>{
  if(SETUP.n_instances===0){ $("#setupClusterNote").textContent="nothing to cluster yet — do step 2 first"; return; }
  // This button borrows the Curate pane's checkboxes, which the user cannot see from here — so an empty
  // selection must not surface as the server's "no usable features" error on a pane with no way to fix it.
  let feats=$$(".feat:checked").map(e=>e.value);
  if(!feats.length) feats=[...defaultFeatSet()];
  if(!feats.length){ $("#setupClusterNote").textContent="no usable features — compute an embedding first"; return; }
  const r=await withBusy("#setupCluster", ()=>post("/api/cluster",{features:feats}));
  if(r.detail){ $("#setupClusterNote").innerHTML=`<span style="color:var(--warn)">${escAttr(r.detail)}</span>`; return; }
  await refreshState();
  showRoute("partitions");
};

// ---- ingest: where a project's masks come from -----------------------------
// Availability comes from the server, so an uninstalled backend is offered with its install hint
// rather than silently missing — the same contract as the extractor dropdown below.
let BACKENDS = [];
async function loadBackends(){
  const sel=$("#ingBackend"); if(!sel) return;
  try{
    const r=await api("/api/backends");
    BACKENDS = r.backends || [];
    // box-only backends (MedSAM) have nothing to prompt with on their own: re-masking only
    sel.innerHTML = BACKENDS.map(b=>
      `<option value="${escAttr(b.name)}" ${b.available&&!b.box_only?"":"disabled"}>${escAttr(b.label||b.name)}${b.available?(b.box_only?" — box re-masking only":""):" — not installed"}</option>`).join("");
    const first = BACKENDS.find(b=>b.available&&!b.box_only);
    if(first) sel.value = first.name;
    sel.onchange = ingSyncForm;
    // anything that makes masks can re-mask boxes; the COCO/whole-image sources only echo boxes back
    const rw=$("#ingRemaskWith");
    if(rw){
      const opts = BACKENDS.filter(b=>b.name!=="coco" && b.name!=="whole_image");
      rw.innerHTML = opts.map(b=>
        `<option value="${escAttr(b.name)}" ${b.available?"":"disabled"}>${escAttr(b.label||b.name)}${b.available?"":" — not installed"}</option>`).join("");
      rw.size = Math.max(2, Math.min(opts.length, 6));   // show every model, not a scrolled 3-row window
      const pick = opts.find(b=>b.available&&b.promptable) || opts.find(b=>b.available);
      if(pick) rw.value = pick.name;
    }
    const cb=$("#ingRemask"); if(cb) cb.onchange = ingSyncForm;
    if(rw) rw.onchange = ingSyncForm;
    ingSyncForm();
  }catch(e){}
}
// The re-mask model(s): one name, or several whose candidates get pooled.
function remaskModels(){
  const rw=$("#ingRemaskWith"); if(!rw) return [];
  return rw.selectedOptions ? [...rw.selectedOptions].map(o=>o.value) : (rw.value ? [rw.value] : []);
}
// Per-model settings for whichever of qseg / HF is in play; the server applies each only to its own
// model (qseg's ckpt/overrides are popped before any other backend sees the request).
function modelCfg(){
  const val=id=>{ const el=$(id); return el ? (el.value||"").trim() : ""; };
  const cfg={}, ck=val("#ingQsegCkpt"), tj=val("#ingQsegJson"), ov=val("#ingQsegOvr"), mid=val("#ingHfModel");
  if(ck) cfg.ckpt=ck; if(tj) cfg.train_json=tj; if(ov) cfg.overrides=ov; if(mid) cfg.model_id=mid;
  return Object.keys(cfg).length ? cfg : undefined;   // undefined drops out of the JSON body
}
function ingSyncForm(){
  const name = $("#ingBackend").value;
  const b = BACKENDS.find(x=>x.name===name);
  $("#ingCocoRow").style.display = (name==="coco") ? "flex" : "none";
  // a model's settings show whenever it is in play — as the proposal source OR a re-mask model
  const inPlay = new Set([name, ...remaskModels()]);
  for(const id of ["#ingQsegRow","#ingQsegOvrRow"]){ const el=$(id); if(el) el.style.display = inPlay.has("qseg") ? "flex" : "none"; }
  const hr=$("#ingHfRow"); if(hr) hr.style.display = inPlay.has("hf_seg") ? "flex" : "none";
  const opt=$("#ingRemaskOpt"); if(opt) opt.style.display = (name==="coco") ? "" : "none";
  $("#ingNote").textContent = !b ? ""
    : (b.available ? (b.detail||"")
       : [b.detail, b.requires || "not installed"].filter(Boolean).join(" — "));
}
$("#ingRun").onclick=async()=>{
  const backend=$("#ingBackend").value;
  if(!backend){ $("#ingMsg").textContent="pick a proposal source first"; return; }
  const body={backend};
  if(backend==="coco"){
    const c=$("#ingCoco").value.trim();
    if(!c){ $("#ingMsg").innerHTML=`<span style="color:var(--warn)">a COCO file path is required</span>`; return; }
    body.coco_path=c;
    if($("#ingAssignCats") && $("#ingAssignCats").checked) body.assign_categories=true;
    const rms=remaskModels();
    if($("#ingRemask") && $("#ingRemask").checked && rms.length){
      body.remask_with = rms.length===1 ? rms[0] : rms;
      const pad=parseFloat($("#ingBoxPad").value); if(pad>=0) body.box_pad=pad;
    }
  }
  body.cfg = modelCfg();
  const root=$("#ingRoot").value.trim(); if(root) body.image_root=root;
  const lim=parseInt($("#ingLimit").value,10); if(lim>0) body.limit=lim;
  const sc=parseFloat($("#ingScore").value); if(sc>0) body.score_thresh=sc;
  const src=$("#ingSource").value.trim(); if(src) body.source=src;
  $("#ingMsg").textContent=`getting masks with ${backend}… (the first run may download a checkpoint)`;
  const r=await withProgress("#ingBar","#ingMsg",()=>post("/api/propose",body),"#ingRun");
  if(!r || r.error || r.detail){
    $("#ingMsg").innerHTML=`<span style="color:var(--warn)">${escAttr((r&&(r.error||r.detail))||"failed")}</span>`;
    return;
  }
  $("#ingMsg").innerHTML=`<b>${r.n_instances}</b> instances from <b>${r.n_images}</b> image(s)`
    + (r.n_assigned ? `, <b>${r.n_assigned}</b> assigned from the file's categories` : "") + `. `
    + (r.n_candidates ? `<b>${r.n_remasked}</b> re-masked by ${escAttr((r.backends||[]).join(" + "))} — `
       + `<a href="#/curate/refine">review the alternatives in Refine</a>. ` : "")
    + `Next: <b>Compute features</b> below, then <b>Cluster</b> in Curate.`
    + (r.n_box_only ? `<br><span style="color:var(--warn)"><b>${r.n_box_only}</b> instance(s) are still just their box — no mask was found. Tick <b>only box-shaped</b> and re-mask them with another model.</span>` : "");
  refreshState();
};
// Label what the project already has with a COCO's categories (the path may be blank: the server
// then uses the COCO the instances were ingested from).
if($("#ingAssignCatsRun")) $("#ingAssignCatsRun").onclick=async()=>{
  const body={};
  const c=$("#ingCoco").value.trim(); if(c) body.coco_path=c;
  if($("#ingAssignOverwrite").checked) body.overwrite=true;
  const src=$("#ingSource").value.trim(); if(src) body.source=src;
  $("#ingMsg").textContent="labelling existing instances from the COCO's categories…";
  const r=await withProgress("#ingBar","#ingMsg",()=>post("/api/assign_coco_categories",body),"#ingAssignCatsRun");
  if(!r || r.error || r.detail){
    $("#ingMsg").innerHTML=`<span style="color:var(--warn)">${escAttr((r&&(r.error||r.detail))||"failed")}</span>`;
    return;
  }
  $("#ingMsg").innerHTML=`<b>${r.n_assigned}</b> of ${r.n_considered} instance(s) labelled from the file's categories`
    + (r.n_unmatched ? `; ${r.n_unmatched} matched no annotation` : "") + ". Undo reverts it.";
  refreshState();
};
// Re-mask what the project already has: each instance's current box goes to the chosen model.
if($("#ingRemaskRun")) $("#ingRemaskRun").onclick=async()=>{
  const rms=remaskModels();
  if(!rms.length){ $("#ingMsg").textContent="pick a model to re-mask with first"; return; }
  const body={backend: rms.length===1 ? rms[0] : rms}, backend=rms.join(" + ");
  const pad=parseFloat($("#ingBoxPad").value); if(pad>=0) body.box_pad=pad;
  const src=$("#ingSource").value.trim(); if(src) body.source=src;
  if($("#ingRemaskBoxOnly") && $("#ingRemaskBoxOnly").checked) body.only_box=true;
  body.cfg = modelCfg();
  $("#ingMsg").textContent=`re-masking inside each box with ${backend}…`;
  const r=await withProgress("#ingBar","#ingMsg",()=>post("/api/remask",body),"#ingRemaskRun");
  if(!r || r.error || r.detail){
    $("#ingMsg").innerHTML=`<span style="color:var(--warn)">${escAttr((r&&(r.error||r.detail))||"failed")}</span>`;
    return;
  }
  $("#ingMsg").innerHTML=`<b>${r.n_remasked}</b> instance(s) re-masked across <b>${r.n_images}</b> image(s)`
    + (r.n_kept ? `; <b>${r.n_kept}</b> kept their mask (nothing matched the box)` : "")
    + `. Each got its best of <b>${r.n_candidates}</b> candidates`
    + ((r.backends||[]).length>1 ? ` pooled from ${escAttr(r.backends.join(" + "))} (masks they agree on first)` : "") + ` — `
    + `<a href="#/curate/refine">review the alternatives in Refine</a>. Undo reverts it.`
    + (r.n_box_only ? `<br><span style="color:var(--warn)"><b>${r.n_box_only}</b> instance(s) are still just their box — no mask was found. Tick <b>only box-shaped</b> and re-mask them with another model.</span>` : "");
  refreshState();
};

// The embedding-model dropdown. Availability comes from the server so an uninstalled extractor is
// shown with its install hint rather than silently missing.
async function loadExtractors(){
  const sel=$("#cfgExtractor"); if(!sel) return;
  try{
    // Remember what was picked before the options are rebuilt: this reloads after every Compute
    // features, `primary` is null for most projects and the registry is alphabetical, so without
    // this the select fell back to the first option (CLIP) each time — the model the user had just
    // chosen and computed with was quietly replaced.
    const prev=sel.value;
    const r=await api("/api/extractors"), present=new Set(r.present||[]);
    sel.innerHTML=(r.extractors||[]).map(e=>{
      const tag=present.has(e.name)?" ✓ computed":(e.available?"":" — not installed");
      return `<option value="${escAttr(e.name)}" ${e.available?"":"disabled"} title="${escAttr(e.detail||"")}">${escAttr(e.label)}${tag}</option>`;
    }).join("");
    const usable=new Set((r.extractors||[]).filter(e=>e.available).map(e=>e.name));
    const cur=(prev && usable.has(prev)) ? prev : (r.primary || (present.has("raddino")?"raddino":null));
    if(cur) sel.value=cur;
    sel.onchange=()=>{ const e=(r.extractors||[]).find(x=>x.name===sel.value);
      $("#cfgExtractorNote").textContent = e ? `${e.detail}${e.space?` · shares the "${e.space}" image-text space`:""}` : ""; };
    sel.onchange();
  }catch(e){}
}
// Which accelerator the encoders will actually use. Shown HERE, next to the model that uses it,
// rather than in the header: answering imports torch server-side, and a session that only reviews
// and exports should never pay for that. `/api/extractors` above already pays it, so this is free.
async function loadDevice(){
  const el=$("#cfgDevice"); if(!el) return;
  try{
    const d=await api("/api/device");
    const alt=(d.available||[]).filter(x=>x!==d.type);
    el.innerHTML = `compute device: <b>${escAttr(d.device||"?")}</b>`
      + (d.amp ? ` · mixed precision ${escAttr(d.amp)}` : " · full precision")
      + (d.override ? ` · forced by CHEVRON_DEVICE=${escAttr(d.override)}` : "")
      + (alt.length ? ` · also available: ${alt.map(escAttr).join(", ")}` : "")
      + (d.torch ? "" : " · torch is not installed, so no embedding model can run");
  }catch(e){ el.textContent="compute device: unknown"; }
}
$("#cfgRaddino").onclick=async()=>{
  const ex=$("#cfgExtractor").value||"raddino";
  $("#cfgRaddinoMsg").textContent=`extracting ${ex} embeddings (one encoder pass per image)…`;
  const r=await withProgress("#raddinoBar","#cfgRaddinoMsg",()=>post("/api/compute_features",{extractor:ex, force:$("#cfgRaddinoForce").checked, pool:$("#cfgRaddinoPool").value}), "#cfgRaddino");
  if(r.error||r.detail){ $("#cfgRaddinoMsg").innerHTML=`<span style="color:var(--warn)">${r.error||r.detail}</span>`; return; }
  refreshFeatures(r.available);
  SETUP.features = r.available || SETUP.features; setupSync();   // step 3 ticks over without a reload
  $("#cfgRaddinoMsg").innerHTML=`<b>${r.extractor}</b> ready for <b>${r.n||'all'}</b> instances — <code>${r.extractor}</code> is now selectable everywhere.`;
  loadExtractors(); };
$("#cfgShape").onclick=async()=>{
  $("#cfgShapeMsg").textContent="recomputing shape features from masks (CPU)…";
  const r=await withProgress("#shapeBar","#cfgShapeMsg",()=>post("/api/recompute_shape",{}), "#cfgShape");
  if(r.error||r.detail){ $("#cfgShapeMsg").innerHTML=`<span style="color:var(--warn)">${r.error||r.detail}</span>`; return; }
  refreshState();   // refresh feature_nan so `shape` is no longer ⚠NaN / disabled in the selectors
  $("#cfgShapeMsg").innerHTML=`recomputed shape + shapecoord for <b>${r.n}</b> instances (NaN sanitized) — <code>shape</code> is selectable again.`; };
$("#clusterBtn").onclick = async ()=>{
  const feats=$$(".feat:checked").map(e=>e.value); $("#status").textContent="clustering…";
  const r=await withBusy("#clusterBtn", ()=>post("/api/cluster",{features:feats}));
  if(r.detail){ alert(r.detail); } await refreshState();
};
$("#levelSel").onchange = async e=>{ await post("/api/level",{level:+e.target.value}); loadPartitions(true); };
$("#exportBtn").onclick = async ()=>{
  const r=await withBusy("#exportBtn", ()=>post("/api/export",{partial:$("#expPartial").checked, class_agnostic:$("#expAgnostic").checked}));
  const s=r.stats||{}, kind=(r.partial?"partial-label":"curated")+(r.class_agnostic?", class-agnostic":"");
  // Export now has a pane of its own, so the result is reported in place instead of in an alert()
  // you have to dismiss before you can read the path.
  const msg = r.error ? `<span style="color:var(--warn)">${escAttr(r.error)}</span>`
    : `Exported <b>${kind}</b> COCO →<br><code>${escAttr(r.path)}</code>`
      + (r.partial ? `<br><br>${s.n_assigned} positives · ${s.n_unassigned} ignore (unreviewed) · ${s.n_background} rejected→background` : "")
      + (r.held_back ? `<br><br><span style="color:var(--warn)">${r.held_back} image(s) held back by the Release gate (policy: ${escAttr(r.release_policy)}).</span>` : "");
  if(!r.error) expSyncGateNote();
  const el = $("#exportMsg"); if(el) el.innerHTML = msg; else alert(msg.replace(/<[^>]+>/g, " ")); };
if($("#exportPatchedBtn")) $("#exportPatchedBtn").onclick = async ()=>{
  const src=$("#expPatchSrc").value.trim();
  const r=await withBusy("#exportPatchedBtn", ()=>post("/api/export_patched", src?{coco_path:src}:{}));
  const err=r.error||r.detail;
  $("#exportMsg").innerHTML = err ? `<span style="color:var(--warn)">${escAttr(err)}</span>`
    : `Patched <b>${r.n_patched}</b> annotation(s) of <code>${escAttr(r.source)}</code> →<br><code>${escAttr(r.path)}</code>`
      + `<br><br>${r.n_unchanged} unchanged` + (r.n_unmatched ? ` · <span style="color:var(--warn)">${r.n_unmatched} changed mask(s) matched no source annotation</span>` : "")
      + (r.held_back ? `<br><span style="color:var(--warn)">${r.held_back} image(s) held back by the Release gate keep their original masks.</span>` : "");
};
async function doUndo(which){ const r=await post(`/api/${which}`,{}); setStatus(r.stats); setClasses(r.classes); loadPartitions(true); if(INST.pid) selectPartition(INST.pid); }
$("#undoBtn").onclick=()=>doUndo("undo"); $("#redoBtn").onclick=()=>doUndo("redo");

// ---------- Partitions ----------
let PART={offset:0,limit:100,total:0,query:"",kind:"all",predFilter:null}, INST={pid:null,offset:0,limit:60,total:0};
let PART_PRED={}, PART_PRED_META=null;               // selected partition: iuid->{label,pred,score,assigned} + {n_total,truncated}
// Scope kinds that are not FINCH partitions. The rejected bin is just another window of instances,
// so it is a SCOPE in the rail rather than a tab with its own grid, selection and buttons.
const REJECTED_SCOPE = "__rejected__";
const isRejectedScope = () => INST.pid === REJECTED_SCOPE;
const BOX_SCOPE = "__boxes__";                       // "Pick masks": one row per source box, not a grid
const isBoxScope = () => INST.pid === BOX_SCOPE;
const SUB_PREFIX = "sub:";                       // a sub-cluster is a scope like any other
const isSubScope = () => String(INST.pid||"").startsWith(SUB_PREFIX);
const subPidOf   = () => String(INST.pid).slice(SUB_PREFIX.length);

// THE Curate selection. One Set, shared by every view in the area (Grid now; Map and Image in P3.3).
const SEL = new Set();
const pGrid = makeGrid("#pgrid", "#pSelCount", "selected", ()=>{ refreshGates(); renderInspector(); }, SEL);

// The inspector rail is the single place the selection is acted on. It renders from SEL, so any view
// that writes to SEL gets the same verbs for free — no per-view assign/reject/merge buttons.
// Dropping the selection has to live next to the verbs that consume it: the Map paints into SEL with
// a brush and (alt-drag aside) had no way to let go of what it painted, the Grid's "none" only ever
// covered the cells on screen.
function clearSelection(){
  SEL.clear();
  pGrid.syncSel(); if(typeof iiGrid!=="undefined") iiGrid.syncSel();
  renderInspector(); refreshGates();
  if(MAP.loaded) mapDraw(); if(MAP3D) MAP3D.recolor();
}
function renderInspector(){
  const n = SEL.size, strip = $("#inspStrip");
  $("#inspN").textContent = n;
  // With nothing selected the rail is NOT a selection panel: a big "0 / 0 selected" over live verbs reads
  // as "these buttons act on those 0 things". Drop the count entirely and let the scope line carry it —
  // what remains (class field + whole-scope verbs) is honestly about the scope.
  $("#inspN").style.display = n ? "" : "none";
  $("#inspSelLine").style.display = n ? "" : "none";
  $("#inspEmpty").style.display = n ? "none" : "";
  $("#inspActs").style.display = n ? "" : "none";
  syncInspScope();
  if(!strip) return;
  if(!n){ strip.innerHTML = ""; strip.style.display = "none"; return; }   // no empty row eating a gap
  strip.style.display = "";
  // Reuse the crops the grid already fetched — filling the rail must not cost extra requests. A cell
  // whose lazy crop has not arrived yet gets a placeholder rather than a broken image.
  const cssEsc = s => (window.CSS && CSS.escape) ? CSS.escape(s) : String(s).replace(/["\\]/g, "\\$&");
  strip.innerHTML = [...SEL].slice(0, 5).map(u=>{
    // look in whichever grid is on screen — the Image view's cells live in #iigrid, not #pgrid
    const sel = `.cell[data-iuid="${cssEsc(u)}"] img`;
    const img = $(`#pgrid ${sel}`) || $(`#iigrid ${sel}`);
    const src = img && img.getAttribute("src");
    return src ? `<img src="${escAttr(src)}" alt="">` : `<div class="more" title="${escAttr(u)}">…</div>`;
  }).join("") + (n > 5 ? `<div class="more">+${n-5}</div>` : "");
}
const SUBS = { rows: [], target: null, active: false };
// Sub-clusters WITHIN a class, right under its rail row: ▸ expands, each sub is an ordinary scope
// ("csub:<cid>:<k>") — same grid, same verbs. Per-class FINCH in the Map's feature space, so the shades
// match the Map's "class › sub-cluster" colouring.
const CSUB = { open: new Set() };
const csubShade = (cls, k) => [(mapHashHue(cls||"")+((k*47)%60)-30+360)%360, 52+(k*29)%30, 40+(k*23)%34];
async function csubRender(cid){
  const head = [...$$("#plist .prow")].find(e=>e.dataset.pid==="class:"+cid); if(!head) return;
  let r; try{ r = await api(`/api/class_subclusters?cid=${enc(cid)}`); }catch(_){ return; }
  if(!CSUB.open.has(cid)) return;                    // collapsed while fetching
  $$(`#plist .csub[data-of="${CSS.escape(cid)}"]`).forEach(e=>e.remove());
  const cls = head.querySelector("span").textContent.replace(/^[▸▾]/,"").trim();
  // the class's mask methods as toggles (same facet as "filter ▾", so grid / Map / subs all follow)
  const ma = r.methods_active, on = m => ma===null || ma.includes(m);
  const chips = (r.methods||[]).length > 1
    ? `<div class="csub csubm" data-of="${escAttr(cid)}" title="mask method — toggles the mask filter (also under filter ▾)">`+
      r.methods.map(x=>`<a class="srcChip mchip${on(x.method)?" on":""}" data-m="${escAttr(x.method)}">${escAttr(x.method)}<span class="muted"> ${x.n}</span></a>`).join(" ")+`</div>`
    : "";
  const html = chips + (r.rows.length < 2
    ? `<div class="csub muted" data-of="${escAttr(cid)}">one mode — no sub-clusters</div>`
    : r.rows.map(x=>{ const [h,sat,l]=csubShade(cls, x.sub);
        return `<div class="prow csub${INST.pid===x.pid?' sel':''}" data-of="${escAttr(cid)}" data-pid="${escAttr(x.pid)}">`+
               `<span><span class="dot" style="background:hsl(${h},${sat}%,${l}%)"></span>${escAttr(cls)} › ${x.sub+1}</span>`+
               `<span class="sz">${x.size}</span></div>`; }).join(""));
  head.insertAdjacentHTML("afterend", html);
}
let _plGen=0;                                         // render generation: a reset starts a new one
async function loadPartitions(reset){
  if(reset) await loadSubList();          // the rail renders sub-clusters as scopes, so refresh them first
  const gen = reset ? ++_plGen : _plGen;              // a 'load more' rides the current generation
  const off = reset ? 0 : PART.offset;
  const r=await api(`/api/partitions?offset=${off}&limit=${PART.limit}&query=${enc(PART.query)}&kind=${PART.kind}`);
  const BOXN = reset ? await api("/api/boxes?limit=0").catch(()=>null) : null;   // only counts
  if(gen!==_plGen) return;                            // a newer reset superseded this fetch -> drop it (no double-append)
  PART.total=r.total; PART.offset=off+r.rows.length;
  $("#pcount").textContent=`${r.total} scopes${r.total>PART.limit?` (showing ${Math.min(PART.offset,r.total)})`:''}`;
  // The rail groups what used to be one flat list: a class pseudo-partition ("class:<cid>") is a
  // fundamentally different thing to browse than a FINCH cluster, and mixing them buried the clusters.
  const row = p => `<div class="prow${INST.pid===p.pid?' sel':''}" data-pid="${escAttr(p.pid)}">`+
    `<span>${p.pid.startsWith("class:") ? `<span class="tw" data-cid="${escAttr(p.pid.slice(6))}" title="show this class's sub-clusters">${CSUB.open.has(p.pid.slice(6))?"▾":"▸"}</span><span class="dot" style="background:var(--ok)"></span>${escAttr(p.cls||p.pid)}${p.final?` <span class="fin" title="finished: every instance has a reviewed mask">✓</span>`:""}`
                                        : `${escAttr(p.pid)}${p.cls?` <span class=cls>[${escAttr(p.cls)}]</span>`:''}`}</span>`+
    `<span class="sz">${p.size}${p.score!=null&&!p.pid.startsWith("class:")?` · ${p.score}`:''}</span></div>`;
  if(reset){
    const classes = r.rows.filter(p=>p.pid.startsWith("class:"));
    const finch   = r.rows.filter(p=>!p.pid.startsWith("class:"));
    let html = "";
    if(classes.length) html += `<div class="grp">Classes</div>` + classes.map(row).join("");
    if(finch.length)   html += `<div class="grp">Partitions</div>` + finch.map(row).join("");
    if(!html) html = `<div class="muted" style="padding:14px 10px; font-size:12px">No scopes yet — press <b>Cluster</b>.</div>`;
    // Sub-clusters are scopes too: Assist computes them, the rail lists them, the canvas shows them.
    if(SUBS.rows.length){
      html += `<div class="grp">Substructure${SUBS.target?` of ${escAttr(SUBS.target)}`:""}</div>` +
        SUBS.rows.map(x=>{ const pid = SUB_PREFIX + x.subpid;
          return `<div class="prow${INST.pid===pid?' sel':''}" data-pid="${escAttr(pid)}">`+
                 `<span><span class="dot" style="background:#c163d8"></span>sub ${escAttr(x.subpid)}</span>`+
                 `<span class="sz">${x.size}</span></div>`; }).join("");
    }
    // The rejected bin is a scope, not a tab: same grid, same selection, same inspector.
    html += `<div class="grp">Other</div>`;
    // only in projects whose boxes were masked more than once (records carry their source annotation)
    if(BOXN && BOXN.n_boxes)
      html += `<div class="prow${isBoxScope()?' sel':''}" id="scopeBoxes" data-pid="${BOX_SCOPE}" `+
              `title="boxes with several generated masks — pick one per box; the rest are rejected as duplicates">`+
              `<span><span class="dot" style="background:var(--acc)"></span>Pick masks</span>`+
              `<span class="sz">${BOXN.n_todo}</span></div>`;
    html += `<div class="prow${isRejectedScope()?' sel':''}" id="scopeRejected" `+
            `data-pid="${REJECTED_SCOPE}" title="instances you rejected — assign one to a class, or un-reject it">`+
            `<span><span class="dot" style="background:var(--warn)"></span>Rejected</span>`+
            `<span class="sz">${window._nBg||0}</span></div>`;
    $("#plist").innerHTML = html;                    // REPLACE on reset (atomic) instead of clear-then-async-append
    CSUB.open.forEach(cid=>csubRender(cid));          // re-expand open classes (their counts may have moved)
  } else {
    // a 'load more' continues the last group — re-emitting headers would repeat "Classes / Partitions"
    $("#plist").insertAdjacentHTML("beforeend", r.rows.map(row).join(""));
  }
  $("#pmore").style.display = PART.offset<r.total?"inline-block":"none";
}
// pid===null means NO scope: clicking the picked row again lets go of it, so a highlight on the map
// (and a filtered grid) is something you can get out of the same way you got into it.
async function selectPartition(pid){
  dupClear();                                         // a duplicate preview belongs to the scope it was run on
  INST.pid=pid||null; INST.offset=0; clearPredFilter(); pGrid.reset();
  $$(".prow").forEach(e=>e.classList.toggle("sel", !!INST.pid && e.dataset.pid===INST.pid));
  boxMode(isBoxScope());                              // before syncScopeUI: it re-hides what the scope lacks
  syncScopeUI();
  if(isBoxScope()){ PART_PRED={}; $("#psugText").textContent=""; return loadBoxes(true); }
  mapSyncScope();                                     // light this scope on the Map view (no-op until it is loaded)
  if(PANE === "inimage") populateImages($("#imgFilter").value).then(()=>{   // the Image view: its picker follows the rail
    if($("#imgSelect").value && $("#imgSelect").value !== IIMG.id) loadImage(true); });
  if(!INST.pid){ PART_PRED={}; $("#psugText").textContent=""; refreshGates();
                 pGrid.msg("pick a scope in the rail to browse its instances"); return; }
  // the 1-NN "most likely class" hint is a partition notion; the rejected bin has no suggestion
  if(isRejectedScope() || isSubScope()){ PART_PRED={}; $("#psugText").textContent=""; }
  else loadPartitionSuggestion();                     // fire-and-forget: hint + per-crop markers
  await loadInstances(true);
}
// What the inspector offers depends on the scope: in the rejected bin, "unassign" means UN-REJECT,
// and rejecting something already rejected is a no-op worth not offering.
// The rail's header line for the scope: it names what "Assign/Reject every instance" would hit, which
// is the one thing the rail cannot get from the selection.
function syncInspScope(){
  const el = $("#inspScope"); if(!el) return;
  const row = $(".prow.sel"), name = row && row.querySelector("span");
  el.textContent = !INST.pid ? "no scope selected"
                 : `scope: ${(name && name.textContent.replace(/^[▸▾]/,"").trim()) || INST.pid}`;
}
function syncScopeUI(){
  const rej = isRejectedScope(), sub = isSubScope();
  $("#subTarget").textContent = INST.pid || "none";
  $("#unassignBtn").textContent = rej ? "↩ Un-reject" : "↩ Unassign";
  $("#rejectBtn").style.display = rej ? "none" : "";
  $("#rejectAllBtn").style.display = rej ? "none" : "";
  $("#assignAllBtn").style.display = rej ? "none" : "";
  $("#inspScopeBlock").style.display = (rej || isBoxScope()) ? "none" : "";   // no whole-scope verbs there
  syncInspScope();
  $("#psugReport").style.display = (rej || sub || isBoxScope()) ? "none" : "";   // a suggestion is a partition notion
}
// Most-likely-class for the selected partition: 1-NN to labeled instances + reject; "no likely class" when
// too far. Always shows the class % AND the reject %. The gate slider re-fires it for the current partition.
async function loadPartitionSuggestion(){
  const t=$("#psugText"), btn=$("#psugAccept");
  if(!INST.pid){ t.textContent=""; btn.disabled=true; btn.textContent="Accept"; PART_PRED={}; clearPredFilter(); return; }
  const pid=INST.pid; t.innerHTML="<span class='muted'>"+SPIN+"</span>"; btn.disabled=true;
  const gate=parseFloat($("#psugGate").value||"1");
  const r=await api(`/api/partition_suggestion?pid=${enc(pid)}&gate_mult=${gate}`);
  if(INST.pid!==pid) return;                          // a newer partition was selected → drop this stale result
  if(!r || r.verdict==="n/a"){ t.innerHTML="<span class='muted'>no class hint (no labels to compare against yet)</span>"; btn.disabled=true; btn.textContent="Accept"; PART_PRED={}; PART._margin=null; updateGateEff("#psugGateEff", null, gate); clearPredFilter(); return; }
  PART._margin = (r.margin==null ? null : r.margin); updateGateEff("#psugGateEff", PART._margin, gate);   // gate cutoff in match% (ties to the crop badges)
  const pc=v=>Math.round((v||0)*100);
  // class / reject portions are CLICKABLE -> filter the grid to that predicted subset (always both, regardless of verdict)
  const cls = r.top_class!=null ? `<a class="psugPick" data-label="${escAttr(r.top_class)}" title="show only the crops predicted ${escAttr(r.top_class)}"><b style="color:var(--ok)">${escAttr(r.top_class)}</b> ${pc(r.confidence)}%</a>` : "";
  const rejP = r.has_reject ? `<a class="psugPick" data-label="reject" title="show only the crops predicted reject"><span style="color:var(--warn)">reject ${pc(r.reject_likelihood)}%</span></a>` : "";
  if(r.verdict==="class")      t.innerHTML = `most likely: ${cls}${rejP?` · ${rejP}`:""}`;
  else if(r.verdict==="reject")t.innerHTML = `likely ${rejP||'<b style="color:var(--warn)">reject</b>'}`+(r.top_class!=null?` · best class ${cls}`:"");
  else                         t.innerHTML = `<b>no likely class</b>${rejP?` · ${rejP}`:""}`+(r.top_class!=null?` · nearest ${cls}`:"");
  if(r.verdict==="class"){ btn.disabled=false; btn.textContent=`✓ Accept → ${r.top_class}`; }     // 1-click apply (whole partition)
  else if(r.verdict==="reject"){ btn.disabled=false; btn.textContent="✓ Accept → reject"; }
  else { btn.disabled=true; btn.textContent="Accept"; }
  loadPartitionPreds(pid, gate);                      // per-crop markers (+ enables the clickable subset filter)
}
// Per-crop gate markers for the selected partition (mirrors In-image): badges + dashed .willAccept outline.
async function loadPartitionPreds(pid, gate){
  PART_PRED={}; PART_PRED_META=null;
  const r=await api(`/api/partition_predictions?pid=${enc(pid)}&gate_mult=${gate}`);
  if(INST.pid!==pid) return;                          // stale
  if(r && r.items) PART_PRED=r.items;
  if(r) PART_PRED_META={n_total:r.n_total, truncated:r.truncated};
  applyPreds("#pgrid", PART_PRED);
}
function clearPredFilter(){ PART.predFilter=null; $("#psugFilterBar").style.display="none";
  $$("#psugText .psugPick").forEach(a=>a.classList.remove("on")); }
function showPredFilterBar(){
  const f=PART.predFilter; if(!f){ $("#psugFilterBar").style.display="none"; return; }
  const m = PART_PRED_META ? PART_PRED_META.n_total : "?";
  const trunc = (PART_PRED_META && PART_PRED_META.truncated) ? " · first 20000 scanned" : "";
  $("#psugFilterText").textContent=`showing ${INST.total} predicted ${f.label} (of ${m})${trunc}`;
  $("#psugSubsetApply").textContent = f.isReject ? "Reject shown" : `Assign shown → ${f.label}`;
  $("#psugFilterBar").style.display="flex";
}
$("#psugText").addEventListener("click", e=>{        // click the class/reject hint -> filter the grid to that subset
  const a=e.target.closest(".psugPick"); if(!a||!INST.pid) return;
  const label=a.dataset.label;
  if(PART.predFilter && PART.predFilter.label===label){ clearPredFilter(); loadInstances(true); return; }   // toggle off
  PART.predFilter={label, isReject: label==="reject"};
  $$("#psugText .psugPick").forEach(x=>x.classList.toggle("on", x===a));
  loadInstances(true).then(showPredFilterBar);
});
$("#psugFilterClear").onclick=()=>{ clearPredFilter(); loadInstances(true); };
$("#psugSubsetApply").onclick=async()=>{ const f=PART.predFilter; if(!f||!INST.pid)return;
  const gate=parseFloat($("#psugGate").value||"1");
  const what=f.isReject?"reject":`assign to ${f.label}`;
  if(!confirm(`Apply "${what}" to the ${INST.total} shown crop(s)? (already-categorized are skipped; undoable)`))return;
  const r=await post("/api/accept_partition_subset",{pid:INST.pid, label:f.label, gate_mult:gate});
  if(r.detail){alert(r.detail);return;}
  setStatus(r.stats); setClasses(r.classes); clearPredFilter(); pGrid.reset(); loadPartitions(true); selectPartition(INST.pid); };
$("#psugAccept").onclick=async()=>{ const btn=$("#psugAccept"); if(!INST.pid||btn.disabled)return;
  const gate=parseFloat($("#psugGate").value||"1");
  if(!confirm(`Apply the recommendation to the WHOLE partition ${INST.pid}? (undoable)`))return;
  const r=await post("/api/accept_partition_suggestion",{pid:INST.pid, gate_mult:gate});
  if(r.detail){alert(r.detail);return;}
  setStatus(r.stats); setClasses(r.classes); clearPredFilter(); pGrid.reset(); $("#psugText").textContent=""; INST.pid=null; refreshGates(); loadPartitions(true); };
// Translate the gate (a ×multiplier on the auto-calibrated inter-class distance) into the concrete cutoff the
// user reads off the crops: a crop counts as class/reject iff its badge match% ≥ this. threshold = gate×margin
// (cosine distance); match% = (1 − threshold)·100. Empty when there's no margin (no labels / <2 classes).
function updateGateEff(sel, margin, gateMult){
  const el=$(sel); if(!el) return;
  if(margin==null){ el.textContent=""; return; }
  el.textContent=` · keep match ≥ ${Math.max(0, Math.min(100, Math.round((1 - gateMult*margin)*100)))}%`;
}
$("#psugGate").oninput=e=>{ $("#psugGateV").textContent=(+e.target.value).toFixed(2)+"×"; updateGateEff("#psugGateEff", PART._margin, +e.target.value); };
$("#psugGate").onchange=()=>{ const hadFilter=!!PART.predFilter; clearPredFilter();
  if(hadFilter && INST.pid) loadInstances(true);      // a filtered view → back to the full partition at the new gate
  loadPartitionSuggestion(); };
async function loadInstances(reset){
  if(!INST.pid) return; if(reset){ INST.offset=0; pGrid.reset(); }   // reset clears the grid (so filter/clear/gate REPLACE, not append)
  const f = (isRejectedScope() || isSubScope()) ? null : PART.predFilter;  // a class/reject subset filter -> server-side predicted filter
  const predQ = f ? `&pred=${enc(f.label)}&gate_mult=${parseFloat($("#psugGate").value||"1")}` : "";
  const r = isRejectedScope()
    ? await api(`/api/rejected?offset=${INST.offset}&limit=${INST.limit}`)
    : isSubScope()
    ? await api(`/api/subcluster_instances?subpid=${enc(subPidOf())}&offset=${INST.offset}&limit=${INST.limit}`)
    : await api(`/api/instances?pid=${enc(INST.pid)}&offset=${INST.offset}&limit=${INST.limit}${predQ}`);
  INST.total=r.total;
  if(reset && !r.items.length){ pGrid.msg(isRejectedScope() ? "(nothing rejected yet)" : isSubScope() ? "(empty sub-cluster)"
    : f ? `(no crops predicted ${escAttr(f.label)})` : "(empty — assign/reject emptied this partition)"); }
  else pGrid.append(r.items);
  INST.offset+=r.items.length;
  $("#imore").style.display = INST.offset<r.total?"inline-block":"none";
  applyPreds("#pgrid", PART_PRED);                    // mark the (newly paged) crops
  dupMark();
}
// ONE post-mutation refresh for the Curate workspace: drop the cells from the grid that held them,
// clear them out of the shared selection, re-render the inspector, recolor the map if it is loaded,
// and refresh the rail. Replaces the per-view afterMut / iiAfter / mapRefreshAfter trio.
const activeGrid = () => PANE === "inimage" ? iiGrid : pGrid;
async function afterMut(resp, dropped, grid){
  setStatus(resp.stats); setClasses(resp.classes);
  if(dropped){ (grid||activeGrid()).drop(dropped); dropped.forEach(u=>SEL.delete(u)); }
  renderInspector();
  if(MAP.loaded) mapRefreshAfter();          // the points that changed state must recolor
  if(PANE === "inimage") reloadOverlay();    // the image overlay must lose the instances too
  loadPartitions(true);
}
$("#search").oninput=e=>{ PART.query=e.target.value; clearTimeout(window._st); window._st=setTimeout(()=>loadPartitions(true),200); };
$("#plKind").onchange=e=>{ PART.kind=e.target.value; loadPartitions(true); };   // scope: all / partitions-only / classes-only
$("#pmore").onclick=()=>loadPartitions(false);
$("#imore").onclick=()=>loadInstances(false);
$("#plist").onclick=e=>{ if(e.target.closest(".tw")) return; const r=e.target.closest(".prow"); if(r) selectPartition(r.dataset.pid===INST.pid ? null : r.dataset.pid); };
// a method chip inside an expanded class flips that method in the shared mask facet
$("#plist").addEventListener("click", async e=>{ const c=e.target.closest(".mchip"); if(!c) return;
  const all = SRC.mall.length ? SRC.mall : [...$$("#plist .mchip")].map(x=>x.dataset.m);
  let active = SRC.mactive===null ? all.slice() : SRC.mactive.slice(); const m=c.dataset.m;
  active = active.includes(m) ? active.filter(x=>x!==m) : active.concat([m]);
  if(active.length===0 || all.every(x=>active.includes(x))) active=null;
  const r=await post("/api/source_filter",{methods:active}); setStatus(r.stats);
  await loadSources(); srcReloadActive(); });
// ▸/▾ on a class row expands/collapses its sub-clusters; it never picks the class itself
$("#plist").addEventListener("click", e=>{ const tw=e.target.closest(".tw"); if(!tw) return; const cid=tw.dataset.cid;
  if(CSUB.open.has(cid)){ CSUB.open.delete(cid); tw.textContent="▸"; $$(`#plist .csub[data-of="${CSS.escape(cid)}"]`).forEach(x=>x.remove()); }
  else { CSUB.open.add(cid); tw.textContent="▾"; csubRender(cid); } });
$("#selAll").onclick=()=>pGrid.selectPage(); $("#selNone").onclick=()=>pGrid.clearSel();
$("#inspClear").onclick=()=>clearSelection();
// Escape is the universal "never mind" — it drops the painted selection from whichever Curate view
// you are in, as long as you are not typing into a field.
addEventListener("keydown", e=>{ if(e.key!=="Escape" || !SEL.size) return;
  if(VIEW_PANES.indexOf(PANE)<0) return;
  const t=e.target, tag=(t&&t.tagName)||""; if(tag==="INPUT"||tag==="TEXTAREA"||tag==="SELECT"||(t&&t.isContentEditable)) return;
  clearSelection(); });
$("#assignBtn").onclick=async()=>{ const cls=$("#classInput").value.trim(); if(!cls||!pGrid.sel.size)return; const iu=[...SEL]; afterMut(await post("/api/assign",{iuids:iu,cls}),iu,activeGrid()); };
// Every instance in the CURRENT scope, whatever kind of scope it is.
async function scopeIuids(){
  const q = isSubScope() ? `/api/subcluster_instances?subpid=${enc(subPidOf())}&offset=0&limit=1000000`
          : isRejectedScope() ? `/api/rejected?offset=0&limit=1000000`
          : `/api/instances?pid=${enc(INST.pid)}&offset=0&limit=1000000`;
  return (await api(q)).items.map(i=>i.iuid);
}
$("#assignAllBtn").onclick=async()=>{ const cls=$("#classInput").value.trim(); if(!cls||!INST.pid)return;
  const iu=await scopeIuids();
  if(!iu.length) return;
  if(!confirm(`Assign all ${iu.length} instance(s) in this scope to "${cls}"?`)) return;
  afterMut(await post("/api/assign",{iuids:iu,cls}),iu,activeGrid()); };
// Accept the masks as they are: they count as human-reviewed (export flags, dedup anchors, curated %). Cells stay.
function markReviewed(iu){ const set=new Set(iu); $$(".cell[data-iuid]").forEach(c=>{ if(set.has(c.dataset.iuid)) c.classList.add("mrev"); }); }
async function acceptMasks(iu){ if(!iu.length) return;
  const r=await post("/api/accept_masks",{iuids:iu}); if(r.detail) return;
  setStatus(r.stats); markReviewed(iu);
  const b=$("#acceptMaskBtn"); b.textContent=`✓ ${r.n} accepted`+(r.n<iu.length?` (${iu.length-r.n} already)`:"");
  clearTimeout(b._t); b._t=setTimeout(()=>{ b.textContent="✓ Accept masks"; }, 1800);
  loadPartitions(true); }
$("#acceptMaskBtn").onclick=()=>acceptMasks([...SEL]);
$("#acceptAllMasksBtn").onclick=async()=>{ if(!INST.pid) return; const iu=await scopeIuids(); if(!iu.length) return;
  if(!confirm(`Accept the current masks of all ${iu.length} instance(s) in this scope as reviewed? (undoable)`)) return;
  acceptMasks(iu); };
$("#rejectBtn").onclick=async()=>{ if(!pGrid.sel.size)return; const iu=[...SEL]; afterMut(await post("/api/reject",{iuids:iu}),iu,activeGrid()); };
$("#rejectAllBtn").onclick=async()=>{ if(!INST.pid)return;
  if(!confirm(`Reject EVERY instance in this scope? They all go to background (undoable).`))return;
  if(isSubScope()){                                   // no server-side by-pid path for a sub-cluster
    const iu=await scopeIuids(); if(!iu.length) return;
    afterMut(await post("/api/reject",{iuids:iu}),iu,activeGrid()); return;
  }
  const r=await post("/api/reject_partition",{pid:INST.pid});   // server-side, by pid
  if(r.detail){alert(r.detail);return;}
  setStatus(r.stats); pGrid.reset(); $("#psugText").textContent=""; INST.pid=null; refreshGates(); loadPartitions(true); };
// Remove duplicates in the scope: Preview marks the crops that would go (dashed, dimmed), the button
// rejects exactly those. Reviewed masks win — the server ranks them first, even out of scope.
const DUP = { pid: null, set: new Set() };
function dupMark(){ $$("#pgrid .cell[data-iuid]").forEach(c=>c.classList.toggle("dupe", DUP.set.has(c.dataset.iuid))); }
function dupClear(){ DUP.pid=null; DUP.set=new Set(); $("#dupInfo").textContent=""; $("#dupApply").style.display="none"; dupMark(); }
const dupBody = async () => {
  const b = { metric: $("#dupMetric").value, thresh: +$("#dupThr").value || 0.8 };
  if(isSubScope()) b.iuids = await scopeIuids(); else b.pid = INST.pid;   // client-only scopes ship their members
  return b; };
$("#dupPrev").onclick=async()=>{ if(!INST.pid) return; const pid=INST.pid;
  const r=await withBusy("#dupPrev", async()=>post("/api/dedup_scope", await dupBody()));
  if(INST.pid!==pid) return;
  if(r.detail){ alert(r.detail); return; }
  DUP.pid=pid; DUP.set=new Set(r.reject); dupMark();
  $("#dupInfo").textContent = r.n ? `${r.n} duplicate(s) on ${r.n_images} image(s) — marked in the grid`
                                  : `no duplicates at ${$("#dupMetric").selectedOptions[0].textContent} ≥ ${$("#dupThr").value}`;
  $("#dupApply").textContent=`Reject ${r.n} duplicate(s)`; $("#dupApply").style.display = r.n ? "" : "none"; };
["#dupMetric","#dupThr"].forEach(s=>$(s).addEventListener("change", dupClear));   // a preview is for ONE setting
$("#dupApply").onclick=async()=>{ if(!INST.pid || DUP.pid!==INST.pid || !DUP.set.size) return;
  if(!confirm(`Reject ${DUP.set.size} duplicate(s) in this scope? A reviewed mask only goes to another reviewed copy of it (undoable).`)) return;
  const r=await post("/api/dedup_scope", {...await dupBody(), apply:true});
  if(r.detail){ alert(r.detail); return; }
  const gone=r.reject; dupClear(); afterMut(r, gone, activeGrid()); };
// ---- Pick masks: one row per source box, its masks side by side ------------------------------------
// A pick keeps that mask on one instance (now mask-reviewed) and rejects the box's other instances as
// duplicates — server-side, one undo step. Rows leave as they are picked; the next page tops the list up.
const BOX = { cls:"", items:[], cur:0, total:0, todo:0, wins:{}, gen:0, loading:false,
              sort:(()=>{ try{ return localStorage.getItem("box.sort")||"class"; }catch(_){ return "class"; } })(),   // LS is declared further down
              drawKey:null, nAgree:0, nMulti:0 };
function boxMode(on){
  const tab=$("#tab-partitions");
  $("#boxView").style.display = on ? "flex" : "none";
  for(const sel of ["#pgrid", ".pager", "#psugReport", "#psugFilterBar"]) { const e=tab.querySelector(sel); if(e) e.style.display = on ? "none" : ""; }
  tab.querySelector(".toolbar").style.display = on ? "none" : "";
  if(!on) $("#psugFilterBar").style.display = PART.predFilter ? "flex" : "none";
}
const boxWinsText = w => { const e=Object.entries(w||{}); return e.length ? " · picked so far: "+e.map(([k,n])=>`${k} ${n}`).join(" · ") : ""; };
function boxInfo(){ $("#boxInfo").textContent = `${BOX.total} box(es) to go${boxWinsText(BOX.wins)}`;
  const n=$("#scopeBoxes .sz"); if(n) n.textContent = BOX.todo;
  const a=$("#boxAgreeBtn"), c=$("#boxCollapseBtn");
  a.style.display = BOX.nAgree ? "" : "none"; a.textContent = `✓ Accept ${BOX.nAgree} agreeing`;
  c.style.display = BOX.nMulti ? "" : "none"; c.textContent = `Drop duplicates (${BOX.nMulti} box${BOX.nMulti===1?"":"es"})`; }
function boxRowHTML(it){
  return `<div class="boxrow" data-key="${escAttr(it.key)}"><div class="bl"><b title="${escAttr(it.cls)}">${escAttr(it.cls||"(no class)")}</b>`+
    `<small>img ${escAttr(it.image_id.slice(0,8))} · ${it.n_members} instance${it.n_members===1?"":"s"}</small>`+
    `<button class="boxDraw" title="none fits: draw it (D) — opens the editor on choice 1, framed on the annotated box">✏️ draw</button> `+
    `<button class="boxRej" title="not worth a mask: reject the whole box (X)">✕</button></div><div class="ch">`+
    it.choices.map((c,i)=>{ const by=c.by.join(" + ");
      const ag = c.reviewed ? ` <span class="ag" title="you drew or accepted this mask already">· ✓ yours</span>`
               : c.agree>1 ? ` <span class="ag" title="${c.agree} models made this mask alike (IoU ≥ 0.8)">· ${c.agree} agree</span>` : "";
      return `<div class="tile" data-i="${i}" title="${escAttr(by)} — ${c.current?"an instance's current mask":"an alternative kept by re-masking"} · blue = the annotated box">`+
        `<img loading="lazy" alt="" src="/api/box_crop?key=${enc(it.key)}&i=${i}&g=${BOX.gen}">`+
        `<span class="k">${i+1}</span>${escAttr(by)}${ag}</div>`; }).join("")+`</div></div>`;
}
function boxMark(){ const rows=$$("#boxList .boxrow"); BOX.cur=Math.max(0, Math.min(BOX.cur, rows.length-1));
  rows.forEach((r,i)=>r.classList.toggle("cur", i===BOX.cur));
  if(rows[BOX.cur]) rows[BOX.cur].scrollIntoView({block:"nearest"}); }
async function loadBoxes(reset){
  if(BOX.loading && !reset) return; BOX.loading=true;
  try{
    const off = reset ? 0 : $$("#boxList .boxrow").length;   // picked rows left the todo list
    const r = await api(`/api/boxes?cls=${enc(BOX.cls)}&offset=${off}&limit=20&sort=${enc(BOX.sort)}`);
    if(!isBoxScope()) return;
    if(reset){ BOX.cur=0; BOX.gen=Date.now(); $("#boxList").innerHTML="";
      $("#boxCls").innerHTML = `<option value="">all classes (${r.n_todo})</option>` +
        r.classes.map(c=>`<option value="${escAttr(c.cid)}">${escAttr(c.name)} (${c.todo})</option>`).join("");
      $("#boxCls").value = BOX.cls; }
    BOX.total=r.total; BOX.todo=r.n_todo; BOX.wins=r.wins; BOX.nAgree=r.n_agree||0; BOX.nMulti=r.n_multi||0;
    $("#boxList").insertAdjacentHTML("beforeend", r.items.map(boxRowHTML).join(""));
    if(reset && !r.items.length) $("#boxList").innerHTML = `<div class="muted" style="padding:14px">Nothing left to pick${BOX.cls?" in this class":""} — every box has one reviewed mask.</div>`;
    boxInfo(); boxMark();
  } finally { BOX.loading=false; }
}
async function boxPick(row, i){
  if(!row || row.dataset.busy) return; row.dataset.busy="1"; row.style.opacity=".35";
  const tile=row.querySelector(`.tile[data-i="${i}"]`); if(!tile){ delete row.dataset.busy; row.style.opacity=""; return; }
  const r = await post("/api/box_pick", {key: row.dataset.key, index: i});
  if(r.detail){ alert(r.detail); delete row.dataset.busy; row.style.opacity=""; return; }
  setStatus(r.stats);
  (tile.title.split(" — ")[0]||"").split(" + ").forEach(m=>{ BOX.wins[m]=(BOX.wins[m]||0)+1; });
  boxGone(row);
}
// a row that left the to-do list: drop it, top the list up. The header counts come back with the next page.
function boxGone(row){
  BOX.total--; BOX.todo--; row.remove(); boxInfo(); boxMark();
  if(MAP.loaded) mapRefreshAfter();
  if($$("#boxList .boxrow").length < 8) loadBoxes(false);
}
async function boxReject(row){
  if(!row || row.dataset.busy) return; row.dataset.busy="1"; row.style.opacity=".35";
  const r = await post("/api/box_reject", {key: row.dataset.key});
  if(r.detail){ alert(r.detail); delete row.dataset.busy; row.style.opacity=""; return; }
  setStatus(r.stats); boxGone(row);
}
// None of the masks fits: collapse the box onto its best-ranked instance WITHOUT reviewing it, then draw
// on that one. Save signs it off (a drawn mask is reviewed); Cancel leaves the box to do, alternatives kept.
async function boxDraw(row){
  if(!row || row.dataset.busy) return; row.dataset.busy="1";
  const r = await post("/api/box_pick", {key: row.dataset.key, index: 0, review: false});
  delete row.dataset.busy;
  if(r.detail){ alert(r.detail); return; }
  setStatus(r.stats); BOX.drawKey = row.dataset.key;
  openMaskEditor(r.kept, null, LS.get("me.mode")||"brush");
}
// the editor closed on a box opened with D: a save finished it, a cancel leaves it (now one instance) to do
function boxDrawDone(saved){
  if(!BOX.drawKey) return;
  const row=[...$$("#boxList .boxrow")].find(r=>r.dataset.key===BOX.drawKey); BOX.drawKey=null;
  if(saved && row) boxGone(row); else if(isBoxScope()){ const cur=BOX.cur; loadBoxes(true).then(()=>{ BOX.cur=cur; boxMark(); }); }
}
$("#boxSort").value = BOX.sort;
$("#boxSort").onchange = e=>{ BOX.sort=e.target.value; LS.set("box.sort", BOX.sort); loadBoxes(true); };
async function boxBulk(url, what){
  const body={cls: BOX.cls};
  const d = await post(url, body); if(d.detail){ alert(d.detail); return; }
  if(!d.n) { loadBoxes(true); return; }
  if(!confirm(what(d.n))) return;
  const r = await post(url, {...body, apply:true}); if(r.detail){ alert(r.detail); return; }
  setStatus(r.stats); loadBoxes(true); loadPartitions(true); if(MAP.loaded) mapRefreshAfter();
}
$("#boxAgreeBtn").onclick = ()=>boxBulk("/api/box_accept_agree", n=>
  `Accept the best mask of ${n} box(es)${BOX.cls?" in this class":""} where two or more models agree (or you already drew / accepted one), and reject each box's other instances? (one undo step)`);
$("#boxCollapseBtn").onclick = ()=>boxBulk("/api/box_collapse", n=>
  `Keep one instance for each of ${n} box(es)${BOX.cls?" in this class":""} and reject the duplicates? Masks stay unreviewed; every alternative stays on offer here. (one undo step)`);
$("#boxCls").onchange = e=>{ BOX.cls=e.target.value; loadBoxes(true); };
$("#boxList").onclick = e=>{ const row=e.target.closest(".boxrow"); if(!row) return;
  const t=e.target.closest(".tile"); if(t) return boxPick(row, +t.dataset.i);
  if(e.target.closest(".boxDraw")) return boxDraw(row);
  if(e.target.closest(".boxRej")) return boxReject(row);
  BOX.cur=[...$$("#boxList .boxrow")].indexOf(row); boxMark(); };
addEventListener("keydown", e=>{
  if(PANE!=="partitions" || !isBoxScope() || e.metaKey || e.ctrlKey || e.altKey) return;
  if($("#maskEditor").classList.contains("on")) return;          // the editor's keys, not the list's
  const t=e.target, tag=(t&&t.tagName)||""; if(tag==="INPUT"||tag==="TEXTAREA"||tag==="SELECT"||(t&&t.isContentEditable)) return;
  const rows=$$("#boxList .boxrow");
  if(/^[1-9]$/.test(e.key)){ e.preventDefault(); boxPick(rows[BOX.cur], +e.key-1); }
  else if(e.key==="Enter"||e.key===" "){ e.preventDefault(); boxPick(rows[BOX.cur], 0); }
  else if(e.key==="d"||e.key==="D"){ e.preventDefault(); boxDraw(rows[BOX.cur]); }
  else if(e.key==="x"||e.key==="X"){ e.preventDefault(); boxReject(rows[BOX.cur]); }
  else if(e.key==="ArrowDown"||e.key==="j"){ e.preventDefault(); BOX.cur++; boxMark(); }
  else if(e.key==="ArrowUp"||e.key==="k"){ e.preventDefault(); BOX.cur--; boxMark(); }
});
// One button, two verbs by scope: un-reject in the rejected bin, unassign everywhere else.
$("#unassignBtn").onclick=async()=>{ if(!pGrid.sel.size)return; const iu=[...pGrid.sel];
  afterMut(await post(isRejectedScope()?"/api/unreject":"/api/unassign",{iuids:iu}),iu,activeGrid()); };
$("#mergeBtn").onclick=async()=>{ if(SEL.size<2)return; const iu=[...SEL];
  const r=await post("/api/merge",{iuids:iu, mode:$("#iiMergeMode").value});
  if(!r.n_groups){ alert("nothing merged — merge only combines instances from the SAME image (the selection spans different images, or no image had ≥2 selected)."); return; }
  setStatus(r.stats); $("#iiPrevWrap").style.display="none";
  if(PANE === "inimage") loadImage(true); else selectPartition(INST.pid);
  loadPartitions(true); };
$("#toRefineBtn").onclick=()=>refineFocus(SEL.size ? SEL : pGrid.sel);
// find-partition-by-reference-image — RAD-DINO NN, the SAME retrieval mechanism as the Reference tab
// (falls back to roialign/decoder server-side when RAD-DINO isn't computed)
$("#matchBtn").onclick=()=>$("#matchFile").click();
$("#matchFile").onchange=async e=>{ const f=e.target.files[0]; if(!f)return; e.target.value="";
  const dataURL=await new Promise(res=>{ const r=new FileReader(); r.onload=()=>res(r.result); r.readAsDataURL(f); });
  $("#matchResults").innerHTML=loadingBox("matching (running model on the upload)…");
  const r=await withBusy("#matchBtn", ()=>post("/api/match_image",{image:dataURL, k:8}));
  if(r.error){ $("#matchResults").innerHTML=`<div class=muted style="padding:6px;color:var(--warn)">${r.error}</div>`; return; }
  const row=m=>`<div class="mrow" data-pid="${m.pid||''}"><img src="${m.crop}"><span>${m.cls?('<b>'+m.cls+'</b>'):(m.pid||'(rejected/merged)')}<br><small>cos ${m.score}</small></span></div>`;
  const sec=(title,arr)=> arr&&arr.length ? `<div style="color:var(--mut);font-size:11px;padding:4px 2px 2px">${title}</div>`+arr.map(row).join("") : "";
  // show BOTH matching CLASSES and matching UNANNOTATED partitions (classes alone crowd out the pool)
  $("#matchResults").innerHTML=`<div style="color:var(--mut);font-size:11px;padding:2px">detected score ${r.query_score} · ${r.feature||'raddino'} NN — click a row → its partition:</div>`+
    sec("▣ matching classes", r.matches_class)+sec("◇ matching unannotated partitions", r.matches_pool);
  const top=(r.matches_pool&&r.matches_pool[0])||(r.matches_class&&r.matches_class[0]); if(top&&top.pid){ $("#search").value=top.pid; PART.query=top.pid; loadPartitions(true).then(()=>selectPartition(top.pid)); }
};
$("#matchResults").onclick=e=>{ const row=e.target.closest(".mrow"); if(row&&row.dataset.pid){ $("#search").value=row.dataset.pid; PART.query=row.dataset.pid; loadPartitions(true).then(()=>selectPartition(row.dataset.pid)); } };
$("#toInimgBtn").onclick=async()=>{ const img=pGrid.firstSelImg(); if(!img)return;
  $('nav button[data-tab="inimage"]').click();
  await populateImages(String(img));
  if(![...$("#imgSelect").options].some(o=>o.value===String(img))) $("#imgSelect").insertAdjacentHTML("afterbegin",`<option value="${img}">${img}</option>`);
  $("#imgSelect").value=String(img); loadImage(true); };

// ---------- In-image ----------
let IIMG={id:null,offset:0,limit:120,total:0};
let MR={cands:[]}, IIREC={cands:[]};                  // last-shown merge-recommender candidates (Merge-rec tab / In-image)
let IIMERGE_PREV=false, _iiMergeT=null, _iiMergeGen=0;   // live merge-preview toggle for the In-image selection
const iiGrid = makeGrid("#iigrid","#pSelCount","selected", iiOnSelChange, SEL);   // THE shared selection
function iiOnSelChange(){ refreshGates(); renderInspector(); scheduleMergePreview(); }   // gating is in the central gate registry
// Live merge preview: while the toggle is ON, (re)render the merge of the current instance selection.
// Debounced so a drag-select fires one request; a generation token drops stale in-flight responses.
function scheduleMergePreview(){ if(IIMERGE_PREV){ clearTimeout(_iiMergeT); _iiMergeT=setTimeout(refreshMergePreview, 150); } }
async function refreshMergePreview(){
  if(!IIMERGE_PREV || iiGrid.sel.size<1){ $("#iiPrevWrap").style.display="none"; return; }
  const gen=++_iiMergeGen, sel=[...iiGrid.sel];
  const r=await post("/api/merge_preview",{iuids:sel, mode:$("#iiMergeMode").value});
  if(gen!==_iiMergeGen || !IIMERGE_PREV) return;       // superseded by a newer selection, or toggled off
  if(r.img){ $("#iiPrevImg").src=r.img; $("#iiPrevWrap").style.display="block"; }
}
// Build a picker <option>. count mode -> "id (n)". work mode -> annotate the estimated manual decisions left
// (work_est) + dominant predicted class, or "✓ ready" for a fully-categorized image. Driven by the 1-NN classifier.
// A finished image (every instance rejected, or classed with a reviewed mask) gets a leading ✓.
function imgOpt(it, mode){
  const ck = it.final ? "✓ " : "";
  if(mode==="count" || it.n_uncat==null) return `<option value="${it.image_id}">${ck}${it.image_id} (${it.n_inst??it.n})</option>`;
  if(it.done) return `<option value="${it.image_id}">${ck}${it.image_id} · ${it.final?"finished":"✓ ready"}</option>`;
  const cls = it.top_class ? ` · ${it.top_class}${it.n_pred_classes>1?"+":""}` : "";
  return `<option value="${it.image_id}">${ck}${it.image_id} · ${it.work_est} left${cls}</option>`;
}
// The rail's scope narrows the image picker to the images holding it — otherwise the rail would sit
// beside the Image view doing nothing. The rejected bin and sub-clusters have no image index: unscoped.
const imgScopePid = () => (INST.pid && !isRejectedScope() && !isSubScope()) ? INST.pid : null;
async function populateImages(query=""){            // windowed image picker: most-populated, or ranked by work left
  const mode = $("#imgSort") ? $("#imgSort").value : "count";
  const keep = $("#imgSelect").value;               // preserve the open image across a re-rank (e.g. gate move)
  const pid = imgScopePid(), pidQ = pid ? `&pid=${enc(pid)}` : "";
  let r, m=mode;
  if(mode==="count"){ r=await api(`/api/images?query=${enc(query)}&limit=200${pidQ}`); }
  else {
    const gate=parseFloat($("#imgPredGate").value||"1");
    const div=$("#imgVariety")?parseFloat($("#imgVariety").value||"0"):0;   // class-variety re-rank strength
    r=await api(`/api/image_ranking?order=${mode}&gate_mult=${gate}&diversity=${div}&query=${enc(query)}&limit=200${pidQ}`);
    if(r.fallback) m="count";                        // no labels yet -> server returned count-style items
  }
  $("#imgSelect").innerHTML = r.items.map(it=>imgOpt(it, m)).join("");
  if(keep && [...$("#imgSelect").options].some(o=>o.value===keep)) $("#imgSelect").value=keep;
  updateImgNav();
  const note=$("#imgSortNote");
  const row=pid && $(`.prow[data-pid="${(window.CSS&&CSS.escape)?CSS.escape(pid):pid}"] span`), name=(row && row.textContent.trim()) || pid;
  if(note) note.textContent = pid ? (r.items.length ? `(images with ${name} — click it in the rail again for all)`
                                                     : `(no image holds ${name})`)
                            : (mode!=="count" && r.fallback) ? "(label some instances to rank by work left)"
                            : (mode!=="count" && r.truncated) ? "(large pool — counts approximate)" : "";
}
// overlay busy state: spinner on + Load button disabled while the server-rendered overlay <img> loads
function _ovBusy(on){ const sp=$("#ovBusy"), lb=$("#ovLoad"); if(sp) sp.style.display=on?"block":"none"; if(lb) lb.disabled=on; }
function reloadOverlay(){ if(!IIMG.id) return; _ovBusy(true);
  $("#ovImg").src=`/api/image_overlay?image_id=${enc(IIMG.id)}&color_by=${$("#ovColor").value}&masks=${MASKS?1:0}&_=${Date.now()}`; }
$("#ovImg").addEventListener("load",  ()=>_ovBusy(false));
$("#ovImg").addEventListener("error", ()=>_ovBusy(false));
let IMG_PRED={}, IMG_MARGIN=null;                    // iuid -> {label, pred, score} for the loaded image (+ gate margin)
async function loadImage(reset=true){
  const id=$("#imgSelect").value; if(!id)return;
  if(id!==IIMG.id){ $("#iiOvCards").innerHTML=""; $("#iiOvMsg").textContent=""; $("#iiOvAll").disabled=true; }  // overlap cards survive per-card accepts, not an image switch
  IIMG.id=id; reloadOverlay(); updateImgNav();
  if(reset){ IIMG.offset=0; iiGrid.reset(); $("#iiPrevWrap").style.display="none";
             $("#iiRecCards").innerHTML=""; $("#iiRecMsg").textContent=""; IIREC.cands=[];   // clear stale per-image merge suggestions
             loadImagePredictions(id); }             // 1-NN classifier prediction per instance of this image
  const r=await api(`/api/image_instances?image_id=${enc(id)}&offset=${IIMG.offset}&limit=${IIMG.limit}`);
  IIMG.total=r.total; if(reset && !r.items.length) iiGrid.msg("(no instances on this image)"); else iiGrid.append(r.items);
  applyImagePreds();                                 // badge the (newly appended) crops with their predicted class
  IIMG.offset+=r.items.length; $("#iimore").style.display=IIMG.offset<r.total?"inline-block":"none";
}
// Apply the 1-NN classifier to every instance of the image -> a per-instance "→ class %"/"→ reject"/"→ ?"
// badge on each crop + a summary line. Mirrors the partition suggestion; the gate slider re-runs it.
async function loadImagePredictions(id){
  IMG_PRED={}; const t=$("#imgPredText"), btn=$("#imgPredAccept");
  t.innerHTML="<span class='muted'>"+SPIN+"predicting…</span>"; btn.disabled=true;
  const gate=parseFloat($("#imgPredGate").value||"1");
  const r=await api(`/api/image_suggestion?image_id=${enc(id)}&gate_mult=${gate}`);
  if(IIMG.id!==id) return;                            // a newer image was loaded -> drop this stale result
  if(!r || r.error || r.note==="no labels yet"){ t.innerHTML="<span class='muted'>no predictions (no labels to compare against yet)</span>"; btn.disabled=true; IMG_MARGIN=null; updateGateEff("#imgPredGateEff", null, gate); return; }
  IMG_MARGIN = (r.margin==null ? null : r.margin); updateGateEff("#imgPredGateEff", IMG_MARGIN, gate);   // gate cutoff in match%
  for(const it of (r.items||[])) IMG_PRED[it.iuid]=it;
  applyImagePreds(); refreshPredSummary();           // badges per crop + the summary line / Accept-all count (both off IMG_PRED)
}
// Summary line + Accept-all count, derived from IMG_PRED so per-instance accept/reject keeps it live without a refetch.
function refreshPredSummary(){
  const t=$("#imgPredText"), btn=$("#imgPredAccept"), counts={}; let assigned=0, actionable=0;
  for(const u in IMG_PRED){ const it=IMG_PRED[u];
    if(it.assigned){ assigned++; continue; }            // already categorized -> outside the Accept-all gate
    counts[it.label]=(counts[it.label]||0)+1;
    if(it.label!=="none") actionable++;                 // class/reject preds are what Accept all applies
  }
  const parts=Object.entries(counts).sort((a,b)=>b[1]-a[1]).map(([k,v])=>{
    const col=k==="reject"?"var(--warn)":(k==="none"?"var(--mut)":"var(--ok)");
    return `<span style="color:${col}">${k==="none"?"?":escAttr(k)} ${v}</span>`; });
  let msg = parts.length ? `predicted: `+parts.join(" · ") : "<span class='muted'>nothing to apply</span>";
  if(assigned) msg += ` <span class="muted">· ${assigned} already categorized (skipped)</span>`;
  t.innerHTML = msg;
  btn.disabled = actionable===0; btn.textContent = actionable ? `✓ Accept all (${actionable})` : "Accept all";
}
$("#imgPredAccept").onclick=async()=>{ const btn=$("#imgPredAccept"); if(!IIMG.id||btn.disabled)return;
  const gate=parseFloat($("#imgPredGate").value||"1");
  if(!confirm("Apply the prediction to every UNcategorized instance (the dashed crops)? Already-categorized instances and 'no likely class' are left as is. (undoable)"))return;
  const r=await post("/api/accept_image_predictions",{image_id:IIMG.id, gate_mult:gate});
  if(r.detail){alert(r.detail);return;}
  setStatus(r.stats); setClasses(r.classes); loadImage(true); };
// Badge crops in `gridSel` from a {iuid:{label,pred,score,assigned}} map: → class %/reject/? text, an
// .isAssigned tint for already-categorized, and a dashed .willAccept outline on crops the gate would
// assign/reject. opts.actions adds the per-crop ✓/✗ buttons (In-image); markers-only otherwise (Partitions).
function applyPreds(gridSel, predMap, opts={}){
  const actions = !!opts.actions;
  $(gridSel).querySelectorAll(".cell[data-iuid]").forEach(c=>{
    const it=predMap[c.dataset.iuid];
    c.classList.remove("isAssigned","willAccept");
    let b=c.querySelector(".predbadge");
    if(!it){ if(b) b.remove(); return; }               // no prediction for this crop -> clear any stale badge
    if(!b){ b=document.createElement("div"); b.className="predbadge"; c.appendChild(b); }
    const rej = actions ? `<button class="pRej" title="reject this instance">✗</button>` : "";
    if(it.assigned){                                   // already categorized: ✓ badge + tint, outside the accept gate
      c.classList.add("isAssigned");
      b.innerHTML=`<span class="pbtxt" style="color:var(--acc)" title="already categorized">✓ ${escAttr(it.assigned)}</span>`
                + (actions?`<span class="predact">${rej}</span>`:"");
      return;
    }
    let txt, col, acc="", ttl="";
    if(it.label==="reject"){ txt="→ reject"; col="var(--warn)"; c.classList.add("willAccept");
                             ttl="looks like already-rejected junk (1-NN)"; }
    else if(it.label==="none"){ txt="→ ?"; col="var(--mut)"; ttl="no likely class — match below the gate cutoff; the gate leaves it"; }
    else { txt=`→ ${it.pred} ${Math.round((it.score||0)*100)}%`; col="var(--ok)"; c.classList.add("willAccept");
           ttl="match confidence = 1 − distance to the nearest labeled instance; kept when ≥ the gate cutoff";
           if(actions) acc=`<button class="pAcc" title="accept → assign to ${escAttr(it.pred)}">✓</button>`; }
    b.innerHTML=`<span class="pbtxt" style="color:${col}" title="${ttl}">${txt}</span>`
              + (actions?`<span class="predact">${acc}${rej}</span>`:"");
  });
}
function applyImagePreds(){ applyPreds("#iigrid", IMG_PRED, {actions:true}); }
// Per-instance accept(✓)/reject(✗): apply ONE crop's recommendation. ✓ assigns it to the predicted class;
// ✗ sends it to background. Drops the crop afterward, like the bulk Assign/Reject (iiAfter), and keeps the
// summary live. Reuses the cached suggestion so it's a single cheap server round-trip per click.
$("#iigrid").addEventListener("click", async e=>{
  const acc=e.target.closest(".pAcc"), rej=e.target.closest(".pRej"); if(!acc&&!rej) return;
  const cell=e.target.closest(".cell"), u=cell&&cell.dataset.iuid; if(!u) return;
  const it=IMG_PRED[u];
  const r = (acc && it && it.pred) ? await post("/api/assign",{iuids:[u],cls:it.pred})
                                   : await post("/api/reject",{iuids:[u]});
  if(r&&r.detail){ alert(r.detail); return; }
  delete IMG_PRED[u]; iiAfter(r,[u]); refreshPredSummary();
});
$("#imgPredGate").oninput=e=>{ $("#imgPredGateV").textContent=(+e.target.value).toFixed(2)+"×"; updateGateEff("#imgPredGateEff", IMG_MARGIN, +e.target.value); };
$("#imgPredGate").onchange=()=>{ if(IIMG.id) loadImagePredictions(IIMG.id);
  if($("#imgSort")&&$("#imgSort").value!=="count") populateImages($("#imgFilter").value); };  // re-rank picker to the new gate
$("#imgSort").onchange=()=>populateImages($("#imgFilter").value);
$("#imgVariety").oninput=e=>{ $("#imgVarietyV").textContent=(+e.target.value).toFixed(2); };
$("#imgVariety").onchange=()=>{ if($("#imgSort").value!=="count") populateImages($("#imgFilter").value); };  // re-rank for class spread
$("#imgFilter").oninput=e=>{ clearTimeout(window._if); window._if=setTimeout(()=>populateImages(e.target.value),200); };
$("#imgSelect").onchange=()=>loadImage(true);
// Prev/Next: step the picker to the adjacent option (within the loaded window) and load it.
function updateImgNav(){ const s=$("#imgSelect"); if(!s) return; const n=s.options.length, i=s.selectedIndex;
  const p=$("#imgPrev"), nx=$("#imgNext"); if(p) p.disabled=(n===0||i<=0); if(nx) nx.disabled=(n===0||i>=n-1); }
function stepImage(delta){ const s=$("#imgSelect"), n=s.options.length; if(!n) return;
  const i=Math.min(n-1, Math.max(0, (s.selectedIndex<0?0:s.selectedIndex)+delta));
  if(i===s.selectedIndex && IIMG.id) return; s.selectedIndex=i; loadImage(true); }
$("#imgPrev").onclick=()=>stepImage(-1);
$("#imgNext").onclick=()=>stepImage(1);
$("#ovLoad").onclick=()=>loadImage(true);
$("#ovColor").onchange=reloadOverlay;
$("#ovMasks").onchange=e=>{ MASKS=e.target.checked; refreshVisibleCrops(); };
(function(){                                          // drag the bar between the image overlay and the annotation grid to adapt their split
  const ov=$("#iiOverlay"), sp=$("#iiSplit"); if(!ov||!sp) return;
  let y0=0,h0=0,on=false;
  sp.addEventListener("mousedown",e=>{ on=true; y0=e.clientY; h0=ov.getBoundingClientRect().height; e.preventDefault(); document.body.style.cursor="row-resize"; });
  document.addEventListener("mousemove",e=>{ if(!on)return; ov.style.height=Math.max(80,h0+e.clientY-y0)+"px"; });
  document.addEventListener("mouseup",()=>{ if(on){ on=false; document.body.style.cursor=""; } });
})();
// Amazon-style hover magnifier: a lens over a source image + a separate pane showing that region
// magnified. Reuses the already-loaded image bitmap (no extra request), so it tracks colour/mask toggles.
// Shared lens/pane across the overlay (#ovImg) and the merge-preview (#iiPrevImg) — only one is hovered at a time.
(function(){
  const ZOOM=2.5, lens=$("#ovLens"), pane=$("#ovZoom"); if(!lens||!pane) return;
  const hide=()=>{ lens.style.display="none"; pane.style.display="none"; };
  function move(img, e){
    if(!img.src || !img.complete || !img.naturalWidth){ hide(); return; }
    const r=img.getBoundingClientRect(); if(r.width<8||r.height<8){ hide(); return; }
    const P=Math.min(380, Math.round(Math.min(window.innerWidth,window.innerHeight)*0.4));   // square pane
    const lw=Math.min(r.width, P/ZOOM), lh=Math.min(r.height, P/ZOOM);
    const lx=Math.max(0, Math.min(e.clientX-r.left-lw/2, r.width-lw));
    const ly=Math.max(0, Math.min(e.clientY-r.top -lh/2, r.height-lh));
    lens.style.display=pane.style.display="block";
    lens.style.left=(r.left+lx)+"px"; lens.style.top=(r.top+ly)+"px"; lens.style.width=lw+"px"; lens.style.height=lh+"px";
    let zx=r.right+12; if(zx+P>window.innerWidth-4) zx=r.left-P-12; zx=Math.max(4, Math.min(zx, window.innerWidth-P-4));
    const zy=Math.max(4, Math.min(r.top, window.innerHeight-P-4));
    pane.style.left=zx+"px"; pane.style.top=zy+"px"; pane.style.width=pane.style.height=P+"px";
    pane.style.backgroundImage=`url("${img.currentSrc||img.src}")`;
    pane.style.backgroundSize=(r.width*ZOOM)+"px "+(r.height*ZOOM)+"px";
    pane.style.backgroundPosition=`-${lx*ZOOM}px -${ly*ZOOM}px`;
  }
  function attach(sel){ const img=$(sel); if(!img) return; const m=e=>move(img,e);
    img.addEventListener("mouseenter", m); img.addEventListener("mousemove", m); img.addEventListener("mouseleave", hide); }
  attach("#ovImg"); attach("#iiPrevImg");
  window.addEventListener("scroll", hide, true);
})();
$("#iimore").onclick=()=>loadImage(false);
const iiAfter = (resp, dropped) => afterMut(resp, dropped, iiGrid);   // kept: the per-crop accept/reject path uses it
// Toggle: ON = live-preview the merge of the current selection (auto-updates as the selection/mode changes);
// click again to turn it OFF and hide the preview.
$("#iiMergePrev").onclick=()=>{ IIMERGE_PREV=!IIMERGE_PREV;
  $("#iiMergePrev").classList.toggle("primary", IIMERGE_PREV);
  $("#iiMergePrev").textContent = IIMERGE_PREV ? "Preview merge: ON" : "Preview merge";
  if(IIMERGE_PREV) refreshMergePreview(); else $("#iiPrevWrap").style.display="none"; };
$("#iiMergeMode").onchange=()=>{ if(IIMERGE_PREV) refreshMergePreview(); };

// ---------- Refine ----------
// ONE instance in focus (RF.cur). The queue (left) decides what comes next, the stage (middle) shows that
// instance's mask, the panel (right) fixes it — pick a re-mask candidate, Auto, Draw or a Recipe — and
// "apply to others" replays the fix on its group after a sampled dry-run. Everything here reads RF.cur;
// there is no second "current" instance (the old tab had three unrelated ones on screen at once).
const RF = {src:"", items:[], i:-1, cur:"", done:new Set(), cands:null, before:"", after:"", cls:null,
            mode:"auto", pending:null, entered:false};
let RF_CHAIN=[], RF_BA_OPS=[], RF_MASK_STATE="";
const LS = { get(k,d){ try{ return localStorage.getItem(k) ?? d; }catch(_){ return d; } },
             set(k,v){ try{ localStorage.setItem(k,v); }catch(_){} } };
function syncSeg(sel, attr, val){ $$(`${sel} button`).forEach(b=>b.classList.toggle("on", b.dataset[attr]===val)); }
// caption = "file.png · 846017 s=0.57 [class]" → the class (or file) as the row title, the rest underneath
function capParts(c){ const m=/^(.*?) · (\w+) s=([\d.]+)(?: \[(.*)\])?$/.exec(c||"");
  return m ? {title:m[4]||m[1], sub:(m[4]?m[1]+" · ":"")+"s="+m[3], file:m[1], score:m[3]} : {title:c||"", sub:"", file:c||"", score:""}; }
// in a group every row has the same class — the file tells them apart there
const grpRow = it=>{ const c=capParts(it.caption); return {iuid:it.iuid, title:c.file, sub:`s=${c.score}`}; };

// ---- queue ----
async function rqLoad(src, {keep=false}={}){
  if(src) RF.src=src; LS.set("rf.src", RF.src); syncSeg("#rqSrc","src",RF.src);
  $("#rfSearch").style.display = RF.src==="search" ? "" : "none";
  $("#rqOnlyNewL").style.display = RF.src==="review" ? "" : "none";
  let items=[], info="", empty="";
  if(RF.src==="review"){
    await mcLoad();
    items = MC.items.map(x=>({iuid:x.iuid, title:x["class"]||"unassigned", sub:`${x.iuid.slice(0,8)} · ${x.n} candidate${x.n===1?"":"s"}`, done:x.reviewed}));
    info = MC.total ? `${MC.nUnrev} of ${MC.total} still to review` : "";
    empty = MC.total ? "Nothing left to review." : "No re-mask candidates yet — Set up ▸ Get masks ▸ Re-mask existing instances.";
  } else if(RF.src==="group"){
    if(!RF.cur){ empty="Open an instance first (Search or Selection) — its group then shows here."; }
    else { const r=await api(`/api/instance_peers?iuid=${enc(RF.cur)}&limit=1000`);
      items=(r.items||[]).map(grpRow);
      info = r.label ? `${r.label} · ${r.total} instances` : "this instance has no group (rejected, merged or not clustered)"; }
  } else if(RF.src==="selection"){
    const iu=[...SEL];
    if(!iu.length) empty="Nothing is selected in Curate. Select instances in Grid, Map or Image view.";
    else { const r=await post("/api/instances_info",{iuids:iu.slice(0,1000)});
      items=(r.items||[]).map(it=>({iuid:it.iuid, ...capParts(it.caption)})); info=`${iu.length} selected in Curate`; }
  } else {
    const q=$("#rfSearch").value.trim();
    const r=await api(`/api/find_instances?query=${enc(q)}&limit=200`);
    items=(r.items||[]).map(it=>({iuid:it.iuid, ...capParts(it.caption)}));
    info=`${r.total}${r.total>=200?"+":""} match${r.total===1?"":"es"}`; empty="No matches.";
  }
  RF.items=items; $("#rqInfo").textContent=info; rqRender(empty);
  RF.i=RF.items.findIndex(x=>x.iuid===RF.cur);
  if(RF.i<0 && !keep && RF.items.length) rfGo(0); else rqMark();
}
function rqRender(empty){
  $("#rfFind").innerHTML = RF.items.length ? RF.items.map((it,i)=>
    `<div class="cell qrow" data-iuid="${escAttr(it.iuid)}" data-i="${i}"><img loading="lazy" class="imgld" alt="">`+
    `<div class="qtxt"><b title="${escAttr(it.title)}">${escAttr(it.title)}</b><span>${escAttr(it.sub)}</span></div>`+
    `<span class="qst">${(it.done||RF.done.has(it.iuid))?"✓":""}</span></div>`).join("")
    : `<div class="muted" style="padding:12px">${escAttr(empty||"")}</div>`;
  observeCrops($("#rfFind")); rqMark();
}
function rqMark(){ $$("#rfFind .qrow").forEach(r=>r.classList.toggle("cur", r.dataset.iuid===RF.cur));
  const c=$("#rfFind .qrow.cur"); if(c) c.scrollIntoView({block:"nearest"});
  $("#rsPos").textContent = RF.i>=0 ? `${RF.i+1} of ${RF.items.length}` : (RF.cur?"not in this queue":""); }
$("#rqSrc").onclick=e=>{ const b=e.target.closest("button[data-src]"); if(b && !b.disabled) rqLoad(b.dataset.src); };
$("#rfFind").onclick=e=>{ const r=e.target.closest(".qrow"); if(r) rfGo(+r.dataset.i); };
$("#rfSearch").oninput=()=>{ clearTimeout(window._rfs); window._rfs=setTimeout(()=>rqLoad("search",{keep:true}),200); };
$("#mcOnlyNew").onchange=()=>rqLoad("review",{keep:true});
$("#mcPrev").onclick=()=>rfGo(RF.i-1);
$("#mcNext").onclick=()=>rfGo(RF.i+1);
// Entering the tab: a hand-off from elsewhere (→ Refine) wins; else the last source, else the review queue
// when there is one, else the selection, else search.
async function rqEnter(){
  await mcLoad(); $('#rqSrc button[data-src="review"]').disabled = !MC.total;
  const p=RF.pending; RF.pending=null;
  if(p){ if(p.iuid) await rfOpen(p.iuid); return rqLoad(p.src); }
  if(RF.entered) return rqLoad(RF.src,{keep:true});
  RF.entered=true;
  const last=LS.get("rf.src","");
  const src = (last && !(last==="review" && !MC.total)) ? last : (MC.nUnrev ? "review" : SEL.size ? "selection" : "search");
  return rqLoad(src);
}
// hand-off from Curate (the inspector's → Refine): several selected → queue = selection; one → its group
function refineFocus(iuids){ iuids=[...iuids]; if(!iuids.length) return;
  RF.pending = {src: iuids.length>1 ? "selection" : "group", iuid: iuids[0]};
  $('nav button[data-tab="refine"]').click(); }

// ---- focus + stage ----
async function rfGo(i){ if(i<0 || i>=RF.items.length) return; RF.i=i; await rfOpen(RF.items[i].iuid); }
async function rfOpen(iuid){ if(!iuid) return;
  const changed = iuid!==RF.cur;
  RF.cur=iuid; $("#rfIuid").value=iuid; RF.i=RF.items.findIndex(x=>x.iuid===iuid); rqMark();
  if(changed){ $("#rfHint").textContent=""; raStale();
    // never leave the previous case's mask/candidates on screen while the new one loads
    const it=RF.items[RF.i]; RF.after=null; RF.cands=null;
    $("#rsTitle").textContent = it ? it.title||"" : ""; $("#rsSub").textContent = it ? it.sub||iuid.slice(0,8) : iuid.slice(0,8);
    $("#rfBA").innerHTML=`<div class="muted rsEmpty">${SPIN}loading…</div>`;
    $("#mcView").innerHTML=""; $("#rpCandsSec").style.display="none"; }
  refreshGates();
  await Promise.all([rfDoPreview(), rfLoadCands()]);
}
async function rfDoPreview(trial){ const iuid=RF.cur; if(!iuid) return; refreshGates();
  if(!trial) RF.trial=false;
  const ops=activeOps().concat(trial||[]);
  let r; try{ r=await post("/api/refine_preview",{iuid, ops, mask:MASKS?1:0, context:VIEW==='context'?1:0, max_side:900}); }
  catch(e){ r={detail:"preview failed — "+(e.message||e)}; }
  if(iuid!==RF.cur) return;                                  // a newer focus superseded this preview
  if(r.detail){ $("#rfBA").innerHTML=`<div class="muted rsEmpty" style="color:var(--warn)">${escAttr(r.detail)}</div>`; return; }
  RF.before=r.before; RF.after=ops.length ? r.after : r.before; RF.cls=r.cls;
  RF_BA_OPS=ops.map(o=>({name:o.name, kw:o.kw||{}}));
  if(trial) $("#rfHint").textContent=`previewing ${trial.map(o=>o.name).join(" → ")} — not in the recipe yet: + Add keeps it`;
  const cp=capParts(r.caption);
  $("#rsTitle").textContent = r.cls || cp.title;
  $("#rsSub").textContent = `image …${String(r.image_id).slice(-6)} · ${iuid.slice(0,8)} · ${cp.sub}`;
  rsShow(false); rfSetHandState(r.mask_state); rfClassRuleHint(r.cls); rqMark();
  if(ops.some(o=>o.name==="sam") && RF.mode==="recipe"){ const f=await rfSamPointsFigure(); if(f) $("#rfHint").innerHTML=f; }
}
function rsShow(before){
  if(!RF.after){ return; }
  const touch = !before && RF_BA_OPS.length;
  const tag = before ? "before" : RF_BA_OPS.length ? "after: "+RF_BA_OPS.map(o=>o.name).join(" → ") : "current mask";
  $("#rfBA").innerHTML = `<img src="${before?RF.before:RF.after}" alt="${tag}"><span class="rsTag">${escAttr(tag)}</span>`+
    `<button class="editPen" data-seed="${touch?1:0}">✏️ ${touch?"touch up this result":"edit by hand"}</button>`; }
$("#rfBA").addEventListener("click", e=>{ if(!RF.cur || !(e.target.closest(".editPen") || e.target.tagName==="IMG")) return;
  const seed = (e.target.closest(".editPen")||$("#rfBA .editPen"))?.dataset.seed==="1";
  openMaskEditor(RF.cur, seed ? RF_BA_OPS : null); });
function rfSetHandState(st){ RF_MASK_STATE=st||""; $("#rfHandState").textContent = st ? `mask: ${st}` : ""; refreshGates(); }

// ---- re-mask candidates for the instance in focus (step 1, only when it has some) ----
let MC = {items:[], total:0, nUnrev:0};
async function mcLoad(){
  const only = $("#mcOnlyNew").checked ? 1 : 0;
  let r; try{ r = await api(`/api/mask_candidates?only_unreviewed=${only}&limit=5000`); }catch(e){ r = {}; }
  MC.items = r.items || []; MC.total = r.total || 0; MC.nUnrev = r.n_unreviewed || 0;
  return MC;
}
// which model(s) proposed candidate k — only worth saying when several were pooled
function mcBy(v, k){
  const by=(v.by||[])[k];
  return by && String(v.backend||"").includes("+") ? escAttr(by.join(" + "))+" · " : "";
}
async function rfLoadCands(){ const iuid=RF.cur; if(!iuid) return;
  let v; try{ v = await post("/api/mask_candidates/view", {iuid}); }catch(e){ v = {detail:String(e)}; }
  if(iuid!==RF.cur) return;
  RF.cands = v.detail ? null : v;
  $("#rpCandsSec").style.display = RF.cands ? "" : "none";
  $("#rpFixN").textContent = RF.cands ? "2" : "1"; $("#rpFixT").textContent = RF.cands ? "Or fix it" : "Fix it";
  $("#rpApplyN").textContent = RF.cands ? "3" : "2";
  $("#mcView").innerHTML = RF.cands ? v.thumbs.map((t, k)=>`<figure data-k="${k}" class="${k===v.current?"cur":""}"><img src="${t}" alt="">`
      + `<figcaption><b>${k}</b> · ${k===0 ? (v.original_box_only ? "box only" : "original") : mcBy(v, k)+v.scores[k]}${k===v.current?" · now":""}</figcaption></figure>`).join("") : "";
}
async function mcPick(k){ const iuid=RF.cur; if(!iuid || !RF.cands) return;
  const r = await post("/api/mask_candidates/pick", k==null ? {iuid} : {iuid, index: k});
  if(r.detail){ $("#rfHint").textContent = r.detail; return; }
  if(r.stats) setStatus(r.stats);
  rfAdvance(iuid);
}
$("#mcView").onclick = e=>{ const f=e.target.closest("figure[data-k]"); if(f) mcPick(+f.dataset.k); };

// ---- accept & move on ----
async function rfAccept(){ const iuid=RF.cur; if(!iuid) return;
  const ops=activeOps();
  if(ops.length){
    const r=await withBusy("#rpAccept", ()=>post("/api/apply_refine",{iuid, ops}));
    if(r.detail){ $("#rfHint").textContent=r.detail; return; }
    setStatus(r.stats);
  }
  if(RF.cands){ const r=await post("/api/mask_candidates/pick",{iuid}); if(r.stats) setStatus(r.stats); }   // mark reviewed
  rfAdvance(iuid);
}
// Done with `iuid`: tick it, drop it from an "unreviewed only" review queue, focus the next one.
function rfAdvance(iuid){ RF.done.add(iuid);
  let i=RF.items.findIndex(x=>x.iuid===iuid);
  if(RF.src==="review" && $("#mcOnlyNew").checked && i>=0){ RF.items.splice(i,1); MC.nUnrev=Math.max(0,MC.nUnrev-1);
    $("#rqInfo").textContent=`${MC.nUnrev} of ${MC.total} still to review`; rqRender("Nothing left to review."); }
  else i=i+1;
  if(i>=0 && i<RF.items.length) rfGo(i);
  else { rqMark(); $("#rfHint").textContent="End of the queue."; }
}
$("#rpAccept").onclick=rfAccept;
function rfAcceptLabel(){ const ops=activeOps();
  $("#rpAccept").textContent = ops.length ? `Apply ${ops.map(o=>o.name).join(" → ")} & next ↵` : "Accept & next ↵"; }

// ---- step 2: fix it (Auto / Draw / Recipe) ----
function rfSetMode(m){ RF.mode=m; LS.set("rf.mode", m); syncSeg("#rpMode","mode",m);
  $$("#rp .rpPane").forEach(p=>p.style.display = p.dataset.pane===m ? "" : "none"); }
$("#rpMode").onclick=e=>{ const b=e.target.closest("button[data-mode]"); if(b) rfSetMode(b.dataset.mode); };
$("#rfAuto").onclick=async()=>{ const iuid=RF.cur; if(!iuid) return;
  $("#rfHint").innerHTML=SPIN+"trying candidate recipes on this mask…";
  const r=await withBusy("#rfAuto", ()=>post("/api/auto_refine_preview",{iuid, kind:"auto"}));
  if(iuid!==RF.cur) return;
  if(r.detail){ $("#rfHint").textContent=r.detail; return; }
  const p=r.pick;
  RF_CHAIN=(p.ops||[]).map(o=>({name:o.name, kw:o.kw||{}, on:true})); renderChain(false);
  RF.after = p.ops && p.ops.length ? r.after : RF.before; RF_BA_OPS=activeOps().map(o=>({name:o.name, kw:o.kw||{}})); rsShow(false);
  $("#rfHint").innerHTML = p.ops && p.ops.length
    ? `Best: <b>${escAttr(p.chain.join(" → "))}</b> (score ${p.score}). Loaded into Recipe — adjust it there, or Accept.`
    : `The current mask already scores best (${p.score}) — nothing to change.`; };
$("#rfEditMask").onclick=()=>openMaskEditor(RF.cur);
$("#rfTouchUp").onclick=()=>openMaskEditor(RF.cur, activeOps());

// recipe: per-op tunable parameters (rendered under the step picker; captured into the op's kw on "+ Add")
const OP_PARAMS = {
  vessel_extend: [{k:"high",label:"seed",def:0.7,step:0.05,min:0,max:3},{k:"low",label:"grow",def:0.4,step:0.05,min:0,max:3},
                  {k:"max_gap",label:"gap",def:40,step:5,min:0,max:300},{k:"max_width",label:"width",def:8,step:1,min:1,max:40}],
  line_centerline: [{k:"alpha",label:"mask-trust",def:0.7,step:0.05,min:0,max:1},{k:"width",label:"width",def:0,step:1,min:0,max:40},
                    {k:"curvature",label:"curve-stiff",def:0,step:0.5,min:0,max:20}],
  sam:        [{k:"n_pos",label:"+pts",def:1,step:1,min:1,max:60},{k:"n_neg",label:"−pts",def:0,step:1,min:0,max:60},
               {k:"margin",label:"neg-gap",def:24,step:2,min:2,max:80},{k:"mask_prior",label:"mask-prior",def:1,step:1,min:0,max:1},
               {k:"keep",label:"keep∪",def:0,step:1,min:0,max:1}],
  dilate:     [{k:"k",label:"k",def:3,step:1,min:1,max:25},{k:"max_contrast",label:"maxΔ",def:0.15,step:0.02,min:0,max:1}],
  erode:      [{k:"k",label:"k",def:3,step:1,min:1,max:25},{k:"min_contrast",label:"minΔ",def:0.15,step:0.02,min:0,max:1}],
  contrast:   [{k:"clip",label:"clip",def:2.0,step:0.5,min:1,max:10}],
  threshold:  [{k:"method",label:"method",type:"select",def:"otsu",opts:[["otsu","Otsu (auto)"],["manual","manual val"],["ght","GHT (Barron)"]]},
               {k:"region",label:"region",type:"select",def:"in_mask",opts:[["in_mask","in-mask (carve)"],["in_bb","in-bbox"],["any","any"]]},
               {k:"direction",label:"keep",type:"select",def:"auto",opts:[["auto","auto"],["above","≥ thr (bright)"],["below","< thr (dark)"]]},
               {k:"val",label:"val",def:128,step:4,min:0,max:255,when:"manual"},
               {k:"nu",label:"reg ν",def:64,step:8,min:0,max:1024,when:"ght"},
               {k:"omega",label:"bias ω",def:0.5,step:0.05,min:0,max:1,when:"ght"}],
  top_k_cc:   [{k:"k",label:"k",def:2,step:1,min:1,max:10}],
  magic_wand: [{k:"tol",label:"tol",def:0.08,step:0.01,min:0,max:1}],
  grabcut:    [{k:"iters",label:"iters",def:5,step:1,min:1,max:20}],
  snap_edges: [{k:"iters",label:"iters",def:20,step:5,min:1,max:200}],
};
const OP_HINT = {
  contrast: "Local contrast (CLAHE) on the image the later steps see — put it first. Higher clip = stronger.",
  vessel_extend: "Grows the tube along vesselness. If it over-extends, raise seed/grow and lower gap; raise width for thick tubes.",
  line_centerline: "Reduces a line to the single shortest path between its two tips. Lower mask-trust bridges gaps via image lines. curve-stiff > 0 needs `pip install agd`.",
  sam: "The result replaces the mask (can shrink and grow). keep∪=1 never shrinks; if it echoes the input, set mask-prior=0. Best on compact parts.",
  threshold: "Otsu (auto) · manual (val 0–255) · GHT. 'keep' picks the bright or dark side; region in-mask only carves. Add contrast first to sharpen the split.",
};
function renderRfParams(){
  const op=$("#rfOp").value, ps=OP_PARAMS[op]||[];
  $("#rfParams").innerHTML = ps.map(p=>{
    const w = p.when?` data-when="${p.when}"`:"";
    if(p.type==="select")
      return `<label${w}>${p.label} <select class=rfp data-k="${p.k}" data-type="select">`+
             p.opts.map(([v,t])=>`<option value="${v}"${v===p.def?" selected":""}>${t}</option>`).join("")+`</select></label>`;
    return `<label${w}>${p.label} <input class=rfp data-k="${p.k}" type=number value="${p.def}" step="${p.step}" min="${p.min}" max="${p.max}" style="width:62px"></label>`;
  }).join("");
  $("#rfHint").textContent = OP_HINT[op]||"";
  $("#rfSamBar").style.display = op==="sam" ? "flex" : "none";
  if(op==="sam") refreshSamStatus();
  rfToggleWhen();
}
function rfToggleWhen(){          // show val (manual) / ν,ω (ght) only for the selected method
  const sel=$('#rfParams .rfp[data-k="method"]'); if(!sel) return;
  $$('#rfParams [data-when]').forEach(el=>{ el.style.display = el.getAttribute("data-when")===sel.value ? "" : "none"; });
}
function readRfKw(){ const kw={};
  $$("#rfParams .rfp").forEach(i=>{
    const lab=i.closest("[data-when]"); if(lab && lab.style.display==="none") return;   // skip hidden (irrelevant) params
    kw[i.dataset.k] = i.dataset.type==="select" ? i.value : +i.value;
  });
  return kw; }
$("#rfOp").onchange=()=>{ renderRfParams(); if(RF.trial) rfDoPreview(); };   // another step: drop the stale trial
$("#rfParams").addEventListener("change", e=>{ if(!e.target.classList.contains("rfp")) return;
  if(e.target.dataset.k==="method") rfToggleWhen();
  if(RF.trial) rfDoPreview([rfTrialOp()]); });                       // a trial preview follows its params
const activeOps = ()=> RF_CHAIN.filter(o=>o.on!==false);          // enabled ops only (toggled-off are skipped)
function renderChain(preview=true){
  $("#rfChain").innerHTML = RF_CHAIN.length ? RF_CHAIN.map((o,i)=>{
      const kv=Object.entries(o.kw||{}).map(([k,v])=>`${k}=${v}`).join(" ");
      return `<div class="step${o.on===false?" off":""}" data-i="${i}" title="click to ${o.on===false?"enable":"disable"}">`+
             `<span>${i+1}. ${escAttr(o.name)}${kv?`<small>${escAttr(kv)}</small>`:""}</span><button class="link x" data-rm="${i}" aria-label="remove step">×</button></div>`;
    }).join("") : `<div class="muted rpNote">No steps yet. Add one below, or use Auto.</div>`;
  rfAcceptLabel(); raSync(); refreshGates();
  if(preview) rfDoPreview();
}
$("#rfChain").onclick=e=>{
  const rm=e.target.closest("[data-rm]"); if(rm){ RF_CHAIN.splice(+rm.dataset.rm,1); renderChain(); return; }
  const st=e.target.closest(".step"); if(st){ const i=+st.dataset.i; RF_CHAIN[i].on=(RF_CHAIN[i].on===false); renderChain(); } };
$("#rfAdd").onclick=()=>{ const kw=readRfKw(); if($("#rfOp").value==="sam") kw.model=$("#rfSamModel").value;   // SAM vs MedSAM recipe
  RF_CHAIN.push({name:$("#rfOp").value, kw, on:true}); renderChain(); };
$("#rfClear").onclick=()=>{ RF_CHAIN=[]; renderChain(); };
// Preview = "show me the step I'm setting up": chain + the dropdown step, WITHOUT adding it (+ Add commits it).
// While such a trial is on screen, changing its params re-previews; Clear/Add/chain edits end it.
function rfTrialOp(){ const name=$("#rfOp").value, kw=readRfKw(); if(name==="sam") kw.model=$("#rfSamModel").value; return {name, kw}; }
$("#rfPreview").onclick=()=>{ RF.trial=true; return withBusy("#rfPreview", ()=>rfDoPreview([rfTrialOp()])); };
function rfRepreviewIfShown(){ if(RF.cur) rfDoPreview(); }
// a class with a saved rule offers it where the recipe is built
async function rfClassRuleHint(cls){
  if(!cls){ $("#rfRules").innerHTML=""; return; }
  const r=await api(`/api/class_rule?cls=${enc(cls)}`);
  $("#rfRules").innerHTML = r.ops && r.ops.length
    ? `${escAttr(cls)} has a saved rule: ${escAttr(r.ops.map(o=>o.name).join(" → "))} <button class="link" id="rfLoadRule">load it</button>` : "";
  const b=$("#rfLoadRule"); if(b) b.onclick=()=>{ RF_CHAIN=r.ops.map(o=>({name:o.name, kw:o.kw||{}, on:o.on!==false})); renderChain(); };
}
async function loadClassRules(){}   // kept for callers: the saved rule now shows per instance (rfClassRuleHint)
// SAM prompt visualisation: where the +/- points and box come from. Pure geometry → works before a checkpoint.
async function rfSamPointsFigure(){
  const iuid=RF.cur; if(!iuid) return "";
  if($("#rfSamModel").value==="medsam") return `MedSAM uses the bounding-box prompt only — no points.`;
  let kw = (activeOps().find(o=>o.name==="sam")||{}).kw;
  if(!kw && $("#rfOp").value==="sam") kw = readRfKw();
  kw = kw || {};
  const r=await post("/api/sam_prompt_preview",{iuid, ops:activeOps(), n_pos:kw.n_pos??1, n_neg:kw.n_neg??0, margin:kw.margin??24});
  if(r.detail) return "";
  return `SAM prompts: <b style="color:#2dd24d">●</b> ${r.n_pos} inside · <b style="color:#eb4a3d">●</b> ${r.n_neg} outside · <b style="color:#ffd000">▭</b> box<br><img src="${r.img}" alt="SAM prompt points" style="max-width:100%;margin-top:4px;border-radius:6px">`; }
$("#rfSamPts").onclick=async()=>{ const f=await rfSamPointsFigure(); $("#rfHint").innerHTML = f || "open an instance first"; };
async function refreshSamStatus(){
  const fam = $("#rfSamModel").value;
  const s=await api(`/api/sam_status?family=${fam==="auto"?"":fam}`);
  const has = f => (s.families||[]).includes(f);
  const label = {samhq:"SAM-HQ", medsam:"MedSAM"}[s.family] || "SAM";
  let msg;
  if(fam==="samhq" && !s.samhq_installed) msg = "the `segment-anything-hq` package is not installed (pip install 'chevron-curator[sam]')";
  else if(!s.installed) msg = "the `segment-anything` package is not installed (pip install 'chevron-curator[sam]')";
  else if(s.ckpt) msg = `${label} ready: ${s.model_type} · ${s.ckpt.split("/").pop()}`;
  else if(fam==="medsam") msg = "no MedSAM checkpoint — drop a *medsam*.pth in CURATOR_SAM_DIR or set CURATOR_MEDSAM_CKPT";
  else if(fam==="samhq") msg = "no SAM-HQ checkpoint yet — set it up ↑";
  else msg = "no checkpoint yet — set up SAM ↑";
  if(fam==="medsam") msg += " · box prompt only";
  $("#rfSamMsg").textContent = msg;
  const needs = fam==="samhq" ? (s.samhq_installed && !has("samhq")) : (fam!=="medsam" && s.installed && !has("sam"));
  $("#rfSamSetup").style.display = needs ? "" : "none";
}
$("#rfSamModel").onchange = refreshSamStatus;
$("#rfSamSetup").onclick=async()=>{ const fam=$("#rfSamModel").value==="samhq"?"samhq":"sam";
  $("#rfSamMsg").innerHTML=SPIN+`downloading ${fam==="samhq"?"SAM-HQ":"SAM"} checkpoint (~375 MB), one-time…`;
  const r=await post("/api/sam_setup",{family:fam});
  if(r.detail){ $("#rfSamMsg").textContent="error: "+r.detail; return; }
  $("#rfSamMsg").textContent=`${fam==="samhq"?"SAM-HQ":"SAM"} ready: ${r.ckpt}`; $("#rfSamSetup").style.display="none"; };

// ---- secondary actions on the instance in focus ----
$("#rfRevertMask").onclick=async()=>{ const iuid=RF.cur; if(!iuid) return;
  const r=await withBusy("#rfRevertMask", ()=>post("/api/revert_mask",{iuid}));
  if(r.detail){ $("#rfHint").textContent=r.detail; return; }
  setStatus(r.stats); rfSetHandState(r.mask_state); $("#rfHint").textContent="Back to the original mask (Undo brings the edit back).";
  rfDoPreview(); };
$("#rfSplit").onclick=async()=>{ const iuid=RF.cur; if(!iuid) return;
  if(!confirm("Split this instance into its connected components?")) return;
  const r=await post("/api/split",{iuids:[iuid]});
  if(r.detail){ $("#rfHint").textContent=r.detail; return; }
  setStatus(r.stats); loadPartitions(true);
  $("#rfHint").textContent=`Split into ${r.n} new instances (re-cluster to see them in groups).`;
  rfAdvance(iuid); };

// ---- apply to others: method × who, a sampled dry-run first, then one undoable commit ----
const RA = {m:"recipe", picked:false, key:null, seed:0, n:0, label:""};   // picked: the user chose a method
function raBody(){ const scope=$("#raScope").value;
  return {ref:RF.cur, method:RA.m, scope, ops:activeOps(),
          match_thresh: scope==="similar" ? parseFloat($("#raTau").value) : null,
          iuids: scope==="selection" ? [...SEL] : null, sam_model:"auto"}; }
const raKey = ()=>JSON.stringify(raBody());
function raSync(){ const scope=$("#raScope").value, hasOps=activeOps().length>0;
  $('#raMethod button[data-m="recipe"]').disabled = !hasOps;
  if(!RA.picked || (RA.m==="recipe" && !hasOps)) RA.m = hasOps ? "recipe" : "transfer";   // follow the fix until chosen
  syncSeg("#raMethod","m",RA.m);
  const sel=$('#raScope option[value="selection"]'); sel.disabled=!SEL.size; sel.textContent=`the current selection (${SEL.size})`;
  if(scope==="selection" && !SEL.size) $("#raScope").value="similar";
  $("#raTauL").style.display = $("#raScope").value==="similar" ? "" : "none";
  $("#raRuleL").style.display = (RA.m==="recipe" && $("#raScope").value==="group" && RF.cls) ? "" : "none";
  const fresh = RA.key && RA.key===raKey();
  $("#raApply").disabled = !fresh;
  $("#raApply").textContent = fresh ? `Apply to ${RA.n} instance${RA.n===1?"":"s"}` : "Apply (preview first)";
  $("#rsSample").style.opacity = RA.key && !fresh ? .45 : 1;
  if(RA.key && !fresh) $("#raSummary").textContent="Settings changed — preview again.";
}
function raStale(){ raSync(); }
$("#raMethod").onclick=e=>{ const b=e.target.closest("button[data-m]"); if(b && !b.disabled){ RA.m=b.dataset.m; RA.picked=true; raSync(); } };
["#raScope","#raTau","#raRule"].forEach(s=>$(s).addEventListener("change", raSync));
$("#rpApply").addEventListener("toggle", raSync);
const RA_NAME = {recipe:"the same recipe", transfer:"the shape of this mask", auto:"Auto on each"};
async function raPreview(){ if(!RF.cur) return;
  const body=raBody();
  $("#raSummary").innerHTML=SPIN+(RA.m==="recipe" ? "running it on a sample…" : "running it on a sample — SAM / chain search per instance, this can take a minute…");
  const r=await withBusy("#raPreview", ()=>post("/api/refine_scope/preview",{...body, sample:8, seed:RA.seed}));
  if(r.detail){ $("#raSummary").textContent=r.detail; RA.key=null; raSync(); return; }
  RA.key=JSON.stringify(body); RA.n=r.n_members; RA.label=r.label;
  const s=r.summary, est = r.shown ? Math.round(s.changed / r.shown * r.n_members) : 0;
  $("#raSummary").innerHTML = r.n_members
    ? `≈ <b>${est} of ${r.n_members}</b> would change (sample: ${s.changed} of ${r.shown}${s.large?`, <b style="color:var(--warn)">${s.large} large</b>`:""}, median IoU ${s.median_iou ?? "—"}).`
    : "No other instances in this scope.";
  const box=$("#rsSample"); box.style.display="";
  box.innerHTML = `<div class="smHead"><b>What ${escAttr(RA_NAME[RA.m])} would do to ${escAttr(r.label)}</b>`+
    `<span class="muted">${r.shown} of ${r.n_members} sampled${r.skipped?` · ${r.skipped} below τ left out`:""} · left = now, right = after</span>`+
    `<span class="grow" style="flex:1"></span><button id="smAgain">Another sample</button><button id="smHide" class="link">hide</button></div>`+
    `<div class="smGrid">`+r.items.map(it=>{ const cp=capParts(it.caption);
      return `<figure class="${it.iou<0.5?"big":""}" title="${escAttr(it.caption)}"><div><img src="${it.before}" alt="now"><img src="${it.after}" alt="after"></div>`+
             `<figcaption><span>${escAttr(cp.title)}</span><span>IoU ${it.iou}</span></figcaption></figure>`; }).join("")+`</div>`;
  $("#smAgain").onclick=()=>{ RA.seed++; raPreview(); };
  $("#smHide").onclick=()=>{ box.style.display="none"; };
  raSync();
}
$("#raPreview").onclick=raPreview;
$("#raApply").onclick=async()=>{ if(RA.key!==raKey()){ raSync(); return; }
  if(!confirm(`Apply ${RA_NAME[RA.m]} to ${RA.n} instance(s) of ${RA.label}? Undo reverts it.`)) return;
  $("#raSummary").innerHTML=SPIN+"applying…";
  const r=await withBusy("#raApply", ()=>post("/api/refine_scope/apply",{...raBody(), save_rule:$("#raRule").checked}));
  if(r.detail){ $("#raSummary").textContent=r.detail; return; }
  setStatus(r.stats); if(r.classes) setClasses(r.classes); loadPartitions(true);
  RA.key=null; $("#rsSample").style.display="none";
  $("#raSummary").textContent=`Applied to ${r.applied} of ${RA.label}${r.skipped?` · ${r.skipped} left out (below τ)`:""}${r.gated_out?` · ${r.gated_out} dropped`:""}${r.rule_saved?" · saved as the class rule":""}.`;
  raSync(); rqLoad(RF.src,{keep:true}); };

// ---- keys (Refine only; never while typing or while the mask editor is open) ----
let RF_SPACE=false;
document.addEventListener("keydown", e=>{
  if(document.querySelector(".tab.active")?.id!=="tab-refine" || $("#maskEditor").classList.contains("on")) return;
  const tn=e.target.tagName;
  if(tn==="INPUT"||tn==="TEXTAREA"||tn==="SELECT"||e.target.isContentEditable||e.metaKey||e.ctrlKey||e.altKey) return;
  if(e.key===" "){ e.preventDefault(); if(!RF_SPACE && RF.cur){ RF_SPACE=true; rsShow(true); } }
  else if(e.key==="Enter"){ e.preventDefault(); rfAccept(); }
  else if(e.key==="ArrowRight"){ e.preventDefault(); rfGo(RF.i+1); }
  else if(e.key==="ArrowLeft"){ e.preventDefault(); rfGo(RF.i-1); }
  else if(/^[0-9]$/.test(e.key) && RF.cands && +e.key < RF.cands.thumbs.length){ e.preventDefault(); mcPick(+e.key); }
});
document.addEventListener("keyup", e=>{ if(e.key===" " && RF_SPACE){ RF_SPACE=false; e.preventDefault(); rsShow(false); } });

rfSetMode(LS.get("rf.mode","auto")); renderRfParams(); renderChain(false);
[["#rpAccept"],["#rfAuto"],["#rfPreview"],["#rfSplit"],["#raPreview"]].forEach(([s])=>gate(s, ()=>!!RF.cur));
gate("#rfRevertMask", ()=>!!RF.cur && !!RF_MASK_STATE && RF_MASK_STATE!=="original");

// ---------- hand-draw mask editor (brush + eraser; zoomed crop with a context toggle) ----------
// Canvas pixels are SOLID red where the mask is on (alpha 0/255 -> crisp binary); CSS opacity makes it
// see-through over the image. Save reads the alpha channel and posts a canvas-res binary PNG + the crop box.
let ME = {iuid:null, ops:null, box:null, ctx:null, painting:false, mode:"brush", context:false, last:[0,0], dirty:false,
          pts:[], hist:[], samBusy:false, line:[], src:null};
// paint a grayscale mask PNG (white = on) into the edit canvas, REPLACING what is there
function mePaintMask(uri){ const cv=$("#meCanvas"), ctx=ME.ctx, w=cv.width, h=cv.height;
  return new Promise(res=>{ const mk=new Image(); mk.onerror=()=>res(); mk.onload=()=>{
    const tmp=document.createElement("canvas"); tmp.width=w; tmp.height=h; const tc=tmp.getContext("2d");
    tc.drawImage(mk,0,0,w,h); const d=tc.getImageData(0,0,w,h).data, out=ctx.createImageData(w,h);
    for(let i=0;i<w*h;i++){ if(d[i*4]>127){ out.data[i*4]=235; out.data[i*4+1]=50; out.data[i*4+2]=40; out.data[i*4+3]=255; } }
    ctx.putImageData(out,0,0); res(); }; mk.src=uri; }); }
// the canvas as the binary PNG the server takes (white = on), at canvas resolution
function meCanvasPng(){ const cv=$("#meCanvas"),W=cv.width,Hh=cv.height,d=ME.ctx.getImageData(0,0,W,Hh).data;
  const tmp=document.createElement("canvas"); tmp.width=W; tmp.height=Hh; const tc=tmp.getContext("2d"), out=tc.createImageData(W,Hh);
  for(let i=0;i<W*Hh;i++){ const v=d[i*4+3]>127?255:0; out.data[i*4]=out.data[i*4+1]=out.data[i*4+2]=v; out.data[i*4+3]=255; }
  tc.putImageData(out,0,0); return tmp.toDataURL("image/png"); }
// undo: a snapshot of the canvas + the SAM clicks before every stroke / click / fill / clear / invert
function meSnap(){ const cv=$("#meCanvas"); ME.hist.push({img:ME.ctx.getImageData(0,0,cv.width,cv.height), pts:ME.pts.slice()});
  if(ME.hist.length>40) ME.hist.shift(); }
function meUndo(){ const h=ME.hist.pop(); if(!h) return; ME.ctx.globalCompositeOperation="source-over";
  ME.ctx.putImageData(h.img,0,0); ME.pts=h.pts; meDrawDots(); ME.dirty=true; }
function meDrawDots(){ const cv=$("#meDots"), c=cv.getContext("2d"); c.clearRect(0,0,cv.width,cv.height);
  if(ME.src){ const [x0,y0,x1,y1]=ME.src; c.save(); c.setLineDash([6,4]); c.lineWidth=1.5; c.strokeStyle="#50c8ff";
    c.strokeRect(x0,y0,x1-x0,y1-y0); c.restore(); }                  // the annotated box: the object is in there
  for(const [x,y,l] of ME.pts){ c.beginPath(); c.arc(x,y,5,0,7); c.fillStyle=l?"#2dd24d":"#eb4a3d"; c.fill();
    c.lineWidth=1.5; c.strokeStyle="#fff"; c.stroke(); }
  if(ME.line.length){                                                // the centre line being placed
    c.save(); c.globalAlpha=.45; meTrace(c, ME.line); c.strokeStyle="#ffc400"; c.lineWidth=+$("#meSize").value;
    c.lineCap="round"; c.lineJoin="round"; c.stroke(); c.restore();
    for(const [x,y] of ME.line){ c.beginPath(); c.arc(x,y,3.5,0,7); c.fillStyle="#ffc400"; c.fill(); c.lineWidth=1; c.strokeStyle="#000"; c.stroke(); } } }
function meSetMode(m){ ME.mode=m; [["#meBrush","brush"],["#meErase","erase"],["#meSam","sam"],["#meLine","line"]].forEach(([s,k])=>$(s).classList.toggle("on", k===m));
  if(m!=="erase") LS.set("me.mode", m);                             // a box opened with D starts in the last tool
  if(m!=="line" && ME.line.length){ ME.line=[]; meDrawDots(); }
  if(m==="line") $("#meMsg").textContent="click along the centre line · double-click / Enter = paint (shift = erase) · Backspace = drop point · Esc = drop line"; }
// Tubes, wires and leads: a few clicks along the centre line, painted as a smooth (Catmull-Rom) stroke of
// the brush size — what brushing a 2 px-wide catheter by hand amounts to, in a handful of clicks.
function meTrace(c, P){ c.beginPath(); c.moveTo(P[0][0], P[0][1]);
  if(P.length===1){ c.lineTo(P[0][0]+.01, P[0][1]); return; }
  for(let i=0;i<P.length-1;i++){ const p0=P[Math.max(0,i-1)], p1=P[i], p2=P[i+1], p3=P[Math.min(P.length-1,i+2)];
    c.bezierCurveTo(p1[0]+(p2[0]-p0[0])/6, p1[1]+(p2[1]-p0[1])/6, p2[0]-(p3[0]-p1[0])/6, p2[1]-(p3[1]-p1[1])/6, p2[0], p2[1]); } }
function meLineFinish(erase){
  // a double-click lands two clicks on the same spot first: drop those repeats
  const P=ME.line.filter((p,i,a)=>!i || Math.hypot(p[0]-a[i-1][0], p[1]-a[i-1][1])>2);
  ME.line=[]; if(!P.length){ meDrawDots(); return; }
  meSnap(); const ctx=ME.ctx; ctx.save(); ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
  meTrace(ctx, P); ctx.strokeStyle="rgb(235,50,40)"; ctx.lineWidth=+$("#meSize").value; ctx.lineCap="round"; ctx.lineJoin="round";
  ctx.stroke(); ctx.restore(); ME.dirty=true; meDrawDots(); }
async function meSamClick(x,y,label){ if(ME.samBusy) return;
  meSnap(); ME.pts.push([x,y,label]); meDrawDots(); ME.samBusy=true; $("#meMsg").innerHTML=SPIN+"SAM…";
  const r=await post("/api/edit_sam",{iuid:ME.iuid, png:meCanvasPng(), box:ME.box, model:$("#meSamModel").value,
                                      points:ME.pts.map(p=>[p[0],p[1]]), labels:ME.pts.map(p=>p[2])});
  ME.samBusy=false;
  if(r.detail){ meUndo(); $("#meMsg").textContent=r.detail; return; }
  ME.ctx.globalCompositeOperation="source-over"; ME.ctx.clearRect(0,0,$("#meCanvas").width,$("#meCanvas").height);
  await mePaintMask(r.mask); ME.dirty=true;
  const np=ME.pts.filter(p=>p[2]).length; $("#meMsg").textContent=`${np} include · ${ME.pts.length-np} exclude click(s)`; }
async function meLoad(){
  const seed = ME.ops&&ME.ops.length ? `&ops=${enc(JSON.stringify(ME.ops))}` : "";
  const r=await api(`/api/edit_view?iuid=${enc(ME.iuid)}&context=${ME.context?1:0}${seed}`);
  if(r.detail){ alert(r.detail); return false; }
  ME.box=r.box;
  const bg=$("#meBg"), cv=$("#meCanvas");
  bg.src=r.img; bg.width=r.w; bg.height=r.h; cv.width=r.w; cv.height=r.h;
  const dots=$("#meDots"); dots.width=r.w; dots.height=r.h;
  const ctx=cv.getContext("2d"); ME.ctx=ctx; ctx.clearRect(0,0,r.w,r.h);
  await mePaintMask(r.mask);
  ME.src=r.src_box||null; ME.line=[];
  ME.pts=[]; ME.hist=[]; meDrawDots(); $("#meMsg").textContent="";   // clicks are in canvas pixels → reset with the view
  ME.dirty=false; return true;
}
async function openMaskEditor(iuid, ops=null, mode="brush"){ if(!iuid){alert("load an instance (iuid above) first");return;}
  ME.iuid=iuid; ME.ops=ops&&ops.length?ops:null; ME.context=false; ME.line=[];
  $("#meContext").textContent="show full image";
  $("#meId").textContent=iuid.slice(0,8)+(ME.ops?` · touching up ${ME.ops.map(o=>o.name).join("→")}`:"");
  if(await meLoad()){ meSetMode(mode); $("#maskEditor").classList.add("on"); } else boxDrawDone(false); }
gate("#rfEditMask", ()=>!!RF.cur);
gate("#rfTouchUp", ()=>!!RF.cur && activeOps().length>0);
// Outside Refine: every grid cell carries a hover ✏️, and the inspector an "Edit mask" for a single selection
// — you notice a bad mask where you browse, so that is where editing has to start.
document.addEventListener("click", e=>{ const b=e.target.closest(".cell .cellEdit"); if(!b) return;
  e.stopPropagation(); openMaskEditor(b.closest(".cell").dataset.iuid); }, true);   // capture: beat the grids' own cell-click handlers
$("#inspEditMask").onclick=()=>{ if(SEL.size===1) openMaskEditor([...SEL][0]); };
gate("#inspEditMask", ()=>SEL.size===1);
function mePos(e){ const cv=$("#meCanvas"), r=cv.getBoundingClientRect();
  return [ (e.clientX-r.left)*cv.width/r.width, (e.clientY-r.top)*cv.height/r.height ]; }
function meStyle(){ ME.ctx.globalCompositeOperation = ME.mode==="erase" ? "destination-out" : "source-over"; }
function meDab(x,y){ const ctx=ME.ctx,s=+$("#meSize").value; meStyle(); ctx.fillStyle="rgb(235,50,40)"; ctx.beginPath(); ctx.arc(x,y,s/2,0,7); ctx.fill(); ME.dirty=true; }
function meLine(a,b){ const ctx=ME.ctx,s=+$("#meSize").value; meStyle(); ctx.strokeStyle="rgb(235,50,40)"; ctx.lineWidth=s; ctx.lineCap="round"; ctx.lineJoin="round"; ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke(); ME.dirty=true; }
$("#meCanvas").addEventListener("pointerdown",e=>{ e.preventDefault();
  if(ME.mode==="sam"){ const [x,y]=mePos(e); meSamClick(x,y, (e.button===2||e.shiftKey)?0:1); return; }
  if(ME.mode==="line"){ if(e.button===0){ ME.line.push(mePos(e)); meDrawDots(); } return; }
  if(e.button!==0) return;
  meSnap(); ME.painting=true; ME.last=mePos(e); meDab(ME.last[0],ME.last[1]); try{$("#meCanvas").setPointerCapture(e.pointerId);}catch(_){} });
$("#meCanvas").addEventListener("contextmenu", e=>e.preventDefault());       // right-click = exclude in SAM mode
$("#meCanvas").addEventListener("dblclick", e=>{ if(ME.mode==="line"){ e.preventDefault(); meLineFinish(e.shiftKey); } });
$("#meCanvas").addEventListener("pointermove",e=>{ if(!ME.painting)return; const p=mePos(e); meLine(ME.last,p); ME.last=p; });
$("#meCanvas").addEventListener("pointerup",()=>{ ME.painting=false; });
$("#meBrush").onclick=()=>meSetMode("brush");
$("#meErase").onclick=()=>meSetMode("erase");
$("#meSam").onclick=()=>{ meSetMode("sam"); $("#meMsg").textContent="click = include · shift / right-click = exclude"; };
$("#meLine").onclick=()=>meSetMode("line");
$("#meUndo").onclick=meUndo;
addEventListener("keydown", e=>{ if(!$("#maskEditor").classList.contains("on")) return;
  if((e.metaKey||e.ctrlKey) && e.key.toLowerCase()==="z"){ e.preventDefault(); meUndo(); return; }
  const t=e.target, tag=(t&&t.tagName)||""; if(tag==="INPUT"||tag==="TEXTAREA"||tag==="SELECT") return;
  if(e.metaKey||e.ctrlKey||e.altKey) return;
  // a centre line in progress owns Enter / Backspace / Esc
  if(ME.line.length && e.key==="Enter"){ e.preventDefault(); meLineFinish(e.shiftKey); }
  else if(ME.line.length && e.key==="Backspace"){ e.preventDefault(); ME.line.pop(); meDrawDots(); }
  else if(ME.line.length && e.key==="Escape"){ e.preventDefault(); ME.line=[]; meDrawDots(); }
  else if(e.key==="Enter"){ e.preventDefault(); $("#meSave").click(); }
  else if(e.key==="Escape" && !ME.dirty){ e.preventDefault(); $("#meCancel").click(); }
  else if(e.key==="b"){ meSetMode("brush"); } else if(e.key==="x"){ meSetMode("erase"); }
  else if(e.key==="l"){ meSetMode("line"); }  else if(e.key==="s"){ $("#meSam").click(); }
  else if(e.key==="[" || e.key==="]"){ const z=$("#meSize"); z.value=Math.max(1, Math.min(80, +z.value+(e.key==="]"?2:-2))); meDrawDots(); } });
$("#meSize").addEventListener("input", ()=>{ if(ME.line.length) meDrawDots(); });
$("#meClear").onclick=()=>{ meSnap(); ME.ctx.clearRect(0,0,$("#meCanvas").width,$("#meCanvas").height); ME.pts=[]; meDrawDots(); ME.dirty=true; };
$("#meInvert").onclick=()=>{ meSnap(); const cv=$("#meCanvas"),ctx=ME.ctx,d=ctx.getImageData(0,0,cv.width,cv.height),a=d.data;
  for(let i=0;i<cv.width*cv.height;i++){ const on=a[i*4+3]>127; a[i*4]=235;a[i*4+1]=50;a[i*4+2]=40;a[i*4+3]=on?0:255; } ctx.putImageData(d,0,0); ME.dirty=true; };
$("#meFill").onclick=()=>{ meSnap(); const cv=$("#meCanvas"),ctx=ME.ctx,W=cv.width,Hh=cv.height,img=ctx.getImageData(0,0,W,Hh),a=img.data,N=W*Hh;
  const on=i=>a[i*4+3]>127, seen=new Uint8Array(N), st=[];
  for(let x=0;x<W;x++){ st.push(x,(Hh-1)*W+x); } for(let y=0;y<Hh;y++){ st.push(y*W,y*W+W-1); }
  while(st.length){ const p=st.pop(); if(p<0||p>=N||seen[p]||on(p))continue; seen[p]=1; const x=p%W,y=(p-x)/W;
    if(x>0)st.push(p-1); if(x<W-1)st.push(p+1); if(y>0)st.push(p-W); if(y<Hh-1)st.push(p+W); }
  for(let i=0;i<N;i++){ if(!on(i)&&!seen[i]){ a[i*4]=235;a[i*4+1]=50;a[i*4+2]=40;a[i*4+3]=255; } } ctx.putImageData(img,0,0); ME.dirty=true; };
$("#meContext").onclick=async()=>{ if(ME.dirty && !confirm("Switching view discards unsaved strokes. Continue?"))return;
  ME.context=!ME.context; $("#meContext").textContent=ME.context?"show crop":"show full image"; await meLoad(); };
$("#meCancel").onclick=()=>{ $("#maskEditor").classList.remove("on"); ME.line=[]; boxDrawDone(false); };
$("#meSave").onclick=async()=>{
  if(ME.line.length>1) meLineFinish(false);                          // a line left unfinished is meant to be in
  const r=await withBusy("#meSave", ()=>post("/api/set_mask",{iuid:ME.iuid, png:meCanvasPng(), box:ME.box}));
  if(r.detail){ alert(r.detail); return; }
  setStatus(r.stats); $("#maskEditor").classList.remove("on"); ME.line=[];
  if(BOX.drawKey){ boxDrawDone(true); return; }                      // the box list is where you are: no grid reloads
  if(ME.ops && RF.cur===ME.iuid){     // the saved mask already contains the chain → don't let Apply redo it
    RF_CHAIN=[]; renderChain(false); $("#rfHint").textContent=`touched-up result saved for ${ME.iuid.slice(0,6)} — chain cleared (it is baked into the mask)`; }
  if(RF.cur===ME.iuid) rfDoPreview();                  // refresh the Refine stage
  if(typeof INST!=="undefined" && INST.pid) selectPartition(INST.pid);
  if(typeof IIMG!=="undefined" && IIMG.id) loadImage(true); };

// ---------- latent-space Map (projection + paint-select -> the curator's existing actions) ----------
const MAP = { pts:[], loaded:false, view:{s:1,ox:0,oy:0}, mode:false, dragging:false, last:[0,0],
              dpr:1, grid:null, gcol:64, sel:SEL, colorBy:"state",
              // the scope picked in the rail, lit on the canvas: a Set of iuids, or null for "no scope
              // highlighted". scopeMissing says the scope IS picked but none of it made this projection.
              scope:null, scopePid:null, scopeMissing:false, info:"" };   // paints into THE shared selection
function mapCanvasSize(){ const cv=$("#mapCanvas"), st=$("#mapStage"), dpr=window.devicePixelRatio||1;
  MAP.dpr=dpr; cv.width=Math.max(1,Math.round(st.clientWidth*dpr)); cv.height=Math.max(1,Math.round(st.clientHeight*dpr)); }
function mapOnShow(){ mapCanvasSize(); if(!MAP.loaded) mapLoad(); else { mapFit(); mapDraw(); } }
addEventListener("resize", ()=>{ if(document.querySelector(".tab.active")?.id==="tab-map" && MAP.loaded){ mapCanvasSize(); mapFit(); mapDraw(); } });
async function mapLoad(){
  $("#mapInfo").innerHTML = SPIN+"projecting instances (h-NNE / UMAP)…";
  const r = await withBusy("#mapLoad", ()=>api(`/api/projection_points?method=hnne`));
  if(!r || r.detail){ $("#mapInfo").innerHTML=`<span style="color:var(--warn)">${(r&&r.detail)||"projection failed"}</span>`; return; }
  // NB: do NOT clear the selection here. MAP.sel IS the shared selection now, and mapOnShow()
  // auto-loads on the first Grid->Map switch — clearing would wipe what you just selected in the
  // Grid, which is precisely the behaviour making Map a view rather than a tab is meant to give.
  // Selected iuids missing from a re-projection simply are not drawn; nothing is corrupted.
  MAP.pts=r.points||[]; MAP.loaded=true; mapBuildGrid(); mapCanvasSize(); mapFit(); mapDraw(); mapRenderSel();
  MAP.info = `${r.n} instances · ${r.method}${r.truncated?` · first ${r.n} (capped)`:""} · features: ${Object.keys(r.spec||{}).join("+")||"—"} · wheel=zoom, drag=pan, ✏️=paint-select, alt-drag=unpaint`;
  mapSyncScope();                                     // a scope picked before the map loaded still lights up
}
// Clicking a scope in the rail — a class, a FINCH partition, a sub-cluster, the rejected bin — asks
// "where does this sit in the latent space?", so the map answers by lighting those points and muting
// the rest. Membership comes out of the projection payload itself (a point carries its pid/state), so
// a class or a partition costs nothing; only a sub-cluster, which the projection knows nothing about,
// has to ask the server for the scope's iuids.
async function mapSyncScope(){
  // `MAP` is a const declared in this block, i.e. BELOW selectPartition. A deep link that selects a
  // scope during the top-level pass would reach it inside its temporal dead zone, so read it guarded
  // rather than throwing on boot; there is no map to light at that point anyway.
  try{ if(!MAP.loaded) return; }catch(_){ return; }
  const pid = (typeof INST!=="undefined" && INST.pid && !isBoxScope()) ? INST.pid : null;
  MAP.scopePid = pid; MAP.scopeMissing = false;
  let s = null;
  if(pid){
    if(isSubScope()){
      try{ const iu = await scopeIuids(); if(MAP.scopePid!==pid) return;   // a newer scope won the race → drop this
           s = new Set(iu); }
      catch(_){ s = null; }
    } else {
      const cs = /^csub:(.*):(\d+)$/.exec(pid);
      const hit = isRejectedScope() ? (p=>p.state==="reject")
                : cs ? (p=>p.pid==="class:"+cs[1] && (p.sub||0)===+cs[2]) : (p=>p.pid===pid);
      s = new Set(MAP.pts.filter(hit).map(p=>p.iuid));
    }
    // Muting EVERY point because the scope missed this projection (stale coords, a capped map, an
    // instance assigned since) reads as a broken map. Say what happened and leave the colors alone.
    if(s && !s.size){ s = null; MAP.scopeMissing = true; }
  }
  MAP.scope = s;
  mapDraw(); if(MAP3D) MAP3D.recolor(); mapScopeInfo();
}
function mapScopeLabel(pid){
  if(pid===REJECTED_SCOPE) return "Rejected";
  if(String(pid).startsWith(SUB_PREFIX)) return `sub ${String(pid).slice(SUB_PREFIX.length)}`;
  if(String(pid).startsWith("class:")) return (MAP.pts.find(p=>p.pid===pid)||{}).cls || pid;
  const cs = /^csub:(.*):(\d+)$/.exec(pid);
  if(cs) return `${(MAP.pts.find(p=>p.pid==="class:"+cs[1])||{}).cls || cs[1]} › ${+cs[2]+1}`;
  return `partition ${pid}`;
}
function mapScopeInfo(){
  const el=$("#mapInfo"); if(!el||!MAP.loaded) return;
  // The scope is picked in the RAIL, so the map is where you notice the highlight and the last place
  // you would look for the way out of it. Say it here, next to the thing it undoes.
  const drop = `<a id="mapScopeClear" class="psugPick" title="stop highlighting this scope — show every point again">✕ clear</a>`;
  if(MAP.scope) el.innerHTML = `${escAttr(MAP.info)} · highlighting ${escAttr(mapScopeLabel(MAP.scopePid))} — ${MAP.scope.size} of ${MAP.pts.length} · ${drop}`;
  else if(MAP.scopeMissing) el.innerHTML = `${escAttr(MAP.info)} · ${escAttr(mapScopeLabel(MAP.scopePid))} has no points on this map — reload it · ${drop}`;
  else el.innerHTML = escAttr(MAP.info);
}
// delegated: mapScopeInfo() rewrites this line every time the scope changes
$("#mapInfo").addEventListener("click", e=>{ if(e.target.closest("#mapScopeClear")) selectPartition(null); });
function mapBuildGrid(){ const G=MAP.gcol, b=Array.from({length:G*G},()=>[]);
  MAP.pts.forEach((p,i)=>{ const gx=Math.min(G-1,Math.max(0,(p.x*G)|0)), gy=Math.min(G-1,Math.max(0,(p.y*G)|0)); b[gy*G+gx].push(i); });
  MAP.grid=b; }
function mapQuery(wx,wy,wr){ const G=MAP.gcol, out=[], r=Math.ceil(wr*G)+1, cx=(wx*G)|0, cy=(wy*G)|0;
  for(let gy=Math.max(0,cy-r); gy<=Math.min(G-1,cy+r); gy++) for(let gx=Math.max(0,cx-r); gx<=Math.min(G-1,cx+r); gx++)
    for(const i of MAP.grid[gy*G+gx]){ const p=MAP.pts[i]; if((p.x-wx)**2+(p.y-wy)**2 <= wr*wr) out.push(i); }
  return out; }
function mapFit(){ const cv=$("#mapCanvas"), W=cv.width, H=cv.height, m=0.06*Math.min(W,H), s=Math.min(W,H)-2*m;
  MAP.view={ s, ox:(W-s)/2, oy:(H-s)/2 }; }
// FNV-1a plus murmur3's finaliser, not the usual `h*31+c`: partition ids are small integers, and a
// polynomial hash of "0".."9" lands within six degrees of hue, so "colour by partition" painted every
// cluster the same yellow. The finaliser is what spreads consecutive ids across the wheel.
function mapHashHue(s){ let h=2166136261; for(let i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619); }
  h^=h>>>16; h=Math.imul(h,2246822507); h^=h>>>13; h=Math.imul(h,3266489909); h^=h>>>16; return (h>>>0)%360; }
const mapInScope = p => !MAP.scope || MAP.scope.has(p.iuid);
// Every palette entry is HSL so that "the same color, muted" is one expression rather than a second
// palette: a dimmed point keeps its hue, and stays readable as the class/partition it belongs to.
function mapHslOf(p){
  if(MAP.colorBy==="state") return p.state==="class"?[152,48,47]:(p.state==="reject"?[9,73,56]:[218,13,56]);
  if(MAP.colorBy==="score"){ const v=Math.max(0,Math.min(1,p.score||0)); return [(v*130)|0,70,55]; }
  // class › sub-cluster: the class keeps its hue, each sub-cluster nudges hue/lightness around it, so a
  // class reads as one family and its modes as shades of it. Unassigned points sink to grey.
  if(MAP.colorBy==="subclass") return p.cls ? csubShade(p.cls, p.sub||0) : [220,10,30];
  const key = MAP.colorBy==="class" ? p.cls : MAP.colorBy==="source" ? p.source
            : MAP.colorBy==="method" ? p.method : p.pid;
  return key ? [mapHashHue(key),64,58] : [220,16,27];
}
function mapColorOf(p){
  const [h,s,l] = mapHslOf(p);
  return mapInScope(p) ? `hsl(${h},${s}%,${l}%)` : `hsl(${h},${Math.round(s*0.22)}%,${Math.round(l*0.42)}%)`;
}
function mapDraw(){ if(!MAP.loaded) return; const cv=$("#mapCanvas"), ctx=cv.getContext("2d"), v=MAP.view;
  ctx.clearRect(0,0,cv.width,cv.height);
  const r=(+$("#mapPtSize").value)*MAP.dpr, d=Math.max(1,r*2);
  // Two passes: the dimmed rest is painted FIRST so the highlighted scope sits on top of it instead
  // of being buried under whatever the projection happened to order later. In-scope points also get
  // a slightly bigger square — at 1-2 px, color alone is not enough to pick a cluster out.
  for(const p of MAP.pts) if(!mapInScope(p)){ ctx.fillStyle=mapColorOf(p); ctx.fillRect(p.x*v.s+v.ox-r, p.y*v.s+v.oy-r, d, d); }
  const rs = MAP.scope ? r+0.75*MAP.dpr : r, ds = Math.max(1,rs*2);
  for(const p of MAP.pts) if(mapInScope(p)){ ctx.fillStyle=mapColorOf(p); ctx.fillRect(p.x*v.s+v.ox-rs, p.y*v.s+v.oy-rs, ds, ds); }
  if(MAP.sel.size){ ctx.strokeStyle="#fff"; ctx.lineWidth=MAP.dpr;
    for(const p of MAP.pts) if(MAP.sel.has(p.iuid)){ ctx.beginPath(); ctx.arc(p.x*v.s+v.ox, p.y*v.s+v.oy, r+1.5*MAP.dpr, 0, 7); ctx.stroke(); } }
}
function mapEvtPos(e){ const cv=$("#mapCanvas"), rect=cv.getBoundingClientRect();
  return [ (e.clientX-rect.left)*cv.width/rect.width, (e.clientY-rect.top)*cv.height/rect.height ]; }
function mapS2W(sx,sy){ const v=MAP.view; return [ (sx-v.ox)/v.s, (sy-v.oy)/v.s ]; }
function mapPaint(e){ const [sx,sy]=mapEvtPos(e), [wx,wy]=mapS2W(sx,sy), wr=((+$("#mapBrush").value)*MAP.dpr)/MAP.view.s;
  const erase=e.altKey; for(const i of mapQuery(wx,wy,wr)){ const u=MAP.pts[i].iuid; erase?MAP.sel.delete(u):MAP.sel.add(u); }
  renderInspector(); mapDraw(); }
function mapHover(e){ const [sx,sy]=mapEvtPos(e), [wx,wy]=mapS2W(sx,sy), wr=(8*MAP.dpr)/MAP.view.s, idxs=mapQuery(wx,wy,wr), tip=$("#mapTip");
  if(!idxs.length){ tip.style.display="none"; return; }
  let best=idxs[0], bd=1e18; for(const i of idxs){ const p=MAP.pts[i], dd=(p.x-wx)**2+(p.y-wy)**2; if(dd<bd){bd=dd;best=i;} }
  const p=MAP.pts[best], rect=$("#mapCanvas").getBoundingClientRect(), cx=e.clientX-rect.left, cy=e.clientY-rect.top;
  tip.style.left=Math.min(rect.width-140, cx+12)+"px"; tip.style.top=Math.min(rect.height-160, cy+12)+"px";
  $("#mapTipCap").textContent=`${p.cls||p.state}${p.nsub>1?` › ${(p.sub||0)+1}/${p.nsub}`:""} · ${p.iuid.slice(0,6)} · s=${p.score}`;
  $("#mapTipImg").src=cropUrl(p.iuid); tip.style.display="block"; }
$("#mapCanvas").addEventListener("wheel", e=>{ if(!MAP.loaded)return; e.preventDefault();
  const [sx,sy]=mapEvtPos(e), v=MAP.view, k=Math.exp(-e.deltaY*0.0015), [wx,wy]=mapS2W(sx,sy);
  v.s*=k; v.ox=sx-wx*v.s; v.oy=sy-wy*v.s; mapDraw(); }, {passive:false});
$("#mapCanvas").addEventListener("pointerdown", e=>{ if(!MAP.loaded)return; MAP.dragging=true; MAP.last=mapEvtPos(e);
  try{$("#mapCanvas").setPointerCapture(e.pointerId);}catch(_){} if(MAP.mode) mapPaint(e); });
$("#mapCanvas").addEventListener("pointermove", e=>{ if(!MAP.loaded)return;
  if(MAP.dragging){ const p=mapEvtPos(e); if(MAP.mode){ mapPaint(e); } else { const v=MAP.view; v.ox+=p[0]-MAP.last[0]; v.oy+=p[1]-MAP.last[1]; MAP.last=p; mapDraw(); } }
  else if(!MAP.mode){ mapHover(e); } });
$("#mapCanvas").addEventListener("pointerup", ()=>{ MAP.dragging=false; if(MAP.mode) mapRenderSel(); });
$("#mapCanvas").addEventListener("pointerleave", ()=>{ $("#mapTip").style.display="none"; });
$("#mapMode").onclick=()=>{ MAP.mode=!MAP.mode; $("#mapMode").textContent=MAP.mode?"✏️ select":"✋ pan";
  $("#mapMode").classList.toggle("primary",MAP.mode); $("#mapCanvas").style.cursor=MAP.mode?"crosshair":"grab"; $("#mapTip").style.display="none"; };
// ---------- 3D latent walk (Spacewalker's viewer, over instances) ----------
// three.js is ~330 KB and most sessions never open 3D, so the module is imported on FIRST switch
// only. The 3D view paints into SEL — the same selection the Grid, Map-2D and Image views use, which
// is the whole point of it being a view rather than a separate tool.
let MAP3D = null, MAP_IS_3D = false;
async function map3dEnsure(){
  if(MAP3D) return MAP3D;
  const { createMap3D } = await import("/map3d.js");
  MAP3D = createMap3D($("#mapCanvas3d"), {
    getSelected: ()=>SEL,
    onSelectionChange: ()=>{ renderInspector(); refreshGates(); },
    colorOf: p=>mapColorOf(p),
  });
  return MAP3D;
}
async function mapSet3D(on){
  // The renderer is built BEFORE the view state flips. Not all machines have a usable WebGL device
  // — software/remote GL, a driver blocklist, a locked-down browser — and there three.js throws.
  // Flipping first would hide the 2D canvas, show an empty one and leave the toggle reading "2D":
  // a map that looks broken, with nothing said.
  let v = null;
  if(on){
    try{ v = await map3dEnsure(); }
    catch(e){
      console.error("[map3d]", e);
      setStatus("3D needs WebGL, which this browser or display cannot provide — staying in 2D");
      on = false;
    }
  }
  MAP_IS_3D = !!on;
  $("#mapCanvas").style.display = on ? "none" : "block";
  $("#mapCanvas3d").style.display = on ? "block" : "none";
  $("#map3d").classList.toggle("primary", on);
  $("#map3d").textContent = on ? "2D" : "3D";
  if(!on){ mapCanvasSize(); mapDraw(); return; }
  // 3D needs its own projection (dims=3) — the 2D coords have no z
  const r = await withBusy("#map3d", ()=>api("/api/projection_points?method=hnne&dims=3"));
  if(r && !r.detail){ v.build(r.points||[]); }
  v.resize(); v.frame();
}
$("#map3d").onclick=()=>mapSet3D(!MAP_IS_3D);
$("#mapBrush").addEventListener("input", ()=>{ if(MAP3D) MAP3D.setBrush(+$("#mapBrush").value/260); });
$("#mapPtSize").addEventListener("input", ()=>{ if(MAP3D) MAP3D.setPointSize(+$("#mapPtSize").value/220); });

// Query pin (P6): place a phrase or an instance on the map and jump to its neighbours.
$("#mapQGo").onclick=async()=>{
  const q=$("#mapQ").value.trim(); if(!q) return;
  const body={ method:"hnne", dims: MAP_IS_3D?3:2, k:24 };
  if(/^[0-9a-f]{16,}$/i.test(q)) body.iuid=q; else body.text=q;
  const r=await withBusy("#mapQGo", ()=>post("/api/project_query", body));
  if(r.detail||r.error){ $("#mapInfo").innerHTML=`<span style="color:var(--warn)">${escAttr(r.detail||r.error)}</span>`; return; }
  SEL.clear(); (r.neighbors||[]).forEach(n=>SEL.add(n.iuid));
  renderInspector(); refreshGates();
  if(MAP_IS_3D && MAP3D){ MAP3D.pin(r.point); MAP3D.recolor(); } else { mapDraw(); }
  $("#mapInfo").textContent=`query placed at (${r.point.x.toFixed(2)}, ${r.point.y.toFixed(2)}) · ${r.neighbors.length} nearest selected`;
};

$("#mapColor").onchange=e=>{ MAP.colorBy=e.target.value; mapDraw(); if(MAP3D) MAP3D.recolor(); };
$("#mapPtSize").oninput=()=>mapDraw();
$("#mapReset").onclick=()=>{ if(MAP.loaded){ mapFit(); mapDraw(); } };
$("#mapLoad").onclick=()=>{ MAP.loaded=false; mapLoad(); };
// The map used to keep its own confirmation grid + count + Assign/Reject. All of that is the
// inspector's job now; what the canvas still owns is drawing white rings on the selected points.
function mapRenderSel(){ renderInspector(); }
async function mapRefreshAfter(){ const r=await api(`/api/projection_points?method=hnne`);   // coords cached server-side -> instant recolor
  if(r && !r.detail){ MAP.pts=r.points||[]; mapBuildGrid(); } mapDraw(); mapRenderSel(); mapSyncScope(); }

// ---------- Classifier ----------
let CLF={offset:0,limit:60,total:0};
const clfGrid = makeGrid("#clfgrid","#clfExclCount","selected", ()=>{ refreshGates(); renderInspector(); }, SEL);
function syncClfFeats(){ if(!window._features)return;
  $("#clfFeats").innerHTML = (D=>featBoxes("clffeat", f=>D.has(f)||f=='shape'))(defaultFeatSet()); }
$("#clfTrain").onclick=async()=>{
  const feats=$$(".clffeat:checked").map(e=>e.value);
  $("#clfReport").innerHTML=SPIN+"training…";
  const r=await withBusy("#clfTrain", ()=>post("/api/train_classifier",{features:feats, algo:$("#clfAlgo").value, openset:$("#clfOpen").checked}));
  if(!r.ok){ $("#clfReport").innerHTML=`<span style="color:var(--warn)">${r.error||'train failed'}</span>`+(r.skipped?.length?` · skipped: ${r.skipped.join(", ")}`:""); return; }
  const yd=Object.entries(r.youden||{}).map(([k,v])=>`${k}: ${v}`).join(" · ");
  // "only class" is a SELECT (a datalist only suggests, it can't restrict) offering ONLY the classifier's
  // trained classes (they have instances and are the only classes apply can predict); "(all)" stays first.
  const prev=$("#clfOnly").value;
  $("#clfOnly").innerHTML = `<option value="">(all)</option>` + (r.classes||[]).map(c=>`<option value="${escAttr(c)}">${escAttr(c)}</option>`).join("");
  if((r.classes||[]).includes(prev)) $("#clfOnly").value=prev;     // keep the prior pick if still trained
  $("#clfReport").innerHTML=`trained <b>${escAttr(r.algo||$("#clfAlgo").value)}</b> on ${r.n_classes} classes: ${r.classes.join(", ")}`+
    (r.dropped_nan?.length?` · <span style="color:var(--warn)">dropped (NaN): ${r.dropped_nan.join(", ")}</span>`:"")+
    (r.skipped?.length?` · skipped (&lt;2): ${r.skipped.join(", ")}`:"")+(yd?`<br>recommended thresholds (Youden J): ${yd}`:""); };
$("#clfThr").oninput=e=>$("#clfThrV").textContent=(+e.target.value).toFixed(2);
async function clfLoad(reset){ if(reset){CLF.offset=0;clfGrid.reset();}
  const r=await withBusy("#clfPredict", ()=>api(`/api/predict?thresh=${$("#clfThr").value}&only_class=${enc($("#clfOnly").value.trim())}&offset=${CLF.offset}&limit=${CLF.limit}`));
  CLF.total=r.total;
  if(reset && !r.items.length) clfGrid.msg("no unassigned instances pass this threshold");
  else clfGrid.append(r.items, it=>`${it.cls} · ${it.conf}`);
  CLF.offset+=r.items.length; $("#clfMore").style.display=CLF.offset<r.total?"inline-block":"none";
  if(reset) $("#clfReport").insertAdjacentHTML("beforeend", ` — <b>${r.total}</b> would be assigned (tick crops to EXCLUDE).`); }
$("#clfPredict").onclick=()=>clfLoad(true);
$("#clfMore").onclick=()=>clfLoad(false);
$("#clfApply").onclick=async()=>{
  const r=await withBusy("#clfApply", ()=>post("/api/apply_predictions",{thresh:+$("#clfThr").value, only_class:$("#clfOnly").value.trim(), exclude:[...SEL]   /* the singled-out crops are held back from the bulk apply */}));
  setStatus(r.stats); setClasses(r.classes); clfGrid.reset(); loadPartitions(true); $("#clfReport").innerHTML=`assigned <b>${r.n}</b> instances.`; };
// fix misclassifications: assign the SELECTED preview instances to a chosen class (overrides the prediction)
$("#clfAssignSel").onclick=async()=>{
  const cls=$("#clfAssignClass").value.trim(); if(!cls||!clfGrid.sel.size) return;
  const iu=[...clfGrid.sel];
  const r=await post("/api/assign",{iuids:iu, cls});
  setStatus(r.stats); setClasses(r.classes); clfGrid.drop(iu); loadPartitions(true);   // drop the now-assigned ones from the preview
  $("#clfReport").innerHTML=`assigned <b>${iu.length}</b> selected → <b>${cls}</b>.`; };
// reject the selected predictions (a wrong/garbage prediction -> background) straight from the preview
$("#clfReject").onclick=async()=>{ const iu=[...clfGrid.sel]; if(!iu.length){alert("select predictions to reject");return;}
  const r=await post("/api/reject",{iuids:iu}); setStatus(r.stats); setClasses(r.classes); clfGrid.drop(iu); loadPartitions(true);
  $("#clfReport").innerHTML=`rejected <b>${iu.length}</b> selected → background.`; };
// reject suggestions: the complement of the assign preview — unassigned instances the classifier is
// confident match NO curated class (low max-probability), surfaced as background/noise to reject.
let CLFREJ={offset:0,limit:60,total:0};
const clfRejGrid = makeGrid("#clfRejGrid","#clfRejSelCount","selected", ()=>{ refreshGates(); renderInspector(); }, SEL);
$("#clfRejThr").oninput=e=>$("#clfRejThrV").textContent=(+e.target.value).toFixed(2);
async function clfRejLoad(reset){ if(reset){CLFREJ.offset=0;clfRejGrid.reset();}
  const r=await withBusy("#clfRecReject", ()=>api(`/api/recommend_rejections?max_conf=${$("#clfRejThr").value}&offset=${CLFREJ.offset}&limit=${CLFREJ.limit}`));
  CLFREJ.total=r.total;
  if(reset && !r.items.length) clfRejGrid.msg("no low-confidence candidates — train the classifier first, or raise max conf");
  else clfRejGrid.append(r.items, it=>`~${it.cls} · ${it.conf}`);
  CLFREJ.offset+=r.items.length; $("#clfRejMore").style.display=CLFREJ.offset<r.total?"inline-block":"none";
  if(reset) $("#clfRejReport").innerHTML=`<b>${r.total}</b> instance(s) below max-conf ${(+$("#clfRejThr").value).toFixed(2)} (most-confidently-not-a-class first). Tick the ones to reject, then "Reject selected".`; }
$("#clfRecReject").onclick=()=>clfRejLoad(true);
$("#clfRejMore").onclick=()=>clfRejLoad(false);
$("#clfRejSelAll").onclick=()=>clfRejGrid.selectPage();
$("#clfRejSel").onclick=async()=>{ const iu=[...clfRejGrid.sel]; if(!iu.length){alert("tick the candidates to reject (or 'select all shown')");return;}
  const r=await post("/api/reject",{iuids:iu}); setStatus(r.stats); setClasses(r.classes); clfRejGrid.drop(iu); loadPartitions(true);
  $("#clfRejReport").innerHTML=`rejected <b>${iu.length}</b> instance(s) → background.`; };

// "interesting to classify" (active learning): unassigned instances the classifier is most UNCERTAIN about
// (entropy/margin/least-conf) — labelling these is most informative. Select + assign to a class right here.
let CLFINT={offset:0,limit:60,total:0};
const clfIntGrid = makeGrid("#clfIntGrid","#clfIntSelCount","selected", ()=>{ refreshGates(); renderInspector(); }, SEL);
async function clfIntLoad(reset){ if(reset){CLFINT.offset=0;clfIntGrid.reset();}
  const r=await withBusy("#clfRecInt", ()=>api(`/api/recommend_interesting?metric=${$("#clfIntMetric").value}&n=300&offset=${CLFINT.offset}&limit=${CLFINT.limit}`));
  CLFINT.total=r.total;
  if(reset && !r.items.length) clfIntGrid.msg("no unassigned instances to suggest");
  else clfIntGrid.append(r.items, it=>`${it.cls==='?'?'?':'~'+it.cls} · u=${it.score}`);
  CLFINT.offset+=r.items.length; $("#clfIntMore").style.display=CLFINT.offset<r.total?"inline-block":"none";
  if(reset) $("#clfIntReport").innerHTML = r.trained
    ? `<b>${r.total}</b> unassigned ranked by classifier uncertainty (most-uncertain first; ~ = predicted class). Tick + assign, or send to a class.`
    : `<b>${r.total}</b> unassigned ranked by LOWEST detection score (no classifier trained yet — train one for true uncertainty sampling).`; }
$("#clfRecInt").onclick=()=>clfIntLoad(true);
$("#clfIntMore").onclick=()=>clfIntLoad(false);
$("#clfIntSelAll").onclick=()=>clfIntGrid.selectPage();
$("#clfIntAssign").onclick=async()=>{ const cls=$("#clfIntClass").value.trim(); const iu=[...clfIntGrid.sel];
  if(!cls||!iu.length){alert("tick instances and type a class to assign them to");return;}
  const r=await post("/api/assign",{iuids:iu, cls}); setStatus(r.stats); setClasses(r.classes); clfIntGrid.drop(iu); loadPartitions(true);
  $("#clfIntReport").innerHTML=`assigned <b>${iu.length}</b> → <b>${cls}</b>. Re-run "Suggest interesting" for the next most-informative batch.`; };
$("#clfIntReject").onclick=async()=>{ const iu=[...clfIntGrid.sel]; if(!iu.length){alert("tick instances to reject");return;}
  const r=await post("/api/reject",{iuids:iu}); setStatus(r.stats); setClasses(r.classes); clfIntGrid.drop(iu); loadPartitions(true);
  $("#clfIntReport").innerHTML=`rejected <b>${iu.length}</b> → background.`; };

// ---------- Merge recommender (learn from past merges → suggest new ones) ----------
function syncMrFeats(){ if(!window._features)return;
  $("#mrFeats").innerHTML = (D=>featBoxes("mrfeat", f=>D.has(f)))(defaultFeatSet()); }
// one card per candidate GROUP. Each input instance is an individually toggleable crop (selected by default):
// "Merge selected" merges only the CHECKED subset (the ones that actually belong), leaving the rest alone.
function mergeCardHTML(c){
  const ius=c.iuids||[];
  const crops = ius.slice(0,30).map(u=>`<div class="mccrop sel" data-iuid="${u}"><img loading="lazy" class="imgld" src="${cropUrl(u)}"><div class="mclbl">${u.slice(0,6)}</div></div>`).join("");
  return `<div class="mcard" data-img="${c.image_id}">`+
    `<div class="mcbar"><b>${c.label||"P(merge)"}=${c.prob}</b> <span class="muted">img ${c.image_id} · ${ius.length} inst · click crops to (de)select</span><span class="grow"></span>`+
    `<button class="mcAcc primary">✓ Merge selected</button><button class="mcRej warn">✗ Dismiss</button></div>`+
    `<div class="mcrops">${crops}</div></div>`;
}
function renderMergeCards(sel, cands){
  const el=$(sel);
  el.innerHTML = (cands && cands.length) ? cands.map(mergeCardHTML).join("") : `<div class="muted">No candidate merges at this threshold.</div>`;
}
// click a crop → toggle its selection; "Merge selected" → merge the SELECTED subset (>=2) of that card only;
// "Dismiss" → log a negative for the group + drop the card. Each card is independent (no all-or-none).
async function onMergeCardClick(e, opts){
  const crop=e.target.closest(".mccrop");
  if(crop){ crop.classList.toggle("sel"); return; }
  const card=e.target.closest(".mcard"); if(!card) return;
  if(e.target.closest(".mcAcc")){
    const ius=[...card.querySelectorAll(".mccrop.sel")].map(c=>c.dataset.iuid);
    if(ius.length<2){ alert("select at least 2 instances to merge (click the crops to toggle)"); return; }
    const r=await post("/api/accept_merge",{iuids:ius, mode:opts.mode(), source:opts.source||"recommended"}); setStatus(r.stats); if(r.classes) setClasses(r.classes);
    card.remove(); loadPartitions(true); if(opts.afterAccept) opts.afterAccept();
  } else if(e.target.closest(".mcRej")){
    const all=[...card.querySelectorAll(".mccrop")].map(c=>c.dataset.iuid);
    if(all.length>=2) await post("/api/reject_merge",{iuids:all, source:opts.source||"recommended"});
    card.remove();
  }
}
$("#mrThr").oninput=e=>$("#mrThrV").textContent=(+e.target.value).toFixed(2);
$("#mrTrain").onclick=async()=>{
  const feats=$$(".mrfeat:checked").map(e=>e.value); $("#mrReport").innerHTML=SPIN+"training…";
  const r=await withBusy("#mrTrain", ()=>post("/api/train_merge_recommender",{features:feats, algo:$("#mrAlgo").value}));
  if(!r.ok){ $("#mrReport").innerHTML=`<span style="color:var(--warn)">${r.error||'train failed'}</span>`; return; }
  $("#mrReport").innerHTML=`trained from <b>${r.n_merge_events}</b> merge event(s) → <b>${r.n_pos}</b> positive pairs / <b>${r.n_neg}</b> negatives`
    + (r.n_rejected_neg?` (incl. <b>${r.n_rejected_neg}</b> rejected)`:"") + `. Recommended P(merge) (Youden J): <b>${r.youden}</b>.`
    + (r.undertrained?` <span style="color:var(--warn)">⚠ few merges recorded — predictions will be noisy; merge/accept a few more then re-train.</span>`:"");
  $("#mrThr").value=r.youden; $("#mrThrV").textContent=(+r.youden).toFixed(2); };
$("#mrRec").onclick=async()=>{
  const r=await withBusy("#mrRec", ()=>api(`/api/recommend_merges?thresh=${$("#mrThr").value}`));
  if(!r.trained){ $("#mrCards").innerHTML=`<div class="muted">Train the merge recommender first.</div>`; return; }
  MR.cands=r.groups; renderMergeCards("#mrCards", r.groups); };
$("#mrCards").addEventListener("click", e=>onMergeCardClick(e, {mode:()=>$("#mrMode").value}));
// In-image per-image suggestions (same trained model, scoped to the current image)
$("#iiRecThr").oninput=e=>$("#iiRecThrV").textContent=(+e.target.value).toFixed(2);
$("#iiRecBtn").onclick=async()=>{
  if(!IIMG.id){ $("#iiRecMsg").textContent="pick an image first"; return; }
  const r=await api(`/api/recommend_merges?image_id=${enc(IIMG.id)}&thresh=${$("#iiRecThr").value}`);
  if(!r.trained){ $("#iiRecMsg").innerHTML=`Train the merge recommender in the <b>Merge-rec</b> tab first.`; $("#iiRecCards").innerHTML=""; return; }
  IIREC.cands=r.groups;
  $("#iiRecMsg").textContent = r.groups.length ? `${r.groups.length} suggested merge(s) for this image at P(merge) ≥ ${(+$("#iiRecThr").value).toFixed(2)}.` : `No suggested merges for this image at P(merge) ≥ ${(+$("#iiRecThr").value).toFixed(2)}.`;
  renderMergeCards("#iiRecCards", r.groups); };
$("#iiRecCards").addEventListener("click", e=>onMergeCardClick(e, {mode:()=>$("#iiMergeMode").value, afterAccept:()=>{ if(IIMG.id) loadImage(true); }}));
// In-image overlap suggestions (model-free): groups of this image's instances chained by mask/box IoU >= threshold
$("#iiOvThr").oninput=e=>$("#iiOvThrV").textContent=(+e.target.value).toFixed(2);
async function iiOverlapFind(){
  if(!IIMG.id){ $("#iiOvMsg").textContent="pick an image first"; return; }
  const metric=$("#iiOvMetric").value, thr=(+$("#iiOvThr").value).toFixed(2);
  const r=await withBusy("#iiOvBtn", ()=>api(`/api/overlap_merges?image_id=${enc(IIMG.id)}&metric=${metric}&thresh=${thr}`));
  const groups=(r.groups||[]).map(g=>({...g, label:`max ${metric} IoU`}));
  $("#iiOvMsg").textContent = groups.length ? `${groups.length} overlap group(s) at ${metric} IoU ≥ ${thr}.` : `No overlaps at ${metric} IoU ≥ ${thr}.`;
  $("#iiOvAll").disabled = !groups.length;
  renderMergeCards("#iiOvCards", groups); }
$("#iiOvBtn").onclick=iiOverlapFind;
$("#iiOvCards").addEventListener("click", e=>onMergeCardClick(e, {mode:()=>$("#iiMergeMode").value, source:"overlap",
  afterAccept:()=>{ $("#iiOvAll").disabled=!$$("#iiOvCards .mcard").length; if(IIMG.id) loadImage(true); }}));
$("#iiOvAll").onclick=async()=>{
  const groups=$$("#iiOvCards .mcard").map(c=>[...c.querySelectorAll(".mccrop.sel")].map(x=>x.dataset.iuid)).filter(g=>g.length>=2);
  if(!groups.length) return;
  const r=await withBusy("#iiOvAll", ()=>post("/api/accept_merge_groups",{groups, mode:$("#iiMergeMode").value, source:"overlap"}));
  setStatus(r.stats); if(r.classes) setClasses(r.classes);
  $("#iiOvCards").innerHTML=""; $("#iiOvAll").disabled=true; $("#iiOvMsg").textContent=`merged ${r.n} group(s).`;
  loadPartitions(true); if(IIMG.id) loadImage(true); };

// ---------- Reference exemplar bank (foreign-object class suggestions) ----------
let REFSUG = {};                                   // iuid -> top suggested class (for "Accept top")
const refSugGrid = makeGrid("#refSugGrid","#refSugSelCount","selected", ()=>{ refreshGates(); renderInspector(); }, SEL);
async function refLoadClasses(){ const r=await api("/api/reference/classes");
  if(r.last_coco_path && !$("#refPath").value) $("#refPath").value = r.last_coco_path;  // remembered path
  if(!r.loaded){ $("#refClassSel").innerHTML=`<option>(load a bank first)</option>`; return; }
  $("#refClassSel").innerHTML = r.rows.map(x=>`<option value="${x.cls}">${x.cls} (${x.n})</option>`).join("");
  refShowExemplars(); }
async function refShowExemplars(){ const cls=$("#refClassSel").value; if(!cls) return;
  const r=await api(`/api/reference/exemplars?cls=${enc(cls)}&limit=12`);
  $("#refExemplars").innerHTML = r.items.length
    ? r.items.map(e=>{ const b=e.bbox||[]; const q=b.length===4?`&x=${b[0]}&y=${b[1]}&w=${b[2]}&h=${b[3]}`:"";
        return `<div class="cell"><img loading="lazy" class="imgld" src="/api/reference/exemplar?file_name=${enc(e.file_name)}${q}"><div class="cap">${cls}</div></div>`; }).join("")
    : `<div class="muted">no exemplars</div>`; }
$("#refClassSel").onchange=refShowExemplars;
// Reverse retrieval: given the selected reference class, rank ALL present instances by similarity — no
// partition preselect needed. Each row is the nearest partition; clicking jumps to it in the Partitions tab.
function refGoToPartition(pid){ if(!pid)return; $('nav button[data-tab="partitions"]').click();
  $("#search").value=pid; PART.query=pid; loadPartitions(true).then(()=>selectPartition(pid)); }
$("#refFind").onclick=async()=>{ const cls=$("#refClassSel").value; if(!cls){alert("load a bank + pick a reference class first");return;}
  $("#refFindReport").innerHTML=SPIN+`embedding instances (RAD-DINO) + ranking against “${escAttr(cls)}”…`; $("#refFindGrid").innerHTML=loadingBox("ranking…");
  const r=await withBusy("#refFind", ()=>post("/api/reference/find",{cls, k:24}));
  if(r.error||r.detail){ $("#refFindReport").innerHTML=`<span style="color:var(--warn)">${r.error||r.detail}</span>`; return; }
  const items=r.items||[];
  $("#refFindGrid").innerHTML = items.length
    ? items.map(m=>`<div class="cell" data-pid="${m.pid||''}" data-iuid="${m.iuid||''}"><img loading="lazy" class="imgld" src="${cropUrl(m.iuid)}"><div class="cap">${m.cls?('['+m.cls+'] '):(m.pid?escAttr(m.pid).slice(0,8)+' ':'')}${m.score}</div></div>`).join("")
    : `<div class="muted">no matching instances</div>`;
  $("#refFindReport").innerHTML=`<b>${items.length}</b> nearest partition(s) to <b>${cls}</b> across all instances (best instance per partition) — click one to open it.${r.truncated?' <span style="color:var(--mut)">(instance pool capped at 4000)</span>':''}`; };
$("#refFindGrid").onclick=e=>{ const c=e.target.closest(".cell"); if(c&&c.dataset.pid) refGoToPartition(c.dataset.pid); };
$("#refLoad").onclick=async()=>{ const p=$("#refPath").value.trim(); if(!p){alert("enter the reference coco.json path");return;}
  $("#refStatus").innerHTML=SPIN+"loading + embedding references (RAD-DINO, one-time)…";
  const r=await withBusy("#refLoad", ()=>post("/api/reference/load",{coco_path:p}));
  if(r.error||!r.ok){ $("#refStatus").innerHTML=`<span style="color:var(--warn)">${r.error||r.detail||'load failed'}</span>`; return; }
  const warn = r.exemplars_ok===false ? ` · <span style="color:var(--warn)">exemplar images NOT found under ${escAttr(r.image_root||"?")} (suggestions still work)</span>` : "";
  $("#refStatus").innerHTML=`bank: <b>${r.n_classes}</b> classes · <b>${r.exemplars}</b> exemplars · added <b>${r.added_classes}</b> to taxonomy · imgs: <code>${escAttr(r.image_root||"?")}</code>${warn}`;
  if(r.class_names) setClasses(r.class_names); refLoadClasses(); };
// Filesystem path autocomplete for the reference coco.json input: datalist on input + shell-like Tab-complete.
async function fsSuggest(val){ try{ return (await api(`/api/fs/suggest?path=${enc(val)}`)).items||[]; }catch(e){ return []; } }
function commonPrefix(a){ if(!a.length)return ""; let p=a[0]; for(const s of a) while(!s.startsWith(p)) p=p.slice(0,-1); return p; }
$("#refPath").addEventListener("input", async e=>{
  const items = await fsSuggest(e.target.value);
  $("#refPathList").innerHTML = items.map(it=>`<option value="${escAttr(it)}">`).join(""); });
$("#refPath").addEventListener("keydown", async e=>{
  if(e.key!=="Tab" || e.shiftKey) return;
  const items = await fsSuggest(e.target.value);
  if(!items.length) return;
  e.preventDefault();
  const cp = commonPrefix(items);                       // complete to the longest shared prefix, else first hit
  e.target.value = (cp && cp.length>e.target.value.length) ? cp : items[0];
  $("#refPathList").innerHTML = items.map(it=>`<option value="${escAttr(it)}">`).join(""); });
$("#refSuggest").onclick=async()=>{ if(!INST.pid){alert("select a partition in the Partitions tab first");return;}
  $("#refSugReport").innerHTML=SPIN+"embedding instances + matching references…"; refSugGrid.reset(); REFSUG={};
  const r=await withBusy("#refSuggest", ()=>post("/api/reference/suggest",{pid:INST.pid, topk:5}));
  if(r.error||r.detail){ $("#refSugReport").innerHTML=`<span style="color:var(--warn)">${r.error||r.detail}</span>`; return; }
  const items=r.items.map(it=>{ const top=(it.suggestions[0]||{}); REFSUG[it.iuid]=top.cls;
    return {iuid:it.iuid, caption:(top.cls?`~${top.cls} ${top.score}`:'?')+(it.suggestions[1]?` · ${it.suggestions[1].cls}`:'')}; });
  refSugGrid.append(items, it=>it.caption);
  $("#refSugReport").innerHTML=`<b>${items.length}</b> instance(s) — top reference class each (~cls · score, then 2nd). Tick + "Accept top" to assign each to its own class, or assign/reject in bulk.`; };
$("#refSelAll").onclick=()=>refSugGrid.selectPage();
$("#refAcceptTop").onclick=async()=>{ const iu=[...refSugGrid.sel]; if(!iu.length){alert("select instances");return;}
  const byCls={}; iu.forEach(u=>{ const c=REFSUG[u]; if(c){(byCls[c]=byCls[c]||[]).push(u);} });
  let n=0; for(const [cls,us] of Object.entries(byCls)){ const r=await post("/api/assign",{iuids:us,cls}); setStatus(r.stats); setClasses(r.classes); n+=us.length; }
  await post("/api/reference/add",{iuids:iu});            // confirmed -> grow the in-domain bank (self-improving)
  refSugGrid.drop(iu); loadPartitions(true);
  $("#refSugReport").innerHTML=`assigned <b>${n}</b> to their top reference class + added to the bank.`; };
$("#refAssign").onclick=async()=>{ const cls=$("#refAssignClass").value.trim(); const iu=[...refSugGrid.sel];
  if(!cls||!iu.length){alert("tick instances + type a class");return;}
  const r=await post("/api/assign",{iuids:iu,cls}); setStatus(r.stats); setClasses(r.classes);
  await post("/api/reference/add",{iuids:iu}); refSugGrid.drop(iu); loadPartitions(true);
  $("#refSugReport").innerHTML=`assigned <b>${iu.length}</b> → <b>${cls}</b> + added to the bank.`; };
$("#refReject").onclick=async()=>{ const iu=[...refSugGrid.sel]; if(!iu.length)return;
  const r=await post("/api/reject",{iuids:iu}); setStatus(r.stats); setClasses(r.classes); refSugGrid.drop(iu); loadPartitions(true);
  $("#refSugReport").innerHTML=`rejected <b>${iu.length}</b> → background.`; };

// ---------- Substructure (within-class self-supervised contrastive + FINCH) ----------
function syncSubFeats(){ if(!window._features)return;
  $("#subFeats").innerHTML = (D=>featBoxes("subfeat", f=>D.has(f)))(defaultFeatSet()); }
$("#subRun").onclick=async()=>{
  if(!INST.pid){ alert("select a partition/class on the Partitions tab first"); return; }
  const feats=$$(".subfeat:checked").map(e=>e.value); if(!feats.length){alert("pick at least one feature");return;}
  $("#subMsg").innerHTML=SPIN+"training contrastive encoder + FINCH… (this can take a few seconds)";
  const r=await withBusy("#subRun", ()=>post("/api/subcluster",{target:INST.pid, features:feats, dim:+$("#subDim").value, epochs:+$("#subEpochs").value, temperature:+$("#subTemp").value}));
  if(r.detail||r.error||!r.ok){ $("#subMsg").innerHTML=`<span style="color:var(--warn)">${r.detail||r.error||'failed'}</span>`; return; }
  $("#subMsg").textContent=`${r.n} instances → ${r.n_levels} FINCH levels${r.capped?" (capped sample)":""}`;
  await loadSubLevels(); await loadPartitions(true); };   // the new sub-clusters appear in the Curate rail
async function loadSubLevels(){ const r=await api("/api/subclusters"); if(!r.active)return;
  $("#subLevel").innerHTML = r.levels.map(l=>`<option value="${l.i}" ${l.i===r.level?'selected':''}>${l.n} sub-clusters</option>`).join(""); }
$("#subLevel").onchange=async()=>{ await post("/api/subcluster_level",{level:+$("#subLevel").value});
  INST.pid=null; pGrid.reset(); await loadPartitions(true); };
// Feeds the rail's Substructure group. No list markup of its own any more.
async function loadSubList(){
  try{ const r=await api("/api/subclusters");
    SUBS.active = !!r.active; SUBS.rows = (r.active && r.rows) ? r.rows : []; SUBS.target = r.target || null;
    const w=$("#subLevelWrap"); if(w) w.style.display = SUBS.active ? "" : "none";
  }catch(e){ SUBS.active=false; SUBS.rows=[]; }
}

// ---------- Loop (launch qseg-train, watch, adopt) ----------
let TR={poll:null};
function trDefaults(){ if(!$("#trCfg").value) $("#trCfg").value="experiments/curator_loop"; }   // turnkey: RAD-DINO + Tversky + balanced
$("#trLaunch").onclick=async()=>{
  if(!confirm("Launch training? This UNLOADS the curator's inference model to free the GPU — inference is unavailable until training finishes or you adopt a checkpoint.")) return;
  $("#trMsg").textContent="exporting curated COCO + launching qseg-train…";
  const r=await post("/api/train/launch",{mode:$("#trMode").value, epochs:$("#trEpochs").value||null,
    config_name:$("#trCfg").value.trim()||null, image_root:$("#trRoot").value.trim()||null,
    json_val:$("#trVal").value.trim()||null, json_test:$("#trTest").value.trim()||null,
    extra_train_json:$("#trExtra").value.trim()||null, extra_image_root:$("#trExtraRoot").value.trim()||null,
    partial:$("#trPartial").checked, class_agnostic:$("#trAgnostic").checked});
  if(r.error||r.detail||!r.ok){ $("#trMsg").innerHTML=`<span style="color:var(--warn)">${r.error||r.detail||'launch failed'}</span>`; return; }
  $("#trMsg").innerHTML=`launched pid <b>${r.pid}</b> → <code>${r.output_dir}</code><br><span class="muted">${r.cmd}</span>`;
  trStartPolling(); };
$("#trStop").onclick=async()=>{ await post("/api/train/stop",{}); setTimeout(trRefresh, 500); };
function trStartPolling(){ if(TR.poll) clearInterval(TR.poll); trRefresh(); TR.poll=setInterval(trRefresh, 4000); }
async function trRefresh(){ const s=await api("/api/train/status");
  if(!s.active){ $("#trLog").textContent="(no training job this session)"; $("#trAdopt").disabled=true; return; }
  $("#trLog").textContent=s.log_tail||"(waiting for log…)"; $("#trLog").scrollTop=$("#trLog").scrollHeight;
  const ck=s.ckpt_best||s.ckpt_final;
  $("#trCkpt").textContent=(s.running?`running (pid ${s.pid})…`:`finished (exit ${s.returncode})`)+(ck?` · checkpoint ready`:(s.running?"":" · no checkpoint produced"));
  $("#trAdopt").disabled=!ck;
  if(!s.running && TR.poll){ clearInterval(TR.poll); TR.poll=null; } }
$("#trAdopt").onclick=async()=>{ const r=await post("/api/train/adopt",{}); if(r.error){ alert(r.error); return; }
  $("#trCkpt").textContent=`adopted ${r.ckpt} — go to Config to re-infer / sample`; refreshState(); };

// ---------- Classes / Taxonomy (superclass -> concept -> part leaves) ----------
let TX_CONCEPTS=[];
async function loadClasses(){
  const r=await api("/api/taxonomy"); window._txtree=r; buildClassList();
  TX_CONCEPTS=(await api("/api/taxonomy/concepts")).concepts||[];
  const rgb=c=>`rgb(${(c||[120,120,120]).join(",")})`;
  let h="";
  for(const s of r.superclasses){
    if(!s.concepts.length) continue;
    h+=`<div class="txsc"><div class="hd"><span class="sw" style="background:${rgb(s.color)}"></span>${s.name} <span class="muted" style="font-weight:400">· ${s.n} inst</span></div>`;
    for(const c of s.concepts){
      const leaves=c.leaves.map(txLeafPill).join(" ") || `<span class="muted" style="font-size:11px">(no annotated parts yet)</span>`;
      const rules=(c.part_rules||[]).map(rl=>`${rl.if}⇒${rl.then.join("+")}`).join(", ");
      h+=`<div class="txcon"><span class="cn">${c.name}</span> <span class="muted">· ${c.n}</span>${c.mimic_family?` <span class="xw">≈${c.mimic_family}</span>`:""}`+
         (c.description?`<div class="desc">${c.description}</div>`:"")+
         (rules?`<div class="desc">rules: ${rules}</div>`:"")+`<div>${leaves}</div><div class="txprev"></div></div>`;
    }
    h+=`</div>`;
  }
  $("#txTree").innerHTML = h || `<div class="muted">No taxonomy yet — click "Seed taxonomy".</div>`;
  // temp / scratch bucket (excluded from export) — tick to merge, or promote into a concept
  const opts=`<option value="">— promote to concept —</option>`+TX_CONCEPTS.map(c=>`<option value="${c.id}">${c.name}</option>`).join("");
  $("#txTree").innerHTML += r.temp.length
    ? `<div class="txtemp"><b>Temp / scratch classes</b> <span class="muted">(usable in the tool, NOT exported)</span>`+
      r.temp.map(t=>`<div class="txtrow"><div class="row"><input type=checkbox class=mccls value="${escAttr(t.name)}"> `+
        `<b class="txname${t.n>0?' clk':''}"${t.n>0?` data-cid="${escAttr(t.id)}" data-name="${escAttr(t.name)}" title="click to preview a sample mask"`:''}>${escAttr(t.name)}</b>`+
        ` <span class="muted">${t.n} inst${t.temp?' · temp':' · ungrouped'}</span>`+
        `<span class="grow"></span><select class="txpromote" data-id="${t.id}">${opts}</select>`+
        `<button class="txtoggle" data-id="${t.id}" data-temp="${t.temp?0:1}">${t.temp?'un-temp':'mark temp'}</button></div><div class="txprev"></div></div>`).join("")+`</div>`
    : "";
  if($("#txSamples") && $("#txSamples").checked) txShowAllSamples();
}

// ---- per-class sample-mask preview (Classes tab) ----
// A leaf/temp class with instances is clickable; clicking toggles a small crop of its highest-score
// instance (mask overlaid) in the adjacent preview strip. Clicking the thumbnail cycles other samples.
function txLeafPill(l){
  const on = l.n>0;
  return `<span class="txleaf${on?' clk':''}"${on?` data-cid="${escAttr(l.id)}" data-name="${escAttr(l.name)}" title="click to preview a sample mask"`:''}>${escAttr(l.name)} <span class="n">${l.n}</span></span>`;
}
function txPrevStrip(pill){ const box = pill.closest(".txcon, .txtrow"); return box && box.querySelector(".txprev"); }
function txToggleSample(pill){
  const cid = pill.dataset.cid, strip = txPrevStrip(pill);
  if(!cid || !strip) return;
  const have = strip.querySelector(`.txcard[data-cid="${CSS.escape(cid)}"]`);
  if(have){ have.remove(); pill.classList.remove("shown"); return; }
  pill.classList.add("shown");
  const card = document.createElement("div");
  card.className = "txcard"; card.dataset.cid = cid; card.dataset.idx = "0";
  card.innerHTML = `<img loading="lazy" class="imgld"><div class="cap" title="${escAttr(pill.dataset.name||'')}">${escAttr(pill.dataset.name||'')}</div>`;
  strip.appendChild(card);
  txLoadCard(card);
}
async function txLoadCard(card){
  const r = await api(`/api/class_samples?class_id=${enc(card.dataset.cid)}&limit=24`);
  card._iuids = r.iuids || []; card.dataset.idx = "0"; txRenderCard(card);
}
function txRenderCard(card){
  const ius = card._iuids || [], i = parseInt(card.dataset.idx||"0", 10);
  const img = card.querySelector("img"), cap = card.querySelector(".cap"), base = cap.title || "";
  if(!ius.length){ img.removeAttribute("src"); img.classList.remove("imgld"); cap.textContent = base+" · (no sample)"; return; }
  img.src = `/api/crop?iuid=${enc(ius[i])}&mask=1&max_side=220`;
  cap.textContent = `${base} · ${i+1}/${ius.length}`;
}
function txCycleCard(card){
  const ius = card._iuids || []; if(ius.length < 2) return;
  card.dataset.idx = String((parseInt(card.dataset.idx||"0", 10) + 1) % ius.length);
  txRenderCard(card);
}
function txShowAllSamples(){ $$("#txTree .clk[data-cid]").forEach(p=>{ if(!p.classList.contains("shown")) txToggleSample(p); }); }
function txClearSamples(){ $$("#txTree .txcard").forEach(c=>c.remove()); $$("#txTree .clk.shown").forEach(p=>p.classList.remove("shown")); }
$("#txSamples").onchange = e=>{ if(e.target.checked) txShowAllSamples(); else txClearSamples(); };
$("#txRefresh").onclick=loadClasses;
$("#txSeed").onclick=async()=>{ $("#txMsg").innerHTML=SPIN+"seeding taxonomy…";
  const r=await withBusy("#txSeed", ()=>post("/api/taxonomy/seed",{})); setClasses((await api("/api/state")).classes);
  $("#txMsg").textContent=`taxonomy: ${r.superclasses} superclasses · ${r.concepts} concepts · ${r.leaves} leaves`; loadClasses(); };
$("#txQc").onclick=async()=>{ const r=await withBusy("#txQc", ()=>api("/api/taxonomy/release_qc"));
  $("#txReport").style.display="block";
  $("#txReport").innerHTML = r.n_violating
    ? `<b style="color:var(--warn)">${r.n_violating}/${r.n_images}</b> image(s) fail part-rules (held back from release). e.g. `+
      r.violations.slice(0,8).map(v=>`img ${v.image_id}: ${v.concept} missing ${v.missing.join("+")}`).join(" · ")
    : `<b style="color:var(--ok)">all ${r.n_images} image(s) pass</b> part-rule completeness.`; };
$("#txTree").addEventListener("change", async e=>{
  const sel=e.target.closest(".txpromote");
  if(sel){ await post("/api/taxonomy/assign_leaf",{class_id:sel.dataset.id, concept:sel.value||null}); loadClasses(); }
});
$("#txTree").addEventListener("click", async e=>{
  const img=e.target.closest(".txcard img");
  if(img){ txCycleCard(img.closest(".txcard")); return; }
  const pill=e.target.closest(".clk[data-cid]");
  if(pill){ txToggleSample(pill); return; }
  const b=e.target.closest(".txtoggle");
  if(b){ await post("/api/taxonomy/temp",{class_ids:[b.dataset.id], temp:b.dataset.temp==="1"}); loadClasses(); }
});
$("#mcMerge").onclick=async()=>{ const sources=$$(".mccls:checked").map(e=>e.value); const into=$("#mcInto").value.trim();
  if(!sources.length||!into){ alert("tick ≥1 source class and enter a target name"); return; }
  if(!confirm(`Merge ${sources.join(", ")} → "${into}"? Their instances move to "${into}" and emptied classes are removed.`)) return;
  const r=await post("/api/merge_classes",{sources, into});
  if(r.detail||r.error){ $("#mcMsg").innerHTML=`<span style="color:var(--warn)">${r.detail||r.error}</span>`; return; }
  setStatus(r.stats); setClasses(r.classes);
  $("#mcMsg").textContent=`moved ${r.moved} → ${r.into}; removed: ${(r.removed||[]).join(", ")||'none'}`;
  $("#mcInto").value=""; loadClasses(); loadPartitions(true); };

// ---------- Rejected ----------
// The rejected bin folded into Curate as a scope (REJECTED_SCOPE) — no pane, grid or verbs of its own.

// ---------- Config: inference model + all inference actions ----------
function showCkpt(){ $("#cfgCkptCur").textContent = window._modelckpt ? `current inference model: ${window._modelckpt}` : "no inference model set"; }
$("#cfgUseCkpt").onclick=async()=>{ const ckpt=$("#cfgCkpt").value.trim(); if(!ckpt){alert("enter a checkpoint path");return;}
  const r=await post("/api/train/adopt",{ckpt}); if(r.error){ $("#inferStatus").innerHTML=`<span style="color:var(--warn)">${r.error}</span>`; return; }
  $("#inferStatus").textContent=`inference model set: ${r.ckpt}`; await refreshState(); showCkpt(); };
$("#cfgAdoptLast").onclick=async()=>{ const r=await post("/api/train/adopt",{}); if(r.error){ $("#inferStatus").innerHTML=`<span style="color:var(--warn)">${r.error}</span>`; return; }
  $("#cfgCkpt").value=r.ckpt; $("#inferStatus").textContent=`adopted latest trained: ${r.ckpt}`; await refreshState(); showCkpt(); };
$("#impRun").onclick=async()=>{ const path=$("#impPath").value.trim(), source=$("#impSource").value.trim();
  if(!path||!source){alert("enter the COCO path and a source label");return;}
  $("#impMsg").innerHTML=SPIN+"importing proposals…";
  const r=await withBusy("#impRun", ()=>post("/api/import_proposals",{path, source}));
  if(r.detail){ $("#impMsg").innerHTML=`<span style="color:var(--warn)">${r.detail}</span>`; return; }
  setStatus(r.stats); await refreshState(); loadSources();
  const un=(r.unmatched_images||[]).length;
  $("#impMsg").textContent=`imported ${r.n_imported} proposal(s) from "${r.source}" over ${r.n_images} image(s)`
    +(r.raddino?" · raddino computed":"")+(un?` · ${un}+ COCO images unmatched (basename)`:"")
    +" — re-Cluster to see them grouped."; };
function inferDone(r){
  const rad = (r.raddino_error!=null) ? ` · RAD-DINO failed: ${r.raddino_error}`
            : (r.raddino_n!=null) ? ` · RAD-DINO: ${r.raddino_n} embedded` : "";
  $("#inferStatus").textContent =
  (r.error||r.detail) ? `error: ${r.error||r.detail}`
  : `done — +${r.n_new_instances??0} instances on ${r.n_new_images??0} image(s)`+((r.n_replaced)?` · ${r.n_replaced} old hidden`:"")+rad+`. Click Cluster.`;
  refreshState(); }
// detection thresholds shared by all inference actions (blank -> server uses the config default)
function inferThr(){ const s=$("#cfgScore").value.trim(), n=$("#cfgNms").value.trim();
  const o={}; if(s!=="")o.score_thresh=+s; if(n!=="")o.nms_iou=+n; return o; }
// opt-in: chain RAD-DINO feature extraction after the run (reuses the Config pool selector)
function radChain(){ const c=$("#cfgChainRaddino"); return (c&&c.checked) ? {with_raddino:true, raddino_pool:$("#cfgRaddinoPool").value} : {}; }
$("#cfgChainRaddino").addEventListener("change", e=>{ e.target.dataset.touched="1"; });   // remember a manual choice
// poll /api/progress while a long inference/embedding job runs and drive a progress bar.
// The server sends a UNIT with done/total ("bytes", "images", "instances") plus rate, ETA and how
// long since the counter last MOVED — so the bar can say "184 MB / 605 MB at 1.2 MB/s, ~6m left"
// instead of animating a stripe that means nothing, and can call out a transfer that has died.
// decimal MB/kB, matching what Hugging Face itself prints ("605M") — a MiB-based 577 MB next to
// the hub's own 605M reads as a different file
function _fmtQty(n, unit){
  if(unit!=="bytes") return Math.round(n).toLocaleString();
  if(n>=1e6) return (n/1e6).toFixed(n>=1e8?0:1)+" MB";
  return Math.max(0,Math.round(n/1e3))+" kB";
}
function _fmtDur(s){
  s=Math.max(0,Math.round(s));
  if(s<60) return s+"s";
  const m=Math.floor(s/60); return m<60 ? `${m}m ${s%60}s` : `${Math.floor(m/60)}h ${m%60}m`;
}
function _fmtRate(r, unit){
  if(!(r>0)) return "";
  if(unit==="bytes") return `${_fmtQty(r,"bytes")}/s`;
  const noun=unit||"item", plural=noun.endsWith("s")?noun:noun+"s";
  return `${r>=10?Math.round(r):r.toFixed(1)} ${plural}/s`;
}
// A job is "stalled" when the counter has not moved for a while. 20s is well past a slow chunk on
// any working link but short enough that a dead HF download is called out in the UI rather than
// leaving the user watching an animation for ten minutes.
const _STALL_S = 20;
// ...but 20s is the budget for a DOWNLOAD chunk, and other work ticks far more slowly: one SAM
// image is a minute of honest CPU. The phase says how long it may go quiet (`stall_after`), and
// once a few ticks have landed the phase's own pace says it better still — 3x the average gap
// between ticks. Calling a healthy job dead teaches the user to ignore the warning that matters.
function _stallLimit(p){
  const perTick = p.rate>0 ? 3/p.rate : 0;
  return Math.max(_STALL_S, p.stall_after||0, perTick);
}
function _progLine(p){
  const stalled=p.stalled>=_stallLimit(p);
  const unit=p.unit||"", head=escAttr(p.phase)+(p.detail?` · <span class="muted">${escAttr(p.detail)}</span>`:"");
  const bits=[];
  if(p.total>0){
    // a repo's small files download with no known size, so `done` can nose past `total` — clamp.
    // 100% is reserved for actually finished: rounding 898/900 up to it reads as a hung job.
    const raw=100*p.done/p.total, pct=p.done>=p.total ? 100 : Math.min(99.9, raw);
    bits.push(`<b>${_fmtQty(p.done,unit)}</b> / ${_fmtQty(p.total,unit)}${unit&&unit!=="bytes"?" "+escAttr(unit):""} (${pct<10||pct>99?pct.toFixed(1):Math.round(pct)}%)`);
    // rate/ETA are averages over the whole phase, so a job that has just died still shows a healthy
    // "1.1 MB/s · ~8m left". Once it is stalled the average is the misleading part — drop it.
    if(!stalled){
      const rate=_fmtRate(p.rate,unit); if(rate) bits.push(rate);
      if(p.eta>=2) bits.push(`~${_fmtDur(p.eta)} left`);    // sub-2s ETAs are noise, not information
    }
  } else if(p.elapsed>0){
    bits.push(`${_fmtDur(p.elapsed)} elapsed`);
  }
  let line=head+(bits.length?" — "+bits.join(" · "):"…");
  if(stalled){
    // only a phase with a byte TOTAL is a live transfer; a byte phase at 0/0 is the model being
    // built off disk, where network advice would be a wrong guess dressed as a diagnosis.
    const why = (unit==="bytes" && p.total>0)
      ? "the download is not moving — check the network, or set <code>HF_HUB_DISABLE_XET=1</code> and retry"
      : "no progress reported";
    line += `<br><span style="color:var(--warn)">stalled: nothing for ${_fmtDur(p.stalled)} · ${why}</span>`;
  }
  return line;
}
let _progTimer=null;
async function _pollOnce(barSel, statusSel, run){
  try{ const p=await api("/api/progress"); const bar=$(barSel), fill=$(barSel+" > span");
    // a poll still in flight when the job returned must not paint over its result (an error, most
    // visibly: a model that fails to load answers within one poll interval)
    if(run && !run.live) return;
    if(!p.active){ return; }
    const stalled = p.stalled>=_stallLimit(p);
    if(p.total>0){ bar.classList.remove("indet"); fill.style.width=Math.min(100,100*p.done/p.total).toFixed(1)+"%"; }
    // a stalled indeterminate bar stops shimmering: the animation was the thing claiming progress
    else { bar.classList.toggle("indet", !stalled); if(stalled) fill.style.width="0%"; }
    if(statusSel) $(statusSel).innerHTML=_progLine(p);
    // which embeddings the collection already has, and which one this job is filling in — the
    // static "available features" line went stale for the whole run, which is exactly when it matters
    if(Array.isArray(p.have) && $("#cfgFeatList")){
      const done=p.have.length ? p.have.map(f=>`<code>${escAttr(f)}</code>`).join(" · ") : "—";
      const tgt=p.target ? ` · <code>${escAttr(p.target)}</code> <span class="muted">(computing${
        p.unit==="images"&&p.total>0 ? ` — ${p.done}/${p.total} images`:""})</span>` : "";
      $("#cfgFeatList").innerHTML=`available features: ${done}${tgt}`;
    }
  }catch(e){}
}
async function withProgress(barSel, statusSel, fn, trigger){
  const bar=$(barSel); bar.style.display="block"; bar.classList.add("indet"); $(barSel+" > span").style.width="0%";
  const btn = typeof trigger==="string" ? $(trigger) : trigger;   // double-submit guard: disable the launch button
  if(btn){ if(btn._busy) return; btn._busy=true; btn.disabled=true; }
  const run={live:true};
  _progTimer=setInterval(()=>_pollOnce(barSel,statusSel,run), 600);
  try{ return await fn(); }
  finally{ run.live=false; clearInterval(_progTimer); _progTimer=null; bar.style.display="none"; bar.classList.remove("indet");
           refreshFeatures();          // drop any "(computing …)" marker the poll left behind, error path included
           if(btn){ btn._busy=false; btn.disabled=false; } }
}
// Client-side busy indicator for SYNCHRONOUS server ops (FINCH/sklearn/export) where
// /api/progress can't be polled (GIL-bound). Shows the top bar AFTER a delay (anti-flicker),
// disables the trigger, always restores in finally. Returns fn()'s result.
function withBusy(trigger, fn, opts={}){
  const btn = typeof trigger==="string" ? $(trigger) : trigger;
  const delay = opts.delay ?? 180;
  if(btn){ if(btn._busy) return Promise.resolve(); btn._busy=true; btn.disabled=true; }
  let shown=false;
  const t=setTimeout(()=>{ shown=true; $("#busyBar").classList.add("on"); }, delay);
  return Promise.resolve().then(fn).finally(()=>{
    clearTimeout(t); if(shown) $("#busyBar").classList.remove("on");
    if(btn){ btn._busy=false; btn.disabled=false; }
  });
}
$("#smplBtn").onclick=async()=>{ $("#inferStatus").textContent="sampling (loading model)…";
  const r=await withProgress("#inferBar","#inferStatus",()=>post("/api/sample",{n:+$("#smplN").value, smart:$("#smplSmart").checked, ...inferThr(), ...radChain()}), "#smplBtn"); inferDone(r.info||r); };
$("#inferDirBtn").onclick=async()=>{ const d=$("#inferDir").value.trim(); if(!d)return;
  $("#inferStatus").textContent="running inference on folder (loading model)…";
  inferDone(await withProgress("#inferBar","#inferStatus",()=>post("/api/infer_dir",{dir:d, limit:+$("#inferLimit").value, mode:$("#inferDirMode").value, ...inferThr(), ...radChain()}), "#inferDirBtn")); };
$("#prevBtn").onclick=async()=>{ $("#inferStatus").textContent="previewing the model on a random sample (non-destructive)…";
  const r=await withProgress("#inferBar","#inferStatus",()=>post("/api/preview_infer",{n:+$("#prevN").value, ...inferThr()}), "#prevBtn");
  if(r.detail){ $("#inferStatus").innerHTML=`<span style="color:var(--warn)">${r.detail}</span>`; return; }
  $("#inferStatus").textContent=`previewed ${r.sampled} image(s) · ${r.n_before} current → ${r.n_inst} new predicted instances (NOT ingested) — if good, Re-infer below`;
  $("#cfgPreview").innerHTML = (r.items&&r.items.length)
    ? r.items.map(it=>`<div style="border:1px solid var(--line);border-radius:6px;padding:6px;margin:4px 0">`+
        `<div class="muted" style="font-size:11px">${it.caption}</div>`+
        `<div class="ba" style="padding:4px 0">`+
        `<figure><figcaption>before (current)</figcaption><img src="${it.before}"></figure>`+
        `<figure><figcaption>after (new model)</figcaption><img src="${it.after}"></figure></div></div>`).join("")
    : `<div class="muted">no processed images to preview</div>`; };
$("#reinferBtn").onclick=async()=>{ const mode=$("#reMode").value;
  if(!confirm(`Re-infer the processed pool with the current model (mode: ${mode})? Re-runs inference; can take a while.`)) return;
  $("#inferStatus").textContent="re-inferring the processed pool…";
  const lim=$("#reLimit").value.trim();
  inferDone(await withProgress("#inferBar","#inferStatus",()=>post("/api/reinfer",{mode, limit:lim?+lim:null, ...inferThr(), ...radChain()}), "#reinferBtn")); };
$("#inferUploadBtn").onclick=async()=>{ const fs=[...$("#inferFiles").files]; if(!fs.length){ $("#inferStatus").textContent="pick image files first"; return; }
  $("#inferStatus").textContent=`uploading ${fs.length} image(s), running inference…`;
  const imgs=await Promise.all(fs.map(f=>new Promise(res=>{const r=new FileReader(); r.onload=()=>res(r.result); r.readAsDataURL(f);})));
  inferDone(await withProgress("#inferBar","#inferStatus",()=>post("/api/infer_upload",{images:imgs, ...radChain()}), "#inferUploadBtn")); };

// ---------- Release gate: image-level accept/reject of FINAL images ----------
let RELEASE={offset:0,limit:24,total:0,filter:"pending",gen:0};
function relStatsLine(s){ return `Fully categorized: <b>${s.fully_categorized}</b> · Accepted: <b style="color:var(--ok)">${s.accepted}</b> · Rejected: <b style="color:var(--warn)">${s.rejected}</b> · Pending: <b>${s.pending}</b>`; }
// The gate only withholds images when a policy says so, and that is off by default — so the one thing
// worth restating outside this tab is whether the export is currently gated, and by how many images.
const REL_POLICY_TEXT = {
  off: () => "The Release gate is <b>off</b> for this project: every curated image is exported.",
  exclude_rejected: s => `The Release gate is on: images <b>rejected</b> in <b>Ship ▸ Release</b> are left out (<b>${s.held_back}</b> right now).`,
  accepted_only: s => `The Release gate is on: only images <b>accepted</b> in <b>Ship ▸ Release</b> are exported (<b>${s.held_back}</b> held back right now).`,
};
function relSyncPolicy(s){
  const sel=$("#relPolicy"), note=$("#expGateNote");
  if(sel && s.policy) sel.value=s.policy;
  if(note) note.innerHTML = (REL_POLICY_TEXT[s.policy] || REL_POLICY_TEXT.off)(s);
}
async function expSyncGateNote(){
  // limit=0 -> the stats block without a page of thumbnails
  try { relSyncPolicy((await api("/api/release_images?filter=all&offset=0&limit=0")).stats); }
  catch(e){ console.error("[chevron] could not read the release gate", e); }
}
function relCell(it){ const st=it.status||"";
  return `<div class="rcell ${st}" data-img="${it.image_id}">`+
    `<img loading="lazy" class="imgld" src="/api/image_overlay?image_id=${enc(it.image_id)}&color_by=class&masks=1&max_side=300&_=${RELEASE.gen}" title="click to inspect in In-image">`+
    `<div class="rcap"><span>img ${String(it.image_id).slice(0,8)} · ${it.n_assigned} inst</span>`+
    `<span class="rbadge ${st}">${st||"pending"}</span></div>`+
    `<div class="rbtns"><button class="rAcc primary" title="accept this image for release">✓ Accept</button>`+
    `<button class="rRej warn" title="reject this image for release (image-level gate, not instance reject)">✗ Reject</button></div></div>`; }
async function loadRelease(reset){
  const gen = reset ? ++RELEASE.gen : RELEASE.gen;
  const off = reset ? 0 : RELEASE.offset;
  const r = await api(`/api/release_images?filter=${RELEASE.filter}&offset=${off}&limit=${RELEASE.limit}`);
  if(gen!==RELEASE.gen) return;                       // superseded by a newer reload -> drop (no double-append)
  RELEASE.total=r.total; RELEASE.offset=off+r.items.length;
  $("#relStats").innerHTML = relStatsLine(r.stats); relSyncPolicy(r.stats);
  const html = r.items.length ? r.items.map(relCell).join("")
             : (reset?`<div class="muted">no ${RELEASE.filter==='all'?'final':RELEASE.filter} images</div>`:"");
  if(reset) $("#relGrid").innerHTML=html; else $("#relGrid").insertAdjacentHTML("beforeend", html);
  $("#relMore").style.display = RELEASE.offset<r.total?"inline-block":"none";
}
async function setRelease(ids, status){ const r=await post("/api/release_set",{image_ids:ids, status});
  if(r&&r.stats){ $("#relStats").innerHTML = relStatsLine(r.stats); relSyncPolicy(r.stats); } return r; }
function relApplyStatus(cell, st){                     // reflect the decision: drop from a filtered view, else re-badge
  if(RELEASE.filter!=="all" && st!==RELEASE.filter){ cell.remove(); }
  else { cell.className=`rcell ${st}`; const b=cell.querySelector(".rbadge"); if(b){ b.className=`rbadge ${st}`; b.textContent=st; } }
}
function openInImage(img){
  $('nav button[data-tab="inimage"]').click();
  if(![...$("#imgSelect").options].some(o=>o.value===String(img))) $("#imgSelect").insertAdjacentHTML("afterbegin",`<option value="${img}">${img}</option>`);
  $("#imgSelect").value=String(img); loadImage(true);
}
$("#relPolicy").onchange=async e=>{ const r=await post("/api/release_policy",{policy:e.target.value});
  if(r&&r.stats){ $("#relStats").innerHTML = relStatsLine(r.stats); relSyncPolicy(r.stats); } };
$("#relFilter").onchange=e=>{ RELEASE.filter=e.target.value; loadRelease(true); };
$("#relReload").onclick=()=>loadRelease(true);
$("#relMore").onclick=()=>loadRelease(false);
$("#relGrid").onclick=async e=>{
  const cell=e.target.closest(".rcell"); if(!cell) return; const iid=cell.dataset.img;
  if(e.target.classList.contains("rAcc")){ await setRelease([iid],"accepted"); relApplyStatus(cell,"accepted"); }
  else if(e.target.classList.contains("rRej")){ await setRelease([iid],"rejected"); relApplyStatus(cell,"rejected"); }
  else if(e.target.tagName==="IMG"){ openInImage(iid); }
};

// ---------- global mask shortcut ('m') + crop/in-context view toggles ----------
function toggleView(){ VIEW = VIEW==="crop"?"context":"crop"; syncViewButtons(); refreshVisibleCrops(); }
document.addEventListener("keydown", e=>{
  const tn=e.target.tagName;
  if(tn==="INPUT"||tn==="TEXTAREA"||tn==="SELECT"||e.target.isContentEditable) return;
  if(e.key==="e" && !e.metaKey && !e.ctrlKey && !e.altKey && document.querySelector(".tab.active")?.id==="tab-refine"
     && !$("#maskEditor").classList.contains("on") && RF.cur){ e.preventDefault(); openMaskEditor(RF.cur); }
  else if(e.key==="e" && !e.metaKey && !e.ctrlKey && !e.altKey && SEL.size===1 && !$("#maskEditor").classList.contains("on")
     && document.querySelector(".tab.active")?.id!=="tab-refine"){ e.preventDefault(); openMaskEditor([...SEL][0]); }
  else if(e.key==="a" && !e.metaKey && !e.ctrlKey && !e.altKey && SEL.size && document.querySelector(".tab.active")?.id==="tab-partitions"
     && !$("#maskEditor").classList.contains("on")){ e.preventDefault(); acceptMasks([...SEL]); }
  else if(e.key==="m"){ MASKS=!MASKS; const cb=$("#ovMasks"); if(cb) cb.checked=MASKS; refreshVisibleCrops(); }
  else if(e.key==="c"){ toggleView(); }          // c = toggle crop <-> context view
});
$$(".viewToggle").forEach(b=> b.onclick=toggleView);
syncViewButtons();

// ---------- register the button gates (disabled when there's nothing to act on) ----------
// Partitions: selection-acting buttons need ≥1 selected (merge ≥2); partition-scoped actions need a partition.
[["#rejectBtn",1],["#acceptMaskBtn",1],["#unassignBtn",1],["#toRefineBtn",1],["#toInimgBtn",1],["#selNone",1],["#mergeBtn",2]]
  .forEach(([s,m])=>gate(s, ()=>pGrid.sel.size>=m));
gate("#assignBtn", ()=>pGrid.sel.size>=1 && !!$("#classInput").value.trim());
// Assign needs a class name as much as it needs a target: an enabled Assign that silently no-ops
// because #classInput is empty is the same lie as verbs over an empty selection.
const hasCls = ()=>!!$("#classInput").value.trim();
gate("#assignAllBtn", ()=>INST.pid!=null && hasCls()); gate("#rejectAllBtn", ()=>INST.pid!=null); gate("#acceptAllMasksBtn", ()=>INST.pid!=null && !isRejectedScope()); gate("#dupPrev", ()=>INST.pid!=null);
$("#classInput").addEventListener("input", refreshGates);
// In-image: assign/reject/refine/deselect need ≥1, merge + its live preview need ≥2.
// In-image now drives the shared inspector; its own verbs are gone. #iiMergePrev stays a toggle.
// Classifier (preview / reject-suggest / interesting grids): bulk actions need ≥1 selected.
gate("#clfAssignSel", ()=>clfGrid.sel.size>=1); gate("#clfReject", ()=>clfGrid.sel.size>=1);
gate("#clfRejSel", ()=>clfRejGrid.sel.size>=1);
gate("#clfIntAssign", ()=>clfIntGrid.sel.size>=1); gate("#clfIntReject", ()=>clfIntGrid.sel.size>=1);
// Reference suggestions: per-instance acceptance needs ≥1 selected; suggest needs a partition.
[["#refAcceptTop",1],["#refAssign",1],["#refReject",1]].forEach(([s,m])=>gate(s, ()=>refSugGrid.sel.size>=m));
gate("#refSuggest", ()=>INST.pid!=null);
// Substructure: clear needs a selection; running needs a partition/class target. (Assign/Reject/Unassign
// intentionally NOT gated — they fall back to the whole sub-cluster when nothing is ticked.)
gate("#subRun", ()=>INST.pid!=null);   // substructure needs a target scope selected in Curate
// Rejected bin: unreject + clear need a selection.
// Undo/redo: disabled when the server's undo/redo stack is empty (depths come back in stats.undo/redo).
gate("#undoBtn", ()=>(window._undoN||0)>0); gate("#redoBtn", ()=>(window._redoN||0)>0);
refreshGates();
renderInspector();          // start in the empty state rather than whatever the markup defaults to
// The boot route goes LAST: it must run after every pane's state has been declared. Dispatched up
// next to the router, deep links to Map/Release/In-image hit the temporal dead zone of `MAP`,
// `RELEASE`, `iiGrid` and friends, the on-show hook threw, and the pane never loaded.
routeFromHash();

refreshState();
