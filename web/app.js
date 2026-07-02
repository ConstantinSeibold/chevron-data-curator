// qseg curator — custom frontend logic. Loads only windowed JSON + lazy per-instance crops, so
// responsiveness is independent of instance/partition count.
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const api = async (u, o) => (await fetch(u, o)).json();
const post = (u, b) => api(u, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(b||{})});
const enc = encodeURIComponent;

function setStatus(s){ if(!s) return; $("#status").textContent =
  `${s.n_instances} inst · ${s.n_assigned} assigned · ${s.n_unassigned} unassigned · ${s.n_background} rejected · ${s.n_classes} classes`;
  window._undoN = s.undo|0; window._redoN = s.redo|0; refreshGates(); }   // gate undo/redo on the server stack depths
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
  return `<div class="cell" data-iuid="${it.iuid}" data-img="${it.image_id??''}">`+
    `<img loading="lazy" class="imgld">`+            // src set by the batched crop loader (observeCrops on append); imgld = shimmer until loaded
    `<div class="cap" title="${cap}">${cap}</div></div>`;
}
function makeGrid(gridSel, countSel, noun="selected", onChange){
  const el = $(gridSel), sel = new Set();
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
    reset(){ el.innerHTML=""; sel.clear(); anchor=null; upd(); },
    append(items, capFn){ el.insertAdjacentHTML("beforeend", items.map(it=>cell(it, capFn?capFn(it):it.caption)).join("")); observeCrops(el); },
    drop(iuids){ const s=new Set(iuids); el.querySelectorAll(".cell").forEach(c=>{ if(s.has(c.dataset.iuid)) c.remove(); }); iuids.forEach(u=>sel.delete(u)); upd(); },
    selectPage(){ el.querySelectorAll(".cell").forEach(c=>{ sel.add(c.dataset.iuid); c.classList.add("sel"); }); upd(); },
    clearSel(){ sel.clear(); el.querySelectorAll(".cell").forEach(c=>c.classList.remove("sel")); upd(); },
    firstSelImg(){ for(const c of el.querySelectorAll(".cell.sel")) return c.dataset.img; return null; },
    msg(h){ el.innerHTML=`<div class="muted">${h}</div>`; }
  };
}

// ---------- tabs ----------
$("#nav").onclick = (e)=>{ const b=e.target.closest("button[data-tab]"); if(!b) return;
  $$("nav button").forEach(x=>x.classList.toggle("active", x===b));
  $$(".tab").forEach(t=>t.classList.toggle("active", t.id===`tab-${b.dataset.tab}`));
  if(b.dataset.tab==="classifier") syncClfFeats();
  if(b.dataset.tab==="mergerec") syncMrFeats();
  if(b.dataset.tab==="substructure"){ syncSubFeats(); $("#subTarget").textContent=INST.pid||"none"; loadSubLevels(); loadSubList(); }
  if(b.dataset.tab==="classes") loadClasses();
  if(b.dataset.tab==="reference") refLoadClasses();
  if(b.dataset.tab==="loop"){ trDefaults(); trRefresh(); }
  if(b.dataset.tab==="config") showCkpt();
  if(b.dataset.tab==="inimage" && !$("#imgSelect").options.length) populateImages("");
  if(b.dataset.tab==="stats") loadStats();
  if(b.dataset.tab==="activity") loadActivity();
  if(b.dataset.tab==="refine"){ loadClassRules(); if($("#rfIuid").value.trim()) rfLoadPeers($("#rfIuid").value.trim()); }
  if(b.dataset.tab==="release") loadRelease(true);
  if(b.dataset.tab==="map") mapOnShow();
};

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
async function refreshState(){
  const st = await api("/api/state");
  setStatus(st.stats); setClasses(st.classes); loadTxLeaves();
  window._modelcfg = st.model_config; window._modelckpt = st.model_ckpt;
  $("#levelSel").innerHTML = st.levels.map(l=>`<option value="${l.i}" ${l.i===st.level?'selected':''}>L${l.i} (${l.n})</option>`).join("");
  window._featureNan = st.feature_nan || [];       // features with NaN/inf -> non-selectable in the classifier
  refreshFeatures(st.features);                    // builds #feats + all selectors + the Config readout
  loadIngests(); loadSources();
  if(st.clustered) loadPartitions(true);
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
let SRC = { all:[], active:null };                 // active: null = all sources
async function loadSources(){
  let r; try{ r = await api("/api/sources"); }catch(e){ return; }
  SRC.all=(r.sources||[]).map(s=>s.source); SRC.active=r.active;
  const bar=$("#srcFacet");
  if(SRC.all.length<=1){ bar.style.display="none"; return; }          // facet only meaningful with >1 source
  bar.style.display="";
  const on=s=> SRC.active===null || SRC.active.includes(s);
  bar.innerHTML = `source: `+(r.sources||[]).map(s=>
    `<a class="srcChip${on(s.source)?" on":""}" data-src="${escAttr(s.source)}">${s.source}<span class="muted"> ${s.n}</span></a>`).join(" ")
    + (SRC.active!==null?` <a class="srcChip" data-src="__all__">all</a>`:"");
}
$("#srcFacet").onclick=async e=>{ const a=e.target.closest("[data-src]"); if(!a)return;
  let active = SRC.active===null ? SRC.all.slice() : SRC.active.slice();
  if(a.dataset.src==="__all__"){ active=null; }
  else { const s=a.dataset.src; active = active.includes(s) ? active.filter(x=>x!==s) : active.concat([s]);
    if(active.length===0 || active.length===SRC.all.length) active=null; }   // none / all -> clear facet
  const r=await post("/api/source_filter",{sources:active}); setStatus(r.stats);
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
function featBoxes(cls, isDefault){
  const nan = new Set(window._featureNan||[]);
  return (window._features||[]).map(f=>{ const bad=nan.has(f);
    const checked = (!bad && isDefault(f)) ? 'checked' : '';
    return `<label title="${bad?'contains NaN/inf — not usable':''}" style="${bad?'opacity:.45':''}">`
      + `<input type=checkbox class=${cls} value="${f}" ${checked} ${bad?'disabled':''}>${f}${bad?' ⚠NaN':''}</label>`;
  }).join("");
}
// single source of truth for the feature selectors: rebuild #feats (cluster) + classifier/sub/merge-rec
// from window._features, and show what's available (so computed embeddings like raddino are visible).
function refreshFeatures(list){
  if(list) window._features = list;
  const fs = window._features || [];
  $("#feats").innerHTML = featBoxes("feat", f=>f=='decoder');
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
$("#cfgRaddino").onclick=async()=>{
  $("#cfgRaddinoMsg").textContent="extracting RAD-DINO embeddings (one RAD-DINO pass per image, GPU)…";
  const r=await withProgress("#raddinoBar","#cfgRaddinoMsg",()=>post("/api/compute_raddino",{force:$("#cfgRaddinoForce").checked, pool:$("#cfgRaddinoPool").value}), "#cfgRaddino");
  if(r.error||r.detail){ $("#cfgRaddinoMsg").innerHTML=`<span style="color:var(--warn)">${r.error||r.detail}</span>`; return; }
  refreshFeatures(r.available);
  $("#cfgRaddinoMsg").innerHTML=`RAD-DINO ready for <b>${r.n||'all'}</b> instances — <code>raddino</code> is now selectable everywhere.`; };
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
  alert(`Exported ${kind} COCO → ${r.path}`+(r.partial?`\n\n${s.n_assigned} positives · ${s.n_unassigned} ignore (unreviewed) · ${s.n_background} rejected→background`:"")); };
async function doUndo(which){ const r=await post(`/api/${which}`,{}); setStatus(r.stats); setClasses(r.classes); loadPartitions(true); if(INST.pid) selectPartition(INST.pid); }
$("#undoBtn").onclick=()=>doUndo("undo"); $("#redoBtn").onclick=()=>doUndo("redo");

// ---------- Partitions ----------
let PART={offset:0,limit:100,total:0,query:"",kind:"all",predFilter:null}, INST={pid:null,offset:0,limit:60,total:0};
let PART_PRED={}, PART_PRED_META=null;               // selected partition: iuid->{label,pred,score,assigned} + {n_total,truncated}
const pGrid = makeGrid("#pgrid", "#pSelCount", "selected", refreshGates);
let _plGen=0;                                         // render generation: a reset starts a new one
async function loadPartitions(reset){
  const gen = reset ? ++_plGen : _plGen;              // a 'load more' rides the current generation
  const off = reset ? 0 : PART.offset;
  const r=await api(`/api/partitions?offset=${off}&limit=${PART.limit}&query=${enc(PART.query)}&kind=${PART.kind}`);
  if(gen!==_plGen) return;                            // a newer reset superseded this fetch -> drop it (no double-append)
  PART.total=r.total; PART.offset=off+r.rows.length;
  $("#pcount").textContent=`${r.total} partitions${r.total>PART.limit?` (showing ${Math.min(PART.offset,r.total)})`:''}`;
  const html = r.rows.map(p=>
    `<div class="prow${INST.pid===p.pid?' sel':''}" data-pid="${p.pid}"><span>${p.pid}${p.cls?` <span class=cls>[${p.cls}]</span>`:''}</span><span class="sz">${p.size} · ${p.score??''}</span></div>`).join("");
  if(reset) $("#plist").innerHTML=html;              // REPLACE on reset (atomic) instead of clear-then-async-append
  else $("#plist").insertAdjacentHTML("beforeend", html);
  $("#pmore").style.display = PART.offset<r.total?"inline-block":"none";
}
async function selectPartition(pid){
  INST.pid=pid; INST.offset=0; clearPredFilter(); pGrid.reset();
  $$(".prow").forEach(e=>e.classList.toggle("sel", e.dataset.pid===pid));
  loadPartitionSuggestion();                          // 1-NN "most likely class" hint + per-crop markers (fire-and-forget)
  await loadInstances(true);
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
  const f=PART.predFilter;                            // a class/reject subset filter -> server-side predicted filter
  const predQ = f ? `&pred=${enc(f.label)}&gate_mult=${parseFloat($("#psugGate").value||"1")}` : "";
  const r=await api(`/api/instances?pid=${enc(INST.pid)}&offset=${INST.offset}&limit=${INST.limit}${predQ}`);
  INST.total=r.total;
  if(reset && !r.items.length){ pGrid.msg(f?`(no crops predicted ${escAttr(f.label)})`:"(empty — assign/reject emptied this partition)"); }
  else pGrid.append(r.items);
  INST.offset+=r.items.length;
  $("#imore").style.display = INST.offset<r.total?"inline-block":"none";
  applyPreds("#pgrid", PART_PRED);                    // mark the (newly paged) crops
}
async function afterMut(resp, dropped, grid){ setStatus(resp.stats); setClasses(resp.classes); if(dropped) grid.drop(dropped); loadPartitions(true); }
$("#search").oninput=e=>{ PART.query=e.target.value; clearTimeout(window._st); window._st=setTimeout(()=>loadPartitions(true),200); };
$("#plKind").onchange=e=>{ PART.kind=e.target.value; loadPartitions(true); };   // scope: all / partitions-only / classes-only
$("#pmore").onclick=()=>loadPartitions(false);
$("#imore").onclick=()=>loadInstances(false);
$("#plist").onclick=e=>{ const r=e.target.closest(".prow"); if(r) selectPartition(r.dataset.pid); };
$("#selAll").onclick=()=>pGrid.selectPage(); $("#selNone").onclick=()=>pGrid.clearSel();
$("#assignBtn").onclick=async()=>{ const cls=$("#classInput").value.trim(); if(!cls||!pGrid.sel.size)return; const iu=[...pGrid.sel]; afterMut(await post("/api/assign",{iuids:iu,cls}),iu,pGrid); };
$("#assignAllBtn").onclick=async()=>{ const cls=$("#classInput").value.trim(); if(!cls||!INST.pid)return;
  const all=await api(`/api/instances?pid=${enc(INST.pid)}&offset=0&limit=1000000`); const iu=all.items.map(i=>i.iuid);
  afterMut(await post("/api/assign",{iuids:iu,cls}),iu,pGrid); };
$("#rejectBtn").onclick=async()=>{ if(!pGrid.sel.size)return; const iu=[...pGrid.sel]; afterMut(await post("/api/reject",{iuids:iu}),iu,pGrid); };
$("#rejectAllBtn").onclick=async()=>{ if(!INST.pid)return;          // reject the WHOLE partition (server-side, by pid)
  if(!confirm(`Reject the ENTIRE partition ${INST.pid}? Every instance in it goes to background (undoable).`))return;
  const r=await post("/api/reject_partition",{pid:INST.pid});
  if(r.detail){alert(r.detail);return;}
  setStatus(r.stats); pGrid.reset(); $("#psugText").textContent=""; INST.pid=null; refreshGates(); loadPartitions(true); };
$("#unassignBtn").onclick=async()=>{ if(!pGrid.sel.size)return; const iu=[...pGrid.sel]; afterMut(await post("/api/unassign",{iuids:iu}),iu,pGrid); };
$("#mergeBtn").onclick=async()=>{ if(pGrid.sel.size<2)return; const iu=[...pGrid.sel];
  const r=await post("/api/merge",{iuids:iu});
  if(!r.n_groups){ alert("nothing merged — merge only combines instances from the SAME image (the selection spans different images, or no image had ≥2 selected)."); return; }
  setStatus(r.stats); selectPartition(INST.pid); loadPartitions(true); };
$("#toRefineBtn").onclick=()=>{ const u=[...pGrid.sel][0]; if(!u)return; $("#rfIuid").value=u; $('nav button[data-tab="refine"]').click(); rfDoPreview(); };
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
const iiGrid = makeGrid("#iigrid","#iiSelCount","selected", iiOnSelChange);
function iiOnSelChange(){ refreshGates(); scheduleMergePreview(); }   // In-image gating is in the central gate registry
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
function imgOpt(it, mode){
  if(mode==="count" || it.n_uncat==null) return `<option value="${it.image_id}">${it.image_id} (${it.n_inst??it.n})</option>`;
  if(it.done) return `<option value="${it.image_id}">${it.image_id} · ✓ ready</option>`;
  const cls = it.top_class ? ` · ${it.top_class}${it.n_pred_classes>1?"+":""}` : "";
  return `<option value="${it.image_id}">${it.image_id} · ${it.work_est} left${cls}</option>`;
}
async function populateImages(query=""){            // windowed image picker: most-populated, or ranked by work left
  const mode = $("#imgSort") ? $("#imgSort").value : "count";
  const keep = $("#imgSelect").value;               // preserve the open image across a re-rank (e.g. gate move)
  let r, m=mode;
  if(mode==="count"){ r=await api(`/api/images?query=${enc(query)}&limit=200`); }
  else {
    const gate=parseFloat($("#imgPredGate").value||"1");
    const div=$("#imgVariety")?parseFloat($("#imgVariety").value||"0"):0;   // class-variety re-rank strength
    r=await api(`/api/image_ranking?order=${mode}&gate_mult=${gate}&diversity=${div}&query=${enc(query)}&limit=200`);
    if(r.fallback) m="count";                        // no labels yet -> server returned count-style items
  }
  $("#imgSelect").innerHTML = r.items.map(it=>imgOpt(it, m)).join("");
  if(keep && [...$("#imgSelect").options].some(o=>o.value===keep)) $("#imgSelect").value=keep;
  updateImgNav();
  const note=$("#imgSortNote");
  if(note) note.textContent = (mode!=="count" && r.fallback) ? "(label some instances to rank by work left)"
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
  const id=$("#imgSelect").value; if(!id)return; IIMG.id=id; reloadOverlay(); updateImgNav();
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
async function iiAfter(resp,dropped){ setStatus(resp.stats); setClasses(resp.classes); iiGrid.drop(dropped); reloadOverlay(); loadPartitions(true); }
$("#iiAssign").onclick=async()=>{ const cls=$("#iiClass").value.trim(); if(!cls||!iiGrid.sel.size)return; const iu=[...iiGrid.sel]; iiAfter(await post("/api/assign",{iuids:iu,cls}),iu); };
$("#iiReject").onclick=async()=>{ if(!iiGrid.sel.size)return; const iu=[...iiGrid.sel]; iiAfter(await post("/api/reject",{iuids:iu}),iu); };
$("#iiToRefine").onclick=()=>{ const u=[...iiGrid.sel][0]; if(!u){alert("select an instance");return;}
  $("#rfIuid").value=u; $('nav button[data-tab="refine"]').click(); rfDoPreview(); };
// Toggle: ON = live-preview the merge of the current selection (auto-updates as the selection/mode changes);
// click again to turn it OFF and hide the preview.
$("#iiMergePrev").onclick=()=>{ IIMERGE_PREV=!IIMERGE_PREV;
  $("#iiMergePrev").classList.toggle("primary", IIMERGE_PREV);
  $("#iiMergePrev").textContent = IIMERGE_PREV ? "Preview merge: ON" : "Preview merge";
  if(IIMERGE_PREV) refreshMergePreview(); else $("#iiPrevWrap").style.display="none"; };
$("#iiMergeMode").onchange=()=>{ if(IIMERGE_PREV) refreshMergePreview(); };
$("#iiDeselect").onclick=()=>iiGrid.clearSel();     // clear the current instance selection (→ refreshGates via onChange)
$("#iiMerge").onclick=async()=>{ if(iiGrid.sel.size<2)return; const iu=[...iiGrid.sel];
  await post("/api/merge",{iuids:iu, mode:$("#iiMergeMode").value}); $("#iiPrevWrap").style.display="none"; loadImage(true); loadPartitions(true); };

// ---------- Refine ----------
let RF_CHAIN=[];
// Left list = the partition PEERS of the currently previewed instance (not a global search). Re-render only
// when the partition changes, so clicking between peers of one partition doesn't reshuffle the grid.
let RF_PEERS_IUID="", RF_PEERS_PID=null;
async function rfLoadPeers(iuid){
  iuid=(iuid||"").trim(); if(!iuid) return;
  const r=await api(`/api/instance_peers?iuid=${enc(iuid)}&limit=200`);
  if(r.detail) return;
  if(r.pid===null || r.pid!==RF_PEERS_PID){
    RF_PEERS_PID=r.pid;
    $("#rfFindCount").textContent = r.pid!==null
      ? `partition ${r.pid} — ${r.total} sample${r.total===1?"":"s"} (peers of the previewed instance)`
      : `1 sample — this instance has no partition (rejected / merged / not clustered)`;
    $("#rfFind").innerHTML = r.items.length ? r.items.map(it=>cell(it,it.caption)).join("") : `<div class="muted">no peers</div>`;
    observeCrops($("#rfFind"));
  }
  $$("#rfFind .cell").forEach(x=>x.classList.toggle("sel", x.dataset.iuid===iuid));
}
// per-op tunable parameters (rendered next to the op picker; captured into the op's kw on "+ add op")
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
  contrast: "local contrast (CLAHE) on the image the LATER ops see — add it FIRST, then threshold/vessel/sam. Higher clip = stronger. The preview shows the enhanced image.",
  vessel_extend: "GROWS the tube along vesselness — tune per image: raise seed/grow and lower gap if it over-extends; raise width for thick tubes.",
  line_centerline: "REDUCES a line to the single shortest path between its two tips — deterministic, cannot branch/mesh. ONE class-wide knob: mask-trust (higher = stay on mask / bridge less; lower = bridge gaps via image lines). width 0 = auto from mask. curve-stiff>0 needs `pip install agd` (won't jump onto crossing tubes), else plain.",
  sam: "boundary-free refine: result REPLACES the mask (can shrink+grow); SAM's best of several proposals is taken. keep∪=1 unions with the original (never shrinks); if it still echoes the input, set mask-prior=0. Compact parts > thin shafts.",
  threshold: "intensity threshold. method: Otsu (auto split) · manual (val 0–255) · GHT (Barron — ν reg, ω bias). keep: auto picks the side matching the mask interior, or force ≥thr (bright) / <thr (dark). region bounds the RESULT: in-mask CARVES (never grows) · in-bbox fills the box · any = whole image. Add a `contrast` op first to sharpen the split.",
};
function renderRfParams(){
  const op=$("#rfOp").value, ps=OP_PARAMS[op]||[];
  $("#rfParams").innerHTML = ps.map(p=>{
    const w = p.when?` data-when="${p.when}"`:"";
    if(p.type==="select")
      return `<label${w}>${p.label} <select class=rfp data-k="${p.k}" data-type="select">`+
             p.opts.map(([v,t])=>`<option value="${v}"${v===p.def?" selected":""}>${t}</option>`).join("")+`</select></label>`;
    return `<label${w}>${p.label} <input class=rfp data-k="${p.k}" type=number value="${p.def}" step="${p.step}" min="${p.min}" max="${p.max}"></label>`;
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
$("#rfOp").onchange=renderRfParams;
$("#rfParams").addEventListener("change", e=>{ if(e.target.classList.contains("rfp") && e.target.dataset.k==="method") rfToggleWhen(); });
const activeOps = ()=> RF_CHAIN.filter(o=>o.on!==false);          // enabled ops only (toggled-off are skipped)
function renderChain(){
  if(!RF_CHAIN.length){ $("#rfChain").innerHTML="chain: (empty)"; return; }
  $("#rfChain").innerHTML = `chain <span class=muted style="padding:0;font-size:10px">(click an op to disable · × to remove)</span>: ` +
    RF_CHAIN.map((o,i)=>{
      const kv=Object.entries(o.kw||{}).map(([k,v])=>`${k}=${v}`).join(" ");
      return `<span class="chip${o.on===false?" off":""}" data-i="${i}" title="${o.on===false?'enable':'disable'}">`+
             `${o.name}${kv?` <small>(${kv})</small>`:""}<span class="x" data-rm="${i}" title="remove">×</span></span>`;
    }).join(""); }
function rfRepreviewIfShown(){ if($("#rfIuid").value.trim() && $("#rfBA figure")) rfDoPreview(); }
$("#rfChain").onclick=e=>{
  const rm=e.target.closest(".x"); if(rm){ RF_CHAIN.splice(+rm.dataset.rm,1); renderChain(); rfRepreviewIfShown(); return; }
  const chip=e.target.closest(".chip"); if(chip){ const i=+chip.dataset.i; RF_CHAIN[i].on=(RF_CHAIN[i].on===false); renderChain(); rfRepreviewIfShown(); } };
$("#rfAdd").onclick=()=>{ const kw=readRfKw(); if($("#rfOp").value==="sam") kw.model=$("#rfSamModel").value;   // SAM vs MedSAM recipe
  RF_CHAIN.push({name:$("#rfOp").value, kw, on:true}); renderChain(); };
$("#rfClear").onclick=()=>{ RF_CHAIN=[]; renderChain(); };
async function rfDoPreview(){ const iuid=$("#rfIuid").value.trim(); if(!iuid)return;
  if(iuid!==RF_PEERS_IUID){ RF_PEERS_IUID=iuid; rfLoadPeers(iuid); }   // refresh the left peer list when the instance changes
  const ops=activeOps();
  const r=await post("/api/refine_preview",{iuid, ops, mask:MASKS?1:0, context:VIEW==='context'?1:0});
  if(r.detail){ $("#rfBA").innerHTML=`<div class="muted" style="color:var(--warn)">${r.detail}</div>`; return; }
  $("#rfBA").innerHTML=`<figure><figcaption>before</figcaption><img src="${r.before}"></figure>`+
    `<figure><figcaption>after (${ops.map(o=>o.name).join("→")||'no ops'}) — `+
    `<b style="color:#e8c000">▦ same</b> · <b style="color:#2dd24d">▦ added</b> · <b style="color:#eb4a3d">▦ removed</b></figcaption>`+
    `<img src="${r.after}"></figure>`;
  if(ops.some(o=>o.name==="sam")){ const f=await rfSamPointsFigure(); if(f) $("#rfBA").insertAdjacentHTML("beforeend", f); } }
$("#rfPreview").onclick=rfDoPreview;
// SAM prompt visualisation: where the +/- points and box come from (green=positive on the skeleton,
// red=negative on the ring, yellow=box). Pure geometry → works even before a checkpoint is downloaded.
async function rfSamPointsFigure(){
  const iuid=$("#rfIuid").value.trim(); if(!iuid) return "";
  if($("#rfSamModel") && $("#rfSamModel").value==="medsam")
    return `<div class="muted">MedSAM uses the bounding-box prompt only — no sample points.</div>`;
  let kw = (activeOps().find(o=>o.name==="sam")||{}).kw;  // use the chained sam op's kw, else the live params
  if(!kw && $("#rfOp").value==="sam") kw = readRfKw();
  kw = kw || {};
  const r=await post("/api/sam_prompt_preview",{iuid, ops:activeOps(), n_pos:kw.n_pos??1, n_neg:kw.n_neg??0, margin:kw.margin??24});
  if(r.detail) return "";
  return `<figure><figcaption>SAM prompts — <b style="color:#2dd24d">●</b> ${r.n_pos} pos (interior) · <b style="color:#eb4a3d">●</b> ${r.n_neg} neg (beyond ${kw.margin??24}px gap) · <b style="color:#ffd000">▭</b> box · rim left free</figcaption><img src="${r.img}"></figure>`; }
$("#rfSamPts").onclick=async()=>{ const f=await rfSamPointsFigure();
  $("#rfBA").innerHTML = f || `<div class="muted">pick an instance first</div>`; };
$("#rfApply").onclick=async()=>{ const iuid=$("#rfIuid").value.trim(); if(!iuid)return;
  const r=await post("/api/apply_refine",{iuid,ops:activeOps()});
  if(r.detail){ alert(r.detail); return; }
  setStatus(r.stats); if(INST.pid)selectPartition(INST.pid); $("#rfHint").textContent=`refined ${iuid.slice(0,6)} ✓`;
  RF_PEERS_PID=null; rfLoadPeers(iuid); };       // refresh peer captions/membership after the edit
// Stage-1 auto-refine: search candidate chains, show the chosen one's before/after, and LOAD it into the
// editable chain (so the human can tweak then Apply — the Apply records meta.rule_ops, a Stage-2 demo).
$("#rfAuto").onclick=async()=>{ const iuid=$("#rfIuid").value.trim(); if(!iuid){alert("load an instance first");return;}
  $("#rfHint").innerHTML=SPIN+"auto-refine: searching candidate chains…";
  const r=await withBusy("#rfAuto", ()=>post("/api/auto_refine_preview",{iuid, kind:"auto"}));
  if(r.detail){ $("#rfBA").innerHTML=`<div class="muted" style="color:var(--warn)">${r.detail}</div>`; return; }
  const p=r.pick, chain=p.chain.join("→");
  $("#rfBA").innerHTML=`<figure><figcaption>before</figcaption><img src="${r.before}"></figure>`+
    `<figure><figcaption>auto pick [${p.kind}/${p.reward}]: <b>${chain}</b> (score ${p.score}) — `+
    `<b style="color:#e8c000">▦ same</b> · <b style="color:#2dd24d">▦ added</b> · <b style="color:#eb4a3d">▦ removed</b></figcaption>`+
    `<img src="${r.after}"></figure>`;
  RF_CHAIN = (p.ops||[]).map(o=>({name:o.name, kw:o.kw||{}, on:true})); renderChain();   // load editable (empty = leave as-is)
  $("#rfHint").innerHTML = `auto [${p.kind}/${p.reward}]: <b>${chain}</b> loaded — edit / <b>Apply</b> to accept. `+
    `ranked: `+p.candidates.slice(0,5).map(c=>`${c.chain.join("→")}·${c.score}`).join("  ") ; };
// bulk auto-refine: each instance in the partition (or the class field) gets its OWN searched-best chain
$("#rfAutoMany").onclick=async()=>{ const cls=$("#rfClass").value.trim();
  const body = cls ? {cls} : (INST.pid ? {pid:INST.pid} : null);
  if(!body){alert("select a partition (Partitions tab) or type a class name first");return;}
  const tgt = cls ? `class "${cls}"` : `partition ${INST.pid}`;
  if(!confirm(`Auto-refine ALL instances in ${tgt} — each gets its own best chain (decided in ${tgt} context)?`))return;
  $("#rfHint").innerHTML=SPIN+`auto-refining ${tgt}…`;
  const r=await withBusy("#rfAutoMany", ()=>post("/api/auto_refine_many",{...body, kind:"auto"}));
  if(r.detail){alert(r.detail);return;}
  setStatus(r.stats); if(INST.pid)selectPartition(INST.pid); loadPartitions(true);
  $("#rfHint").innerHTML=`auto-refined ${r.n} in ${tgt} [${r.kind}/${r.reward}] — chains: `+
    r.summary.map(s=>`${s.chain.join("→")}×${s.n}`).join("  ·  "); };
// category consensus: one MODAL chain for the whole class (preview the vote, then apply + save as the rule)
$("#rfConsensus").onclick=async()=>{ const cls=$("#rfClass").value.trim(); if(!cls){alert("enter a class name");return;}
  $("#rfHint").innerHTML=SPIN+`consensus: searching class "${escAttr(cls)}"…`;
  const pre=await withBusy("#rfConsensus", ()=>post("/api/auto_refine_consensus",{cls, apply:false}));
  if(pre.detail){alert(pre.detail);return;}
  if(!pre.n){ $("#rfHint").textContent=`class "${cls}": no instances`; return; }
  const chain=pre.chain.join("→");
  if(!confirm(`Class "${cls}" [${pre.kind}/${pre.reward}] consensus over ${pre.n} — apply "${chain}" `+
    `(won ${pre.votes}/${pre.n}) to ALL + save as the class rule?\n\nvotes: `+
    pre.summary.map(s=>`${s.chain.join("→")}×${s.n}`).join("   ")))return;
  const r=await withBusy("#rfConsensus", ()=>post("/api/auto_refine_consensus",{cls, apply:true}));
  if(r.detail){alert(r.detail);return;}
  setStatus(r.stats); setClasses(r.classes); loadClassRules(); loadPartitions(true);
  $("#rfHint").innerHTML=`class "${cls}" [${r.kind}/${r.reward}]: consensus <b>${r.chain.join("→")}</b> applied to ${r.applied}, saved as rule`; };
$("#rfSplit").onclick=async()=>{
  const cur=$("#rfIuid").value.trim();                       // split the instance LOADED in Refine (e.g. arrived via → Refine from In-image)
  const iu = cur ? [cur] : [...pGrid.sel];                   // else fall back to the Partitions-grid selection
  if(!iu.length){ alert("load an instance into Refine (its iuid above — e.g. via → Refine from In-image), or select instances in the Partitions grid, then Split."); return; }
  const r=await post("/api/split",{iuids:iu});
  setStatus(r.stats); loadPartitions(true); if(INST.pid)selectPartition(INST.pid);
  if(cur){ $("#rfIuid").value=""; $("#rfBA").innerHTML=""; }  // the split original became background → clear the stale target/preview
  alert(`split → ${r.n} new instances (re-cluster to see them in partitions)`); };
// bulk-apply the current chain to a whole partition or a whole class (stored as the class's rule)
$("#rfApplyPart").onclick=async()=>{ const ops=activeOps(); if(!ops.length){alert("add ops to the chain first");return;}
  if(!INST.pid){alert("select a partition in the Partitions tab first");return;}
  if(!confirm(`Apply ${ops.length} op(s) to ALL instances in partition ${INST.pid}?`))return;
  const r=await withBusy("#rfApplyPart", ()=>post("/api/apply_refine_partition",{pid:INST.pid, ops}));
  if(r.detail){alert(r.detail);return;}
  setStatus(r.stats); selectPartition(INST.pid); $("#rfHint").textContent=`applied chain to ${r.n} instance(s) in partition ${INST.pid}`; };
$("#rfApplyClass").onclick=async()=>{ const cls=$("#rfClass").value.trim(); const ops=activeOps();
  if(!cls){alert("enter a class name");return;} if(!ops.length){alert("add ops to the chain first");return;}
  if(!confirm(`Save & apply ${ops.length} op(s) as the rule for class "${cls}" (all its instances)?`))return;
  const r=await post("/api/apply_class_rule",{cls, ops});
  if(r.detail){alert(r.detail);return;}
  setStatus(r.stats); setClasses(r.classes); loadClassRules(); loadPartitions(true);
  $("#rfHint").textContent=`class "${cls}": rule saved, applied to ${r.n} instance(s)`; };
async function loadClassRules(){ const r=await api("/api/class_rules");
  $("#rfRules").innerHTML = r.rules.length
    ? "saved rules: "+r.rules.map(x=>`<b>${x.cls}</b> [${x.ops.join("→")||'—'}]×${x.n}`).join(" · ")+" · <i>(type a class below to load its rule)</i>"
    : "no saved class rules yet"; }
// reselecting a class in Refine LOADS its saved rule-chain into the live chain so the preview shows it
// (the summary only carries op names; this fetches the full ops + kw). Empty class with no rule -> no-op.
async function loadClassRuleIntoChain(cls){
  cls=(cls||"").trim(); if(!cls) return;
  const r=await api(`/api/class_rule?cls=${enc(cls)}`);
  if(!r.ops || !r.ops.length){ $("#rfHint").textContent=`class "${cls}": no saved rule`; return; }
  RF_CHAIN = r.ops.map(o=>({name:o.name, kw:o.kw||{}, on:o.on!==false}));
  renderChain(); rfRepreviewIfShown();
  $("#rfHint").textContent=`loaded saved rule for "${cls}" (${RF_CHAIN.length} op(s)) — Preview / edit / re-apply`;
}
$("#rfClass").onchange=e=>loadClassRuleIntoChain(e.target.value);
// instance picker / search (iuids are opaque → search by file / class / image-id / iuid-prefix). Search
// REPLACES the peer list (it's how you find a starting instance); picking one then shows its partition peers.
async function rfFind(q=""){
  RF_PEERS_PID=null; RF_PEERS_IUID="";                  // leaving the peer view → next preview re-renders peers
  const r=await api(`/api/find_instances?query=${enc(q)}&limit=60`);
  $("#rfFindCount").textContent = `${r.total}${r.total>=60?"+":""} match${r.total===1?"":"es"} — click one to refine (its partition peers then show here)`;
  $("#rfIuidList").innerHTML = r.items.map(it=>`<option value="${it.iuid}">`).join("");
  $("#rfFind").innerHTML = r.items.length ? r.items.map(it=>cell(it,it.caption)).join("") : `<div class="muted">no matches</div>`;
  observeCrops($("#rfFind"));
}
$("#rfSearch").oninput=e=>{ clearTimeout(window._rfs); window._rfs=setTimeout(()=>rfFind(e.target.value.trim()),200); };
$("#rfFind").onclick=e=>{ const c=e.target.closest(".cell"); if(!c)return;
  $$("#rfFind .cell").forEach(x=>x.classList.remove("sel")); c.classList.add("sel");
  $("#rfIuid").value=c.dataset.iuid; rfDoPreview(); };
$("#rfIuid").onchange=rfDoPreview;
// SAM checkpoint setup (so the `sam` op works without manual env wiring)
async function refreshSamStatus(){
  const fam = $("#rfSamModel") ? $("#rfSamModel").value : "auto";
  const s=await api(`/api/sam_status?family=${fam==="auto"?"":fam}`);
  const has = f => (s.families||[]).includes(f);
  const label = {samhq:"SAM-HQ", medsam:"MedSAM"}[s.family] || "SAM";
  let msg;
  if(fam==="samhq" && !s.samhq_installed) msg = "the `segment-anything-hq` package is not installed (pip install segment-anything-hq)";
  else if(!s.installed) msg = "the `segment-anything` package is not installed (pip install segment-anything)";
  else if(s.ckpt) msg = `${label} ready: ${s.model_type} · ${s.ckpt.split("/").pop()}`;
  else if(fam==="medsam") msg = "no MedSAM checkpoint — drop a *medsam*.pth in CURATOR_SAM_DIR or set CURATOR_MEDSAM_CKPT (not auto-downloadable)";
  else if(fam==="samhq") msg = "no SAM-HQ checkpoint yet — download ↓ (HQ token = crisper masks, incl. thin structures)";
  else msg = "no checkpoint yet — download SAM ↓";
  if(fam==="medsam") msg += " · box-prompt, medical-tuned (points ignored)";
  $("#rfSamMsg").textContent = msg;
  // show the setup button when the family's package is installed but its checkpoint is missing
  const needs = fam==="samhq" ? (s.samhq_installed && !has("samhq")) : (fam!=="medsam" && s.installed && !has("sam"));
  $("#rfSamSetup").style.display = needs ? "inline-block" : "none";
}
$("#rfSamModel").onchange = refreshSamStatus;
$("#rfSamSetup").onclick=async()=>{ const fam=$("#rfSamModel").value==="samhq"?"samhq":"sam";
  $("#rfSamMsg").innerHTML=SPIN+`downloading ${fam==="samhq"?"SAM-HQ":"SAM"} checkpoint (~375 MB), one-time…`;
  const r=await post("/api/sam_setup",{family:fam});
  if(r.detail){ $("#rfSamMsg").textContent="error: "+r.detail; return; }
  $("#rfSamMsg").textContent=`${fam==="samhq"?"SAM-HQ":"SAM"} ready: ${r.ckpt}`; $("#rfSamSetup").style.display="none"; };
// Propagate the LOADED reference instance's refinement across its partition, RAD-DINO-gated (Task 1).
$("#rfPropMatch").onclick=async()=>{ const ref=$("#rfIuid").value.trim(); if(!ref){alert("load a refined reference instance (set its iuid / arrive via → Refine) first");return;}
  const ops=activeOps();                                   // live chain; empty => server uses the ref's recorded rule_ops
  const thr=parseFloat($("#rfMatchThr").value);
  if(!confirm(`Propagate ${ops.length||"the reference's"} op(s) to RAD-DINO-similar members (τ=${thr}) of ${ref.slice(0,6)}…'s partition?`))return;
  $("#rfHint").innerHTML=SPIN+"matching (RAD-DINO) + propagating…";
  const r=await withBusy("#rfPropMatch", ()=>post("/api/propagate_refinement",{ref_iuid:ref, ops:(ops.length?ops:null), match_thresh:thr}));
  if(r.detail){ $("#rfHint").innerHTML=`<span style="color:var(--warn)">${r.detail}</span>`; return; }
  setStatus(r.stats); if(INST.pid)selectPartition(INST.pid); loadPartitions(true);
  $("#rfHint").textContent=`propagated to ${r.applied} member(s) of partition ${r.pid} · skipped ${r.skipped} (below τ)`; };

// ---- few-shot shape transfer: reference mask(s) -> partition peers (SAM/SAM-HQ within each bbox) ----
let XFER_REFS = new Set();            // collected reference iuids; empty => the loaded #rfIuid is the sole ref
let XFER_PREVIEW_KEY = null;          // key of the last successful preview; Commit is gated to match it (preview-first)
const xferRefs = ()=>{ const r=[...XFER_REFS]; const u=$("#rfIuid").value.trim(); return r.length?r:(u?[u]:[]); };
const xferKey = ()=> JSON.stringify([xferRefs(), $("#rfXferModel").value, $("#rfXferTau").value, $("#rfXferIou").value, INST.pid||null]);
function xferGate(){ $("#rfXferCommit").disabled = !(XFER_PREVIEW_KEY && XFER_PREVIEW_KEY===xferKey()); }
function renderXferRefs(){ const r=[...XFER_REFS];
  $("#rfXferRefs").innerHTML = r.length
    ? `refs: `+r.map(u=>`<span class="chip" data-rmref="${u}" title="remove">${u.slice(0,6)} <span class="x">×</span></span>`).join(" ")
    : `refs: <i>(loaded instance)</i>`; }
function xferBody(){ const tau=$("#rfXferTau").value.trim(), iou=$("#rfXferIou").value.trim();
  return {ref_iuids:xferRefs(), pid:INST.pid||null, sam_model:$("#rfXferModel").value,
          match_thresh:(tau===""?null:parseFloat(tau)), agree_iou:(iou===""?null:parseFloat(iou))}; }
function xferItemFig(it){ const tag=it.keep?`<b style="color:#2dd24d">keep</b>`:`<b style="color:#eb4a3d">drop (low IoU)</b>`;
  return `<figure><figcaption>${it.iuid.slice(0,6)} · IoU ${it.iou} · ${tag}</figcaption>`+
         `<div style="display:flex;gap:4px"><img src="${it.before}" style="max-height:130px"><img src="${it.after}" style="max-height:130px"></div></figure>`; }
$("#rfXferRefs").onclick=e=>{ const c=e.target.closest("[data-rmref]"); if(c){ XFER_REFS.delete(c.dataset.rmref); renderXferRefs(); XFER_PREVIEW_KEY=null; xferGate(); } };
$("#rfXferAddRef").onclick=()=>{ const u=$("#rfIuid").value.trim(); if(!u){alert("load an instance (iuid above) first");return;} XFER_REFS.add(u); renderXferRefs(); XFER_PREVIEW_KEY=null; xferGate(); };
$("#rfXferClearRefs").onclick=()=>{ XFER_REFS.clear(); renderXferRefs(); XFER_PREVIEW_KEY=null; xferGate(); };
["#rfXferModel","#rfXferTau","#rfXferIou"].forEach(s=>$(s).addEventListener("change",()=>{ XFER_PREVIEW_KEY=null; xferGate(); }));
$("#rfIuid").addEventListener("input", ()=>{ XFER_PREVIEW_KEY=null; xferGate(); });   // changing the loaded ref invalidates the preview
$("#rfXferPreview").onclick=async()=>{ const refs=xferRefs(); if(!refs.length){alert("load an instance or add a reference first");return;}
  $("#rfHint").innerHTML=SPIN+"shape transfer: building template + SAM-decoding a sample…";
  const r=await withBusy("#rfXferPreview", ()=>post("/api/shape_transfer_preview", {...xferBody(), sample:12}));
  if(r.detail){ $("#rfHint").innerHTML=`<span style="color:var(--warn)">${r.detail}</span>`; XFER_PREVIEW_KEY=null; xferGate(); return; }
  const drop=r.items.filter(it=>!it.keep).length;
  const mode = r.kind==="line" ? `<b style="color:var(--acc)">LINE mode</b> (vessel trace, width ${r.width}px — SAM not used)` : `shape mode (SAM/SAM-HQ)`;
  $("#rfBA").innerHTML=`<div class="report" style="padding:4px">transfer preview · partition ${r.pid} · ${mode} · `+
    `${r.n_members} member(s)${r.gate_skipped?` (τ-skipped ${r.gate_skipped})`:""} · showing ${r.shown}${r.truncated?` of ${r.shown+r.truncated}`:""}`+
    `${r.agree_iou!=null?` · would drop ${drop} below IoU ${r.agree_iou}`:""} — left = before, right = after</div>`+
    `<div class="ba">`+(r.items.map(xferItemFig).join("")||"<div class='muted'>no members to transfer to</div>")+`</div>`;
  XFER_PREVIEW_KEY=xferKey(); xferGate();
  $("#rfHint").textContent=`previewed ${r.shown} member(s) — review, then Commit to apply to the whole partition`; };
$("#rfXferCommit").onclick=async()=>{ if(XFER_PREVIEW_KEY!==xferKey()){ alert("Preview the transfer first (settings changed since the last preview)."); xferGate(); return; }
  if(!confirm(`Commit shape transfer to partition ${INST.pid||"(of the reference)"} — SAM/SAM-HQ refines each member toward the reference shape?`))return;
  $("#rfHint").innerHTML=SPIN+"shape transfer: committing to the partition…";
  const r=await withBusy("#rfXferCommit", ()=>post("/api/shape_transfer", xferBody()));
  if(r.detail){ $("#rfHint").innerHTML=`<span style="color:var(--warn)">${r.detail}</span>`; return; }
  setStatus(r.stats); if(INST.pid)selectPartition(INST.pid); loadPartitions(true);
  XFER_PREVIEW_KEY=null; xferGate();
  $("#rfHint").textContent=`transferred to ${r.applied} member(s) of ${r.pid} · τ-skipped ${r.skipped} · IoU-dropped ${r.gated_out}`; };
renderRfParams(); renderXferRefs(); xferGate();

// ---------- hand-draw mask editor (brush + eraser; zoomed crop with a context toggle) ----------
// Canvas pixels are SOLID red where the mask is on (alpha 0/255 -> crisp binary); CSS opacity makes it
// see-through over the image. Save reads the alpha channel and posts a canvas-res binary PNG + the crop box.
let ME = {iuid:null, box:null, ctx:null, painting:false, mode:"brush", context:false, last:[0,0], dirty:false};
async function meLoad(){
  const r=await api(`/api/edit_view?iuid=${enc(ME.iuid)}&context=${ME.context?1:0}`);
  ME.box=r.box;
  const bg=$("#meBg"), cv=$("#meCanvas");
  bg.src=r.img; bg.width=r.w; bg.height=r.h; cv.width=r.w; cv.height=r.h;
  const ctx=cv.getContext("2d"); ME.ctx=ctx; ctx.clearRect(0,0,r.w,r.h);
  await new Promise(res=>{ const mk=new Image(); mk.onerror=()=>res(); mk.onload=()=>{
    const tmp=document.createElement("canvas"); tmp.width=r.w; tmp.height=r.h; const tc=tmp.getContext("2d");
    tc.drawImage(mk,0,0,r.w,r.h); const d=tc.getImageData(0,0,r.w,r.h).data, out=ctx.createImageData(r.w,r.h);
    for(let i=0;i<r.w*r.h;i++){ if(d[i*4]>127){ out.data[i*4]=235; out.data[i*4+1]=50; out.data[i*4+2]=40; out.data[i*4+3]=255; } }
    ctx.putImageData(out,0,0); res(); }; mk.src=r.mask; });
  ME.dirty=false;
}
async function openMaskEditor(iuid){ if(!iuid){alert("load an instance (iuid above) first");return;}
  ME.iuid=iuid; ME.context=false; ME.mode="brush"; $("#meBrush").classList.add("on"); $("#meErase").classList.remove("on");
  $("#meContext").textContent="show full image"; $("#meId").textContent=iuid.slice(0,8);
  await meLoad(); $("#maskEditor").classList.add("on"); }
$("#rfEditMask").onclick=()=>openMaskEditor($("#rfIuid").value.trim());
function mePos(e){ const cv=$("#meCanvas"), r=cv.getBoundingClientRect();
  return [ (e.clientX-r.left)*cv.width/r.width, (e.clientY-r.top)*cv.height/r.height ]; }
function meStyle(){ ME.ctx.globalCompositeOperation = ME.mode==="erase" ? "destination-out" : "source-over"; }
function meDab(x,y){ const ctx=ME.ctx,s=+$("#meSize").value; meStyle(); ctx.fillStyle="rgb(235,50,40)"; ctx.beginPath(); ctx.arc(x,y,s/2,0,7); ctx.fill(); ME.dirty=true; }
function meLine(a,b){ const ctx=ME.ctx,s=+$("#meSize").value; meStyle(); ctx.strokeStyle="rgb(235,50,40)"; ctx.lineWidth=s; ctx.lineCap="round"; ctx.lineJoin="round"; ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke(); ME.dirty=true; }
$("#meCanvas").addEventListener("pointerdown",e=>{ e.preventDefault(); ME.painting=true; ME.last=mePos(e); meDab(ME.last[0],ME.last[1]); try{$("#meCanvas").setPointerCapture(e.pointerId);}catch(_){} });
$("#meCanvas").addEventListener("pointermove",e=>{ if(!ME.painting)return; const p=mePos(e); meLine(ME.last,p); ME.last=p; });
$("#meCanvas").addEventListener("pointerup",()=>{ ME.painting=false; });
$("#meBrush").onclick=()=>{ ME.mode="brush"; $("#meBrush").classList.add("on"); $("#meErase").classList.remove("on"); };
$("#meErase").onclick=()=>{ ME.mode="erase"; $("#meErase").classList.add("on"); $("#meBrush").classList.remove("on"); };
$("#meClear").onclick=()=>{ ME.ctx.clearRect(0,0,$("#meCanvas").width,$("#meCanvas").height); ME.dirty=true; };
$("#meInvert").onclick=()=>{ const cv=$("#meCanvas"),ctx=ME.ctx,d=ctx.getImageData(0,0,cv.width,cv.height),a=d.data;
  for(let i=0;i<cv.width*cv.height;i++){ const on=a[i*4+3]>127; a[i*4]=235;a[i*4+1]=50;a[i*4+2]=40;a[i*4+3]=on?0:255; } ctx.putImageData(d,0,0); ME.dirty=true; };
$("#meFill").onclick=()=>{ const cv=$("#meCanvas"),ctx=ME.ctx,W=cv.width,Hh=cv.height,img=ctx.getImageData(0,0,W,Hh),a=img.data,N=W*Hh;
  const on=i=>a[i*4+3]>127, seen=new Uint8Array(N), st=[];
  for(let x=0;x<W;x++){ st.push(x,(Hh-1)*W+x); } for(let y=0;y<Hh;y++){ st.push(y*W,y*W+W-1); }
  while(st.length){ const p=st.pop(); if(p<0||p>=N||seen[p]||on(p))continue; seen[p]=1; const x=p%W,y=(p-x)/W;
    if(x>0)st.push(p-1); if(x<W-1)st.push(p+1); if(y>0)st.push(p-W); if(y<Hh-1)st.push(p+W); }
  for(let i=0;i<N;i++){ if(!on(i)&&!seen[i]){ a[i*4]=235;a[i*4+1]=50;a[i*4+2]=40;a[i*4+3]=255; } } ctx.putImageData(img,0,0); ME.dirty=true; };
$("#meContext").onclick=async()=>{ if(ME.dirty && !confirm("Switching view discards unsaved strokes. Continue?"))return;
  ME.context=!ME.context; $("#meContext").textContent=ME.context?"show crop":"show full image"; await meLoad(); };
$("#meCancel").onclick=()=>{ $("#maskEditor").classList.remove("on"); };
$("#meSave").onclick=async()=>{ const cv=$("#meCanvas"),ctx=ME.ctx,W=cv.width,Hh=cv.height,d=ctx.getImageData(0,0,W,Hh).data;
  const tmp=document.createElement("canvas"); tmp.width=W; tmp.height=Hh; const tc=tmp.getContext("2d"), out=tc.createImageData(W,Hh);
  for(let i=0;i<W*Hh;i++){ const v=d[i*4+3]>127?255:0; out.data[i*4]=out.data[i*4+1]=out.data[i*4+2]=v; out.data[i*4+3]=255; }
  tc.putImageData(out,0,0);
  const r=await withBusy("#meSave", ()=>post("/api/set_mask",{iuid:ME.iuid, png:tmp.toDataURL("image/png"), box:ME.box}));
  if(r.detail){ alert(r.detail); return; }
  setStatus(r.stats); $("#maskEditor").classList.remove("on");
  if($("#rfIuid").value.trim()===ME.iuid){ RF_PEERS_PID=null; rfDoPreview(); }   // refresh refine before/after + peers
  if(typeof INST!=="undefined" && INST.pid) selectPartition(INST.pid);
  if(typeof IIMG!=="undefined" && IIMG.id) loadImage(true); };

// ---------- latent-space Map (projection + paint-select -> the curator's existing actions) ----------
const MAP = { pts:[], loaded:false, view:{s:1,ox:0,oy:0}, mode:false, dragging:false, last:[0,0],
              dpr:1, grid:null, gcol:64, sel:new Set(), colorBy:"state" };
function mapCanvasSize(){ const cv=$("#mapCanvas"), st=$("#mapStage"), dpr=window.devicePixelRatio||1;
  MAP.dpr=dpr; cv.width=Math.max(1,Math.round(st.clientWidth*dpr)); cv.height=Math.max(1,Math.round(st.clientHeight*dpr)); }
function mapOnShow(){ mapCanvasSize(); if(!MAP.loaded) mapLoad(); else { mapFit(); mapDraw(); } }
addEventListener("resize", ()=>{ if(document.querySelector(".tab.active")?.id==="tab-map" && MAP.loaded){ mapCanvasSize(); mapFit(); mapDraw(); } });
async function mapLoad(){
  $("#mapInfo").innerHTML = SPIN+"projecting instances (h-NNE / UMAP)…";
  const r = await withBusy("#mapLoad", ()=>api(`/api/projection_points?method=hnne`));
  if(!r || r.detail){ $("#mapInfo").innerHTML=`<span style="color:var(--warn)">${(r&&r.detail)||"projection failed"}</span>`; return; }
  MAP.pts=r.points||[]; MAP.loaded=true; MAP.sel.clear(); mapBuildGrid(); mapCanvasSize(); mapFit(); mapDraw(); mapRenderSel();
  $("#mapInfo").textContent = `${r.n} instances · ${r.method}${r.truncated?` · first ${r.n} (capped)`:""} · features: ${Object.keys(r.spec||{}).join("+")||"—"} · wheel=zoom, drag=pan, ✏️=paint-select`;
}
function mapBuildGrid(){ const G=MAP.gcol, b=Array.from({length:G*G},()=>[]);
  MAP.pts.forEach((p,i)=>{ const gx=Math.min(G-1,Math.max(0,(p.x*G)|0)), gy=Math.min(G-1,Math.max(0,(p.y*G)|0)); b[gy*G+gx].push(i); });
  MAP.grid=b; }
function mapQuery(wx,wy,wr){ const G=MAP.gcol, out=[], r=Math.ceil(wr*G)+1, cx=(wx*G)|0, cy=(wy*G)|0;
  for(let gy=Math.max(0,cy-r); gy<=Math.min(G-1,cy+r); gy++) for(let gx=Math.max(0,cx-r); gx<=Math.min(G-1,cx+r); gx++)
    for(const i of MAP.grid[gy*G+gx]){ const p=MAP.pts[i]; if((p.x-wx)**2+(p.y-wy)**2 <= wr*wr) out.push(i); }
  return out; }
function mapFit(){ const cv=$("#mapCanvas"), W=cv.width, H=cv.height, m=0.06*Math.min(W,H), s=Math.min(W,H)-2*m;
  MAP.view={ s, ox:(W-s)/2, oy:(H-s)/2 }; }
function mapHashColor(s){ let h=0; for(let i=0;i<s.length;i++) h=(h*31+s.charCodeAt(i))|0; return `hsl(${((h%360)+360)%360},64%,58%)`; }
function mapColorOf(p){
  if(MAP.colorBy==="state") return p.state==="class"?"#3fb27f":(p.state==="reject"?"#e0533d":"#7f8aa0");
  if(MAP.colorBy==="score"){ const v=Math.max(0,Math.min(1,p.score||0)); return `hsl(${(v*130)|0},70%,55%)`; }
  const key = MAP.colorBy==="class" ? p.cls : (MAP.colorBy==="source" ? p.source : p.pid);
  return key ? mapHashColor(key) : "#3a4150";
}
function mapDraw(){ if(!MAP.loaded) return; const cv=$("#mapCanvas"), ctx=cv.getContext("2d"), v=MAP.view;
  ctx.clearRect(0,0,cv.width,cv.height);
  const r=(+$("#mapPtSize").value)*MAP.dpr, d=Math.max(1,r*2);
  for(const p of MAP.pts){ ctx.fillStyle=mapColorOf(p); ctx.fillRect(p.x*v.s+v.ox-r, p.y*v.s+v.oy-r, d, d); }
  if(MAP.sel.size){ ctx.strokeStyle="#fff"; ctx.lineWidth=MAP.dpr;
    for(const p of MAP.pts) if(MAP.sel.has(p.iuid)){ ctx.beginPath(); ctx.arc(p.x*v.s+v.ox, p.y*v.s+v.oy, r+1.5*MAP.dpr, 0, 7); ctx.stroke(); } }
}
function mapEvtPos(e){ const cv=$("#mapCanvas"), rect=cv.getBoundingClientRect();
  return [ (e.clientX-rect.left)*cv.width/rect.width, (e.clientY-rect.top)*cv.height/rect.height ]; }
function mapS2W(sx,sy){ const v=MAP.view; return [ (sx-v.ox)/v.s, (sy-v.oy)/v.s ]; }
function mapPaint(e){ const [sx,sy]=mapEvtPos(e), [wx,wy]=mapS2W(sx,sy), wr=((+$("#mapBrush").value)*MAP.dpr)/MAP.view.s;
  const erase=e.altKey; for(const i of mapQuery(wx,wy,wr)){ const u=MAP.pts[i].iuid; erase?MAP.sel.delete(u):MAP.sel.add(u); }
  $("#mapSelCount").textContent=`${MAP.sel.size} selected`; mapDraw(); }
function mapHover(e){ const [sx,sy]=mapEvtPos(e), [wx,wy]=mapS2W(sx,sy), wr=(8*MAP.dpr)/MAP.view.s, idxs=mapQuery(wx,wy,wr), tip=$("#mapTip");
  if(!idxs.length){ tip.style.display="none"; return; }
  let best=idxs[0], bd=1e18; for(const i of idxs){ const p=MAP.pts[i], dd=(p.x-wx)**2+(p.y-wy)**2; if(dd<bd){bd=dd;best=i;} }
  const p=MAP.pts[best], rect=$("#mapCanvas").getBoundingClientRect(), cx=e.clientX-rect.left, cy=e.clientY-rect.top;
  tip.style.left=Math.min(rect.width-140, cx+12)+"px"; tip.style.top=Math.min(rect.height-160, cy+12)+"px";
  $("#mapTipCap").textContent=`${p.cls||p.state} · ${p.iuid.slice(0,6)} · s=${p.score}`;
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
$("#mapColor").onchange=e=>{ MAP.colorBy=e.target.value; mapDraw(); };
$("#mapPtSize").oninput=()=>mapDraw();
$("#mapReset").onclick=()=>{ if(MAP.loaded){ mapFit(); mapDraw(); } };
$("#mapLoad").onclick=()=>{ MAP.loaded=false; mapLoad(); };
function mapRenderSel(){ const g=$("#mapGrid"); $("#mapSelCount").textContent=`${MAP.sel.size} selected`;
  const ius=[...MAP.sel]; if(!ius.length){ g.innerHTML=`<div class="muted">drag over a region (paint-select) to fill this — then Assign / Reject.</div>`; return; }
  const show=ius.slice(0,120);
  g.innerHTML=show.map(u=>cell({iuid:u})).join("")+(ius.length>show.length?`<div class="muted" style="grid-column:1/-1">+${ius.length-show.length} more selected (not shown)</div>`:"");
  observeCrops(g); }
async function mapRefreshAfter(){ const r=await api(`/api/projection_points?method=hnne`);   // coords cached server-side -> instant recolor
  if(r && !r.detail){ MAP.pts=r.points||[]; mapBuildGrid(); } mapDraw(); mapRenderSel();
  if(typeof loadPartitions==="function") loadPartitions(true); }
$("#mapAssign").onclick=async()=>{ const cls=$("#mapClass").value.trim(), ius=[...MAP.sel];
  if(!ius.length){alert("paint-select some points first (turn on ✏️ select)");return;} if(!cls){alert("enter or pick a class name");return;}
  if(!confirm(`Assign ${ius.length} selected instance(s) to "${cls}"?`))return;
  const r=await withBusy("#mapAssign", ()=>post("/api/assign",{iuids:ius, cls})); setStatus(r.stats); setClasses(r.classes);
  MAP.sel.clear(); await mapRefreshAfter(); };
$("#mapReject").onclick=async()=>{ const ius=[...MAP.sel];
  if(!ius.length){alert("paint-select some points first (turn on ✏️ select)");return;}
  if(!confirm(`Reject (send to background) ${ius.length} selected instance(s)?`))return;
  const r=await withBusy("#mapReject", ()=>post("/api/reject",{iuids:ius})); setStatus(r.stats);
  MAP.sel.clear(); await mapRefreshAfter(); };
$("#mapClear").onclick=()=>{ MAP.sel.clear(); mapDraw(); mapRenderSel(); };

// ---------- Classifier ----------
let CLF={offset:0,limit:60,total:0};
const clfGrid = makeGrid("#clfgrid","#clfExclCount","excluded", refreshGates);
function syncClfFeats(){ if(!window._features)return;
  $("#clfFeats").innerHTML = featBoxes("clffeat", f=>f=='decoder'||f=='shape'); }
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
  $("#clfReport").innerHTML=`trained <b>${r.algo}</b> on ${r.n_classes} classes: ${r.classes.join(", ")}`+
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
  const r=await withBusy("#clfApply", ()=>post("/api/apply_predictions",{thresh:+$("#clfThr").value, only_class:$("#clfOnly").value.trim(), exclude:[...clfGrid.sel]}));
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
const clfRejGrid = makeGrid("#clfRejGrid","#clfRejSelCount","selected", refreshGates);
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
const clfIntGrid = makeGrid("#clfIntGrid","#clfIntSelCount","selected", refreshGates);
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
  $("#mrFeats").innerHTML = featBoxes("mrfeat", f=>f=='decoder'); }
// one card per candidate GROUP. Each input instance is an individually toggleable crop (selected by default):
// "Merge selected" merges only the CHECKED subset (the ones that actually belong), leaving the rest alone.
function mergeCardHTML(c){
  const ius=c.iuids||[];
  const crops = ius.slice(0,30).map(u=>`<div class="mccrop sel" data-iuid="${u}"><img loading="lazy" class="imgld" src="${cropUrl(u)}"><div class="mclbl">${u.slice(0,6)}</div></div>`).join("");
  return `<div class="mcard" data-img="${c.image_id}">`+
    `<div class="mcbar"><b>P(merge)=${c.prob}</b> <span class="muted">img ${c.image_id} · ${ius.length} inst · click crops to (de)select</span><span class="grow"></span>`+
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
    const r=await post("/api/accept_merge",{iuids:ius, mode:opts.mode()}); setStatus(r.stats); if(r.classes) setClasses(r.classes);
    card.remove(); loadPartitions(true); if(opts.afterAccept) opts.afterAccept();
  } else if(e.target.closest(".mcRej")){
    const all=[...card.querySelectorAll(".mccrop")].map(c=>c.dataset.iuid);
    if(all.length>=2) await post("/api/reject_merge",{iuids:all});
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

// ---------- Reference exemplar bank (foreign-object class suggestions) ----------
let REFSUG = {};                                   // iuid -> top suggested class (for "Accept top")
const refSugGrid = makeGrid("#refSugGrid","#refSugSelCount","selected", refreshGates);
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
let SUB={subpid:null, offset:0, limit:60, total:0};
const subGrid = makeGrid("#subgrid","#subSelCount","selected", refreshGates);
function syncSubFeats(){ if(!window._features)return;
  $("#subFeats").innerHTML = featBoxes("subfeat", f=>f=='decoder'); }
$("#subRun").onclick=async()=>{
  if(!INST.pid){ alert("select a partition/class on the Partitions tab first"); return; }
  const feats=$$(".subfeat:checked").map(e=>e.value); if(!feats.length){alert("pick at least one feature");return;}
  $("#subMsg").innerHTML=SPIN+"training contrastive encoder + FINCH… (this can take a few seconds)";
  const r=await withBusy("#subRun", ()=>post("/api/subcluster",{target:INST.pid, features:feats, dim:+$("#subDim").value, epochs:+$("#subEpochs").value, temperature:+$("#subTemp").value}));
  if(r.detail||r.error||!r.ok){ $("#subMsg").innerHTML=`<span style="color:var(--warn)">${r.detail||r.error||'failed'}</span>`; return; }
  $("#subMsg").textContent=`${r.n} instances → ${r.n_levels} FINCH levels${r.capped?" (capped sample)":""}`;
  await loadSubLevels(); loadSubList(); subGrid.reset(); };
async function loadSubLevels(){ const r=await api("/api/subclusters"); if(!r.active)return;
  $("#subLevel").innerHTML = r.levels.map(l=>`<option value="${l.i}" ${l.i===r.level?'selected':''}>${l.n} sub-clusters</option>`).join(""); }
$("#subLevel").onchange=async()=>{ await post("/api/subcluster_level",{level:+$("#subLevel").value}); loadSubList(); subGrid.reset(); SUB.subpid=null; };
async function loadSubList(){ const r=await api("/api/subclusters");
  $("#subList").innerHTML = (r.rows&&r.rows.length)
    ? r.rows.map(x=>`<div class="prow subrow" data-subpid="${x.subpid}"><span>sub ${x.subpid}</span><span class="sz">${x.size}</span></div>`).join("")
    : `<div class="muted">no sub-clusters yet — Run above</div>`; }
$("#subList").onclick=e=>{ const row=e.target.closest(".subrow"); if(!row)return;
  $$("#subList .subrow").forEach(x=>x.classList.toggle("sel", x===row)); selectSub(row.dataset.subpid); };
async function selectSub(subpid, reset=true){ SUB.subpid=subpid; if(reset){ SUB.offset=0; subGrid.reset(); }
  const r=await api(`/api/subcluster_instances?subpid=${enc(subpid)}&offset=${SUB.offset}&limit=${SUB.limit}`);
  SUB.total=r.total; if(reset && !r.items.length) subGrid.msg("(empty)"); else subGrid.append(r.items);
  SUB.offset+=r.items.length; $("#subMore").style.display=SUB.offset<r.total?"inline-block":"none"; }
$("#subMore").onclick=()=>selectSub(SUB.subpid,false);
$("#subSelAll").onclick=()=>subGrid.selectPage(); $("#subNone").onclick=()=>subGrid.clearSel();
// actions operate on the SELECTED crops, or the WHOLE current sub-cluster when nothing is selected
async function subActionIuids(verb){
  if(subGrid.sel.size) return [...subGrid.sel];
  if(!SUB.subpid){ alert("select a sub-cluster (or tick instances) first"); return []; }
  const all=await api(`/api/subcluster_instances?subpid=${enc(SUB.subpid)}&offset=0&limit=1000000`);
  const iu=all.items.map(i=>i.iuid);
  if(iu.length && !confirm(`${verb} all ${iu.length} instances in sub-cluster ${SUB.subpid}?`)) return [];
  return iu;
}
function subAfter(r, iu, msg){ setStatus(r.stats); if(r.classes) setClasses(r.classes); subGrid.drop(iu); loadPartitions(true); $("#subMsg").textContent=msg; }
$("#subAssign").onclick=async()=>{ const cls=$("#subClass").value.trim(); if(!cls){alert("enter a sub-class name");return;}
  const iu=await subActionIuids("Assign"); if(!iu.length)return;
  subAfter(await post("/api/assign",{iuids:iu, cls}), iu, `assigned ${iu.length} → ${cls}`); };
$("#subReject").onclick=async()=>{ const iu=await subActionIuids("Reject"); if(!iu.length)return;
  subAfter(await post("/api/reject",{iuids:iu}), iu, `rejected ${iu.length}`); };
$("#subUnassign").onclick=async()=>{ const iu=await subActionIuids("Unassign"); if(!iu.length)return;
  subAfter(await post("/api/unassign",{iuids:iu}), iu, `unassigned ${iu.length} (back to the pool)`); };

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
let RJ={offset:0,limit:60,total:0};
const rjGrid = makeGrid("#rjgrid","#rjSelCount","selected", refreshGates);
async function rjLoad(reset){ if(reset){RJ.offset=0;rjGrid.reset();}
  const r=await api(`/api/rejected?offset=${RJ.offset}&limit=${RJ.limit}`); RJ.total=r.total;
  if(reset && !r.items.length) rjGrid.msg("no rejected instances"); else rjGrid.append(r.items);
  RJ.offset+=r.items.length; $("#rjMore").style.display=RJ.offset<r.total?"inline-block":"none"; }
$("#rjLoad").onclick=()=>rjLoad(true); $("#rjMore").onclick=()=>rjLoad(false);
$("#rjSelAll").onclick=()=>rjGrid.selectPage(); $("#rjNone").onclick=()=>rjGrid.clearSel();
$("#rjUnreject").onclick=async()=>{ if(!rjGrid.sel.size)return; const iu=[...rjGrid.sel]; const r=await post("/api/unreject",{iuids:iu}); setStatus(r.stats); rjGrid.drop(iu); loadPartitions(true); };

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
// poll /api/progress while a long inference/RAD-DINO job runs and drive a progress bar
let _progTimer=null;
async function _pollOnce(barSel, statusSel){
  try{ const p=await api("/api/progress"); const bar=$(barSel), fill=$(barSel+" > span");
    if(!p.active){ return; }
    if(p.total>0){ bar.classList.remove("indet"); const pct=Math.round(100*p.done/p.total);
      fill.style.width=pct+"%"; if(statusSel)$(statusSel).textContent=`${p.phase}: ${p.done}/${p.total} (${pct}%)`; }
    else { bar.classList.add("indet"); if(statusSel)$(statusSel).textContent=`${p.phase}…`; }
  }catch(e){}
}
async function withProgress(barSel, statusSel, fn, trigger){
  const bar=$(barSel); bar.style.display="block"; bar.classList.add("indet"); $(barSel+" > span").style.width="0%";
  const btn = typeof trigger==="string" ? $(trigger) : trigger;   // double-submit guard: disable the launch button
  if(btn){ if(btn._busy) return; btn._busy=true; btn.disabled=true; }
  _progTimer=setInterval(()=>_pollOnce(barSel,statusSel), 600);
  try{ return await fn(); }
  finally{ clearInterval(_progTimer); _progTimer=null; bar.style.display="none"; bar.classList.remove("indet");
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
  $("#relStats").innerHTML = relStatsLine(r.stats);
  const html = r.items.length ? r.items.map(relCell).join("")
             : (reset?`<div class="muted">no ${RELEASE.filter==='all'?'final':RELEASE.filter} images</div>`:"");
  if(reset) $("#relGrid").innerHTML=html; else $("#relGrid").insertAdjacentHTML("beforeend", html);
  $("#relMore").style.display = RELEASE.offset<r.total?"inline-block":"none";
}
async function setRelease(ids, status){ const r=await post("/api/release_set",{image_ids:ids, status});
  if(r&&r.stats) $("#relStats").innerHTML = relStatsLine(r.stats); return r; }
function relApplyStatus(cell, st){                     // reflect the decision: drop from a filtered view, else re-badge
  if(RELEASE.filter!=="all" && st!==RELEASE.filter){ cell.remove(); }
  else { cell.className=`rcell ${st}`; const b=cell.querySelector(".rbadge"); if(b){ b.className=`rbadge ${st}`; b.textContent=st; } }
}
function openInImage(img){
  $('nav button[data-tab="inimage"]').click();
  if(![...$("#imgSelect").options].some(o=>o.value===String(img))) $("#imgSelect").insertAdjacentHTML("afterbegin",`<option value="${img}">${img}</option>`);
  $("#imgSelect").value=String(img); loadImage(true);
}
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
  if(e.key==="m"){ MASKS=!MASKS; const cb=$("#ovMasks"); if(cb) cb.checked=MASKS; refreshVisibleCrops(); }
  else if(e.key==="c"){ toggleView(); }          // c = toggle crop <-> context view
});
$$(".viewToggle").forEach(b=> b.onclick=toggleView);
syncViewButtons();

// ---------- register the button gates (disabled when there's nothing to act on) ----------
// Partitions: selection-acting buttons need ≥1 selected (merge ≥2); partition-scoped actions need a partition.
[["#assignBtn",1],["#rejectBtn",1],["#unassignBtn",1],["#toRefineBtn",1],["#toInimgBtn",1],["#selNone",1],["#mergeBtn",2]]
  .forEach(([s,m])=>gate(s, ()=>pGrid.sel.size>=m));
gate("#assignAllBtn", ()=>INST.pid!=null); gate("#rejectAllBtn", ()=>INST.pid!=null);
// In-image: assign/reject/refine/deselect need ≥1, merge + its live preview need ≥2.
[["#iiAssign",1],["#iiReject",1],["#iiToRefine",1],["#iiDeselect",1],["#iiMerge",2]]
  .forEach(([s,m])=>gate(s, ()=>iiGrid.sel.size>=m));   // #iiMergePrev is a toggle: always clickable
// Classifier (preview / reject-suggest / interesting grids): bulk actions need ≥1 selected.
gate("#clfAssignSel", ()=>clfGrid.sel.size>=1); gate("#clfReject", ()=>clfGrid.sel.size>=1);
gate("#clfRejSel", ()=>clfRejGrid.sel.size>=1);
gate("#clfIntAssign", ()=>clfIntGrid.sel.size>=1); gate("#clfIntReject", ()=>clfIntGrid.sel.size>=1);
// Reference suggestions: per-instance acceptance needs ≥1 selected; suggest needs a partition.
[["#refAcceptTop",1],["#refAssign",1],["#refReject",1]].forEach(([s,m])=>gate(s, ()=>refSugGrid.sel.size>=m));
gate("#refSuggest", ()=>INST.pid!=null);
// Substructure: clear needs a selection; running needs a partition/class target. (Assign/Reject/Unassign
// intentionally NOT gated — they fall back to the whole sub-cluster when nothing is ticked.)
gate("#subNone", ()=>subGrid.sel.size>=1); gate("#subRun", ()=>INST.pid!=null);
// Rejected bin: unreject + clear need a selection.
gate("#rjUnreject", ()=>rjGrid.sel.size>=1); gate("#rjNone", ()=>rjGrid.sel.size>=1);
// Undo/redo: disabled when the server's undo/redo stack is empty (depths come back in stats.undo/redo).
gate("#undoBtn", ()=>(window._undoN||0)>0); gate("#redoBtn", ()=>(window._redoN||0)>0);
refreshGates();

refreshState();
