// qseg curator — custom frontend logic. Loads only windowed JSON + lazy per-instance crops, so
// responsiveness is independent of instance/partition count.
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const api = async (u, o) => (await fetch(u, o)).json();
const post = (u, b) => api(u, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(b||{})});
const enc = encodeURIComponent;

function setStatus(s){ if(!s) return; $("#status").textContent =
  `${s.n_instances} inst · ${s.n_assigned} assigned · ${s.n_unassigned} unassigned · ${s.n_background} rejected · ${s.n_classes} classes`; }
const escAttr = s => String(s).replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;");
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
function refreshVisibleCrops(){                     // re-point img src in the ACTIVE tab (no JSON re-fetch)
  const tab = document.querySelector(".tab.active"); if(!tab) return;
  tab.querySelectorAll(".cell img").forEach(img=>{ img.src = cropUrl(img.closest(".cell").dataset.iuid); });
  if(tab.id==="tab-inimage") reloadOverlay();
  if(tab.id==="tab-refine") rfDoPreview();          // before/after are not .cell imgs → re-render with the mask flag
}
function syncViewButtons(){ $$(".viewToggle").forEach(b=> b.textContent = `view: ${VIEW} (c)`); }

// ---------- reusable selectable image grid ----------
function cell(it, cap){
  return `<div class="cell" data-iuid="${it.iuid}" data-img="${it.image_id??''}">`+
    `<img loading="lazy" src="${cropUrl(it.iuid)}">`+
    `<div class="cap" title="${cap}">${cap}</div></div>`;
}
function makeGrid(gridSel, countSel, noun="selected"){
  const el = $(gridSel), sel = new Set();
  const upd = ()=>{ if(countSel) $(countSel).textContent = `${sel.size} ${noun}`; };
  // Selection: click to toggle · press-and-DRAG to paint a run (first cell sets direction:
  // press an UNselected cell to sweep-select, a selected one to sweep-deselect) · SHIFT+click
  // to select the whole run between the last click (anchor) and the shift-clicked cell.
  let dragging = false, paintSel = true, anchor = null;
  const cells = ()=>[...el.querySelectorAll(".cell")];
  const setSel = (c, on)=>{ const u = c.dataset.iuid;
    if(on){ if(!sel.has(u)){ sel.add(u); c.classList.add("sel"); } }
    else  { if(sel.has(u)){ sel.delete(u); c.classList.remove("sel"); } } };
  el.addEventListener("mousedown", e=>{ const c = e.target.closest(".cell"); if(!c) return; e.preventDefault();
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
    append(items, capFn){ el.insertAdjacentHTML("beforeend", items.map(it=>cell(it, capFn?capFn(it):it.caption)).join("")); },
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
  if(b.dataset.tab==="refine"){ loadClassRules(); if(!$("#rfFind").dataset.loaded){ rfFind(""); $("#rfFind").dataset.loaded="1"; } }
};

// ---------- Statistics ----------
async function loadStats(){
  $("#statsBody").innerHTML = "<div class='muted'>computing…</div>";
  const s = await api("/api/statistics"), o = s.overview;
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

// ---------- state / cluster / undo ----------
async function refreshState(){
  const st = await api("/api/state");
  setStatus(st.stats); setClasses(st.classes); loadTxLeaves();
  window._modelcfg = st.model_config; window._modelckpt = st.model_ckpt;
  $("#levelSel").innerHTML = st.levels.map(l=>`<option value="${l.i}" ${l.i===st.level?'selected':''}>L${l.i} (${l.n})</option>`).join("");
  refreshFeatures(st.features);                    // builds #feats + all selectors + the Config readout
  loadIngests();
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
  $("#resetMsg").textContent="resetting…";
  const r = await post("/api/reset",{confirm:true});
  if(!r.ok){ $("#resetMsg").innerHTML=`<span style="color:var(--warn)">${r.detail||'reset failed'}</span>`; return; }
  await refreshState(); loadPartitions(true); if(typeof pGrid!=='undefined') pGrid.reset();
  $("#resetMsg").innerHTML='<b>done</b> — project emptied (config kept). Sample &amp; extract to start again.';
};
// single source of truth for the feature selectors: rebuild #feats (cluster) + classifier/sub/merge-rec
// from window._features, and show what's available (so computed embeddings like raddino are visible).
function refreshFeatures(list){
  if(list) window._features = list;
  const fs = window._features || [];
  $("#feats").innerHTML = fs.map(f=>`<label><input type=checkbox class=feat value="${f}" ${f=='decoder'?'checked':''}>${f}</label>`).join("");
  if($("#cfgFeatList")) $("#cfgFeatList").innerHTML = "available features: "+(fs.length?fs.map(f=>`<code>${f}</code>`).join(" · "):"— (Sample &amp; extract first)");
  syncClfFeats(); syncMrFeats(); syncSubFeats();
}
$("#plRun").onclick=async()=>{
  const body={dir:$("#plDir").value.trim(), shard_size:+$("#plShard").value, method:$("#plMethod").value,
    thresh:+$("#plThresh").value, pool:$("#plPool").value, class_agnostic:$("#plAgnostic").checked,
    limit:($("#plLimit").value.trim()?+$("#plLimit").value:null), ...inferThr()};
  $("#plStatus").textContent="sharded pseudo-labeling (loading model)…";
  const r=await withProgress("#plBar","#plStatus",()=>post("/api/scaled_pseudolabel",body));
  if(r.error||r.detail){ $("#plStatus").innerHTML=`<span style="color:var(--warn)">${r.error||r.detail}</span>`; return; }
  $("#plStatus").innerHTML=`done: <b>${r.n_images}</b> imgs · <b>${r.n_instances}</b> instances · <b>${r.n_labeled}</b> labeled (${r.method}) · ${r.shards} shards → <code>${r.merged.path}</code> (${r.merged.annotations} anns, ${r.merged.categories} classes)`; };
$("#cfgRaddino").onclick=async()=>{
  $("#cfgRaddinoMsg").textContent="extracting RAD-DINO embeddings (one RAD-DINO pass per image, GPU)…";
  const r=await withProgress("#raddinoBar","#cfgRaddinoMsg",()=>post("/api/compute_raddino",{force:$("#cfgRaddinoForce").checked, pool:$("#cfgRaddinoPool").value}));
  if(r.error||r.detail){ $("#cfgRaddinoMsg").innerHTML=`<span style="color:var(--warn)">${r.error||r.detail}</span>`; return; }
  refreshFeatures(r.available);
  $("#cfgRaddinoMsg").innerHTML=`RAD-DINO ready for <b>${r.n||'all'}</b> instances — <code>raddino</code> is now selectable everywhere.`; };
$("#clusterBtn").onclick = async ()=>{
  const feats=$$(".feat:checked").map(e=>e.value); $("#status").textContent="clustering…";
  const r=await post("/api/cluster",{features:feats});
  if(r.detail){ alert(r.detail); } await refreshState();
};
$("#levelSel").onchange = async e=>{ await post("/api/level",{level:+e.target.value}); loadPartitions(true); };
$("#exportBtn").onclick = async ()=>{
  const r=await post("/api/export",{partial:$("#expPartial").checked, class_agnostic:$("#expAgnostic").checked});
  const s=r.stats||{}, kind=(r.partial?"partial-label":"curated")+(r.class_agnostic?", class-agnostic":"");
  alert(`Exported ${kind} COCO → ${r.path}`+(r.partial?`\n\n${s.n_assigned} positives · ${s.n_unassigned} ignore (unreviewed) · ${s.n_background} rejected→background`:"")); };
async function doUndo(which){ const r=await post(`/api/${which}`,{}); setStatus(r.stats); setClasses(r.classes); loadPartitions(true); if(INST.pid) selectPartition(INST.pid); }
$("#undoBtn").onclick=()=>doUndo("undo"); $("#redoBtn").onclick=()=>doUndo("redo");

// ---------- Partitions ----------
let PART={offset:0,limit:100,total:0,query:""}, INST={pid:null,offset:0,limit:60,total:0};
const pGrid = makeGrid("#pgrid", "#pSelCount");
async function loadPartitions(reset){
  if(reset){ PART.offset=0; $("#plist").innerHTML=""; }
  const r=await api(`/api/partitions?offset=${PART.offset}&limit=${PART.limit}&query=${enc(PART.query)}`);
  PART.total=r.total;
  $("#pcount").textContent=`${r.total} partitions${r.total>PART.limit?` (showing ${Math.min(PART.offset+PART.limit,r.total)})`:''}`;
  $("#plist").insertAdjacentHTML("beforeend", r.rows.map(p=>
    `<div class="prow" data-pid="${p.pid}"><span>${p.pid}${p.cls?` <span class=cls>[${p.cls}]</span>`:''}</span><span class="sz">${p.size} · ${p.score??''}</span></div>`).join(""));
  PART.offset+=r.rows.length;
  $("#pmore").style.display = PART.offset<r.total?"inline-block":"none";
}
async function selectPartition(pid){
  INST.pid=pid; INST.offset=0; pGrid.reset();
  $$(".prow").forEach(e=>e.classList.toggle("sel", e.dataset.pid===pid));
  await loadInstances(true);
}
async function loadInstances(reset){
  if(!INST.pid) return; if(reset) INST.offset=0;
  const r=await api(`/api/instances?pid=${enc(INST.pid)}&offset=${INST.offset}&limit=${INST.limit}`);
  INST.total=r.total;
  if(reset && !r.items.length){ pGrid.msg("(empty — assign/reject emptied this partition)"); }
  else pGrid.append(r.items);
  INST.offset+=r.items.length;
  $("#imore").style.display = INST.offset<r.total?"inline-block":"none";
}
async function afterMut(resp, dropped, grid){ setStatus(resp.stats); setClasses(resp.classes); if(dropped) grid.drop(dropped); loadPartitions(true); }
$("#search").oninput=e=>{ PART.query=e.target.value; clearTimeout(window._st); window._st=setTimeout(()=>loadPartitions(true),200); };
$("#pmore").onclick=()=>loadPartitions(false);
$("#imore").onclick=()=>loadInstances(false);
$("#plist").onclick=e=>{ const r=e.target.closest(".prow"); if(r) selectPartition(r.dataset.pid); };
$("#selAll").onclick=()=>pGrid.selectPage(); $("#selNone").onclick=()=>pGrid.clearSel();
$("#assignBtn").onclick=async()=>{ const cls=$("#classInput").value.trim(); if(!cls||!pGrid.sel.size)return; const iu=[...pGrid.sel]; afterMut(await post("/api/assign",{iuids:iu,cls}),iu,pGrid); };
$("#assignAllBtn").onclick=async()=>{ const cls=$("#classInput").value.trim(); if(!cls||!INST.pid)return;
  const all=await api(`/api/instances?pid=${enc(INST.pid)}&offset=0&limit=1000000`); const iu=all.items.map(i=>i.iuid);
  afterMut(await post("/api/assign",{iuids:iu,cls}),iu,pGrid); };
$("#rejectBtn").onclick=async()=>{ if(!pGrid.sel.size)return; const iu=[...pGrid.sel]; afterMut(await post("/api/reject",{iuids:iu}),iu,pGrid); };
$("#unassignBtn").onclick=async()=>{ if(!pGrid.sel.size)return; const iu=[...pGrid.sel]; afterMut(await post("/api/unassign",{iuids:iu}),iu,pGrid); };
$("#mergeBtn").onclick=async()=>{ if(pGrid.sel.size<2)return; const iu=[...pGrid.sel];
  const r=await post("/api/merge",{iuids:iu});
  if(!r.n_groups){ alert("nothing merged — merge only combines instances from the SAME image (the selection spans different images, or no image had ≥2 selected)."); return; }
  setStatus(r.stats); selectPartition(INST.pid); loadPartitions(true); };
$("#toRefineBtn").onclick=()=>{ const u=[...pGrid.sel][0]; if(!u)return; $("#rfIuid").value=u; $('nav button[data-tab="refine"]').click(); rfDoPreview(); };
// find-partition-by-reference-image (NN over the roialign feature space)
$("#matchBtn").onclick=()=>$("#matchFile").click();
$("#matchFile").onchange=async e=>{ const f=e.target.files[0]; if(!f)return; e.target.value="";
  const dataURL=await new Promise(res=>{ const r=new FileReader(); r.onload=()=>res(r.result); r.readAsDataURL(f); });
  $("#matchResults").innerHTML="<div class=muted style='padding:6px'>matching (running model on the upload)…</div>";
  const r=await post("/api/match_image",{image:dataURL, feature:"roialign", k:8});
  if(r.error){ $("#matchResults").innerHTML=`<div class=muted style="padding:6px;color:var(--warn)">${r.error}</div>`; return; }
  const row=m=>`<div class="mrow" data-pid="${m.pid||''}"><img src="${m.crop}"><span>${m.cls?('<b>'+m.cls+'</b>'):(m.pid||'(rejected/merged)')}<br><small>cos ${m.score}</small></span></div>`;
  const sec=(title,arr)=> arr&&arr.length ? `<div style="color:var(--mut);font-size:11px;padding:4px 2px 2px">${title}</div>`+arr.map(row).join("") : "";
  // show BOTH matching CLASSES and matching UNANNOTATED partitions (classes alone crowd out the pool)
  $("#matchResults").innerHTML=`<div style="color:var(--mut);font-size:11px;padding:2px">detected score ${r.query_score} — click a row → its partition:</div>`+
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
const iiGrid = makeGrid("#iigrid","#iiSelCount");
async function populateImages(query=""){            // windowed image picker (most-populated first)
  const r=await api(`/api/images?query=${enc(query)}&limit=200`);
  $("#imgSelect").innerHTML = r.items.map(it=>`<option value="${it.image_id}">${it.image_id} (${it.n})</option>`).join("");
}
function reloadOverlay(){ if(IIMG.id) $("#ovImg").src=`/api/image_overlay?image_id=${enc(IIMG.id)}&color_by=${$("#ovColor").value}&masks=${MASKS?1:0}&_=${Date.now()}`; }
async function loadImage(reset=true){
  const id=$("#imgSelect").value; if(!id)return; IIMG.id=id; reloadOverlay();
  if(reset){ IIMG.offset=0; iiGrid.reset(); $("#iiPrevWrap").style.display="none";
             $("#iiRecCards").innerHTML=""; $("#iiRecMsg").textContent=""; IIREC.cands=[]; }   // clear stale per-image merge suggestions
  const r=await api(`/api/image_instances?image_id=${enc(id)}&offset=${IIMG.offset}&limit=${IIMG.limit}`);
  IIMG.total=r.total; if(reset && !r.items.length) iiGrid.msg("(no instances on this image)"); else iiGrid.append(r.items);
  IIMG.offset+=r.items.length; $("#iimore").style.display=IIMG.offset<r.total?"inline-block":"none";
}
$("#imgFilter").oninput=e=>{ clearTimeout(window._if); window._if=setTimeout(()=>populateImages(e.target.value),200); };
$("#imgSelect").onchange=()=>loadImage(true);
$("#ovLoad").onclick=()=>loadImage(true);
$("#ovColor").onchange=reloadOverlay;
$("#ovMasks").onchange=e=>{ MASKS=e.target.checked; refreshVisibleCrops(); };
$("#iimore").onclick=()=>loadImage(false);
async function iiAfter(resp,dropped){ setStatus(resp.stats); setClasses(resp.classes); iiGrid.drop(dropped); reloadOverlay(); loadPartitions(true); }
$("#iiAssign").onclick=async()=>{ const cls=$("#iiClass").value.trim(); if(!cls||!iiGrid.sel.size)return; const iu=[...iiGrid.sel]; iiAfter(await post("/api/assign",{iuids:iu,cls}),iu); };
$("#iiReject").onclick=async()=>{ if(!iiGrid.sel.size)return; const iu=[...iiGrid.sel]; iiAfter(await post("/api/reject",{iuids:iu}),iu); };
$("#iiToRefine").onclick=()=>{ const u=[...iiGrid.sel][0]; if(!u){alert("select an instance");return;}
  $("#rfIuid").value=u; $('nav button[data-tab="refine"]').click(); rfDoPreview(); };
$("#iiMergePrev").onclick=async()=>{ if(iiGrid.sel.size<2){ $("#iiPrevWrap").style.display="none"; return; }
  const r=await post("/api/merge_preview",{iuids:[...iiGrid.sel], mode:$("#iiMergeMode").value});
  if(r.img){ $("#iiPrevImg").src=r.img; $("#iiPrevWrap").style.display="block"; } };
$("#iiMerge").onclick=async()=>{ if(iiGrid.sel.size<2)return; const iu=[...iiGrid.sel];
  await post("/api/merge",{iuids:iu, mode:$("#iiMergeMode").value}); $("#iiPrevWrap").style.display="none"; loadImage(true); loadPartitions(true); };

// ---------- Refine ----------
let RF_CHAIN=[];
// per-op tunable parameters (rendered next to the op picker; captured into the op's kw on "+ add op")
const OP_PARAMS = {
  vessel_extend: [{k:"high",label:"seed",def:0.7,step:0.05,min:0,max:3},{k:"low",label:"grow",def:0.4,step:0.05,min:0,max:3},
                  {k:"max_gap",label:"gap",def:40,step:5,min:0,max:300},{k:"max_width",label:"width",def:8,step:1,min:1,max:40}],
  sam:        [{k:"n_pos",label:"+pts",def:10,step:1,min:1,max:60},{k:"n_neg",label:"−pts",def:12,step:1,min:0,max:60},
               {k:"margin",label:"neg-gap",def:24,step:2,min:2,max:80},{k:"mask_prior",label:"mask-prior",def:1,step:1,min:0,max:1},
               {k:"keep",label:"keep∪",def:0,step:1,min:0,max:1}],
  dilate:     [{k:"k",label:"k",def:3,step:1,min:1,max:25},{k:"max_contrast",label:"maxΔ",def:0.15,step:0.02,min:0,max:1}],
  erode:      [{k:"k",label:"k",def:3,step:1,min:1,max:25},{k:"min_contrast",label:"minΔ",def:0.15,step:0.02,min:0,max:1}],
  contrast:   [{k:"clip",label:"clip",def:2.0,step:0.5,min:1,max:10}],
  threshold:  [{k:"val",label:"val",def:128,step:4,min:0,max:255}],
  top_k_cc:   [{k:"k",label:"k",def:2,step:1,min:1,max:10}],
  magic_wand: [{k:"tol",label:"tol",def:0.08,step:0.01,min:0,max:1}],
  grabcut:    [{k:"iters",label:"iters",def:5,step:1,min:1,max:20}],
  snap_edges: [{k:"iters",label:"iters",def:20,step:5,min:1,max:200}],
};
const OP_HINT = {
  contrast: "local contrast (CLAHE) on the image the LATER ops see — add it FIRST, then threshold/vessel/sam. Higher clip = stronger. The preview shows the enhanced image.",
  vessel_extend: "tune per image: raise seed/grow and lower gap if it over-extends; raise width for thick tubes.",
  sam: "boundary-free refine: result REPLACES the mask (can shrink+grow); SAM's best of several proposals is taken. keep∪=1 unions with the original (never shrinks); if it still echoes the input, set mask-prior=0. Compact parts > thin shafts.",
};
function renderRfParams(){
  const op=$("#rfOp").value, ps=OP_PARAMS[op]||[];
  $("#rfParams").innerHTML = ps.map(p=>`<label>${p.label} <input class=rfp data-k="${p.k}" type=number value="${p.def}" step="${p.step}" min="${p.min}" max="${p.max}"></label>`).join("");
  $("#rfHint").textContent = OP_HINT[op]||"";
  $("#rfSamBar").style.display = op==="sam" ? "flex" : "none";
  if(op==="sam") refreshSamStatus();
}
function readRfKw(){ const kw={}; $$("#rfParams .rfp").forEach(i=>{ kw[i.dataset.k]=+i.value; }); return kw; }
$("#rfOp").onchange=renderRfParams;
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
async function rfDoPreview(){ const iuid=$("#rfIuid").value.trim(); if(!iuid)return; const ops=activeOps();
  const r=await post("/api/refine_preview",{iuid, ops, mask:MASKS?1:0});
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
  const r=await post("/api/sam_prompt_preview",{iuid, ops:activeOps(), n_pos:kw.n_pos??10, n_neg:kw.n_neg??12, margin:kw.margin??24});
  if(r.detail) return "";
  return `<figure><figcaption>SAM prompts — <b style="color:#2dd24d">●</b> ${r.n_pos} pos (interior) · <b style="color:#eb4a3d">●</b> ${r.n_neg} neg (beyond ${kw.margin??24}px gap) · <b style="color:#ffd000">▭</b> box · rim left free</figcaption><img src="${r.img}"></figure>`; }
$("#rfSamPts").onclick=async()=>{ const f=await rfSamPointsFigure();
  $("#rfBA").innerHTML = f || `<div class="muted">pick an instance first</div>`; };
$("#rfApply").onclick=async()=>{ const iuid=$("#rfIuid").value.trim(); if(!iuid)return;
  const r=await post("/api/apply_refine",{iuid,ops:activeOps()});
  if(r.detail){ alert(r.detail); return; }
  setStatus(r.stats); if(INST.pid)selectPartition(INST.pid); $("#rfHint").textContent=`refined ${iuid.slice(0,6)} ✓`; };
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
  const r=await post("/api/apply_refine_partition",{pid:INST.pid, ops});
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
// instance picker / search (iuids are opaque → search by file / class / image-id / iuid-prefix)
async function rfFind(q=""){
  const r=await api(`/api/find_instances?query=${enc(q)}&limit=60`);
  $("#rfFindCount").textContent = `${r.total}${r.total>=60?"+":""} match${r.total===1?"":"es"} — click one to refine`;
  $("#rfIuidList").innerHTML = r.items.map(it=>`<option value="${it.iuid}">`).join("");
  $("#rfFind").innerHTML = r.items.length ? r.items.map(it=>cell(it,it.caption)).join("") : `<div class="muted">no matches</div>`;
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
  let msg;
  if(!s.installed) msg = "the `segment-anything` package is not installed (pip install segment-anything)";
  else if(s.ckpt) msg = `${(s.family||"sam")==="medsam"?"MedSAM":"SAM"} ready: ${s.model_type} · ${s.ckpt.split("/").pop()}`;
  else if(fam==="medsam") msg = "no MedSAM checkpoint — drop a *medsam*.pth in CURATOR_SAM_DIR or set CURATOR_MEDSAM_CKPT (not auto-downloadable)";
  else msg = "no checkpoint yet — download SAM ↓";
  if(fam==="medsam") msg += " · box-prompt, medical-tuned (points ignored)";
  $("#rfSamMsg").textContent = msg;
  $("#rfSamSetup").style.display = (s.installed && !has("sam")) ? "inline-block" : "none";   // setup downloads VANILLA SAM
}
$("#rfSamModel").onchange = refreshSamStatus;
$("#rfSamSetup").onclick=async()=>{ $("#rfSamMsg").textContent="downloading SAM checkpoint (~375 MB), one-time…";
  const r=await post("/api/sam_setup",{});
  if(r.detail){ $("#rfSamMsg").textContent="error: "+r.detail; return; }
  $("#rfSamMsg").textContent=`SAM ready: ${r.ckpt}`; $("#rfSamSetup").style.display="none"; };
renderRfParams();

// ---------- Classifier ----------
let CLF={offset:0,limit:60,total:0};
const clfGrid = makeGrid("#clfgrid","#clfExclCount","excluded");
function syncClfFeats(){ if(!window._features)return;
  $("#clfFeats").innerHTML = window._features.map(f=>`<label><input type=checkbox class=clffeat value="${f}" ${(f=='decoder'||f=='shape')?'checked':''}>${f}</label>`).join(""); }
$("#clfTrain").onclick=async()=>{
  const feats=$$(".clffeat:checked").map(e=>e.value);
  $("#clfReport").textContent="training…";
  const r=await post("/api/train_classifier",{features:feats, algo:$("#clfAlgo").value, openset:$("#clfOpen").checked});
  if(!r.ok){ $("#clfReport").innerHTML=`<span style="color:var(--warn)">${r.error||'train failed'}</span>`+(r.skipped?.length?` · skipped: ${r.skipped.join(", ")}`:""); return; }
  const yd=Object.entries(r.youden||{}).map(([k,v])=>`${k}: ${v}`).join(" · ");
  $("#clfReport").innerHTML=`trained <b>${r.algo}</b> on ${r.n_classes} classes: ${r.classes.join(", ")}`+
    (r.skipped?.length?` · skipped (&lt;2): ${r.skipped.join(", ")}`:"")+(yd?`<br>recommended thresholds (Youden J): ${yd}`:""); };
$("#clfThr").oninput=e=>$("#clfThrV").textContent=(+e.target.value).toFixed(2);
async function clfLoad(reset){ if(reset){CLF.offset=0;clfGrid.reset();}
  const r=await api(`/api/predict?thresh=${$("#clfThr").value}&only_class=${enc($("#clfOnly").value.trim())}&offset=${CLF.offset}&limit=${CLF.limit}`);
  CLF.total=r.total;
  if(reset && !r.items.length) clfGrid.msg("no unassigned instances pass this threshold");
  else clfGrid.append(r.items, it=>`${it.cls} · ${it.conf}`);
  CLF.offset+=r.items.length; $("#clfMore").style.display=CLF.offset<r.total?"inline-block":"none";
  if(reset) $("#clfReport").insertAdjacentHTML("beforeend", ` — <b>${r.total}</b> would be assigned (tick crops to EXCLUDE).`); }
$("#clfPredict").onclick=()=>clfLoad(true);
$("#clfMore").onclick=()=>clfLoad(false);
$("#clfApply").onclick=async()=>{
  const r=await post("/api/apply_predictions",{thresh:+$("#clfThr").value, only_class:$("#clfOnly").value.trim(), exclude:[...clfGrid.sel]});
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
const clfRejGrid = makeGrid("#clfRejGrid","#clfRejSelCount");
$("#clfRejThr").oninput=e=>$("#clfRejThrV").textContent=(+e.target.value).toFixed(2);
async function clfRejLoad(reset){ if(reset){CLFREJ.offset=0;clfRejGrid.reset();}
  const r=await api(`/api/recommend_rejections?max_conf=${$("#clfRejThr").value}&offset=${CLFREJ.offset}&limit=${CLFREJ.limit}`);
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
const clfIntGrid = makeGrid("#clfIntGrid","#clfIntSelCount");
async function clfIntLoad(reset){ if(reset){CLFINT.offset=0;clfIntGrid.reset();}
  const r=await api(`/api/recommend_interesting?metric=${$("#clfIntMetric").value}&n=300&offset=${CLFINT.offset}&limit=${CLFINT.limit}`);
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
  $("#mrFeats").innerHTML = window._features.map(f=>`<label><input type=checkbox class=mrfeat value="${f}" ${(f=='decoder')?'checked':''}>${f}</label>`).join(""); }
// one card per candidate GROUP. Each input instance is an individually toggleable crop (selected by default):
// "Merge selected" merges only the CHECKED subset (the ones that actually belong), leaving the rest alone.
function mergeCardHTML(c){
  const ius=c.iuids||[];
  const crops = ius.slice(0,30).map(u=>`<div class="mccrop sel" data-iuid="${u}"><img loading="lazy" src="${cropUrl(u)}"><div class="mclbl">${u.slice(0,6)}</div></div>`).join("");
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
  const feats=$$(".mrfeat:checked").map(e=>e.value); $("#mrReport").textContent="training…";
  const r=await post("/api/train_merge_recommender",{features:feats, algo:$("#mrAlgo").value});
  if(!r.ok){ $("#mrReport").innerHTML=`<span style="color:var(--warn)">${r.error||'train failed'}</span>`; return; }
  $("#mrReport").innerHTML=`trained from <b>${r.n_merge_events}</b> merge event(s) → <b>${r.n_pos}</b> positive pairs / <b>${r.n_neg}</b> negatives. Recommended P(merge) (Youden J): <b>${r.youden}</b>.`;
  $("#mrThr").value=r.youden; $("#mrThrV").textContent=(+r.youden).toFixed(2); };
$("#mrRec").onclick=async()=>{
  const r=await api(`/api/recommend_merges?thresh=${$("#mrThr").value}`);
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
const refSugGrid = makeGrid("#refSugGrid","#refSugSelCount");
async function refLoadClasses(){ const r=await api("/api/reference/classes");
  if(r.last_coco_path && !$("#refPath").value) $("#refPath").value = r.last_coco_path;  // remembered path
  if(!r.loaded){ $("#refClassSel").innerHTML=`<option>(load a bank first)</option>`; return; }
  $("#refClassSel").innerHTML = r.rows.map(x=>`<option value="${x.cls}">${x.cls} (${x.n})</option>`).join("");
  refShowExemplars(); }
async function refShowExemplars(){ const cls=$("#refClassSel").value; if(!cls) return;
  const r=await api(`/api/reference/exemplars?cls=${enc(cls)}&limit=12`);
  $("#refExemplars").innerHTML = r.items.length
    ? r.items.map(e=>{ const b=e.bbox||[]; const q=b.length===4?`&x=${b[0]}&y=${b[1]}&w=${b[2]}&h=${b[3]}`:"";
        return `<div class="cell"><img loading="lazy" src="/api/reference/exemplar?file_name=${enc(e.file_name)}${q}"><div class="cap">${cls}</div></div>`; }).join("")
    : `<div class="muted">no exemplars</div>`; }
$("#refClassSel").onchange=refShowExemplars;
$("#refLoad").onclick=async()=>{ const p=$("#refPath").value.trim(); if(!p){alert("enter the reference coco.json path");return;}
  $("#refStatus").textContent="loading + embedding references (RAD-DINO, one-time)…";
  const r=await post("/api/reference/load",{coco_path:p});
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
  $("#refSugReport").textContent="embedding instances + matching references…"; refSugGrid.reset(); REFSUG={};
  const r=await post("/api/reference/suggest",{pid:INST.pid, topk:5});
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
const subGrid = makeGrid("#subgrid","#subSelCount");
function syncSubFeats(){ if(!window._features)return;
  $("#subFeats").innerHTML = window._features.map(f=>`<label><input type=checkbox class=subfeat value="${f}" ${(f=='decoder')?'checked':''}>${f}</label>`).join(""); }
$("#subRun").onclick=async()=>{
  if(!INST.pid){ alert("select a partition/class on the Partitions tab first"); return; }
  const feats=$$(".subfeat:checked").map(e=>e.value); if(!feats.length){alert("pick at least one feature");return;}
  $("#subMsg").textContent="training contrastive encoder + FINCH… (this can take a few seconds)";
  const r=await post("/api/subcluster",{target:INST.pid, features:feats, dim:+$("#subDim").value, epochs:+$("#subEpochs").value, temperature:+$("#subTemp").value});
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
      const leaves=c.leaves.map(l=>`<span class="txleaf">${l.name} <span class="n">${l.n}</span></span>`).join(" ") || `<span class="muted" style="font-size:11px">(no annotated parts yet)</span>`;
      const rules=(c.part_rules||[]).map(rl=>`${rl.if}⇒${rl.then.join("+")}`).join(", ");
      h+=`<div class="txcon"><span class="cn">${c.name}</span> <span class="muted">· ${c.n}</span>${c.mimic_family?` <span class="xw">≈${c.mimic_family}</span>`:""}`+
         (c.description?`<div class="desc">${c.description}</div>`:"")+
         (rules?`<div class="desc">rules: ${rules}</div>`:"")+`<div>${leaves}</div></div>`;
    }
    h+=`</div>`;
  }
  $("#txTree").innerHTML = h || `<div class="muted">No taxonomy yet — click "Seed taxonomy".</div>`;
  // temp / scratch bucket (excluded from export) — tick to merge, or promote into a concept
  const opts=`<option value="">— promote to concept —</option>`+TX_CONCEPTS.map(c=>`<option value="${c.id}">${c.name}</option>`).join("");
  $("#txTree").innerHTML += r.temp.length
    ? `<div class="txtemp"><b>Temp / scratch classes</b> <span class="muted">(usable in the tool, NOT exported)</span>`+
      r.temp.map(t=>`<div class="row"><input type=checkbox class=mccls value="${t.name}"> <b>${t.name}</b> <span class="muted">${t.n} inst${t.temp?' · temp':' · ungrouped'}</span>`+
        `<span class="grow"></span><select class="txpromote" data-id="${t.id}">${opts}</select>`+
        `<button class="txtoggle" data-id="${t.id}" data-temp="${t.temp?0:1}">${t.temp?'un-temp':'mark temp'}</button></div>`).join("")+`</div>`
    : "";
}
$("#txRefresh").onclick=loadClasses;
$("#txSeed").onclick=async()=>{ $("#txMsg").textContent="seeding taxonomy…";
  const r=await post("/api/taxonomy/seed",{}); setClasses((await api("/api/state")).classes);
  $("#txMsg").textContent=`taxonomy: ${r.superclasses} superclasses · ${r.concepts} concepts · ${r.leaves} leaves`; loadClasses(); };
$("#txQc").onclick=async()=>{ const r=await api("/api/taxonomy/release_qc");
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
const rjGrid = makeGrid("#rjgrid","#rjSelCount");
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
function inferDone(r){ $("#inferStatus").textContent =
  (r.error||r.detail) ? `error: ${r.error||r.detail}`
  : `done — +${r.n_new_instances??0} instances on ${r.n_new_images??0} image(s)`+((r.n_replaced)?` · ${r.n_replaced} old hidden`:"")+`. Click Cluster.`;
  refreshState(); }
// detection thresholds shared by all inference actions (blank -> server uses the config default)
function inferThr(){ const s=$("#cfgScore").value.trim(), n=$("#cfgNms").value.trim();
  const o={}; if(s!=="")o.score_thresh=+s; if(n!=="")o.nms_iou=+n; return o; }
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
async function withProgress(barSel, statusSel, fn){
  const bar=$(barSel); bar.style.display="block"; bar.classList.add("indet"); $(barSel+" > span").style.width="0%";
  _progTimer=setInterval(()=>_pollOnce(barSel,statusSel), 600);
  try{ return await fn(); }
  finally{ clearInterval(_progTimer); _progTimer=null; bar.style.display="none"; bar.classList.remove("indet"); }
}
$("#smplBtn").onclick=async()=>{ $("#inferStatus").textContent="sampling (loading model)…";
  const r=await withProgress("#inferBar","#inferStatus",()=>post("/api/sample",{n:+$("#smplN").value, smart:$("#smplSmart").checked, ...inferThr()})); inferDone(r.info||r); };
$("#inferDirBtn").onclick=async()=>{ const d=$("#inferDir").value.trim(); if(!d)return;
  $("#inferStatus").textContent="running inference on folder (loading model)…";
  inferDone(await withProgress("#inferBar","#inferStatus",()=>post("/api/infer_dir",{dir:d, limit:+$("#inferLimit").value, mode:$("#inferDirMode").value, ...inferThr()}))); };
$("#prevBtn").onclick=async()=>{ $("#inferStatus").textContent="previewing the model on a random sample (non-destructive)…";
  const r=await withProgress("#inferBar","#inferStatus",()=>post("/api/preview_infer",{n:+$("#prevN").value, ...inferThr()}));
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
  inferDone(await withProgress("#inferBar","#inferStatus",()=>post("/api/reinfer",{mode, limit:lim?+lim:null, ...inferThr()}))); };
$("#inferUploadBtn").onclick=async()=>{ const fs=[...$("#inferFiles").files]; if(!fs.length){ $("#inferStatus").textContent="pick image files first"; return; }
  $("#inferStatus").textContent=`uploading ${fs.length} image(s), running inference…`;
  const imgs=await Promise.all(fs.map(f=>new Promise(res=>{const r=new FileReader(); r.onload=()=>res(r.result); r.readAsDataURL(f);})));
  inferDone(await post("/api/infer_upload",{images:imgs})); };

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

refreshState();
