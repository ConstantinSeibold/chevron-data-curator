// qseg curator — custom frontend logic. Loads only windowed JSON + lazy per-instance crops, so
// responsiveness is independent of instance/partition count.
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const api = async (u, o) => (await fetch(u, o)).json();
const post = (u, b) => api(u, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(b||{})});
const enc = encodeURIComponent;

function setStatus(s){ if(!s) return; $("#status").textContent =
  `${s.n_instances} inst · ${s.n_assigned} assigned · ${s.n_unassigned} unassigned · ${s.n_background} rejected · ${s.n_classes} classes`; }
function setClasses(cls){ if(cls) $("#classList").innerHTML = cls.map(c=>`<option value="${c}">`).join(""); }

// ---------- global view state: masks on/off ('m' shortcut) + crop vs in-context ----------
let MASKS = true, VIEW = "crop";                    // VIEW: "crop" (bbox) | "context" (whole image)
function cropUrl(iuid){ return `/api/crop?iuid=${enc(iuid)}&max_side=256&mask=${MASKS?1:0}&context=${VIEW==='context'?1:0}`; }
function refreshVisibleCrops(){                     // re-point img src in the ACTIVE tab (no JSON re-fetch)
  const tab = document.querySelector(".tab.active"); if(!tab) return;
  tab.querySelectorAll(".cell img").forEach(img=>{ img.src = cropUrl(img.closest(".cell").dataset.iuid); });
  if(tab.id==="tab-inimage") reloadOverlay();
  if(tab.id==="tab-refine") rfDoPreview();          // before/after are not .cell imgs → re-render with the mask flag
}
function syncViewButtons(){ $$(".viewToggle").forEach(b=> b.textContent = `view: ${VIEW}`); }

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
  if(b.dataset.tab==="inimage" && !$("#imgSelect").options.length) populateImages("");
  if(b.dataset.tab==="stats") loadStats();
  if(b.dataset.tab==="refine" && !$("#rfFind").dataset.loaded){ rfFind(""); $("#rfFind").dataset.loaded="1"; }
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
  setStatus(st.stats); setClasses(st.classes);
  window._features = st.features;
  $("#feats").innerHTML = st.features.map(f=>`<label><input type=checkbox class=feat value="${f}" ${f=='decoder'?'checked':''}>${f}</label>`).join("");
  $("#levelSel").innerHTML = st.levels.map(l=>`<option value="${l.i}" ${l.i===st.level?'selected':''}>L${l.i} (${l.n})</option>`).join("");
  syncClfFeats();
  if(st.clustered) loadPartitions(true);
}
$("#clusterBtn").onclick = async ()=>{
  const feats=$$(".feat:checked").map(e=>e.value); $("#status").textContent="clustering…";
  const r=await post("/api/cluster",{features:feats});
  if(r.detail){ alert(r.detail); } await refreshState();
};
$("#levelSel").onchange = async e=>{ await post("/api/level",{level:+e.target.value}); loadPartitions(true); };
$("#exportBtn").onclick = async ()=>{ const r=await post("/api/export",{}); alert("Exported COCO → "+r.path); };
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
  const r=await post("/api/match_image",{image:dataURL, feature:"roialign", k:12});
  if(r.error){ $("#matchResults").innerHTML=`<div class=muted style="padding:6px;color:var(--warn)">${r.error}</div>`; return; }
  $("#matchResults").innerHTML=`<div style="color:var(--mut);font-size:11px;padding:2px">top matches (detected score ${r.query_score}) — click → its partition:</div>`+
    r.matches.map(m=>`<div class="mrow" data-pid="${m.pid||''}"><img src="${m.crop}"><span>${m.pid||'(rejected/merged)'}<br><small>cos ${m.score}</small></span></div>`).join("");
  const top=r.matches.find(m=>m.pid); if(top){ $("#search").value=top.pid; PART.query=top.pid; loadPartitions(true).then(()=>selectPartition(top.pid)); }
};
$("#matchResults").onclick=e=>{ const row=e.target.closest(".mrow"); if(row&&row.dataset.pid){ $("#search").value=row.dataset.pid; PART.query=row.dataset.pid; loadPartitions(true).then(()=>selectPartition(row.dataset.pid)); } };
$("#toInimgBtn").onclick=async()=>{ const img=pGrid.firstSelImg(); if(!img)return;
  $('nav button[data-tab="inimage"]').click();
  await populateImages(String(img));
  if(![...$("#imgSelect").options].some(o=>o.value===String(img))) $("#imgSelect").insertAdjacentHTML("afterbegin",`<option value="${img}">${img}</option>`);
  $("#imgSelect").value=String(img); loadImage(true); };

// ---------- In-image ----------
let IIMG={id:null,offset:0,limit:120,total:0};
const iiGrid = makeGrid("#iigrid","#iiSelCount");
async function populateImages(query=""){            // windowed image picker (most-populated first)
  const r=await api(`/api/images?query=${enc(query)}&limit=200`);
  $("#imgSelect").innerHTML = r.items.map(it=>`<option value="${it.image_id}">${it.image_id} (${it.n})</option>`).join("");
}
function reloadOverlay(){ if(IIMG.id) $("#ovImg").src=`/api/image_overlay?image_id=${enc(IIMG.id)}&color_by=${$("#ovColor").value}&masks=${MASKS?1:0}&_=${Date.now()}`; }
async function loadImage(reset=true){
  const id=$("#imgSelect").value; if(!id)return; IIMG.id=id; reloadOverlay();
  if(reset){ IIMG.offset=0; iiGrid.reset(); $("#iiPrevWrap").style.display="none"; }
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
  threshold:  [{k:"val",label:"val",def:128,step:4,min:0,max:255}],
  top_k_cc:   [{k:"k",label:"k",def:2,step:1,min:1,max:10}],
  magic_wand: [{k:"tol",label:"tol",def:0.08,step:0.01,min:0,max:1}],
  grabcut:    [{k:"iters",label:"iters",def:5,step:1,min:1,max:20}],
  snap_edges: [{k:"iters",label:"iters",def:20,step:5,min:1,max:200}],
};
const OP_HINT = {
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
$("#rfAdd").onclick=()=>{ RF_CHAIN.push({name:$("#rfOp").value, kw:readRfKw(), on:true}); renderChain(); };
$("#rfClear").onclick=()=>{ RF_CHAIN=[]; renderChain(); };
async function rfDoPreview(){ const iuid=$("#rfIuid").value.trim(); if(!iuid)return; const ops=activeOps();
  const r=await post("/api/refine_preview",{iuid, ops, mask:MASKS?1:0});
  if(r.detail){ $("#rfBA").innerHTML=`<div class="muted" style="color:var(--warn)">${r.detail}</div>`; return; }
  $("#rfBA").innerHTML=`<figure><figcaption>before</figcaption><img src="${r.before}"></figure><figure><figcaption>after (${ops.map(o=>o.name).join("→")||'no ops'})</figcaption><img src="${r.after}"></figure>`;
  if(ops.some(o=>o.name==="sam")){ const f=await rfSamPointsFigure(); if(f) $("#rfBA").insertAdjacentHTML("beforeend", f); } }
$("#rfPreview").onclick=rfDoPreview;
// SAM prompt visualisation: where the +/- points and box come from (green=positive on the skeleton,
// red=negative on the ring, yellow=box). Pure geometry → works even before a checkpoint is downloaded.
async function rfSamPointsFigure(){
  const iuid=$("#rfIuid").value.trim(); if(!iuid) return "";
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
$("#rfSplit").onclick=async()=>{ const iu=[...pGrid.sel]; if(!iu.length){alert("select instances in Partitions first");return;} const r=await post("/api/split",{iuids:iu}); setStatus(r.stats); loadPartitions(true); if(INST.pid)selectPartition(INST.pid); alert(`split → ${r.n} new instances (re-cluster to see them in partitions)`); };
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
  const s=await api("/api/sam_status");
  $("#rfSamMsg").textContent = s.ckpt ? `SAM ready: ${s.model_type} · ${s.ckpt}`
    : (s.installed ? "no checkpoint yet — download one ↓" : "the `segment-anything` package is not installed (pip install segment-anything)");
  $("#rfSamSetup").style.display = s.ckpt ? "none" : "inline-block";
}
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

// ---------- Config / inference on new images ----------
function inferDone(r){ $("#inferStatus").textContent =
  r.error ? `error: ${r.error}` : `done — +${r.n_new_instances??0} instances from ${r.n_new_images??0} image(s). Click Cluster.`;
  refreshState(); }
$("#smplBtn").onclick=async()=>{ $("#inferStatus").textContent="sampling (loading model)…";
  const r=await post("/api/sample",{n:+$("#smplN").value, smart:$("#smplSmart").checked}); inferDone(r.info||r); };
$("#inferDirBtn").onclick=async()=>{ const d=$("#inferDir").value.trim(); if(!d)return;
  $("#inferStatus").textContent="running inference on folder (loading model)…";
  inferDone(await post("/api/infer_dir",{dir:d, limit:+$("#inferLimit").value})); };
$("#inferUploadBtn").onclick=async()=>{ const fs=[...$("#inferFiles").files]; if(!fs.length){ $("#inferStatus").textContent="pick image files first"; return; }
  $("#inferStatus").textContent=`uploading ${fs.length} image(s), running inference…`;
  const imgs=await Promise.all(fs.map(f=>new Promise(res=>{const r=new FileReader(); r.onload=()=>res(r.result); r.readAsDataURL(f);})));
  inferDone(await post("/api/infer_upload",{images:imgs})); };

// ---------- global mask shortcut ('m') + crop/in-context view toggles ----------
document.addEventListener("keydown", e=>{
  const tn=e.target.tagName;
  if(tn==="INPUT"||tn==="TEXTAREA"||tn==="SELECT"||e.target.isContentEditable) return;
  if(e.key==="m"){ MASKS=!MASKS; const cb=$("#ovMasks"); if(cb) cb.checked=MASKS; refreshVisibleCrops(); }
});
$$(".viewToggle").forEach(b=> b.onclick=()=>{ VIEW = VIEW==="crop"?"context":"crop"; syncViewButtons(); refreshVisibleCrops(); });
syncViewButtons();

refreshState();
