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
  el.onclick = (e)=>{ const c=e.target.closest(".cell"); if(!c) return; const u=c.dataset.iuid;
    if(sel.has(u)){sel.delete(u);c.classList.remove("sel");} else {sel.add(u);c.classList.add("sel");} upd(); };
  return {
    sel, el,
    reset(){ el.innerHTML=""; sel.clear(); upd(); },
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
};

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
$("#mergeBtn").onclick=async()=>{ if(pGrid.sel.size<2)return; const iu=[...pGrid.sel]; await post("/api/merge",{iuids:iu}); selectPartition(INST.pid); loadPartitions(true); };
$("#toRefineBtn").onclick=()=>{ const u=[...pGrid.sel][0]; if(!u)return; $("#rfIuid").value=u; $('nav button[data-tab="refine"]').click(); };
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
function renderChain(){ $("#rfChain").innerHTML = "chain: " + (RF_CHAIN.length? RF_CHAIN.map(o=>`<span class=chip>${o.name}</span>`).join("") : "(empty)"); }
$("#rfAdd").onclick=()=>{ RF_CHAIN.push({name:$("#rfOp").value, kw:{}}); renderChain(); };
$("#rfClear").onclick=()=>{ RF_CHAIN=[]; renderChain(); };
$("#rfPreview").onclick=async()=>{ const iuid=$("#rfIuid").value.trim(); if(!iuid)return;
  const r=await post("/api/refine_preview",{iuid, ops:RF_CHAIN});
  $("#rfBA").innerHTML=`<figure><figcaption>before</figcaption><img src="${r.before}"></figure><figure><figcaption>after (${RF_CHAIN.map(o=>o.name).join("→")||'no ops'})</figcaption><img src="${r.after}"></figure>`; };
$("#rfApply").onclick=async()=>{ const iuid=$("#rfIuid").value.trim(); if(!iuid)return; const r=await post("/api/apply_refine",{iuid,ops:RF_CHAIN}); setStatus(r.stats); if(INST.pid)selectPartition(INST.pid); alert("refined "+iuid.slice(0,6)); };
$("#rfSplit").onclick=async()=>{ const iu=[...pGrid.sel]; if(!iu.length){alert("select instances in Partitions first");return;} const r=await post("/api/split",{iuids:iu}); setStatus(r.stats); loadPartitions(true); if(INST.pid)selectPartition(INST.pid); alert(`split → ${r.n} new instances (re-cluster to see them in partitions)`); };

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

// ---------- Config / sampling ----------
$("#smplBtn").onclick=async()=>{ $("#smplStatus").textContent="sampling (loading model)…";
  const r=await post("/api/sample",{n:+$("#smplN").value, smart:$("#smplSmart").checked});
  $("#smplStatus").textContent=`done — ${JSON.stringify(r.info||{})}`; await refreshState(); };

// ---------- global mask shortcut ('m') + crop/in-context view toggles ----------
document.addEventListener("keydown", e=>{
  const tn=e.target.tagName;
  if(tn==="INPUT"||tn==="TEXTAREA"||tn==="SELECT"||e.target.isContentEditable) return;
  if(e.key==="m"){ MASKS=!MASKS; const cb=$("#ovMasks"); if(cb) cb.checked=MASKS; refreshVisibleCrops(); }
});
$$(".viewToggle").forEach(b=> b.onclick=()=>{ VIEW = VIEW==="crop"?"context":"crop"; syncViewButtons(); refreshVisibleCrops(); });
syncViewButtons();

refreshState();
