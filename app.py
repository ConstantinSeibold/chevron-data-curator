"""Gradio web app for the qseg instance curator.

Run:  python -m tools.curator.app  [--project DIR] [--port 7860] [--share]

ONE server-side CuratorEngine (single user, GPU). Tabs: Config | Partitions | In-image |
Refine | Classifier | Map | Export | Rejected.

Selection model (rewritten 2026-06-17): instances render as a @gr.render grid of
image + per-image CHECKBOX (gr.Gallery.select can't reliably toggle/unselect). Checkboxes
drive a small `selected_iuids` gr.State; the grid re-renders only when the partition / view /
a mutation-nonce changes (NOT on every toggle), so checking/unchecking is instant and there is
no separate "selected" window. Refine previews are also @gr.render-driven off refine_target +
op_stack, so they always paint (no cross-tab gallery hand-off to fail).
"""
from __future__ import annotations

import argparse

import cv2
import gradio as gr
import numpy as np

from .engine import CuratorEngine

ENG: CuratorEngine | None = None

CURATOR_JS = """
<script>
(function(){
  function prev(){let p=document.getElementById('map-hover-prev');
    if(!p){p=document.createElement('div');p.id='map-hover-prev';
      p.style.cssText='position:fixed;z-index:99999;pointer-events:none;display:none;border:1px solid #888;background:#fff;padding:2px;box-shadow:0 2px 8px rgba(0,0,0,.3)';
      p.innerHTML='<img id="map-hover-img" style="max-width:150px;max-height:150px;display:block"><div id="map-hover-txt" style="font:11px sans-serif;text-align:center;color:#000"></div>';
      document.body.appendChild(p);}return p;}
  function bind(gd){if(gd.__hoverBound)return;gd.__hoverBound=true;
    gd.on('plotly_hover',function(ev){var cd=ev.points[0].customdata,p=prev();
      var img=document.getElementById('map-hover-img'),txt=document.getElementById('map-hover-txt');
      if(cd&&cd[0]){img.src=cd[0];img.style.display='block';}else{img.style.display='none';}
      txt.textContent=(cd&&cd[1])?cd[1]:'';p.style.display='block';
      p.style.left=(ev.event.clientX+14)+'px';p.style.top=(ev.event.clientY+14)+'px';});
    gd.on('plotly_unhover',function(){prev().style.display='none';});}
  new MutationObserver(function(){document.querySelectorAll('#map_plot .js-plotly-plot').forEach(bind);})
    .observe(document.body,{childList:true,subtree:true});
  var KMAP={'a':'kb_assign','s':'kb_assignsel','r':'kb_reject','u':'kb_unassign',
            'z':'kb_undo','y':'kb_redo','[':'kb_prev',']':'kb_next'};
  document.addEventListener('keydown',function(e){
    var tn=e.target.tagName; if(tn==='INPUT'||tn==='TEXTAREA'||e.target.isContentEditable)return;
    var id=KMAP[e.key]; if(!id)return; var el=document.getElementById(id);
    if(el){e.preventDefault();(el.querySelector('button')||el).click();}});
})();
</script>
"""

_GRID_CAP = 24          # instances shown in a partition/in-image grid at once (kept low for render fluidity)
_REFINE_CAP = 8         # instances previewed in the Refine tab


# --------------------------------------------------------------------------- #
def _defaults() -> dict:
    try:
        from ._bootstrap import get_P
        P = get_P()
        return {"ckpt": str(P.DEFAULT_CKPT), "overrides": list(P.DEFAULT_OVERRIDES),
                "config_name": "experiments/synthfb_arch3", "root": str(P.RANZCR_IMG_ROOT)}
    except Exception:
        return {"ckpt": "", "overrides": [], "config_name": "experiments/synthfb_arch3", "root": ""}


def _status_md() -> str:
    if ENG is None:
        return "**No project open.** Create/open one in the Config tab."
    s = ENG.stats()
    star = " · clustering **stale** (re-cluster)" if s["dirty"] else ""
    return (f"**{s['n_instances']}** instances · **{s['n_assigned']}** assigned · "
            f"**{s['n_unassigned']}** unassigned · **{s['n_background']}** rejected · "
            f"**{s['n_classes']}** classes · undo {s['undo']}/redo {s['redo']}{star}")


def _class_choices():
    return ENG.state.class_names() if ENG else []


def _refresh_classes():
    ch = _class_choices()
    return [gr.update(choices=ch) for _ in range(5)]   # must equal len(class_dds)


def _img_choices():
    return gr.update(choices=[str(i) for i in ENG.image_ids()]) if ENG else gr.update()


_PREF_CLUSTER = ["decoder", "coords"]
_PREF_CLF = ["decoder", "shape"]
_PREF_MERGE = ["decoder", "shape", "coords"]   # geometry matters for "should these two merge?"


def _avail_features():
    return ENG.available_features() if ENG else []


def _avail_md():
    a = _avail_features()
    return ("**available features:** " + ", ".join(a)) if a else "_available features: (none yet — Sample & extract first)_"


def _feat_update(preferred):
    """Choices = features actually present in the collection; value = preferred ∩ present (else all)."""
    a = _avail_features()
    val = [m for m in preferred if m in a] or a
    return gr.update(choices=a, value=val)


def _partition_rows():
    if ENG is None or ENG._cluster is None:
        return []
    return [[str(r["pid"]), r["size"], (round(r["purity"], 2) if r["purity"] is not None else None),
             r["mean_score"], r["majority_class"] or ""] for r in ENG.partition_view()]


def _bump(n) -> int:
    return int(n or 0) + 1


def _img_id(x):
    """Parse an image_id Dropdown value to int, tolerating stray/non-numeric values (the Dropdowns are
    allow_custom_value=True so a misrouted value like 'selected: 0' reaches handlers instead of crashing
    Gradio's preprocess). Returns None when not a usable image id."""
    s = str(x).strip()
    return int(s) if s.isdigit() else None


def _toggle_factory(u: str):
    """Per-checkbox handler: add/remove this iuid from the selection State (reliable un/select)."""
    def _t(checked, cur):
        s = set(cur or [])
        s.add(u) if checked else s.discard(u)
        s = sorted(s)
        return s, f"selected: {len(s)}"
    return _t


# ---- Config ----------------------------------------------------------------
def do_open_project(project_dir, ckpt, config_name, overrides_text, root, score_thr, nms_iou):
    global ENG
    overrides = [ln.strip() for ln in overrides_text.splitlines() if ln.strip()]
    ENG = CuratorEngine(project_dir)
    if not ENG.store.is_project():
        ENG.init_project({"images": {"root": root},
                          "model": {"ckpt": ckpt, "config_name": config_name, "overrides": overrides,
                                    "score_thresh": float(score_thr), "nms_iou": float(nms_iou)},
                          "features": {"model_features": ["decoder", "maskpool", "roialign", "backbone"],
                                       "handcrafted": {"shape": True, "shape_coords_extra": True}, "raddino": False}})
    return (f"Project **{project_dir}** open.\n\n{_status_md()}", _status_md(), _img_choices(),
            _feat_update(_PREF_CLUSTER), _feat_update(_PREF_CLF), _feat_update(_PREF_MERGE), _avail_md())


def do_sample(n, smart, progress=gr.Progress()):
    if ENG is None:
        return "Open a project first.", _status_md(), gr.update(), gr.update(), gr.update(), gr.update(), _avail_md()
    progress(0.05, desc="loading model + extracting…")
    rep = ENG.sample_more(int(n), smart=bool(smart))
    progress(1.0, desc="done")
    return (f"Added **{rep['n_new_images']}** images / **{rep['n_new_instances']}** instances "
            f"(after class-agnostic NMS).\n\n{_status_md()}", _status_md(), _img_choices(),
            _feat_update(_PREF_CLUSTER), _feat_update(_PREF_CLF), _feat_update(_PREF_MERGE), _avail_md())


def do_compute_raddino(progress=gr.Progress()):
    if ENG is None:
        return "Open a project first.", gr.update(), gr.update(), gr.update(), _avail_md()
    progress(0.05, desc="loading RAD-DINO + pooling masks…")
    rep = ENG.compute_raddino()
    progress(1.0, desc="done")
    if "error" in rep:
        return rep["error"], gr.update(), gr.update(), gr.update(), _avail_md()
    msg = rep.get("msg") or f"Added RAD-DINO features for **{rep['n']}** instances. 'raddino' is now selectable."
    return f"{msg}\n\n{_status_md()}", _feat_update(_PREF_CLUSTER), _feat_update(_PREF_CLF), _feat_update(_PREF_MERGE), _avail_md()


def do_dedup(iou):
    if ENG is None:
        return "Open a project first.", _status_md()
    n = ENG.dedup_current(float(iou))
    return f"Sent **{n}** duplicate instances to background.\n\n{_status_md()}", _status_md()


def do_reset(confirm, nonce):
    if ENG is None:
        return "Open a project first.", _status_md(), gr.update(), None, [], "selected: 0", _bump(nonce)
    if not confirm:
        return ("Tick the confirm box first — this drops the collection, all instances, assignments and classes.",
                _status_md(), gr.update(), None, [], "selected: 0", nonce or 0)
    ENG.reset(keep_config=True)
    return (f"**Dropped everything** (config kept). Re-run Sample & extract.\n\n{_status_md()}",
            _status_md(), gr.update(value=_partition_rows()), None, [], "selected: 0", _bump(nonce))


def do_cluster(feat_methods, distance, per_image, force_n):
    if ENG is None:
        return ("Open a project first.", _status_md(), gr.update(), gr.update(), gr.update(), gr.update(),
                None, [], "selected: 0")
    spec = {m: 1.0 for m in feat_methods} or {"decoder": 1.0}
    try:
        info = ENG.cluster(spec, distance=distance, per_image=bool(per_image),
                           req_clust=(int(force_n) if force_n and int(force_n) > 0 else None))
    except ValueError as e:
        return (str(e), _status_md(), gr.update(), gr.update(), gr.update(), gr.update(),
                None, [], "selected: 0")
    levels = [f"L{i} ({c} clusters)" for i, c in enumerate(info["counts"])]
    pids = [str(r["pid"]) for r in ENG.partition_view()]
    return (f"FINCH on the unassigned pool: levels {info['counts']} (showing L{info['level']}). "
            f"Assigned classes shown as standalone partitions.\n\n{_status_md()}",
            _status_md(), gr.update(choices=levels, value=levels[info["level"]] if levels else None),
            gr.update(value=_partition_rows()), gr.update(choices=pids), _img_choices(),
            None, [], "selected: 0")


# ---- Partitions ------------------------------------------------------------
def on_level_change(level_label):
    try:
        lvl = int(str(level_label).split()[0][1:])                 # "L2 (n clusters)" -> 2
    except (ValueError, IndexError, AttributeError):
        return gr.update(), None, [], "selected: 0", _status_md(), 0
    if ENG is None or ENG._cluster is None:
        return gr.update(), None, [], "selected: 0", _status_md(), 0
    ENG.set_level(lvl)
    return gr.update(value=_partition_rows()), None, [], "selected: 0", _status_md(), 0


def on_partition_select(evt: gr.SelectData):
    if ENG is None or ENG._cluster is None:
        return None, [], "selected: 0", 0
    rows = _partition_rows()
    ridx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if ridx is None or ridx >= len(rows):
        return None, [], "selected: 0", 0
    return str(rows[ridx][0]), [], "selected: 0", 0                # reset to page 0 on a new partition


def do_step_partition(sel_partition, delta):
    pids = [str(r[0]) for r in _partition_rows()]
    if not pids:
        return None, [], "selected: 0", 0
    idx = (pids.index(str(sel_partition)) + delta) if (sel_partition is not None and str(sel_partition) in pids) else 0
    return pids[idx % len(pids)], [], "selected: 0", 0


def do_part_page(sel_partition, page, delta):
    """Step the partition grid's page (the render clamps; this just bounds it)."""
    if ENG is None or sel_partition is None:
        return 0
    n = len(ENG.partition_iuids(str(sel_partition)))
    npages = max(1, -(-n // _GRID_CAP))
    return max(0, min(int(page or 0) + delta, npages - 1))


def do_assign_partition(sel_partition, class_name, nonce):
    if ENG and sel_partition is not None and class_name:
        ENG.assign_partition(sel_partition, class_name)
    return gr.update(value=_partition_rows()), _status_md(), _bump(nonce), [], "selected: 0", *_refresh_classes()


def do_assign_selected(selected, class_name, nonce):
    if ENG and selected and class_name:
        ENG.assign(list(selected), class_name)
    return gr.update(value=_partition_rows()), _status_md(), _bump(nonce), [], "selected: 0", *_refresh_classes()


def do_reject_selected(selected, nonce):
    if ENG and selected:
        ENG.set_background(list(selected))
    return gr.update(value=_partition_rows()), _status_md(), _bump(nonce), [], "selected: 0"


def do_remove_selected(selected, nonce):
    if ENG and selected:
        ENG.remove_from_class(list(selected))
    return gr.update(value=_partition_rows()), _status_md(), _bump(nonce), [], "selected: 0"


def do_merge_same_image(sel_partition, nonce):
    if ENG and sel_partition is not None:
        ENG.merge_partition_by_image(sel_partition)
    return gr.update(value=_partition_rows()), _status_md(), _bump(nonce), [], "selected: 0"


def do_open_source(sel_partition, selected):
    if ENG is None or sel_partition is None:
        return gr.update(), gr.update(), None, 0, [], "selected: 0"
    ius = list(selected) or ENG.partition_iuids(sel_partition)
    if not ius:
        return gr.update(), gr.update(), None, 0, [], "selected: 0"
    iid = ENG.state.meta[ius[0]].image_id
    return gr.Tabs(selected="tab_inimg"), gr.update(value=str(iid)), ENG.image_overlay(iid), 1, [], "selected: 0"


# ---- Refine: ordered op-stack, @gr.render-driven preview -------------------
def _compose(before, after):
    h = max(before.shape[0], after.shape[0])
    def pad(im):
        return cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) if im.shape[0] < h else im
    return np.hstack([pad(before), np.full((h, 3, 3), 255, np.uint8), pad(after)])


def _stack_md(op_stack):
    if not op_stack:
        return "**Op chain (in order):** _(empty — add ops below)_"
    parts = []
    for i, o in enumerate(op_stack):
        kw = o.get("kw", {})
        parts.append(f"{i+1}.{o['name']}" + (f"({','.join(f'{k}={v}' for k, v in kw.items())})" if kw else ""))
    return "**Op chain (in order):** " + " → ".join(parts)


def _refine_target_iuids(target, limit=_REFINE_CAP):
    if not target or ENG is None:
        return []
    if target["kind"] == "partition":
        iuids = ENG.partition_iuids(target["pid"])
    else:
        iuids = [u for u in target["iuids"] if u in ENG.state.meta]   # drop stale/invalid selections
    return iuids[:limit]


def _refine_banner(target) -> str:
    shown = len(_refine_target_iuids(target))
    if target.get("kind") == "partition":
        n = len(ENG.partition_iuids(target["pid"]))
        return f"**partition {target['pid']}** — {n} instance(s); previewing first {shown}  ·  left = before, right = after"
    return f"**{len(target.get('iuids', []))} selected instance(s)**; previewing first {shown}  ·  left = before, right = after"


def do_send_refine_instance(sel_partition, selected):
    if ENG is None or sel_partition is None:
        return gr.update(), None, [], _stack_md([])
    iuids = list(selected) or ENG.partition_iuids(sel_partition)
    return gr.Tabs(selected="tab_refine"), {"kind": "instance", "iuids": iuids}, [], _stack_md([])


def do_send_refine_partition(sel_partition):
    if ENG is None or sel_partition is None:
        return gr.update(), None, [], _stack_md([])
    return gr.Tabs(selected="tab_refine"), {"kind": "partition", "pid": sel_partition}, [], _stack_md([])


def do_add_op(name, op_stack, thr_val, dk, ek, contrast, within, tol, iters):
    st = list(op_stack or [])
    if name == "otsu":
        st.append({"name": "otsu", "kw": {"within_mask": bool(within)}})
    elif name == "threshold":
        st.append({"name": "threshold", "kw": {"val": int(thr_val), "within_mask": bool(within)}})
    elif name == "dilate":
        st.append({"name": "dilate", "kw": {"k": int(dk), "max_contrast": float(contrast)}})
    elif name == "erode":
        st.append({"name": "erode", "kw": {"k": int(ek), "min_contrast": float(contrast)}})
    elif name == "magic_wand":
        st.append({"name": "magic_wand", "kw": {"tol": float(tol)}})
    elif name in ("grabcut", "snap_edges"):
        st.append({"name": name, "kw": {"iters": int(iters)}})
    else:
        st.append({"name": name})
    return st, _stack_md(st)


def do_split(target, nonce):
    if ENG is None or not target:
        return "_(no target — send instances or a partition here first)_", _status_md(), gr.update(), None, nonce or 0
    iuids = _refine_target_iuids(target, limit=10 ** 9)
    n = ENG.split_instances(iuids)
    if n == 0:
        return ("No instance had >1 connected component — nothing to split.",
                _status_md(), gr.update(value=_partition_rows()), target, nonce or 0)
    return (f"Split into **{n}** new instances (originals → background). They're visible now in the "
            f"**In-image** tab; **re-cluster** to see them in Partitions. _(split adds instances → undo cleared)_",
            _status_md(), gr.update(value=_partition_rows()), None, _bump(nonce))


def do_remove_op(op_stack):
    st = list(op_stack or [])[:-1]
    return st, _stack_md(st)


def do_clear_ops():
    return [], _stack_md([])


def do_refine_apply(target, op_stack, nonce):
    if ENG is None or not target:
        return _status_md(), gr.update(), [], _stack_md([]), nonce or 0
    ops = op_stack or []
    if target["kind"] == "partition":
        ENG.apply_refine_partition(target["pid"], ops)
    else:
        ENG.apply_refine_many(list(target["iuids"]), ops)
    return _status_md(), gr.update(value=_partition_rows()), [], _stack_md([]), _bump(nonce)


def do_refine_revert(target, nonce):
    if ENG is None or not target:
        return _status_md(), gr.update(), [], _stack_md([]), nonce or 0
    for u in _refine_target_iuids(target, limit=10 ** 9):
        ENG.revert_refine(u)
    return _status_md(), gr.update(value=_partition_rows()), [], _stack_md([]), _bump(nonce)


# ---- In-image --------------------------------------------------------------
def on_image_pick(image_id, color_by):
    iid = _img_id(image_id)
    if ENG is None or iid is None:
        return None, [], "selected: 0", 0, gr.update()
    return (ENG.image_overlay(iid, color_by=color_by), [], "selected: 0", 0,
            gr.update(choices=_class_choices()))                   # reset page + refresh class choices


def do_recolor(image_id, color_by):
    iid = _img_id(image_id)
    return ENG.image_overlay(iid, color_by=color_by) if (ENG and iid is not None) else None


def do_inimg_page(image_id, page, delta):
    iid = _img_id(image_id)
    if ENG is None or iid is None:
        return 0
    n = len(ENG.image_instance_iuids(iid))
    npages = max(1, -(-n // _GRID_CAP))
    return max(0, min(int(page or 0) + delta, npages - 1))


def do_merge_selected_inimage(image_id, inimg_sel, color_by, nonce):
    iid = _img_id(image_id)
    if ENG and inimg_sel and len(inimg_sel) >= 2:
        ENG.merge_instances(list(inimg_sel))
    ov = ENG.image_overlay(iid, color_by=color_by) if (ENG and iid is not None) else None
    return ov, [], "selected: 0", _bump(nonce), _status_md(), gr.update(value=_partition_rows())


def do_assign_inimage(image_id, inimg_sel, class_name, color_by, inimg_nonce, nonce):
    """Assign the in-image-selected instances (incl. merged reps) to a class."""
    iid = _img_id(image_id)
    if ENG and inimg_sel and class_name:
        ENG.assign(list(inimg_sel), class_name)
    ov = ENG.image_overlay(iid, color_by=color_by) if (ENG and iid is not None) else None
    return (ov, [], "selected: 0", _bump(inimg_nonce), _status_md(), gr.update(value=_partition_rows()),
            _bump(nonce), *_refresh_classes())


def do_merge_preview(image_id, dist_kind, method, thresh, max_grp):
    iid = _img_id(image_id)
    if ENG is None or iid is None:
        return None, None, []
    mg = None if int(max_grp) == 0 else int(max_grp)
    b, a, groups = ENG.merge_preview(iid, dist_kind=dist_kind, method=method, thresh=float(thresh), max_group_size=mg)
    return b, a, groups


def do_commit_merge(image_id, groups, color_by, nonce):
    iid = _img_id(image_id)
    if ENG is None or iid is None:
        return None, _status_md(), gr.update(), nonce or 0
    ENG.commit_merge(iid, groups)
    return ENG.image_overlay(iid, color_by=color_by), _status_md(), gr.update(value=_partition_rows()), _bump(nonce)


# ---- Rejected / unreject ---------------------------------------------------
def do_load_rejected():
    if ENG is None:
        return [], [], "0 rejected", []
    bg = ENG.background_iuids()
    crops = [(ENG.crop(u, mask_overlay=True), ENG._caption(u)) for u in bg]
    return crops, bg, f"{len(bg)} rejected · selected: 0", []


def on_bg_gallery_select(bg_iuids, bg_sel, evt: gr.SelectData):
    idx = int(evt.index); s = set(bg_sel or [])
    if bg_iuids and idx < len(bg_iuids):
        s.symmetric_difference_update({bg_iuids[idx]})
    s = sorted(s)
    return s, f"{len(bg_iuids or [])} rejected · selected: {len(s)}"


def do_unreject(which, bg_iuids, bg_sel, nonce):
    if ENG is None:
        return [], [], "0 rejected", [], _status_md(), gr.update(), nonce or 0
    targets = list(ENG.background_iuids()) if which == "all" else list(bg_sel or [])
    ENG.unreject(targets)
    return *do_load_rejected(), _status_md(), gr.update(value=_partition_rows()), _bump(nonce)


# ---- Classifier ------------------------------------------------------------
def _youden_md(rep):
    cur = (rep.get("pr") or {}).get("curves", {})
    rec = [f"**{ENG.state.class_name(c)}** {v['youden']:.2f}" for c, v in cur.items() if "youden" in v]
    return ("**Recommended thresholds (Youden's J):** " + " · ".join(rec)) if rec else ""


def do_train(feat_methods, algo, openset):
    empty = (gr.update(choices=[], value=None), "")
    if ENG is None:
        return "Open a project first.", None, *empty
    spec = {m: 1.0 for m in feat_methods} or {"decoder": 1.0}
    rep = ENG.train_classifier(spec, algo=algo, use_unassigned_negatives=bool(openset))
    if "error" in rep:
        return rep["error"], None, *empty
    mode = "open-set (this·vs·not-this × this·vs·others)" if openset else "vs-background-only"
    skipped = rep.get("skipped_names") or []
    skip_note = (f" · **skipped {len(skipped)} class(es)** with <2 instances: {', '.join(skipped)} "
                 f"(assign ≥2 each, then retrain)") if skipped else ""
    names = [ENG.state.class_name(c) for c in rep["classes"]]
    return (f"Trained [{mode}]: **{rep['n']}** assigned across **{rep['n_classes']}** classes; "
            f"negatives = {rep['n_background']} bg + {rep['n_unassigned_neg']} unassigned.{skip_note}",
            _pr_fig(rep.get("pr", {})), gr.update(choices=names, value=None), _youden_md(rep))


def _pr_fig(pr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (c, cur) in enumerate(pr.get("curves", {}).items()):
        name = ENG.state.class_name(c) if ENG else str(c)          # show class NAME, not the cid
        col = f"C{i % 10}"
        t = cur.get("thresholds", [])
        if t:
            ax.plot(t, cur["precision"][:len(t)], "-", color=col, label=f"{name} P")
            ax.plot(t, cur["recall"][:len(t)], "--", color=col, label=f"{name} R")
        if cur.get("youden") is not None:                          # mark Youden-J recommended threshold
            ax.axvline(cur["youden"], color=col, ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("threshold"); ax.set_ylabel("P / R"); ax.set_xlim(0, 1); ax.set_title("CV-OOF P/R  (·· = Youden J)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)   # legend OUTSIDE the axes
    fig.tight_layout()
    return fig


def do_predict(thresh, only_class_name=""):
    if ENG is None or getattr(ENG, "_clf", None) is None:
        return "Train a classifier first.", None, []
    cid = ENG.state.class_id_by_name(only_class_name) if only_class_name else None
    preds = sorted(ENG.predict_and_threshold(float(thresh), only_class=cid), key=lambda t: -t[2])  # conf-desc
    rows = [[u[:8], ENG.state.class_name(c), round(conf, 3)] for u, c, conf in preds[:200]]
    scope = f" for **{only_class_name}**" if only_class_name else " (all classes)"
    msg = (f"{len(preds)} unassigned instances would be assigned{scope} at thresh={thresh:.2f}. "
           f"Previews of the highest-confidence ones are below."
           if preds else f"No unassigned instances pass thresh={thresh:.2f}{scope}.")
    return msg, rows, preds


def do_apply_predictions(thresh, only_class_name, nonce):
    if ENG is None or getattr(ENG, "_clf", None) is None:
        return "Train a classifier first.", _status_md(), gr.update(), nonce or 0, None, []
    cid = ENG.state.class_id_by_name(only_class_name) if only_class_name else None
    n = ENG.apply_predictions(float(thresh), only_class=cid)
    scope = f" to **{only_class_name}**" if only_class_name else ""
    _, rows, preds = do_predict(thresh, only_class_name)            # refresh preview over the now-smaller unassigned pool
    return (f"Assigned **{n}** instances{scope} at thresh={thresh:.2f}. Preview refreshed (assigned ones removed).",
            _status_md(), gr.update(value=_partition_rows()), _bump(nonce), rows, preds)


# ---- Merge recommender -----------------------------------------------------
def _merge_pr_fig(curve, youden):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6))
    t = curve.get("thresholds", [])
    if t:
        ax.plot(t, curve["precision"][:len(t)], "-", color="C0", label="precision")
        ax.plot(t, curve["recall"][:len(t)], "--", color="C1", label="recall")
    if youden is not None:
        ax.axvline(youden, color="k", ls=":", lw=1, alpha=0.7, label=f"Youden {youden:.2f}")
    ax.set_xlabel("P(merge) threshold"); ax.set_ylabel("P / R"); ax.set_xlim(0, 1)
    ax.set_title("merge recommender — CV-OOF P/R")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig


def do_train_merge(feat_methods, algo):
    if ENG is None:
        return "Open a project first.", None
    spec = {m: 1.0 for m in feat_methods} or {"decoder": 1.0}
    rep = ENG.train_merge_recommender(spec, algo=algo)
    if "error" in rep:
        return rep["error"], None
    return (f"Trained from **{rep['n_merge_events']}** merge event(s) → **{rep['n_pos']}** positive pairs / "
            f"**{rep['n_neg']}** negatives. Recommended threshold (Youden's J): **{rep.get('youden', 0.5):.2f}**.",
            _merge_pr_fig(rep.get("curve", {}), rep.get("youden")))


def do_recommend_merges(thresh):
    if ENG is None or getattr(ENG, "_merge_clf", None) is None:
        return "Train the merge recommender first.", []
    cands = ENG.recommend_merges(float(thresh))
    msg = (f"**{len(cands)}** candidate merge group(s) at P(merge) ≥ {float(thresh):.2f} — ✓ to merge, ✗ to reject."
           if cands else f"No candidate merges at P(merge) ≥ {float(thresh):.2f}.")
    return msg, cands


def do_accept_merge(iuids, cands, nonce):
    if ENG and iuids and len(iuids) >= 2:
        ENG.accept_merge(iuids)
    cands = [c for c in (cands or []) if c.get("iuids") != iuids]
    return cands, _status_md(), gr.update(value=_partition_rows()), _bump(nonce)


def do_reject_merge(iuids, cands):
    if ENG and iuids and len(iuids) >= 2:
        ENG.reject_merge(iuids)
    return [c for c in (cands or []) if c.get("iuids") != iuids]


# ---- Map -------------------------------------------------------------------
def do_map(method, color_by):
    if ENG is None:
        return None, gr.update(choices=[])
    import plotly.graph_objects as go
    xy, labels, order, thumbs, names = ENG.embed_thumbnails(method=method, color_by=color_by)
    custom = [[t, f"{n} · c{int(l)}"] for t, n, l in zip(thumbs, names, labels)]
    fig = go.Figure(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers",
                               marker=dict(size=7, color=[int(l) for l in labels], colorscale="Turbo"),
                               customdata=custom, hoverinfo="skip"))
    fig.update_layout(width=760, height=560, title=f"{method} · color={color_by} (hover a point)",
                      showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig, gr.update(choices=sorted({str(int(l)) for l in labels if l >= 0}))


def do_assign_cluster(cluster_id, class_name, nonce):
    if ENG and cluster_id is not None and class_name and ENG._cluster is not None:
        ENG.assign_partition(cluster_id, class_name)
    return _status_md(), gr.update(value=_partition_rows()), _bump(nonce), *_refresh_classes()


# ---- Export / undo ---------------------------------------------------------
def do_export(class_subset, instance_scope, include_kpts, mask_fmt):
    if ENG is None:
        return "Open a project first.", None
    classes = [ENG.state.class_id_by_name(n) for n in class_subset if ENG.state.class_id_by_name(n)] if class_subset else None
    p = ENG.export_coco(classes=classes, with_keypoints=bool(include_kpts), polygon=(mask_fmt == "polygon"),
                        include_unassigned=(instance_scope == "all (incl. unassigned)"))
    import json
    coco = json.loads(p.read_text())
    return f"Exported {len(coco['annotations'])} anns / {len(coco['images'])} images -> {p}", str(p)


def do_undo(nonce):
    if ENG:
        ENG.undo()
    return _status_md(), gr.update(value=_partition_rows()), _bump(nonce), [], "selected: 0"


def do_redo(nonce):
    if ENG:
        ENG.redo()
    return _status_md(), gr.update(value=_partition_rows()), _bump(nonce), [], "selected: 0"


# --------------------------------------------------------------------------- #
def build_app(default_project: str = "/tmp/curator_project") -> gr.Blocks:
    d = _defaults()
    with gr.Blocks(title="qseg curator") as demo:
        with gr.Row():
            undo_btn = gr.Button("↶ Undo", scale=0, elem_id="kb_undo")
            redo_btn = gr.Button("↷ Redo", scale=0, elem_id="kb_redo")
            status = gr.Markdown(_status_md())
        class_dds: list = []
        sel_partition = gr.State(None); selected_iuids = gr.State([]); render_nonce = gr.State(0)
        part_page = gr.State(0); inimg_page = gr.State(0)
        pending_groups = gr.State([]); refine_target = gr.State(None); op_stack = gr.State([])
        inimg_sel = gr.State([]); inimg_nonce = gr.State(0)
        bg_iuids = gr.State([]); bg_sel = gr.State([]); pred_state = gr.State([]); merge_cands = gr.State([])

        with gr.Tabs() as tabs:
            with gr.Tab("Config"):
                proj_tb = gr.Textbox(label="Project dir", value=default_project)
                ckpt_tb = gr.Textbox(label="Seg-model weights", value=d["ckpt"])
                cfgname_tb = gr.Textbox(label="config name", value=d["config_name"])
                overrides_tb = gr.Textbox(label="model overrides (one per line)", value="\n".join(d["overrides"]), lines=5)
                root_tb = gr.Textbox(label="Root image folder", value=d["root"])
                with gr.Row():
                    score_sl = gr.Slider(0, 1, value=0.3, step=0.01, label="Score threshold")
                    nms_sl = gr.Slider(0, 1, value=0.8, step=0.05, label="NMS mask-IoU (class-agnostic; 0=off)")
                    nimg_sl = gr.Slider(1, 400, value=40, step=1, label="# images")
                    smart_cb = gr.Checkbox(label="smart sample", value=False)
                with gr.Row():
                    open_btn = gr.Button("Create / Open project", variant="primary")
                    sample_btn = gr.Button("Sample & extract (additive)", variant="primary")
                    dedup_btn = gr.Button("Dedup now")
                    raddino_btn = gr.Button("Compute RAD-DINO features (adds 'raddino')")
                avail_md = gr.Markdown(_avail_md())
                feat_cbg = gr.CheckboxGroup(choices=[], value=[], label="Feature types (clustering)")
                with gr.Row():
                    dist_dd = gr.Dropdown(["cosine", "euclidean"], value="cosine", label="FINCH distance")
                    perimg_cb = gr.Checkbox(label="per-image clustering", value=False)
                    forcen_num = gr.Number(value=0, precision=0, label="force N clusters (0=FINCH levels)")
                    cluster_btn = gr.Button("Cluster (FINCH on unassigned)", variant="primary")
                cfg_status = gr.Markdown()
                gr.Markdown("---")
                with gr.Row():
                    reset_confirm = gr.Checkbox(label="confirm: drop EVERYTHING (instances, assignments, classes) — keep config", value=False)
                    reset_btn = gr.Button("Drop everything (reset project)", variant="stop")

            with gr.Tab("Partitions"):
                gr.Markdown("Shortcuts: **a** assign partition · **s** assign selected · **r** reject · **u** unassign · "
                            "**z/y** undo/redo · **[ ]** prev/next. Tick the checkbox on each instance to (de)select.")
                level_dd = gr.Dropdown(label="FINCH level (unassigned pool)", choices=[], interactive=True, allow_custom_value=True)
                with gr.Row():
                    with gr.Column(scale=1):
                        part_df = gr.Dataframe(headers=["pid", "size", "purity", "score", "class"],
                                               datatype=["str", "number", "number", "number", "str"],
                                               interactive=False, label="partitions (click a row)", max_height=900)
                        with gr.Row():
                            prev_btn = gr.Button("◀ prev", elem_id="kb_prev")
                            next_btn = gr.Button("next ▶", elem_id="kb_next")
                    with gr.Column(scale=3):
                        with gr.Row():
                            mask_toggle = gr.Checkbox(label="mask overlay", value=True)
                            view_mode = gr.Radio(["crop", "in context"], value="crop", label="view")
                            inst_count = gr.Markdown("selected: 0")
                        pclass_dd = gr.Dropdown(choices=_class_choices(), allow_custom_value=True, label="class (type to filter / new)")
                        class_dds.append(pclass_dd)
                        with gr.Row():
                            assign_all = gr.Button("Assign partition", variant="primary", elem_id="kb_assign")
                            assign_sel = gr.Button("Assign selected", elem_id="kb_assignsel")
                            remove_sel = gr.Button("Unassign", elem_id="kb_unassign")
                            reject_sel = gr.Button("Reject", elem_id="kb_reject")
                        with gr.Row():
                            merge_img_btn = gr.Button("Merge same-image instances")
                            open_src_btn = gr.Button("Open source → In-image")
                            send_refine_inst = gr.Button("Send selected → Refine")
                            send_refine_part = gr.Button("Refine whole partition")

                        with gr.Row():
                            gr.Markdown("**Instances in the selected partition** — tick a box to (de)select, then use the buttons above.")
                            part_pageprev = gr.Button("◀ page", scale=0)
                            part_pagenext = gr.Button("page ▶", scale=0)

                        @gr.render(inputs=[sel_partition, mask_toggle, view_mode, render_nonce, part_page])
                        def _partition_grid(pid, mask_overlay, vmode, _n, page):
                            if ENG is None or pid is None:
                                gr.Markdown("_Click a partition row to load its instances._"); return
                            allu = ENG.partition_iuids(str(pid))
                            if not allu:
                                gr.Markdown("_(no unassigned instances here — assign/reject emptied this partition)_"); return
                            npages = max(1, -(-len(allu) // _GRID_CAP))
                            page = max(0, min(int(page or 0), npages - 1))
                            iuids = allu[page * _GRID_CAP:(page + 1) * _GRID_CAP]
                            if npages > 1:
                                gr.Markdown(f"**page {page + 1}/{npages}** · {len(allu)} instances total (use ◀ page / page ▶)")
                            for i in range(0, len(iuids), 6):
                                with gr.Row():
                                    for u in iuids[i:i + 6]:
                                        with gr.Column(min_width=150):
                                            gr.Image(ENG.crop(u, mask_overlay=bool(mask_overlay), context=(vmode == "in context")),
                                                     show_label=False, height=190)
                                            cb = gr.Checkbox(label=ENG._caption(u), value=False)
                                            cb.change(_toggle_factory(u), [cb, selected_iuids], [selected_iuids, inst_count])

            with gr.Tab("In-image", id="tab_inimg"):
                with gr.Row():
                    image_dd = gr.Dropdown(label="image_id", choices=[], interactive=True, allow_custom_value=True)
                    colorby_radio = gr.Radio(["partition", "class"], value="partition", label="color by")
                inimg = gr.Image(label="image overlay (context)", height=440)
                with gr.Row():
                    inimg_count = gr.Markdown("selected: 0")
                    merge_sel_btn = gr.Button("Merge selected → one instance", variant="primary")
                with gr.Row():
                    inimg_class_dd = gr.Dropdown(choices=_class_choices(), allow_custom_value=True,
                                                 label="assign selected to class (type to filter / new)", scale=2)
                    class_dds.append(inimg_class_dd)
                    assign_inimg_btn = gr.Button("Assign selected → class", variant="primary", scale=1)
                    inimg_pageprev = gr.Button("◀ page", scale=0)
                    inimg_pagenext = gr.Button("page ▶", scale=0)

                @gr.render(inputs=[image_dd, inimg_nonce, inimg_page])
                def _inimg_grid(image_id, _n, page):
                    if ENG is None or not image_id:
                        gr.Markdown("_Pick an image_id above._"); return
                    allu = ENG.image_instance_iuids(int(image_id))
                    if not allu:
                        gr.Markdown("_(no instances on this image)_"); return
                    npages = max(1, -(-len(allu) // _GRID_CAP))
                    page = max(0, min(int(page or 0), npages - 1))
                    iuids = allu[page * _GRID_CAP:(page + 1) * _GRID_CAP]
                    if npages > 1:
                        gr.Markdown(f"**page {page + 1}/{npages}** · {len(allu)} instances total (use ◀ page / page ▶)")
                    for i in range(0, len(iuids), 8):
                        with gr.Row():
                            for u in iuids[i:i + 8]:
                                with gr.Column(min_width=120):
                                    gr.Image(ENG.crop(u, mask_overlay=True, max_side=256), show_label=False, height=140)
                                    cb = gr.Checkbox(label=ENG._caption(u), value=False)
                                    cb.change(_toggle_factory(u), [cb, inimg_sel], [inimg_sel, inimg_count])

                gr.Markdown("**Distance merge** — set params, **Preview**, then commit (preview is NOT live, to stay responsive):")
                with gr.Row():
                    mdist_dd = gr.Dropdown(["mask_gap", "feature", "centroid", "combo"], value="mask_gap", label="dist")
                    mmeth_dd = gr.Dropdown(["decoder", "maskpool", "backbone"], value="decoder", label="feature")
                    mthr_sl = gr.Slider(0, 0.5, value=0.05, step=0.005, label="threshold")
                    mgrp_sl = gr.Slider(0, 8, value=3, step=1, label="max group (0=any)")
                merge_prev_btn = gr.Button("Preview merge")
                with gr.Row():
                    before_img = gr.Image(label="before", height=320)
                    after_img = gr.Image(label="after", height=320)
                commit_btn = gr.Button("Commit distance merge", variant="primary")

            with gr.Tab("Refine", id="tab_refine"):
                gr.Markdown("Send instances here from **Partitions** (_Send selected → Refine_ / _Refine whole partition_). "
                            "Add operations **in order** (applied top-to-bottom); the preview updates live.")
                with gr.Row():
                    refine_mask = gr.Checkbox(label="show mask overlay (off = raw image)", value=True)
                    within_cb = gr.Checkbox(label="threshold within current mask (carve, don't grow)", value=True)
                with gr.Row():
                    thr_val = gr.Slider(0, 255, value=128, step=1, label="threshold val")
                    dk_sl = gr.Slider(1, 5, value=2, step=1, label="dilate k")
                    ek_sl = gr.Slider(1, 5, value=2, step=1, label="erode k")
                    contrast_sl = gr.Slider(0, 1, value=0.2, step=0.01, label="contrast gate")
                with gr.Row():
                    tol_sl = gr.Slider(0.01, 0.4, value=0.08, step=0.01, label="magic-wand tolerance")
                    iters_sl = gr.Slider(1, 60, value=20, step=1, label="grabcut/snap iterations")
                with gr.Row():
                    add_otsu = gr.Button("+ otsu"); add_thr = gr.Button("+ threshold")
                    add_dil = gr.Button("+ dilate"); add_ero = gr.Button("+ erode")
                    add_fill = gr.Button("+ fill"); add_lcc = gr.Button("+ largest CC"); add_sm = gr.Button("+ smooth")
                with gr.Row():
                    add_gc = gr.Button("+ grabcut (quick-select)"); add_mw = gr.Button("+ magic wand")
                    add_snap = gr.Button("+ snap to edges (magnetic)")
                with gr.Row():
                    remove_op_btn = gr.Button("remove last"); clear_op_btn = gr.Button("clear chain")
                stack_md = gr.Markdown(_stack_md([]))
                with gr.Row():
                    split_btn = gr.Button("Split → connected components (new instances)", variant="secondary")
                refine_msg = gr.Markdown()

                @gr.render(inputs=[refine_target, op_stack, refine_mask])
                def _refine_preview(target, ops, mask_overlay):
                    if ENG is None or not target:
                        gr.Markdown("_(no target — use the buttons in the Partitions tab to send instances or a whole partition here)_"); return
                    iuids = _refine_target_iuids(target)
                    if not iuids:
                        gr.Markdown("_(target has no instances)_"); return
                    gr.Markdown(_refine_banner(target))
                    for i in range(0, len(iuids), 2):
                        with gr.Row():
                            for u in iuids[i:i + 2]:
                                b, a = ENG.refine_preview(u, ops or [], mask_overlay=bool(mask_overlay))
                                gr.Image(_compose(b, a), label=f"{u[:6]}  ·  left = before / right = after",
                                         height=260)

                with gr.Row():
                    refine_apply = gr.Button("Apply chain", variant="primary"); refine_revert = gr.Button("Revert")

            with gr.Tab("Classifier"):
                clf_feat = gr.CheckboxGroup(choices=[], value=[], label="classifier features (only features present in the collection)")
                with gr.Row():
                    clf_algo = gr.Radio(["logreg", "rf"], value="logreg", label="model")
                    clf_openset = gr.Checkbox(value=True, label="open-set: unassigned+background as negatives")
                train_btn = gr.Button("Train on assigned", variant="primary")
                clf_msg = gr.Markdown(); pr_plot = gr.Plot(label="P/R vs threshold")
                rec_thresh_md = gr.Markdown()
                with gr.Row():
                    clf_thr = gr.Slider(0, 1, value=0.5, step=0.01, label="assignment threshold", scale=3)
                    clf_apply_class = gr.Dropdown(choices=[], value=None, label="apply only this class (empty = all)", scale=2)
                with gr.Row():
                    predict_btn = gr.Button("Preview predictions"); apply_pred_btn = gr.Button("Apply", variant="primary")
                    pred_n = gr.Number(value=12, precision=0, label="# previews", minimum=1, maximum=60)
                pred_df = gr.Dataframe(headers=["iuid", "pred class", "conf"], interactive=False, max_height=360)

                @gr.render(inputs=[pred_state, pred_n])
                def _pred_preview(preds, n):
                    if ENG is None or not preds:
                        gr.Markdown("_Click **Preview predictions** to see the highest-confidence predicted instances._"); return
                    show = preds[:int(n or 12)]
                    gr.Markdown(f"**Top {len(show)} predictions** (highest confidence) — each: class · conf · iuid:")
                    for i in range(0, len(show), 6):
                        with gr.Row():
                            for u, cid, conf in show[i:i + 6]:
                                with gr.Column(min_width=150):
                                    gr.Image(ENG.crop(u, max_side=256), show_label=False, height=170)
                                    gr.Markdown(f"**{ENG.state.class_name(cid)}** · {conf:.2f} · {u[:6]}")

            with gr.Tab("Merge-rec"):
                gr.Markdown("Learn from your **In-image merges** to suggest new merges. Merge a few groups first, "
                            "then Train → Recommend → ✓ accept / ✗ reject (each accept/reject improves the model).")
                mr_feat = gr.CheckboxGroup(choices=[], value=[], label="features (present in collection) + geometry")
                with gr.Row():
                    mr_algo = gr.Radio(["logreg", "rf"], value="logreg", label="model")
                    mr_train_btn = gr.Button("Train merge recommender", variant="primary")
                mr_msg = gr.Markdown(); mr_plot = gr.Plot(label="merge P/R vs threshold")
                with gr.Row():
                    mr_thr = gr.Slider(0, 1, value=0.5, step=0.01, label="P(merge) threshold", scale=3)
                    mr_rec_btn = gr.Button("Recommend merges", variant="primary", scale=1)

                @gr.render(inputs=[merge_cands])
                def _merge_preview(cands):
                    if ENG is None or not cands:
                        gr.Markdown("_Train, then click **Recommend merges**._"); return
                    for c in cands:
                        ius = list(c["iuids"])
                        with gr.Row():
                            for u in ius[:8]:
                                with gr.Column(min_width=120):
                                    gr.Image(ENG.crop(u, max_side=200), show_label=False, height=130)
                            with gr.Column(min_width=180):
                                gr.Markdown(f"**P(merge)={c['prob']:.2f}**\n\nimage {c['image_id']} · {len(ius)} instances")
                                acc = gr.Button("✓ Merge", variant="primary")
                                rej = gr.Button("✗ Reject")
                                acc.click(lambda cs, n, _i=ius: do_accept_merge(_i, cs, n),
                                          [merge_cands, render_nonce], [merge_cands, status, part_df, render_nonce])
                                rej.click(lambda cs, _i=ius: do_reject_merge(_i, cs), [merge_cands], [merge_cands])

            with gr.Tab("Map"):
                with gr.Row():
                    map_method = gr.Radio(["pca", "umap", "tsne"], value="pca", label="embedding")
                    map_colorby = gr.Radio(["cluster", "class"], value="cluster", label="color by")
                    map_btn = gr.Button("Compute map", variant="primary")
                map_plot = gr.Plot(label="2D embedding (hover a point for a preview)", elem_id="map_plot")
                with gr.Row():
                    map_cluster_dd = gr.Dropdown(label="cluster id", choices=[], interactive=True, allow_custom_value=True)
                    map_class_dd = gr.Dropdown(choices=_class_choices(), allow_custom_value=True, label="assign to class")
                    class_dds.append(map_class_dd)
                    map_assign_btn = gr.Button("Assign whole cluster", variant="primary")

            with gr.Tab("Rejected"):
                gr.Markdown("Rejected (background) instances. Unreject sends them back to **unassigned**.")
                with gr.Row():
                    load_bg_btn = gr.Button("Load rejected", variant="primary")
                    bg_count = gr.Markdown("0 rejected")
                    unreject_sel_btn = gr.Button("Unreject selected")
                    unreject_all_btn = gr.Button("Unreject all", variant="primary")
                bg_gallery = gr.Gallery(label="rejected instances (click to (de)select)", columns=8, height=320)

            with gr.Tab("Export"):
                exp_classes = gr.CheckboxGroup(choices=_class_choices(), label="classes (empty = all)")
                class_dds.append(exp_classes)
                exp_scope = gr.Radio(["assigned only", "all (incl. unassigned)"], value="assigned only", label="scope")
                with gr.Row():
                    exp_kpts = gr.Checkbox(label="include keypoints", value=True)
                    exp_fmt = gr.Radio(["RLE", "polygon"], value="RLE", label="mask format")
                export_btn = gr.Button("Export COCO", variant="primary")
                exp_msg = gr.Markdown(); exp_file = gr.File(label="download")

        # hidden 4th class-fanout sink — MUST live outside gr.Tabs (a non-Tab direct child of
        # gr.Tabs corrupts the tab group: breaks gr.Tabs(selected=...) switching + tab render).
        class_dds.append(gr.Dropdown(visible=False, allow_custom_value=True))

        # ---- wiring ----
        open_btn.click(do_open_project, [proj_tb, ckpt_tb, cfgname_tb, overrides_tb, root_tb, score_sl, nms_sl],
                       [cfg_status, status, image_dd, feat_cbg, clf_feat, mr_feat, avail_md])
        sample_btn.click(do_sample, [nimg_sl, smart_cb], [cfg_status, status, image_dd, feat_cbg, clf_feat, mr_feat, avail_md])
        dedup_btn.click(do_dedup, [nms_sl], [cfg_status, status])
        raddino_btn.click(do_compute_raddino, None, [cfg_status, feat_cbg, clf_feat, mr_feat, avail_md])
        reset_btn.click(do_reset, [reset_confirm, render_nonce], [cfg_status, status, part_df, sel_partition, selected_iuids, inst_count, render_nonce])
        cluster_btn.click(do_cluster, [feat_cbg, dist_dd, perimg_cb, forcen_num],
                          [cfg_status, status, level_dd, part_df, map_cluster_dd, image_dd, sel_partition, selected_iuids, inst_count])

        mut = [part_df, status, render_nonce, selected_iuids, inst_count]        # mutation outputs (re-render grid)
        psel = [sel_partition, selected_iuids, inst_count, part_page]            # partition-row-select outputs (resets page)
        level_dd.change(on_level_change, [level_dd], [part_df, sel_partition, selected_iuids, inst_count, status, part_page])
        part_df.select(on_partition_select, None, psel)
        prev_btn.click(lambda sp: do_step_partition(sp, -1), [sel_partition], psel)
        next_btn.click(lambda sp: do_step_partition(sp, 1), [sel_partition], psel)
        part_pageprev.click(lambda sp, pg: do_part_page(sp, pg, -1), [sel_partition, part_page], [part_page])
        part_pagenext.click(lambda sp, pg: do_part_page(sp, pg, 1), [sel_partition, part_page], [part_page])
        assign_all.click(do_assign_partition, [sel_partition, pclass_dd, render_nonce], [*mut, *class_dds])
        assign_sel.click(do_assign_selected, [selected_iuids, pclass_dd, render_nonce], [*mut, *class_dds])
        remove_sel.click(do_remove_selected, [selected_iuids, render_nonce], mut)
        reject_sel.click(do_reject_selected, [selected_iuids, render_nonce], mut)
        merge_img_btn.click(do_merge_same_image, [sel_partition, render_nonce], mut)
        open_src_btn.click(do_open_source, [sel_partition, selected_iuids], [tabs, image_dd, inimg, inimg_nonce, inimg_sel, inimg_count])
        send_refine_inst.click(do_send_refine_instance, [sel_partition, selected_iuids], [tabs, refine_target, op_stack, stack_md])
        send_refine_part.click(do_send_refine_partition, [sel_partition], [tabs, refine_target, op_stack, stack_md])

        image_dd.change(on_image_pick, [image_dd, colorby_radio], [inimg, inimg_sel, inimg_count, inimg_page, inimg_class_dd])
        colorby_radio.change(do_recolor, [image_dd, colorby_radio], [inimg])
        merge_sel_btn.click(do_merge_selected_inimage, [image_dd, inimg_sel, colorby_radio, inimg_nonce],
                            [inimg, inimg_sel, inimg_count, inimg_nonce, status, part_df])
        assign_inimg_btn.click(do_assign_inimage, [image_dd, inimg_sel, inimg_class_dd, colorby_radio, inimg_nonce, render_nonce],
                               [inimg, inimg_sel, inimg_count, inimg_nonce, status, part_df, render_nonce, *class_dds])
        inimg_pageprev.click(lambda iid, pg: do_inimg_page(iid, pg, -1), [image_dd, inimg_page], [inimg_page])
        inimg_pagenext.click(lambda iid, pg: do_inimg_page(iid, pg, 1), [image_dd, inimg_page], [inimg_page])
        merge_prev_btn.click(do_merge_preview, [image_dd, mdist_dd, mmeth_dd, mthr_sl, mgrp_sl], [before_img, after_img, pending_groups])
        commit_btn.click(do_commit_merge, [image_dd, pending_groups, colorby_radio, inimg_nonce], [inimg, status, part_df, inimg_nonce])

        adds = [op_stack, stack_md]
        ain = [op_stack, thr_val, dk_sl, ek_sl, contrast_sl, within_cb, tol_sl, iters_sl]
        addin = lambda nm: (lambda st, tv, d_, e_, c_, w_, t_, it_: do_add_op(nm, st, tv, d_, e_, c_, w_, t_, it_))
        add_otsu.click(addin("otsu"), ain, adds); add_thr.click(addin("threshold"), ain, adds)
        add_dil.click(addin("dilate"), ain, adds); add_ero.click(addin("erode"), ain, adds)
        add_fill.click(addin("fill"), ain, adds); add_lcc.click(addin("largest_cc"), ain, adds); add_sm.click(addin("smooth"), ain, adds)
        add_gc.click(addin("grabcut"), ain, adds); add_mw.click(addin("magic_wand"), ain, adds); add_snap.click(addin("snap_edges"), ain, adds)
        remove_op_btn.click(do_remove_op, [op_stack], adds)
        clear_op_btn.click(do_clear_ops, None, adds)
        refine_apply.click(do_refine_apply, [refine_target, op_stack, render_nonce], [status, part_df, op_stack, stack_md, render_nonce])
        refine_revert.click(do_refine_revert, [refine_target, render_nonce], [status, part_df, op_stack, stack_md, render_nonce])
        split_btn.click(do_split, [refine_target, render_nonce], [refine_msg, status, part_df, refine_target, render_nonce])

        train_btn.click(do_train, [clf_feat, clf_algo, clf_openset], [clf_msg, pr_plot, clf_apply_class, rec_thresh_md])
        predict_btn.click(do_predict, [clf_thr, clf_apply_class], [clf_msg, pred_df, pred_state])
        apply_pred_btn.click(do_apply_predictions, [clf_thr, clf_apply_class, render_nonce],
                             [clf_msg, status, part_df, render_nonce, pred_df, pred_state])

        mr_train_btn.click(do_train_merge, [mr_feat, mr_algo], [mr_msg, mr_plot])
        mr_rec_btn.click(do_recommend_merges, [mr_thr], [mr_msg, merge_cands])

        map_btn.click(do_map, [map_method, map_colorby], [map_plot, map_cluster_dd])
        map_assign_btn.click(do_assign_cluster, [map_cluster_dd, map_class_dd, render_nonce], [status, part_df, render_nonce, *class_dds])

        load_bg_btn.click(do_load_rejected, [], [bg_gallery, bg_iuids, bg_count, bg_sel])
        bg_gallery.select(on_bg_gallery_select, [bg_iuids, bg_sel], [bg_sel, bg_count])
        unreject_sel_btn.click(lambda i, s, n: do_unreject("sel", i, s, n), [bg_iuids, bg_sel, render_nonce], [bg_gallery, bg_iuids, bg_count, bg_sel, status, part_df, render_nonce])
        unreject_all_btn.click(lambda i, s, n: do_unreject("all", i, s, n), [bg_iuids, bg_sel, render_nonce], [bg_gallery, bg_iuids, bg_count, bg_sel, status, part_df, render_nonce])

        export_btn.click(do_export, [exp_classes, exp_scope, exp_kpts, exp_fmt], [exp_msg, exp_file])
        undo_btn.click(do_undo, [render_nonce], [status, part_df, render_nonce, selected_iuids, inst_count])
        redo_btn.click(do_redo, [render_nonce], [status, part_df, render_nonce, selected_iuids, inst_count])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="/tmp/curator_project")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build_app(args.project).queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share, head=CURATOR_JS)


if __name__ == "__main__":
    main()
