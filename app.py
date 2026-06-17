"""Gradio web app for the qseg instance curator.

Run:  python -m tools.curator.app  [--project DIR] [--port 7860] [--share]

ONE server-side CuratorEngine (single user, GPU). Tabs: Config | Partitions | In-image |
Refine | Classifier | Map | Export.

v5 perf fix: gr.Gallery ships FULL-RES images to the browser and re-emitting the whole gallery on
every select overflowed browser RAM. Now: all crops are downscaled thumbnails (engine.crop max_side),
selection NEVER re-emits the main gallery (it updates a small "selected" thumbnail strip + a count),
no image lists are stored in gr.State, and Refine previews are capped + thumbnailed.
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
    return [gr.update(choices=ch) for _ in range(4)]


def _img_choices():
    return gr.update(choices=[str(i) for i in ENG.image_ids()]) if ENG else gr.update()


def _partition_rows():
    if ENG is None or ENG._cluster is None:
        return []
    return [[str(r["pid"]), r["size"], (round(r["purity"], 2) if r["purity"] is not None else None),
             r["mean_score"], r["majority_class"] or ""] for r in ENG.partition_view()]


def _load_partition(pid, mask_overlay, view_mode):
    crops, _ = ENG.partition_crops(pid, mask_overlay=bool(mask_overlay), context=(view_mode == "in context"))
    return crops


def _sel_strip(sel_partition, sel_idx):
    """Small thumbnail strip of the currently-selected instances (cheap; NOT the full gallery)."""
    if ENG is None or sel_partition is None:
        return []
    iuids = ENG.partition_iuids(sel_partition)
    return [(ENG.crop(iuids[i], max_side=200), iuids[i][:6]) for i in (sel_idx or []) if i < len(iuids)]


def _iuid_strip(iuids):
    return [(ENG.crop(u, max_side=200), u[:6]) for u in (iuids or [])] if ENG else []


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
    return f"Project **{project_dir}** open.\n\n{_status_md()}", _status_md(), _img_choices()


def do_sample(n, smart, progress=gr.Progress()):
    if ENG is None:
        return "Open a project first.", _status_md(), gr.update()
    progress(0.05, desc="loading model + extracting…")
    rep = ENG.sample_more(int(n), smart=bool(smart))
    progress(1.0, desc="done")
    return (f"Added **{rep['n_new_images']}** images / **{rep['n_new_instances']}** instances "
            f"(after class-agnostic NMS).\n\n{_status_md()}", _status_md(), _img_choices())


def do_dedup(iou):
    if ENG is None:
        return "Open a project first.", _status_md()
    n = ENG.dedup_current(float(iou))
    return f"Sent **{n}** duplicate instances to background.\n\n{_status_md()}", _status_md()


def do_reset(confirm):
    if ENG is None:
        return "Open a project first.", _status_md()
    if not confirm:
        return "Tick the confirm box first — this drops the collection, all instances, assignments and classes.", _status_md()
    ENG.reset(keep_config=True)
    return f"**Dropped everything** (config kept). Re-run Sample & extract.\n\n{_status_md()}", _status_md()


def do_cluster(feat_methods, distance, per_image, force_n):
    if ENG is None:
        return "Open a project first.", _status_md(), gr.update(), gr.update(), gr.update(), gr.update()
    spec = {m: 1.0 for m in feat_methods} or {"decoder": 1.0}
    info = ENG.cluster(spec, distance=distance, per_image=bool(per_image),
                       req_clust=(int(force_n) if force_n and int(force_n) > 0 else None))
    levels = [f"L{i} ({c} clusters)" for i, c in enumerate(info["counts"])]
    pids = [str(r["pid"]) for r in ENG.partition_view()]
    return (f"FINCH on the unassigned pool: levels {info['counts']} (showing L{info['level']}). "
            f"Assigned classes shown as standalone partitions.\n\n{_status_md()}",
            _status_md(), gr.update(choices=levels, value=levels[info["level"]] if levels else None),
            gr.update(value=_partition_rows()), gr.update(choices=pids), _img_choices())


# ---- Partitions ------------------------------------------------------------
def on_level_change(level_label):
    if ENG is None or ENG._cluster is None or not level_label:
        return gr.update(), [], None, [], "selected: 0", [], _status_md()
    ENG.set_level(int(level_label.split()[0][1:]))
    return gr.update(value=_partition_rows()), [], None, [], "selected: 0", [], _status_md()


def on_partition_select(mask_overlay, view_mode, evt: gr.SelectData):
    if ENG is None or ENG._cluster is None:
        return [], None, [], "selected: 0", []
    rows = _partition_rows()
    ridx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if ridx is None or ridx >= len(rows):
        return [], None, [], "selected: 0", []
    pid = str(rows[ridx][0])
    return _load_partition(pid, mask_overlay, view_mode), pid, [], "selected: 0", []


def on_view_change(sel_partition, sel_indices, mask_overlay, view_mode):
    if ENG is None or sel_partition is None:
        return [], []
    return _load_partition(sel_partition, mask_overlay, view_mode), _sel_strip(sel_partition, sel_indices)


def on_part_gallery_select(sel_partition, sel_indices, evt: gr.SelectData):
    s = set(sel_indices or []); s.symmetric_difference_update({int(evt.index)}); s = sorted(s)
    return s, f"selected: {len(s)}", _sel_strip(sel_partition, s)        # NO main-gallery re-emit


def do_step_partition(sel_partition, mask_overlay, view_mode, delta):
    pids = [str(r[0]) for r in _partition_rows()]
    if not pids:
        return [], None, [], "selected: 0", []
    idx = (pids.index(str(sel_partition)) + delta) if (sel_partition is not None and str(sel_partition) in pids) else 0
    pid = pids[idx % len(pids)]
    return _load_partition(pid, mask_overlay, view_mode), pid, [], "selected: 0", []


def _sel_iuids(sel_partition, sel_indices):
    iuids = ENG.partition_iuids(sel_partition)
    return [iuids[i] for i in (sel_indices or []) if i < len(iuids)]


def _after_part_mutation(sel_partition, mask_overlay, view_mode):
    return (gr.update(value=_partition_rows()), _status_md(),
            _load_partition(sel_partition, mask_overlay, view_mode) if sel_partition is not None else [],
            [], "selected: 0", [])                                       # part_df, status, gallery, sel_indices, count, strip


def do_assign_partition(sel_partition, class_name, mask_overlay, view_mode):
    if ENG and sel_partition is not None and class_name:
        ENG.assign_partition(sel_partition, class_name)
    return *_after_part_mutation(sel_partition, mask_overlay, view_mode), *_refresh_classes()


def do_assign_selected(sel_partition, sel_indices, class_name, mask_overlay, view_mode):
    if ENG and sel_partition is not None and class_name:
        ENG.assign(_sel_iuids(sel_partition, sel_indices), class_name)
    return *_after_part_mutation(sel_partition, mask_overlay, view_mode), *_refresh_classes()


def do_reject_selected(sel_partition, sel_indices, mask_overlay, view_mode):
    if ENG and sel_partition is not None:
        ENG.set_background(_sel_iuids(sel_partition, sel_indices))
    return _after_part_mutation(sel_partition, mask_overlay, view_mode)


def do_remove_selected(sel_partition, sel_indices, mask_overlay, view_mode):
    if ENG and sel_partition is not None:
        ENG.remove_from_class(_sel_iuids(sel_partition, sel_indices))
    return _after_part_mutation(sel_partition, mask_overlay, view_mode)


def do_merge_same_image(sel_partition, mask_overlay, view_mode):
    if ENG and sel_partition is not None:
        ENG.merge_partition_by_image(sel_partition)
    return _after_part_mutation(sel_partition, mask_overlay, view_mode)


def do_open_source(sel_partition, sel_indices):
    if ENG is None or sel_partition is None:
        return gr.update(), gr.update(), None, [], [], "selected: 0", []
    ius = _sel_iuids(sel_partition, sel_indices) or ENG.partition_iuids(sel_partition)
    if not ius:
        return gr.update(), gr.update(), None, [], [], "selected: 0", []
    iid = ENG.state.meta[ius[0]].image_id
    crops, iuids = ENG.image_instance_gallery(iid, mask_overlay=True)
    return gr.Tabs(selected="tab_inimg"), gr.update(value=str(iid)), ENG.image_overlay(iid), crops, iuids, "selected: 0", []


# ---- Refine: ordered op-stack (capped + thumbnailed) -----------------------
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
        parts.append(f"{i+1}.{o['name']}" + (f"({','.join(f'{k}={v}' for k,v in kw.items())})" if kw else ""))
    return "**Op chain (in order):** " + " → ".join(parts)


_REFINE_CAP = 8


def _refine_target_iuids(target, limit=_REFINE_CAP):
    if not target:
        return []
    if target["kind"] == "partition":
        return ENG.partition_iuids(target["pid"])[:limit]
    return list(target["iuids"])[:limit]


def _render_refine(target, op_stack, mask_overlay):
    if ENG is None or not target:
        return [], "(no target — use **Send to Refine** from the Partitions tab)"
    iuids = _refine_target_iuids(target)
    items = []
    for u in iuids:
        b, a = ENG.refine_preview(u, op_stack or [], mask_overlay=mask_overlay)
        items.append((_compose(b, a), u[:6]))
    if target["kind"] == "partition":
        n = len(ENG.partition_iuids(target["pid"]))
        banner = f"**partition {target['pid']}** — {n} instances (previewing first {min(n,_REFINE_CAP)}) · left=before, right=after"
    else:
        n = len(target["iuids"])
        banner = f"**{n} instance(s)** (previewing first {min(n,_REFINE_CAP)}) · left=before, right=after"
    return items, banner


def do_send_refine_instance(sel_partition, sel_indices):
    if ENG is None or sel_partition is None:
        return gr.update(), None, [], "(no target)", [], _stack_md([])
    iuids = _sel_iuids(sel_partition, sel_indices) or ENG.partition_iuids(sel_partition)
    target = {"kind": "instance", "iuids": iuids}
    items, banner = _render_refine(target, [], True)
    return gr.Tabs(selected="tab_refine"), target, items, banner, [], _stack_md([])


def do_send_refine_partition(sel_partition):
    if ENG is None or sel_partition is None:
        return gr.update(), None, [], "(no target)", [], _stack_md([])
    target = {"kind": "partition", "pid": sel_partition}
    items, banner = _render_refine(target, [], True)
    return gr.Tabs(selected="tab_refine"), target, items, banner, [], _stack_md([])


def do_add_op(name, op_stack, thr_val, dk, ek, contrast, target, mask_overlay):
    st = list(op_stack or [])
    if name == "threshold":
        st.append({"name": "threshold", "kw": {"val": int(thr_val)}})
    elif name == "dilate":
        st.append({"name": "dilate", "kw": {"k": int(dk), "max_contrast": float(contrast)}})
    elif name == "erode":
        st.append({"name": "erode", "kw": {"k": int(ek), "min_contrast": float(contrast)}})
    else:
        st.append({"name": name})
    items, banner = _render_refine(target, st, mask_overlay)
    return st, _stack_md(st), items, banner


def do_remove_op(op_stack, target, mask_overlay):
    st = list(op_stack or [])[:-1]
    items, banner = _render_refine(target, st, mask_overlay)
    return st, _stack_md(st), items, banner


def do_clear_ops(target, mask_overlay):
    items, banner = _render_refine(target, [], mask_overlay)
    return [], _stack_md([]), items, banner


def do_refine_rerender(target, op_stack, mask_overlay):
    return _render_refine(target, op_stack, mask_overlay)


def do_refine_apply(target, op_stack, mask_overlay):
    if ENG is None or not target:
        return _status_md(), gr.update(), [], "(no target)"
    ops = op_stack or []
    if target["kind"] == "partition":
        ENG.apply_refine_partition(target["pid"], ops)
    else:
        ENG.apply_refine_many(list(target["iuids"]), ops)
    items, banner = _render_refine(target, [], mask_overlay)
    return _status_md(), gr.update(value=_partition_rows()), items, banner


def do_refine_revert(target, mask_overlay):
    if ENG is None or not target:
        return _status_md(), gr.update(), [], "(no target)"
    for u in _refine_target_iuids(target, limit=10 ** 9):
        ENG.revert_refine(u)
    items, banner = _render_refine(target, [], mask_overlay)
    return _status_md(), gr.update(value=_partition_rows()), items, banner


# ---- In-image --------------------------------------------------------------
def on_image_pick(image_id, color_by):
    if ENG is None or not image_id:
        return None, [], [], "selected: 0", []
    iid = int(image_id)
    crops, iuids = ENG.image_instance_gallery(iid, mask_overlay=True)
    return ENG.image_overlay(iid, color_by=color_by), crops, iuids, "selected: 0", []


def on_inst_gallery_select(inimg_iuids, inimg_sel, evt: gr.SelectData):
    idx = int(evt.index); s = set(inimg_sel or [])
    if inimg_iuids and idx < len(inimg_iuids):
        s.symmetric_difference_update({inimg_iuids[idx]})
    s = sorted(s)
    return s, f"selected: {len(s)}", _iuid_strip(s)                      # NO inst_gallery re-emit


def on_canvas_click(image_id, inimg_iuids, inimg_sel, evt: gr.SelectData):
    s = set(inimg_sel or [])
    idx = evt.index
    if ENG and image_id and isinstance(idx, (list, tuple)) and len(idx) >= 2:
        u = ENG.instance_at_pixel(int(image_id), int(idx[0]), int(idx[1]))
        if u:
            s.symmetric_difference_update({u})
    s = sorted(s)
    return s, f"selected: {len(s)}", _iuid_strip(s)


def do_merge_selected_inimage(image_id, inimg_sel, color_by):
    if ENG and inimg_sel and len(inimg_sel) >= 2:
        ENG.merge_instances(list(inimg_sel))
    iid = int(image_id)
    crops, iuids = ENG.image_instance_gallery(iid, mask_overlay=True)
    return (ENG.image_overlay(iid, color_by=color_by), crops, iuids, [], "selected: 0", [],
            _status_md(), gr.update(value=_partition_rows()))


def do_merge_preview(image_id, dist_kind, method, thresh, max_grp):
    if ENG is None or not image_id:
        return None, None, []
    mg = None if int(max_grp) == 0 else int(max_grp)
    b, a, groups = ENG.merge_preview(int(image_id), dist_kind=dist_kind, method=method, thresh=float(thresh), max_group_size=mg)
    return b, a, groups


def do_commit_merge(image_id, groups, color_by):
    if ENG is None or not image_id:
        return None, _status_md(), gr.update()
    ENG.commit_merge(int(image_id), groups)
    return ENG.image_overlay(int(image_id), color_by=color_by), _status_md(), gr.update(value=_partition_rows())


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


def do_unreject(which, bg_iuids, bg_sel):
    if ENG is None:
        return [], [], "0 rejected", [], _status_md(), gr.update()
    targets = list(ENG.background_iuids()) if which == "all" else list(bg_sel or [])
    ENG.unreject(targets)
    return *do_load_rejected(), _status_md(), gr.update(value=_partition_rows())


# ---- Classifier ------------------------------------------------------------
def do_train(feat_methods, algo, openset):
    if ENG is None:
        return "Open a project first.", None
    spec = {m: 1.0 for m in feat_methods} or {"decoder": 1.0}
    rep = ENG.train_classifier(spec, algo=algo, use_unassigned_negatives=bool(openset))
    if "error" in rep:
        return rep["error"], None
    mode = "open-set (this·vs·not-this × this·vs·others)" if openset else "vs-background-only"
    return (f"Trained [{mode}]: **{rep['n']}** assigned across **{rep['n_classes']}** classes; "
            f"negatives = {rep['n_background']} bg + {rep['n_unassigned_neg']} unassigned.", _pr_fig(rep.get("pr", {})))


def _pr_fig(pr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, cur in pr.get("curves", {}).items():
        t = cur.get("thresholds", [])
        if t:
            ax.plot(t, cur["precision"][:len(t)], "-", label=f"{c} P")
            ax.plot(t, cur["recall"][:len(t)], "--", label=f"{c} R")
    ax.set_xlabel("threshold"); ax.set_ylabel("P / R"); ax.legend(fontsize=7); ax.set_title("CV-OOF P/R (bg=neg)")
    plt.tight_layout()
    return fig


def do_predict(thresh):
    if ENG is None or getattr(ENG, "_clf", None) is None:
        return "Train a classifier first.", None
    preds = ENG.predict_and_threshold(float(thresh))
    return (f"{len(preds)} instances would be assigned at thresh={thresh}.",
            [[u[:8], ENG.state.class_name(cid), round(conf, 3)] for u, cid, conf in preds[:200]])


def do_apply_predictions(thresh):
    if ENG is None or getattr(ENG, "_clf", None) is None:
        return "Train a classifier first.", _status_md(), gr.update()
    n = ENG.apply_predictions(float(thresh))
    return f"Assigned {n} instances.", _status_md(), gr.update(value=_partition_rows())


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


def do_assign_cluster(cluster_id, class_name):
    if ENG and cluster_id is not None and class_name and ENG._cluster is not None:
        ENG.assign_partition(cluster_id, class_name)
    return _status_md(), gr.update(value=_partition_rows()), *_refresh_classes()


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


def do_undo():
    if ENG:
        ENG.undo()
    return _status_md(), gr.update(value=_partition_rows())


def do_redo():
    if ENG:
        ENG.redo()
    return _status_md(), gr.update(value=_partition_rows())


# --------------------------------------------------------------------------- #
def build_app(default_project: str = "/tmp/curator_project") -> gr.Blocks:
    d = _defaults()
    FEATS = ["decoder", "maskpool", "roialign", "backbone", "shape", "shapecoord", "coords", "raddino"]
    with gr.Blocks(title="qseg curator") as demo:
        with gr.Row():
            undo_btn = gr.Button("↶ Undo", scale=0, elem_id="kb_undo")
            redo_btn = gr.Button("↷ Redo", scale=0, elem_id="kb_redo")
            status = gr.Markdown(_status_md())
        class_dds: list = []
        sel_partition = gr.State(None); sel_indices = gr.State([])
        pending_groups = gr.State([]); refine_target = gr.State(None); op_stack = gr.State([])
        inimg_sel = gr.State([]); inimg_iuids = gr.State([])
        bg_iuids = gr.State([]); bg_sel = gr.State([])

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
                feat_cbg = gr.CheckboxGroup(FEATS, value=["decoder", "coords"], label="Feature types")
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
                gr.Markdown("Shortcuts: **a** assign partition · **s** assign selected · **r** reject · **u** unassign · **z/y** undo/redo · **[ ]** prev/next. Selecting shows a thumbnail strip below (the main gallery is not re-rendered, for speed).")
                level_dd = gr.Dropdown(label="FINCH level (unassigned pool)", choices=[], interactive=True)
                with gr.Row():
                    with gr.Column(scale=1):
                        part_df = gr.Dataframe(headers=["pid", "size", "purity", "score", "class"],
                                               datatype=["str", "number", "number", "number", "str"],
                                               interactive=False, label="partitions (click a row)", max_height=900)
                        with gr.Row():
                            prev_btn = gr.Button("◀ prev", elem_id="kb_prev")
                            next_btn = gr.Button("next ▶", elem_id="kb_next")
                    with gr.Column(scale=2):
                        with gr.Row():
                            mask_toggle = gr.Checkbox(label="mask overlay", value=True)
                            view_mode = gr.Radio(["crop", "in context"], value="crop", label="view")
                            part_count = gr.Markdown("selected: 0")
                        part_gallery = gr.Gallery(label="instances (click to (de)select)", columns=6, height=380, allow_preview=True)
                        part_sel_strip = gr.Gallery(label="selected", columns=8, height=110, allow_preview=False)
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

            with gr.Tab("In-image", id="tab_inimg"):
                with gr.Row():
                    image_dd = gr.Dropdown(label="image_id", choices=[], interactive=True)
                    colorby_radio = gr.Radio(["partition", "class"], value="partition", label="color by")
                inimg = gr.Image(label="instances (click an instance to (de)select for merge)", height=440)
                gr.Markdown("**Manual merge** — click instance crops (or the image above), then merge:")
                with gr.Row():
                    inimg_count = gr.Markdown("selected: 0")
                    merge_sel_btn = gr.Button("Merge selected → one instance", variant="primary")
                inst_gallery = gr.Gallery(label="this image's instances", columns=8, height=170, allow_preview=False)
                inimg_sel_strip = gr.Gallery(label="selected", columns=8, height=110, allow_preview=False)
                gr.Markdown("**Distance merge** — auto-group then commit:")
                with gr.Row():
                    mdist_dd = gr.Dropdown(["mask_gap", "feature", "centroid", "combo"], value="mask_gap", label="dist")
                    mmeth_dd = gr.Dropdown(["decoder", "maskpool", "backbone"], value="decoder", label="feature")
                    mthr_sl = gr.Slider(0, 0.5, value=0.05, step=0.005, label="threshold")
                    mgrp_sl = gr.Slider(0, 8, value=3, step=1, label="max group (0=any)")
                with gr.Row():
                    before_img = gr.Image(label="before", height=320)
                    after_img = gr.Image(label="after", height=320)
                commit_btn = gr.Button("Commit distance merge", variant="primary")

            with gr.Tab("Refine", id="tab_refine"):
                refine_banner = gr.Markdown("(no target — use **Send to Refine** from the Partitions tab)")
                refine_mask = gr.Checkbox(label="show mask overlay (off = raw image)", value=True)
                with gr.Row():
                    thr_val = gr.Slider(0, 255, value=128, step=1, label="threshold val")
                    dk_sl = gr.Slider(1, 5, value=2, step=1, label="dilate k")
                    ek_sl = gr.Slider(1, 5, value=2, step=1, label="erode k")
                    contrast_sl = gr.Slider(0, 1, value=0.2, step=0.01, label="contrast gate")
                gr.Markdown("Add operations **in order** (applied top-to-bottom). Preview shows the first 8 instances.")
                with gr.Row():
                    add_otsu = gr.Button("+ otsu"); add_thr = gr.Button("+ threshold")
                    add_dil = gr.Button("+ dilate"); add_ero = gr.Button("+ erode")
                    add_fill = gr.Button("+ fill"); add_lcc = gr.Button("+ largest CC"); add_sm = gr.Button("+ smooth")
                with gr.Row():
                    remove_op_btn = gr.Button("remove last"); clear_op_btn = gr.Button("clear chain")
                stack_md = gr.Markdown(_stack_md([]))
                refine_gallery = gr.Gallery(label="before | after (per instance, synced)", columns=3, height=380, allow_preview=True)
                with gr.Row():
                    refine_apply = gr.Button("Apply chain", variant="primary"); refine_revert = gr.Button("Revert")

            with gr.Tab("Classifier"):
                clf_feat = gr.CheckboxGroup(FEATS, value=["decoder", "shape"], label="classifier features")
                with gr.Row():
                    clf_algo = gr.Radio(["logreg", "rf"], value="logreg", label="model")
                    clf_openset = gr.Checkbox(value=True, label="open-set: unassigned+background as negatives")
                train_btn = gr.Button("Train on assigned", variant="primary")
                clf_msg = gr.Markdown(); pr_plot = gr.Plot(label="P/R vs threshold")
                clf_thr = gr.Slider(0, 1, value=0.5, step=0.01, label="assignment threshold")
                pred_df = gr.Dataframe(headers=["iuid", "pred class", "conf"], interactive=False)
                with gr.Row():
                    predict_btn = gr.Button("Preview predictions"); apply_pred_btn = gr.Button("Apply", variant="primary")

            with gr.Tab("Map"):
                with gr.Row():
                    map_method = gr.Radio(["pca", "umap", "tsne"], value="pca", label="embedding")
                    map_colorby = gr.Radio(["cluster", "class"], value="cluster", label="color by")
                    map_btn = gr.Button("Compute map", variant="primary")
                map_plot = gr.Plot(label="2D embedding (hover a point for a preview)", elem_id="map_plot")
                with gr.Row():
                    map_cluster_dd = gr.Dropdown(label="cluster id", choices=[], interactive=True)
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
            class_dds.append(gr.Dropdown(visible=False, allow_custom_value=True))

        # ---- wiring ----
        open_btn.click(do_open_project, [proj_tb, ckpt_tb, cfgname_tb, overrides_tb, root_tb, score_sl, nms_sl], [cfg_status, status, image_dd])
        sample_btn.click(do_sample, [nimg_sl, smart_cb], [cfg_status, status, image_dd])
        dedup_btn.click(do_dedup, [nms_sl], [cfg_status, status])
        reset_btn.click(do_reset, [reset_confirm], [cfg_status, status])
        cluster_btn.click(do_cluster, [feat_cbg, dist_dd, perimg_cb, forcen_num], [cfg_status, status, level_dd, part_df, map_cluster_dd, image_dd])

        psel = [part_gallery, sel_partition, sel_indices, part_count, part_sel_strip]   # partition-select outputs
        amut = [part_df, status, part_gallery, sel_indices, part_count, part_sel_strip] # mutation outputs (re-render)
        level_dd.change(on_level_change, [level_dd], [part_df, part_gallery, sel_partition, sel_indices, part_count, part_sel_strip, status])
        part_df.select(on_partition_select, [mask_toggle, view_mode], psel)
        mask_toggle.change(on_view_change, [sel_partition, sel_indices, mask_toggle, view_mode], [part_gallery, part_sel_strip])
        view_mode.change(on_view_change, [sel_partition, sel_indices, mask_toggle, view_mode], [part_gallery, part_sel_strip])
        part_gallery.select(on_part_gallery_select, [sel_partition, sel_indices], [sel_indices, part_count, part_sel_strip])
        prev_btn.click(lambda sp, m, v: do_step_partition(sp, m, v, -1), [sel_partition, mask_toggle, view_mode], psel)
        next_btn.click(lambda sp, m, v: do_step_partition(sp, m, v, 1), [sel_partition, mask_toggle, view_mode], psel)
        assign_all.click(do_assign_partition, [sel_partition, pclass_dd, mask_toggle, view_mode], [*amut, *class_dds])
        assign_sel.click(do_assign_selected, [sel_partition, sel_indices, pclass_dd, mask_toggle, view_mode], [*amut, *class_dds])
        remove_sel.click(do_remove_selected, [sel_partition, sel_indices, mask_toggle, view_mode], amut)
        reject_sel.click(do_reject_selected, [sel_partition, sel_indices, mask_toggle, view_mode], amut)
        merge_img_btn.click(do_merge_same_image, [sel_partition, mask_toggle, view_mode], amut)
        open_src_btn.click(do_open_source, [sel_partition, sel_indices], [tabs, image_dd, inimg, inst_gallery, inimg_iuids, inimg_count, inimg_sel_strip])
        send_refine_inst.click(do_send_refine_instance, [sel_partition, sel_indices], [tabs, refine_target, refine_gallery, refine_banner, op_stack, stack_md])
        send_refine_part.click(do_send_refine_partition, [sel_partition], [tabs, refine_target, refine_gallery, refine_banner, op_stack, stack_md])

        image_dd.change(on_image_pick, [image_dd, colorby_radio], [inimg, inst_gallery, inimg_iuids, inimg_count, inimg_sel_strip])
        inst_gallery.select(on_inst_gallery_select, [inimg_iuids, inimg_sel], [inimg_sel, inimg_count, inimg_sel_strip])
        inimg.select(on_canvas_click, [image_dd, inimg_iuids, inimg_sel], [inimg_sel, inimg_count, inimg_sel_strip])
        merge_sel_btn.click(do_merge_selected_inimage, [image_dd, inimg_sel, colorby_radio],
                            [inimg, inst_gallery, inimg_iuids, inimg_sel, inimg_count, inimg_sel_strip, status, part_df])
        for comp in (mdist_dd, mmeth_dd, mthr_sl, mgrp_sl):
            comp.change(do_merge_preview, [image_dd, mdist_dd, mmeth_dd, mthr_sl, mgrp_sl], [before_img, after_img, pending_groups])
        commit_btn.click(do_commit_merge, [image_dd, pending_groups, colorby_radio], [inimg, status, part_df])

        ro = [op_stack, stack_md, refine_gallery, refine_banner]
        ain = [op_stack, thr_val, dk_sl, ek_sl, contrast_sl, refine_target, refine_mask]
        addin = lambda nm: (lambda st, tv, d_, e_, c_, tg, mo: do_add_op(nm, st, tv, d_, e_, c_, tg, mo))
        add_otsu.click(addin("otsu"), ain, ro); add_thr.click(addin("threshold"), ain, ro)
        add_dil.click(addin("dilate"), ain, ro); add_ero.click(addin("erode"), ain, ro)
        add_fill.click(addin("fill"), ain, ro); add_lcc.click(addin("largest_cc"), ain, ro); add_sm.click(addin("smooth"), ain, ro)
        remove_op_btn.click(do_remove_op, [op_stack, refine_target, refine_mask], ro)
        clear_op_btn.click(do_clear_ops, [refine_target, refine_mask], ro)
        refine_mask.change(do_refine_rerender, [refine_target, op_stack, refine_mask], [refine_gallery, refine_banner])
        refine_apply.click(do_refine_apply, [refine_target, op_stack, refine_mask], [status, part_df, refine_gallery, refine_banner])
        refine_revert.click(do_refine_revert, [refine_target, refine_mask], [status, part_df, refine_gallery, refine_banner])

        train_btn.click(do_train, [clf_feat, clf_algo, clf_openset], [clf_msg, pr_plot])
        predict_btn.click(do_predict, [clf_thr], [clf_msg, pred_df])
        apply_pred_btn.click(do_apply_predictions, [clf_thr], [clf_msg, status, part_df])

        map_btn.click(do_map, [map_method, map_colorby], [map_plot, map_cluster_dd])
        map_assign_btn.click(do_assign_cluster, [map_cluster_dd, map_class_dd], [status, part_df, *class_dds])

        load_bg_btn.click(do_load_rejected, [], [bg_gallery, bg_iuids, bg_count, bg_sel])
        bg_gallery.select(on_bg_gallery_select, [bg_iuids, bg_sel], [bg_sel, bg_count])
        unreject_sel_btn.click(lambda i, s: do_unreject("sel", i, s), [bg_iuids, bg_sel], [bg_gallery, bg_iuids, bg_count, bg_sel, status, part_df])
        unreject_all_btn.click(lambda i, s: do_unreject("all", i, s), [bg_iuids, bg_sel], [bg_gallery, bg_iuids, bg_count, bg_sel, status, part_df])

        export_btn.click(do_export, [exp_classes, exp_scope, exp_kpts, exp_fmt], [exp_msg, exp_file])
        undo_btn.click(do_undo, [], [status, part_df]); redo_btn.click(do_redo, [], [status, part_df])
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
