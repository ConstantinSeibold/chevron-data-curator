"""Gradio web app for the qseg instance curator.

Run:  python -m tools.curator.app  [--project DIR] [--port 7860] [--share]

Binds Gradio widgets to ONE server-side CuratorEngine (the project is large + GPU-backed
-> single user). gr.State holds only small cursors (selected partition / gallery indices /
current image / pending merge / map selection). Tabs: Config | Partitions | In-image |
Refine | Classifier | Map | Export.
"""
from __future__ import annotations

import argparse

import gradio as gr
import numpy as np

from .engine import CuratorEngine

ENG: CuratorEngine | None = None        # server-side singleton


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _defaults() -> dict:
    """Pre-fill the Config tab with the synthfb M2F (synth->real) model + RANZCR images."""
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
    star = " · clustering **stale**" if s["dirty"] else ""
    return (f"**{s['n_instances']}** instances · **{s['n_assigned']}** assigned · "
            f"**{s['n_unassigned']}** unassigned · **{s['n_background']}** rejected · "
            f"**{s['n_classes']}** classes · undo {s['undo']}/redo {s['redo']}{star}")


def _class_choices():
    return ENG.state.class_names() if ENG else []


def _refresh_classes():
    """gr.update for every class dropdown (fan-out of the canonical class list)."""
    ch = _class_choices()
    return [gr.update(choices=ch) for _ in range(4)]


def _partition_rows():
    if ENG is None or ENG._cluster is None:
        return []
    return [[r["pid"], r["size"], (round(r["purity"], 2) if r["purity"] is not None else None),
             round(r["mean_score"], 2), r["majority_class"] or ""] for r in ENG.partition_view()]


# --------------------------------------------------------------------------- #
# Config tab
# --------------------------------------------------------------------------- #
def do_open_project(project_dir, ckpt, config_name, overrides_text, root, score_thr):
    global ENG
    overrides = [ln.strip() for ln in overrides_text.splitlines() if ln.strip()]
    ENG = CuratorEngine(project_dir)
    if not ENG.store.is_project():
        ENG.init_project({"images": {"root": root},
                          "model": {"ckpt": ckpt, "config_name": config_name,
                                    "overrides": overrides, "score_thresh": float(score_thr)},
                          "features": {"model_features": ["decoder", "maskpool", "roialign", "backbone"],
                                       "handcrafted": {"shape": True, "shape_coords_extra": True},
                                       "raddino": False}})
    return f"Project **{project_dir}** open.\n\n{_status_md()}", _status_md()


def do_sample(n, smart, progress=gr.Progress()):
    if ENG is None:
        return "Open a project first.", _status_md(), gr.update()
    progress(0.05, desc="loading model + extracting…")
    rep = ENG.sample_more(int(n), smart=bool(smart))
    progress(1.0, desc="done")
    return (f"Added **{rep['n_new_images']}** images / **{rep['n_new_instances']}** instances.\n\n{_status_md()}",
            _status_md(), gr.update())


def do_cluster(feat_methods, distance, per_image):
    if ENG is None:
        return "Open a project first.", _status_md(), gr.update(choices=[], value=None), gr.update()
    spec = {m: 1.0 for m in feat_methods} or {"decoder": 1.0}
    info = ENG.cluster(spec, distance=distance, per_image=bool(per_image))
    levels = [f"L{i} ({c} clusters)" for i, c in enumerate(info["counts"])]
    return (f"FINCH levels: {info['counts']} (default L{info['level']}).\n\n{_status_md()}",
            _status_md(), gr.update(choices=levels, value=levels[info["level"]]),
            gr.update(value=_partition_rows()))


# --------------------------------------------------------------------------- #
# Partitions tab
# --------------------------------------------------------------------------- #
def on_level_change(level_label):
    if ENG is None or ENG._cluster is None:
        return gr.update(), _status_md()
    ENG.set_level(int(level_label.split()[0][1:]))
    return gr.update(value=_partition_rows()), _status_md()


def on_partition_select(mask_overlay, evt: gr.SelectData):
    if ENG is None or ENG._cluster is None:
        return [], None, []
    rows = _partition_rows()
    ridx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if ridx is None or ridx >= len(rows):
        return [], None, []
    pid = int(rows[ridx][0])
    crops, iuids = ENG.partition_crops(pid, mask_overlay=bool(mask_overlay))
    return [im for im, _ in crops], pid, []         # gallery, sel_partition, clear sel_indices


def on_mask_toggle(sel_partition, mask_overlay):
    if ENG is None or sel_partition is None:
        return []
    crops, _ = ENG.partition_crops(int(sel_partition), mask_overlay=bool(mask_overlay))
    return [im for im, _ in crops]


def on_gallery_select(sel_indices, evt: gr.SelectData):
    idx = int(evt.index)
    s = set(sel_indices or [])
    s.symmetric_difference_update({idx})            # toggle
    return sorted(s)


def _sel_iuids(sel_partition, sel_indices):
    iuids = ENG.partition_iuids(int(sel_partition))
    return [iuids[i] for i in (sel_indices or []) if i < len(iuids)]


def do_assign_partition(sel_partition, class_name):
    if ENG is None or sel_partition is None or not class_name:
        return gr.update(value=_partition_rows()), _status_md(), *_refresh_classes()
    ENG.assign_partition(int(sel_partition), class_name)
    return gr.update(value=_partition_rows()), _status_md(), *_refresh_classes()


def do_assign_selected(sel_partition, sel_indices, class_name):
    if ENG is None or sel_partition is None or not class_name:
        return gr.update(value=_partition_rows()), _status_md(), *_refresh_classes()
    ENG.assign(_sel_iuids(sel_partition, sel_indices), class_name)
    return gr.update(value=_partition_rows()), _status_md(), *_refresh_classes()


def do_reject_selected(sel_partition, sel_indices):
    if ENG is None or sel_partition is None:
        return gr.update(value=_partition_rows()), _status_md()
    ENG.set_background(_sel_iuids(sel_partition, sel_indices))
    return gr.update(value=_partition_rows()), _status_md()


def do_remove_selected(sel_partition, sel_indices):
    if ENG is None or sel_partition is None:
        return gr.update(value=_partition_rows()), _status_md()
    ENG.remove_from_class(_sel_iuids(sel_partition, sel_indices))
    return gr.update(value=_partition_rows()), _status_md()


# --------------------------------------------------------------------------- #
# In-image tab
# --------------------------------------------------------------------------- #
def on_image_pick(image_id, color_by):
    if ENG is None or not image_id:
        return None, None
    iid = int(image_id)
    return ENG.image_overlay(iid, color_by=color_by), ENG.keypoints_overlay(iid)


def do_merge_preview(image_id, dist_kind, method, thresh, max_grp):
    if ENG is None or not image_id:
        return None, None, []
    mg = None if int(max_grp) == 0 else int(max_grp)
    before, after, groups = ENG.merge_preview(int(image_id), dist_kind=dist_kind, method=method,
                                              thresh=float(thresh), max_group_size=mg)
    return before, after, groups


def do_commit_merge(image_id, groups, color_by):
    if ENG is None or not image_id:
        return None, _status_md(), gr.update()
    ENG.commit_merge(int(image_id), groups)
    return ENG.image_overlay(int(image_id), color_by=color_by), _status_md(), gr.update(value=_partition_rows())


# --------------------------------------------------------------------------- #
# Refine tab
# --------------------------------------------------------------------------- #
def _ops_from_controls(thr_mode, thr_val, dilate, erode, contrast, fill, largest, smooth):
    ops = []
    if thr_mode == "otsu":
        ops.append({"name": "otsu"})
    elif thr_mode == "manual":
        ops.append({"name": "threshold", "kw": {"val": int(thr_val)}})
    if int(dilate) > 0:
        ops.append({"name": "dilate", "kw": {"k": int(dilate), "max_contrast": float(contrast)}})
    if int(erode) > 0:
        ops.append({"name": "erode", "kw": {"k": int(erode), "min_contrast": float(contrast)}})
    if fill:
        ops.append({"name": "fill"})
    if largest:
        ops.append({"name": "largest_cc"})
    if smooth:
        ops.append({"name": "smooth"})
    return ops


def do_refine_preview(refine_iuid, thr_mode, thr_val, dilate, erode, contrast, fill, largest, smooth):
    if ENG is None or not refine_iuid:
        return None, None, []
    ops = _ops_from_controls(thr_mode, thr_val, dilate, erode, contrast, fill, largest, smooth)
    o, r = ENG.refine_preview(refine_iuid, ops)
    return o, r, ops


def do_refine_apply(refine_iuid, ops):
    if ENG is None or not refine_iuid:
        return _status_md()
    ENG.apply_refine(refine_iuid, ops or [])
    return _status_md()


def do_refine_revert(refine_iuid):
    if ENG is None or not refine_iuid:
        return _status_md()
    ENG.revert_refine(refine_iuid)
    return _status_md()


# --------------------------------------------------------------------------- #
# Classifier tab
# --------------------------------------------------------------------------- #
def do_train(feat_methods, algo):
    if ENG is None:
        return "Open a project first.", None
    spec = {m: 1.0 for m in feat_methods} or {"decoder": 1.0}
    rep = ENG.train_classifier(spec, algo=algo)
    if "error" in rep:
        return rep["error"], None
    ENG._clf_spec_methods = feat_methods
    fig = _pr_fig(rep.get("pr", {}))
    txt = f"Trained ({rep['n']} samples, {rep['n_classes']} classes). cv_acc={rep.get('cv_acc','-')}"
    return txt, fig


def _pr_fig(pr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, cur in pr.get("curves", {}).items():
        if cur.get("thresholds"):
            t = cur["thresholds"]
            ax.plot(t, cur["precision"][:len(t)], "-", label=f"{c} P")
            ax.plot(t, cur["recall"][:len(t)], "--", label=f"{c} R")
    ax.set_xlabel("threshold"); ax.set_ylabel("P / R"); ax.legend(fontsize=7); ax.set_title("CV-OOF P/R vs threshold")
    plt.tight_layout()
    return fig


def do_predict(thresh):
    if ENG is None or getattr(ENG, "_clf", None) is None:
        return "Train a classifier first.", None
    preds = ENG.predict_and_threshold(float(thresh))
    rows = [[u[:8], ENG.state.class_name(cid), round(conf, 3)] for u, cid, conf in preds[:200]]
    return f"{len(preds)} instances would be assigned at thresh={thresh}.", rows


def do_apply_predictions(thresh):
    if ENG is None or getattr(ENG, "_clf", None) is None:
        return "Train a classifier first.", _status_md(), gr.update()
    n = ENG.apply_predictions(float(thresh))
    return f"Assigned {n} instances.", _status_md(), gr.update(value=_partition_rows())


# --------------------------------------------------------------------------- #
# Map tab
# --------------------------------------------------------------------------- #
def do_map(method, color_by):
    if ENG is None:
        return None, gr.update(choices=[])
    import plotly.express as px
    xy, labels, order = ENG.embed2d(method=method, color_by=color_by)
    fig = px.scatter(x=xy[:, 0], y=xy[:, 1], color=[str(int(l)) for l in labels],
                     title=f"{method} · color={color_by}", width=720, height=560)
    fig.update_traces(marker=dict(size=6)); fig.update_layout(showlegend=False)
    clusters = sorted(set(int(l) for l in labels))
    return fig, gr.update(choices=[str(c) for c in clusters])


def do_assign_cluster(cluster_id, class_name):
    if ENG is None or cluster_id is None or not class_name or ENG._cluster is None:
        return _status_md(), gr.update(value=_partition_rows()), *_refresh_classes()
    ENG.assign_partition(int(cluster_id), class_name)
    return _status_md(), gr.update(value=_partition_rows()), *_refresh_classes()


# --------------------------------------------------------------------------- #
# Export tab
# --------------------------------------------------------------------------- #
def do_export(class_subset, instance_scope, include_kpts, mask_fmt):
    if ENG is None:
        return "Open a project first.", None
    classes = None
    if class_subset:
        classes = [ENG.state.class_id_by_name(n) for n in class_subset if ENG.state.class_id_by_name(n)]
    p = ENG.export_coco(classes=classes, with_keypoints=bool(include_kpts),
                        polygon=(mask_fmt == "polygon"),
                        include_unassigned=(instance_scope == "all (incl. unassigned)"))
    import json
    coco = json.loads(p.read_text())
    return f"Exported {len(coco['annotations'])} anns / {len(coco['images'])} images -> {p}", str(p)


def do_undo():
    if ENG is None:
        return _status_md(), gr.update()
    ENG.undo()
    return _status_md(), gr.update(value=_partition_rows())


def do_redo():
    if ENG is None:
        return _status_md(), gr.update()
    ENG.redo()
    return _status_md(), gr.update(value=_partition_rows())


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_app(default_project: str = "/tmp/curator_project") -> gr.Blocks:
    d = _defaults()
    FEATS = ["decoder", "maskpool", "roialign", "backbone", "shape", "shapecoord", "coords", "raddino"]
    with gr.Blocks(title="qseg curator", fill_height=True) as demo:
        # ---- top bar ----
        with gr.Row():
            undo_btn = gr.Button("↶ Undo", scale=0)
            redo_btn = gr.Button("↷ Redo", scale=0)
            status = gr.Markdown(_status_md())
        class_dds: list = []          # collected for fan-out refresh
        sel_partition = gr.State(None)
        sel_indices = gr.State([])
        pending_groups = gr.State([])
        refine_ops = gr.State([])

        with gr.Tabs():
            # ===== Config =====
            with gr.Tab("Config"):
                proj_tb = gr.Textbox(label="Project dir", value=default_project)
                ckpt_tb = gr.Textbox(label="Seg-model weights", value=d["ckpt"])
                cfgname_tb = gr.Textbox(label="config name", value=d["config_name"])
                overrides_tb = gr.Textbox(label="model overrides (one per line)",
                                          value="\n".join(d["overrides"]), lines=6)
                root_tb = gr.Textbox(label="Root image folder", value=d["root"])
                with gr.Row():
                    score_sl = gr.Slider(0, 1, value=0.3, step=0.01, label="Score threshold")
                    nimg_sl = gr.Slider(1, 400, value=40, step=1, label="# images to sample")
                    smart_cb = gr.Checkbox(label="smart (low-confidence first)", value=False)
                with gr.Row():
                    open_btn = gr.Button("Create / Open project", variant="primary")
                    sample_btn = gr.Button("Sample & extract (additive)", variant="primary")
                feat_cbg = gr.CheckboxGroup(FEATS, value=["decoder", "coords"], label="Feature types (clustering/classifier)")
                with gr.Row():
                    dist_dd = gr.Dropdown(["cosine", "euclidean"], value="cosine", label="FINCH distance")
                    perimg_cb = gr.Checkbox(label="per-image clustering", value=False)
                    cluster_btn = gr.Button("Cluster (FINCH)", variant="primary")
                cfg_status = gr.Markdown()

            # ===== Partitions =====
            with gr.Tab("Partitions"):
                level_dd = gr.Dropdown(label="FINCH level", choices=[], interactive=True)
                with gr.Row():
                    with gr.Column(scale=1):
                        part_df = gr.Dataframe(headers=["pid", "size", "purity", "score", "class"],
                                               datatype=["number", "number", "number", "number", "str"],
                                               interactive=False, label="partitions (click a row)")
                    with gr.Column(scale=2):
                        mask_toggle = gr.Checkbox(label="mask overlay", value=True)
                        part_gallery = gr.Gallery(label="instances (click to (de)select)", columns=6, height=460,
                                                  allow_preview=True)
                        pclass_dd = gr.Dropdown(choices=_class_choices(), allow_custom_value=True,
                                                label="class (type to filter / enter new)")
                        class_dds.append(pclass_dd)
                        with gr.Row():
                            assign_all = gr.Button("Assign whole partition", variant="primary")
                            assign_sel = gr.Button("Assign selected")
                            remove_sel = gr.Button("Unassign selected")
                            reject_sel = gr.Button("Reject selected")

            # ===== In-image =====
            with gr.Tab("In-image"):
                with gr.Row():
                    image_dd = gr.Dropdown(label="image_id", choices=[], interactive=True)
                    colorby_radio = gr.Radio(["partition", "class"], value="partition", label="color by")
                with gr.Row():
                    inimg = gr.Image(label="instances", height=460)
                    kptimg = gr.Image(label="pred keypoints / centerlines", height=460)
                with gr.Row():
                    mdist_dd = gr.Dropdown(["mask_gap", "feature", "centroid", "combo"], value="mask_gap", label="dist kind")
                    mmeth_dd = gr.Dropdown(["decoder", "maskpool", "backbone"], value="decoder", label="feature")
                    mthr_sl = gr.Slider(0, 0.5, value=0.05, step=0.005, label="threshold")
                    mgrp_sl = gr.Slider(0, 8, value=3, step=1, label="max group (0=any)")
                with gr.Row():
                    before_img = gr.Image(label="before", height=360)
                    after_img = gr.Image(label="after (merged)", height=360)
                commit_btn = gr.Button("Commit merge", variant="primary")

            # ===== Refine =====
            with gr.Tab("Refine"):
                refine_iuid_tb = gr.Textbox(label="instance iuid (paste from a gallery caption)")
                with gr.Row():
                    refine_orig = gr.Image(label="crop", height=300)
                    refine_out = gr.Image(label="preview", height=300)
                with gr.Row():
                    thr_mode = gr.Radio(["off", "otsu", "manual"], value="off", label="threshold")
                    thr_val = gr.Slider(0, 255, value=128, step=1, label="manual val")
                    contrast_sl = gr.Slider(0, 1, value=0.2, step=0.01, label="contrast gate")
                with gr.Row():
                    dilate_sl = gr.Slider(0, 5, value=0, step=1, label="dilate")
                    erode_sl = gr.Slider(0, 5, value=0, step=1, label="erode")
                    fill_cb = gr.Checkbox(label="fill holes")
                    largest_cb = gr.Checkbox(label="largest CC")
                    smooth_cb = gr.Checkbox(label="smooth")
                with gr.Row():
                    refine_apply = gr.Button("Apply", variant="primary")
                    refine_revert = gr.Button("Revert")

            # ===== Classifier =====
            with gr.Tab("Classifier"):
                clf_feat = gr.CheckboxGroup(FEATS, value=["decoder", "shape"], label="classifier features")
                clf_algo = gr.Radio(["logreg", "rf"], value="logreg", label="model")
                train_btn = gr.Button("Train on assigned", variant="primary")
                clf_msg = gr.Markdown()
                pr_plot = gr.Plot(label="precision/recall vs threshold")
                clf_thr = gr.Slider(0, 1, value=0.5, step=0.01, label="assignment threshold")
                pred_df = gr.Dataframe(headers=["iuid", "pred class", "conf"], interactive=False)
                with gr.Row():
                    predict_btn = gr.Button("Preview predictions")
                    apply_pred_btn = gr.Button("Apply assignments", variant="primary")

            # ===== Map =====
            with gr.Tab("Map"):
                with gr.Row():
                    map_method = gr.Radio(["pca", "umap", "tsne"], value="pca", label="embedding")
                    map_colorby = gr.Radio(["cluster", "class"], value="cluster", label="color by")
                    map_btn = gr.Button("Compute map", variant="primary")
                map_plot = gr.Plot(label="2D embedding")
                with gr.Row():
                    map_cluster_dd = gr.Dropdown(label="cluster id", choices=[], interactive=True)
                    map_class_dd = gr.Dropdown(choices=_class_choices(), allow_custom_value=True, label="assign to class")
                    class_dds.append(map_class_dd)
                    map_assign_btn = gr.Button("Assign whole cluster", variant="primary")

            # ===== Export =====
            with gr.Tab("Export"):
                exp_classes = gr.CheckboxGroup(choices=_class_choices(), label="classes (empty = all)")
                class_dds.append(exp_classes)
                exp_scope = gr.Radio(["assigned only", "all (incl. unassigned)"], value="assigned only", label="scope")
                with gr.Row():
                    exp_kpts = gr.Checkbox(label="include keypoints", value=True)
                    exp_fmt = gr.Radio(["RLE", "polygon"], value="RLE", label="mask format")
                export_btn = gr.Button("Export COCO", variant="primary")
                exp_msg = gr.Markdown()
                exp_file = gr.File(label="download")
            # a 4th class dropdown placeholder so _refresh_classes always returns 4 updates
            class_dds.append(gr.Dropdown(visible=False, allow_custom_value=True))

        # ---- wiring ----
        open_btn.click(do_open_project, [proj_tb, ckpt_tb, cfgname_tb, overrides_tb, root_tb, score_sl],
                       [cfg_status, status])
        sample_btn.click(do_sample, [nimg_sl, smart_cb], [cfg_status, status, part_df])
        cluster_btn.click(do_cluster, [feat_cbg, dist_dd, perimg_cb], [cfg_status, status, level_dd, part_df])

        level_dd.change(on_level_change, [level_dd], [part_df, status])
        part_df.select(on_partition_select, [mask_toggle], [part_gallery, sel_partition, sel_indices])
        mask_toggle.change(on_mask_toggle, [sel_partition, mask_toggle], [part_gallery])
        part_gallery.select(on_gallery_select, [sel_indices], [sel_indices])
        assign_all.click(do_assign_partition, [sel_partition, pclass_dd], [part_df, status, *class_dds])
        assign_sel.click(do_assign_selected, [sel_partition, sel_indices, pclass_dd], [part_df, status, *class_dds])
        remove_sel.click(do_remove_selected, [sel_partition, sel_indices], [part_df, status])
        reject_sel.click(do_reject_selected, [sel_partition, sel_indices], [part_df, status])

        image_dd.change(on_image_pick, [image_dd, colorby_radio], [inimg, kptimg])
        for comp in (mdist_dd, mmeth_dd, mthr_sl, mgrp_sl):
            comp.change(do_merge_preview, [image_dd, mdist_dd, mmeth_dd, mthr_sl, mgrp_sl],
                        [before_img, after_img, pending_groups])
        commit_btn.click(do_commit_merge, [image_dd, pending_groups, colorby_radio], [inimg, status, part_df])

        for comp in (thr_mode, thr_val, dilate_sl, erode_sl, contrast_sl, fill_cb, largest_cb, smooth_cb):
            comp.change(do_refine_preview,
                        [refine_iuid_tb, thr_mode, thr_val, dilate_sl, erode_sl, contrast_sl, fill_cb, largest_cb, smooth_cb],
                        [refine_orig, refine_out, refine_ops])
        refine_iuid_tb.change(do_refine_preview,
                              [refine_iuid_tb, thr_mode, thr_val, dilate_sl, erode_sl, contrast_sl, fill_cb, largest_cb, smooth_cb],
                              [refine_orig, refine_out, refine_ops])
        refine_apply.click(do_refine_apply, [refine_iuid_tb, refine_ops], [status])
        refine_revert.click(do_refine_revert, [refine_iuid_tb], [status])

        train_btn.click(do_train, [clf_feat, clf_algo], [clf_msg, pr_plot])
        predict_btn.click(do_predict, [clf_thr], [clf_msg, pred_df])
        apply_pred_btn.click(do_apply_predictions, [clf_thr], [clf_msg, status, part_df])

        map_btn.click(do_map, [map_method, map_colorby], [map_plot, map_cluster_dd])
        map_assign_btn.click(do_assign_cluster, [map_cluster_dd, map_class_dd], [status, part_df, *class_dds])

        export_btn.click(do_export, [exp_classes, exp_scope, exp_kpts, exp_fmt], [exp_msg, exp_file])
        undo_btn.click(do_undo, [], [status, part_df])
        redo_btn.click(do_redo, [], [status, part_df])

        def _refresh_image_dd():
            return gr.update(choices=[str(i) for i in ENG.image_ids()]) if ENG else gr.update()
        sample_btn.click(_refresh_image_dd, [], [image_dd])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="/tmp/curator_project")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    demo = build_app(args.project)
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
