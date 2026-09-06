"""Offline harvest of a curation project's logs into the aggregates a paper needs (pure-python, no torch/GPU).

Reads ``<project_dir>/{history.jsonl, merge_log.jsonl, state.json, lineage.jsonl}`` and produces:
  - action mix + cumulative-actions-over-time + actions-per-verified-instance   (history.jsonl) -> "effort is on
    decisions, not pixels"
  - labels-by-assign_source breakdown                                           (state.json)    -> one-click cluster
    vs classifier vs manual
  - merge-recommender accept-rate (overall + over session, when events carry ts/source)  (merge_log.jsonl) -> "the
    tool learns from the human"
  - dataset-version -> checkpoint -> eval-metric lineage of the retrain loop    (lineage.jsonl) -> "loop works"

Run: ``python -m chevron.paper_stats <project_dir> [--out <dir>]`` -> writes summary.json + CSVs (+ PNGs if
matplotlib is available). Purely descriptive; never mutates the project. Importable functions
(action_stats/source_stats/merge_stats/summarize) are unit-tested without a model.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .state import CuratorState
from .store import Store

_NON_ACTION_OPS = {"undo", "redo"}                       # bookkeeping events, not curation actions


def action_stats(store: Store) -> dict[str, Any]:
    """From history.jsonl: per-op action tally, cumulative-actions timeline, undo/redo counts, duration."""
    hist = store.read_history()
    cmds = [h for h in hist if h.get("op") not in _NON_ACTION_OPS]
    ts = sorted(float(h["ts"]) for h in cmds if "ts" in h)
    return {
        "n_actions": len(cmds),
        "by_op": dict(Counter(h.get("op", "?") for h in cmds)),
        "n_undo": sum(1 for h in hist if h.get("op") == "undo"),
        "n_redo": sum(1 for h in hist if h.get("op") == "redo"),
        "n_instances_touched": sum(int(h.get("n_instances", 0)) for h in cmds),
        "t_start": ts[0] if ts else None,
        "t_end": ts[-1] if ts else None,
        "duration_s": (ts[-1] - ts[0]) if len(ts) >= 2 else 0.0,
        "timeline": [[t, i + 1] for i, t in enumerate(ts)],          # (ts, cumulative #actions)
    }


def source_stats(state: CuratorState) -> dict[str, Any]:
    """From state.json: how the live labels were produced (manual / one-click partition / classifier / import)."""
    assigned = [m for m in state.meta.values()
                if m.assigned_class is not None and not m.is_background and m.merged_into is None]
    by_source = Counter((m.assign_source or "unknown") for m in assigned)
    by_class = Counter((state.class_name(m.assigned_class) or "?") for m in assigned)
    return {
        "n_assigned": len(assigned),
        "by_source": dict(by_source),
        "by_class": dict(by_class),
        "n_unassigned": sum(1 for m in state.meta.values()
                            if m.assigned_class is None and not m.is_background and m.merged_into is None),
        "n_rejected": sum(1 for m in state.meta.values() if m.is_background),
        "n_merged_children": sum(1 for m in state.meta.values() if m.merged_into is not None),
        "n_total": len(state.meta),
    }


def merge_stats(store: Store) -> dict[str, Any]:
    """From merge_log.jsonl: merge/reject counts, manual-vs-recommended split, and (when events carry ts+source)
    the merge-recommender accept-rate over the session."""
    evs = store.read_merge_events()
    merges = [e for e in evs if e.get("kind") == "merge"]
    rejects = [e for e in evs if e.get("kind") == "reject"]
    rec_acc = sum(1 for e in merges if e.get("source") == "recommended")
    rec_rej = sum(1 for e in rejects if e.get("source") == "recommended")
    rec_total = rec_acc + rec_rej
    rec_events = sorted([e for e in (merges + rejects) if e.get("source") == "recommended" and "ts" in e],
                        key=lambda e: float(e["ts"]))
    timeline, acc = [], 0
    for i, e in enumerate(rec_events):                               # cumulative accept-rate over recommender decisions
        if e.get("kind") == "merge":
            acc += 1
        timeline.append([float(e["ts"]), i + 1, acc / (i + 1)])
    return {
        "n_merge_events": len(merges),
        "n_reject_events": len(rejects),
        "by_merge_source": dict(Counter(e.get("source", "unknown") for e in merges)),
        "recommender": {"accepted": rec_acc, "rejected": rec_rej, "total": rec_total,
                        "accept_rate": (rec_acc / rec_total) if rec_total else None,
                        "timeline": timeline},
    }


def summarize(project_dir: str | Path) -> dict[str, Any]:
    """Full descriptive aggregate over a project's logs. Tolerates missing files (returns the parts that exist)."""
    store = Store(project_dir)
    out: dict[str, Any] = {
        "project_dir": str(project_dir),
        "actions": action_stats(store),
        "merges": merge_stats(store),
        "lineage": store.read_lineage(),
    }
    if store.is_project():
        out["sources"] = source_stats(store.load_state())
        na = out["sources"]["n_assigned"]
        out["actions_per_verified_instance"] = (out["actions"]["n_actions"] / na) if na else None
    return out


def _write_csv(path: Path, header: list[str], rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _maybe_plots(out: dict, od: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    made = []
    tl = out["actions"]["timeline"]
    if tl:
        t0 = tl[0][0]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot([(t - t0) / 60.0 for t, _ in tl], [c for _, c in tl])
        ax.set_xlabel("session minutes"); ax.set_ylabel("cumulative curation actions")
        fig.tight_layout(); fig.savefig(od / "actions_timeline.png", dpi=130); plt.close(fig)
        made.append("actions_timeline.png")
    src = out.get("sources", {}).get("by_source")
    if src:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(list(src.keys()), list(src.values()))
        ax.set_ylabel("labels"); ax.set_title("labels by assign_source")
        fig.tight_layout(); fig.savefig(od / "source_breakdown.png", dpi=130); plt.close(fig)
        made.append("source_breakdown.png")
    rtl = out["merges"]["recommender"]["timeline"]
    if rtl:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot([n for _, n, _ in rtl], [r for _, _, r in rtl])
        ax.set_xlabel("recommender decisions"); ax.set_ylabel("cumulative accept-rate"); ax.set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(od / "recommender_accept_rate.png", dpi=130); plt.close(fig)
        made.append("recommender_accept_rate.png")
    return made


def write_report(project_dir: str | Path, out_dir: str | Path) -> tuple[dict, Path]:
    out = summarize(project_dir)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "summary.json").write_text(json.dumps(out, indent=2))
    _write_csv(od / "action_mix.csv", ["op", "count"], sorted(out["actions"]["by_op"].items()))
    _write_csv(od / "actions_timeline.csv", ["ts", "cumulative_actions"], out["actions"]["timeline"])
    if "sources" in out:
        _write_csv(od / "source_breakdown.csv", ["assign_source", "count"], sorted(out["sources"]["by_source"].items()))
    _write_csv(od / "merge_recommender_timeline.csv", ["ts", "n_decisions", "cumulative_accept_rate"],
               out["merges"]["recommender"]["timeline"])
    if out["lineage"]:
        keys = ["ts", "coll_version", "n_assigned", "export_json", "ckpt", "metric_name", "metric"]
        _write_csv(od / "lineage.csv", keys, [[r.get(k) for k in keys] for r in out["lineage"]])
    out["_plots"] = _maybe_plots(out, od)
    return out, od


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest a curation project's logs into paper aggregates.")
    ap.add_argument("project_dir")
    ap.add_argument("--out", default=None, help="output dir (default: <project_dir>/paper_stats)")
    args = ap.parse_args()
    out, od = write_report(args.project_dir, args.out or (Path(args.project_dir) / "paper_stats"))
    brief = {
        "n_actions": out["actions"]["n_actions"],
        "actions_per_verified_instance": out.get("actions_per_verified_instance"),
        "by_source": out.get("sources", {}).get("by_source"),
        "recommender_accept_rate": out["merges"]["recommender"]["accept_rate"],
        "lineage_runs": len(out["lineage"]),
        "wrote": str(od),
    }
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
