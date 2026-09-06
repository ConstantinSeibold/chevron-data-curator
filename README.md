# Chevron

**Local dataset curation from segmentation proposals.**

Point Chevron at a folder of images and a proposal source. It clusters the resulting class-agnostic
instance masks, lets you assign / reject / merge / refine them through a web UI, and exports COCO.
Everything runs in one local process — no database, no object store, no inference server, no
containers.

```bash
pip install -e .
chevron                       # launcher over ~/.chevron/projects → http://127.0.0.1:7870
```

Pick a project from the launcher, or create one. Each project is its own directory and they share
nothing, so several can coexist without conflicting. To skip the launcher and open one directly:

```bash
chevron --project ~/data/my-dataset      # or: python -m chevron.server --root /somewhere/else
```

> **Status: v0.1, phases P0–P1 complete.** Chevron was extracted from
> [qseg](https://github.com/ConstantinSeibold/qseg)'s `tools/curator` into its own repository with
> its 134-commit history intact, now running with no qseg, detectron2, MaskDINO or torch required,
> and given multi-project support with a starter UI. **284 tests green.** The Spacewalker merge and
> the UI restructure are phases P2–P8 below.

---

## What it does

The core loop is **class-agnostic proposals in, curated labelled dataset out**:

1. **Propose** — a segmentation model emits masks; their labels (if any) are discarded.
2. **Partition** — FINCH clusters the unassigned pool in a chosen feature space.
3. **Curate** — assign a partition to a class, reject it, merge fragments, or refine the masks.
4. **Accelerate** — train a classifier on what you have labelled and apply it to the rest; a merge
   recommender learns from your past merges.
5. **Export** — COCO, optionally partial-label or class-agnostic.

Curation is **reversible**: undo/redo over an append-only history, and mask refinement is a
non-destructive overlay that never overwrites the source RLE.

## Capabilities

- **Partitions** — FINCH-clustered pool with per-crop 1-NN class suggestions and gate markers.
- **In-image** — one image's instances, with a workload ranking that orders images by
  classifier-estimated work remaining.
- **Map** — latent-space projection (h-NNE → UMAP → PCA) on a 2D canvas: pan/zoom, paint-select to
  assign or reject, colour by state / class / partition / source / score, hover crops.
- **Refine** — a per-instance op chain (contrast, threshold, vessel trace, line, GrabCut, SAM/SAM-HQ),
  auto-refine search, per-class rules, few-shot shape transfer, and a hand-draw mask editor.
- **Classifier** — per-class training over labelled instances (factored open-set, or kNN for
  single-example classes), applied to the pool with per-class thresholds.
- **Reference** — RAD-DINO retrieval; find a partition from an uploaded reference image.
- **Multi-model proposals** — import any model's COCO output as a tagged source; a header chip bar
  filters every view by proposing model at once, and the Map's "colour by source" shows where models
  agree.
- Plus taxonomy, release gate, retrain loop, rejected bin, activity log and statistics.

## Architecture

| Path | Role |
|---|---|
| `chevron/engine.py` | `CuratorEngine` — the UI-agnostic facade: data model, clustering, classifier, refine, ingest, projection, sources, release, training |
| `chevron/projects.py` | project registry: discovery, cheap card summaries, create/rename/delete |
| `chevron/server.py` | thin FastAPI layer — windowed JSON, lazy batched crops |
| `chevron/web/` | vanilla-JS frontend, no framework and **no build step** |
| `chevron/core/` | generic machinery: collection/clustering/pair features, morphology, shape priors, class-head surgery |
| `chevron/extractors/` | embedding models — one encoder pass per image, pooled per item |
| `chevron/backends/` | proposal sources; `qseg` is one **optional** backend |

**Project format** — self-contained directory, no database:

```
<project>/
  state.json            source of truth (config + taxonomy + per-instance meta + row order)
  collection.pkl        records + per-extractor feature matrices (row-aligned)
  collection_shards/    append-only ingest shards (crash-recoverable)
  ingests.jsonl         ingest registry — lets a view scope to one inference run
  history.jsonl         append-only audit log (undo/redo + provenance)
  refine/<iuid>.pkl     reversible refine overlays
  cluster_cache/*.npz   cached FINCH partitions
  snapshots/  exports/
```

All small writes are `tmp → os.replace` (atomic on POSIX).

**Design properties worth preserving:** the row-alignment invariant across feature matrices; a
label-*independent* projection cache; one `_in_scope` predicate that makes every view filter apply
everywhere for free; an incrementally-maintained membership index that keeps interaction O(1) at 1M
instances; and lazy imports throughout, so the base install needs no model stack.

## Proposal backends

| Backend | Needs | Status |
|---|---|---|
| COCO import | nothing | available |
| `qseg` (MaskDINO / Mask2Former) | a qseg checkout + detectron2 + MaskDINO | available |
| SAM automatic mask generator | `chevron[sam]` | P5 |
| HF Mask2Former / OneFormer | `chevron[embed]` | P5 |
| Torchvision Mask R-CNN | torchvision | P5 |
| detectron2 model zoo | detectron2 | P5 |

The `qseg` backend is optional and lazily resolved. Point it at a checkout with
`CHEVRON_QSEG_ROOT=/path/to/qseg`; without it every other backend still works.

## Roadmap

| Phase | Content |
|---|---|
| **P0** ✅ | Extract to a standalone repo; vendor the generic qseg modules; 273 tests green with no qseg |
| **P1** ✅ | Multi-project support + starter UI (project cards, new-project dialog, one active engine) |
| P2 | Data-model unification (`granularity`, `modality`) |
| P3 | UI restructure — 15 peer tabs → 5 areas, one workspace, one selection model, inspector rail |
| P4 | Extractor registry — RAD-DINO / DINOv2 / CLIP / SigLIP2 as a dropdown |
| P5 | Off-the-shelf proposal backends (SAM, HF, torchvision, detectron2) |
| P6 | Persisted dimensionality reduction + project a new image/text query into the map |
| P7 | Unified 2D/3D viewer (Spacewalker's latent walk over instances) |
| P8 | Sample mode — label whole images / text / video, not only mask instances |

See `DESIGN.md` for the full design and its rationale.

## Provenance

Chevron merges two projects:

- **qseg curator** — the instance-curation engine, extracted here with its history.
- **[Spacewalker](https://github.com/ConstantinSeibold/Spacewalker)**
  ([arXiv:2409.16793](https://arxiv.org/abs/2409.16793), MIT, © 2024 Lukas Heine) — contributes the
  latent-space viewer, persisted DR with query projection, and the embedding/DR menus, generalised
  so the point cloud holds *instances*, not only whole samples. Triton, Django, Postgres, MinIO and
  docker-compose are deliberately not carried over.

See `THIRD_PARTY_NOTICES.md`.

## Hosting

Binds `127.0.0.1:7870`, **no built-in auth**. The intended multi-user model is one shared project
reached over SSH tunnels:

```bash
ssh -L 7870:localhost:7870 you@host   # then open http://localhost:7870
```

Concurrent mutations are serialised by a reentrant lock, and `GET /api/version` exposes a mutation
stamp the browser polls to show an "updated elsewhere" banner. It remains single-tenant and
last-writer-wins: no per-instance locks, no user identity. For any network exposure, put it behind a
reverse proxy with auth.

## Tests

```bash
pytest tests/ -q                            # 284 tests, CPU-only, no model stack needed
for f in $(find chevron/web -name '*.js'); do node --check "$f"; done
```

If `node` dies with `undefined symbol: sqlite3session_attach`, an activated conda env is shadowing
the system libsqlite3 — run the check with `env -u LD_LIBRARY_PATH PATH=/usr/bin:/bin node --check`.

The suite stubs SAM, RAD-DINO, h-NNE, inference and training. If it ever needs `CHEVRON_QSEG_ROOT`,
the core has regained a qseg dependency — that is the regression to watch for.

## Known limitations

- `CuratorEngine` is a 229-method god object and the server is 110 flat endpoints. Deliberately left
  intact through the extraction so the test suite stayed green; P3 is where this gets carved.
- ~13 hand-rolled caches with bespoke invalidation keys — the main correctness hazard.
- The feature NaN check is global: one non-finite row disables that feature everywhere.
- Stringly-typed API (`body: dict`, no Pydantic models).
- ML paths are under-tested; there are no frontend tests beyond a syntax check.

## Licence

MIT — see `LICENSE` and `THIRD_PARTY_NOTICES.md`.
