# Chevron

**Local dataset curation from segmentation proposals.**

Point Chevron at a set of class-agnostic instance masks — from a COCO you already have, from SAM, or from your own model. It clusters them, lets you assign / reject /
merge / refine them through a web UI, and exports COCO. Everything runs in one local process — no
database, no object store, no inference server, no containers.

> **Status: v0.1, all planned phases (P0–P8) complete.** Extracted from
> [qseg](https://github.com/ConstantinSeibold/qseg)'s `tools/curator` with its 134-commit history,
> now standalone; multi-project launcher; one Curate workspace with a shared selection across
> Grid/Map/Image; model-free proposal backends; an embedding-model dropdown; query projection; a 3D latent walk;
> and sample mode. **418 tests green.**

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

- **Curate** — one workspace: a scope rail (partitions, classes, rejected bin, sub-clusters), a canvas
  with three interchangeable views, and an inspector that acts on the selection. The **selection is
  shared**, so switching view keeps it.
  - *Grid* — crops with per-crop 1-NN class suggestions and gate markers.
  - *Map* — latent-space projection (h-NNE → UMAP → PCA) in **2D or 3D**: pan/zoom or orbit,
    paint-select, colour by state / class / partition / source / score, hover crops. Type a phrase or
    an instance id to **place a query on the map** and select its neighbours.
  - *Image* — one image's instances, with a workload ranking that orders images by
    classifier-estimated work remaining.
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

Where the masks come from. Labels are always **discarded** — a COCO detector's 80 classes are not the
label space you are curating, and the human supplies the taxonomy.

| Backend | Needs | Notes |
|---|---|---|
| **`coco`** | **nothing** | Bootstraps a project straight from a COCO of masks you already have. No model, no GPU, no torch. |
| **`whole_image`** | **nothing** | Sample mode: one item per image, for labelling images rather than masks. |
| `sam_auto` / `samhq_auto` | `chevron[sam]` | SAM automatic mask generation — proposals with **no trained model at all**. Checkpoint auto-downloads. |
| `torchvision_maskrcnn` | `torch` + `torchvision` | COCO-pretrained Mask R-CNN, labels dropped. Runs on CPU. |
| `hf_seg` | `chevron[embed]` | Any HF `AutoModelForUniversalSegmentation` (default Mask2Former-COCO). |
| `qseg` | a qseg checkout + detectron2 + MaskDINO | The original path; set `CHEVRON_QSEG_ROOT`. **CUDA only** — MaskDINO's MSDeformAttn kernel does not build for MPS or CPU. |

`GET /api/backends` lists them with availability and an install hint, so the picker shows what is
*installable*, not only what is installed.

```bash
# start a project from masks you already have — nothing else installed
curl -X POST localhost:7870/api/propose -H 'Content-Type: application/json' \
     -d '{"backend":"coco","coco_path":"/data/masks.json","image_root":"/data/images"}'
```

Adding a backend means implementing `propose(image) -> [Proposal]`; ids, records, geometry features,
NMS and the row-alignment invariant are handled once in `backends/base.py`.

## Hardware — CPU, CUDA and Apple silicon

All three are first-class, and nothing needs configuring: `chevron/device.py` picks the best device
present (CUDA → MPS → CPU), and every extractor, proposal backend and refinement path asks it rather
than deciding for itself. The choice is shown in **Settings → Instance features**, next to the
embedding model that uses it, and served by `GET /api/device`.

| | Selection | Mixed precision |
|---|---|---|
| **CUDA** | used when present; `cuda:N` honoured | bf16 on Ampere and later, fp16 before that |
| **Apple MPS** | used when present | **off by default** — fp16 on Metal has shipped numerical differences, and a pooled embedding goes straight into `feats`, where a bad block poisons clustering, kNN and the map for the whole project. `CHEVRON_AMP=1` opts in |
| **CPU** | always available | none |

torch's Metal backend still has operator gaps. When one is hit, Chevron reruns that call on the CPU
and keeps that model there, so a missing operator costs speed rather than the run — and it says so
once, naming the operator. `PYTORCH_ENABLE_MPS_FALLBACK=1` is the finer-grained per-op alternative.

```bash
CHEVRON_DEVICE=cpu chevron --project ./proj     # escape hatch: force everything onto the CPU
CHEVRON_DEVICE=cuda:1 chevron --project ./proj  # pick a GPU
```

`CHEVRON_DEVICE` overrides automatic selection, but will **not** move work that deliberately asked
for the CPU onto a GPU — the substructure encoder and the shape priors are tiny nets where the
host↔device copy costs more than the arithmetic.

Starting the server imports no torch at all, and neither does `/api/state`: a session that only
reviews and exports an existing project never pays for a model stack it does not use.

The **browser** has its own device requirement: the Map's 3D view needs WebGL, which software or
remote GL, a driver blocklist or a locked-down browser may not provide. When the renderer cannot be
built the view stays in 2D and says why, rather than leaving a blank canvas. Everything else in the
UI is plain 2D canvas and DOM.

## Roadmap

| Phase | Content |
|---|---|
| **P0** ✅ | Extract to a standalone repo; vendor the generic qseg modules; 273 tests green with no qseg |
| **P1** ✅ | Multi-project support + starter UI (project cards, new-project dialog, one active engine) |
| **P2** ✅ | Data-model unification (`granularity`, `modality`, project mode + capabilities) |
| P3 | UI restructure — *(done: 6 areas + router, Curate workspace, one selection across Grid/Map/Image, inspector rail; remaining: Assist grids, command palette)* |
| **P4** ✅ | Extractor registry — RAD-DINO / DINOv2 / CLIP / SigLIP2 as a dropdown |
| **P5** ✅ | Model-free proposal backends — COCO bootstrap, SAM auto-mask, torchvision, HF |
| **P6** ✅ | Persisted dimensionality reduction + project a new image/text/instance query onto the map |
| **P7** ✅ | Unified 2D/3D viewer — Spacewalker's latent walk, over instances, sharing the selection |
| **P8** ✅ | Sample mode — label whole images, not only mask instances *(text/video items not yet)* |

See `DESIGN.md` for the full design and its rationale.

## Sample mode — labelling images, not masks

Set `mode: "sample"` on a project and ingest with the `whole_image` backend: one item per image.

It needs no second data model, grid, selection or feature pipeline, because **a sample is an instance
with a trivial mask** — an all-ones one. Clustering, the projection, the map, the classifier and kNN
work untouched, since they only ever read `feats[name]` and never look at a mask. An extractor pooled
over an all-ones mask *is* a whole-image embedding.

The mask-only tools (Refine, Merge-rec, Substructure) disappear from the nav, driven by the
`capabilities` block the server sends — the UI never re-derives it. Export becomes a classification
manifest (`{file, class}`, JSON + CSV) rather than COCO with empty segmentations.

Text and video items are **not** implemented: both need a records/preview path with no image at all.
The data model carries `modality` for them, and the extractor registry already has the shared
image-text space, but the ingest and rendering are not built.

## Is it tied to chest X-rays?

The curation loop is **domain-agnostic**: proposals in, clusters, labels, COCO out. Nothing in it
knows what the pixels depict. The mask-derived features (`shape`, `shapecoord`, `coords` — PCA axes,
radial signature, contour Fourier) are pure geometry, and clustering, the classifier, the projection
and kNN only ever read the feature matrices. It has been used on chest X-rays because that is where
it was written, not because of an assumption in the core.

What *is* chest-X-ray flavoured, and what it means for, say, a surgical-video or endoscopy dataset:

| Piece | Status on another domain |
|---|---|
| Curation loop, clustering, classifier, projection, export | **domain-agnostic** — use as-is |
| Mask-geometry features | **domain-agnostic** — computed from the mask alone |
| Embedding model | **pick one**: DINOv2 (general purpose), CLIP or SigLIP (also give a shared image-text space), or RAD-DINO (chest X-ray). Config → *Embedding model*. Nothing is computed automatically |
| Shipped `taxonomy_seed.json` | chest foreign bodies (airway tubes, catheters, cardiac implants…). Applied only when you press **Seed**; supply your own JSON, or just create classes as you go |
| `vessel_extend` refine op | tuned for catheters and lines. One op among many; ignore it |
| Anatomy "recipe" profiles in `core/morphology.py` | came along with the vendored module and are **not reachable** from the UI — the refine chain uses only the generic primitives (`largest_cc`, `top_k_cc`, `fill`) |

So the honest summary for a new domain: the curation machinery transfers unchanged, you pick an
encoder that suits your images (DINOv2 is the sane default outside chest X-ray), and you supply your
own taxonomy.

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
pip install -e ".[dev]"                     # pytest + the TestClient's HTTP client
pytest tests/ -q                            # 418 tests, CPU-only, no model stack needed
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
- **MPS has not been run on Apple hardware.** Selection, precision policy and the operator-gap
  fallback are tested by injecting the availability probes and a synthetic gap, which is why they
  are checked on every machine rather than only on a Mac — but no real Metal kernel has executed.
  CUDA and CPU are verified end to end with real weights.

## Licence

MIT — see `LICENSE` and `THIRD_PARTY_NOTICES.md`.
