# Chevron — design

**A local suite for curating datasets from segmentation proposals.**

Chevron merges two existing tools into one local-only application:

- **qseg curator** (`qseg/tools/curator`, ~9.3k LOC Python + 2.6k web, 273 tests) — instance-level
  curation: cluster a segmentation model's raw predictions, assign/reject/merge/refine them, export COCO.
- **Spacewalker** ([github](https://github.com/ConstantinSeibold/Spacewalker),
  [arXiv:2409.16793](https://arxiv.org/abs/2409.16793)) — latent-space exploration and annotation:
  a 3D point cloud of embedded samples you fly through and paint labels onto.

The curator is the base. Spacewalker contributes its **viewer**, its **persisted dimensionality
reduction**, its **embedding/DR menus** and its **multi-modality** — generalised so the point cloud
holds *instances* (mask crops), not just whole samples.

Everything operational in Spacewalker is deleted: Triton, Django, Postgres, MinIO, docker-compose,
the ONNX `model_repository`, the Parcel build. Chevron is `pip install -e .` + `chevron serve`.

> **Status: design only. No implementation has begun.**

---

## 0. Table of contents

1. [Goals and non-goals](#1-goals-and-non-goals)
2. [The core equivalence](#2-the-core-equivalence-a-sample-is-an-instance-with-a-trivial-mask)
3. [Inventory: what is kept, dropped, merged](#3-inventory-what-is-kept-dropped-merged)
4. [Unified data model](#4-unified-data-model)
5. [Extractor registry (the model dropdown)](#5-extractor-registry-the-model-dropdown)
6. [Proposal backends](#6-proposal-backends)
7. [Projection: fit, persist, transform](#7-projection-fit-persist-transform)
8. [The unified viewer (2D + 3D)](#8-the-unified-viewer-2d--3d)
9. [Sample mode](#9-sample-mode)
10. [Repository layout](#10-repository-layout)
11. [Packaging, dependencies, licensing](#11-packaging-dependencies-licensing)
12. [Extraction mechanics from qseg](#12-extraction-mechanics-from-qseg)
13. [Sharp edges and design hazards](#13-sharp-edges-and-design-hazards)
14. [Phasing](#14-phasing)
15. [Open decisions](#15-open-decisions)

---

## 1. Goals and non-goals

### Goals

- **One repo, one app, one process.** `chevron serve --project DIR` on localhost. No containers, no
  database, no object store, no inference server.
- **Model-agnostic.** Segmentation proposals come from a pluggable backend. qseg/MaskDINO is one
  backend, not a dependency of the core.
- **One viewer for instances and samples.** The latent map shows mask crops or whole images/texts/videos
  as the same kind of point, with the same select-and-label verbs.
- **Preserve what works.** 273 green tests, 110 endpoints, the reversible-curation guarantees, the
  on-disk project format, and the O(1)-at-1M interaction properties survive the merge.

### Non-goals (explicitly out of scope for now)

- Scaling, multi-tenancy, auth, horizontal deployment. The hosting model stays *one shared project,
  small team, SSH tunnel* — as documented in the curator README today.
- Triton, ONNX export, or any separate model-serving process.
- A database. `state.json` + `collection.pkl` + append-only logs remain the source of truth.
- A frontend build step. Vanilla JS + vendored ESM; no npm, no Parcel, no bundler.

---

## 2. The core equivalence: a sample IS an instance with a trivial mask

This is the load-bearing idea of the whole merge.

The curator's RAD-DINO path works like this: run the encoder once per image to get a patch-token grid
`(C, g, g)`, then **soft-mask-pool** that grid per instance to get one vector per instance, written into
the row-aligned matrix `collection["feats"]["raddino"]`.

```
image ──encoder──> patch grid (C, g, g) ──pool(mask)──> vector ──> feats["raddino"][row]
```

Change the pooling mask and the same code produces a different granularity:

| Item | Pooling mask | Result |
|---|---|---|
| Segmentation instance | the instance mask | instance embedding (today's behaviour) |
| Whole image | all-ones | sample embedding (Spacewalker's behaviour) |
| Video | all-ones over a frame stack | sample embedding |
| Text | *(no grid; encoder emits a vector directly)* | sample embedding |

So Spacewalker's data points and the curator's instances are **the same object at different pooling
masks**, landing in the same `feats` matrix. And because every downstream consumer in the curator —
FINCH clustering, `fuse_features`, the classifier, the projection, kNN retrieval, the merge
recommender — reads *only* `feats[name]` and never touches masks, they all work unchanged on samples.

Three consequences that define the rest of this document:

1. **P4 (sample mode) is not a second data model.** It is a proposal backend that emits one
   full-image item per file.
2. **P3 (model menu) is not a new subsystem.** It is a registry of things that produce a grid or a
   vector, all writing into the existing `feats` dict. `engine.available_features()` already drives
   every feature selector in the UI.
3. **The viewer does not need to know the difference.** It plots rows of a projected matrix; whether a
   row is a rib mask or a whole chest X-ray only changes what the hover preview shows.

---

## 3. Inventory: what is kept, dropped, merged

### From Spacewalker

| Element | Disposition | Rationale |
|---|---|---|
| three.js 3D point cloud, `OrbitControls`, `InstancedMesh`, raycast tooltips | **Keep — port** | The headline capability the curator lacks. `engine.project(dims=3)` already emits `z`. |
| Right-drag paint annotation + adjustable cursor sphere | **Keep — port** | Curator has 2D paint-select; this is its 3D twin. |
| Persisted fitted DR (`joblib` `dr2d.pkl`/`dr3d.pkl` + `scale_val`) | **Keep — port** | Enables projecting a *new* query into an existing map. Curator only `fit_transform`s. |
| DR menu: hnne / umap / pca / t-SNE / MDS / Isomap | **Keep — port** | Curator has a hnne→umap→pca fallback chain, not a choice. |
| Embedding zoo: ViT, ResNet50, VGG16, DINOv2, CLIP, SigLIP2 | **Keep — reimplement** | As in-process `transformers` extractors, not Triton ONNX. |
| Cross-modal text↔image query (CLIP / SigLIP2) | **Keep — port** | Genuinely new capability; falls out of the extractor registry. |
| Modality handling (image / text / video) | **Keep — generalise** | Becomes a field on the item, not a separate pipeline. |
| Thumbnail generation (`resize_longest_edge`, 128px) | **Merge** | Curator's crop cache already does this for instances; extend to samples. |
| `AnnotationColorDescription` | **Merge** | Curator's `TaxonomyClass.color` already carries this. |
| `DataPoint` model | **Drop** | Its `x/y/z/is3d/model/dr_method` are *derived* projection state; curator computes and caches them label-independently, which is strictly better. |
| Django (views, models, urls, settings, migrations, templates) | **Drop** | Replaced by the curator's FastAPI layer. |
| Postgres | **Drop** | Replaced by `state.json`. |
| MinIO + `MinIOWebhook` + bucket events | **Drop** | Replaced by the local project directory. |
| Triton + `tritonclient` + `model_repository` + `Triton/Dockerfile` | **Drop** | 74 LOC, 2 functions, 3 call sites. Replaced by in-process torch. Removes the Google-Drive ONNX download entirely. |
| `docker-compose*.yml`, `Dockerfile`, `.devcontainer` | **Drop** | Local-only install. |
| Parcel / npm / `package.json` | **Drop** | Vendored ESM instead; keeps the curator's no-build-step property. |
| `multiprocessing.Pool` datapoint writes | **Drop** | An artefact of per-row DB inserts; there are no rows to insert. |

**Deleting Triton is genuinely cheap** — it is one 74-line module (`triton_inference`,
`triton_inference_text`) called from three places in `views.py`. The replacement (an in-process
`AutoModel`) is what the curator already does for RAD-DINO.

### From the curator — all kept

Engine, store, history/undo, taxonomy, FINCH partitions, classifier + active-learning suggestions,
refine chain (`refine.py`, `autorefine.py`, SAM/SAM-HQ, shape transfer, mask editor), merge
recommender, reference bank, release gate, COCO import/export, training-loop orchestration,
multi-source proposal ingest, the `_in_scope` facet, the live membership index, the 273 tests.

---

## 4. Unified data model

### 4.1 Item metadata

The curator's `InstanceMeta` gains **two fields**; nothing is removed.

```
ItemMeta:
    iuid            str          immutable id
    batch_id        str          ingest batch
    row             int          index into every feats matrix (THE INVARIANT)
    source_id       int          image / document / video id   (was: image_id)

    granularity     str          "instance" | "sample"          ← NEW
    modality        str          "image" | "text" | "video"     ← NEW

    assigned_class  str | None
    is_background   bool
    assign_source   str | None   manual | partition | classifier | merge | import
    assign_score    float | None
    merged_into     str | None   ┐
    merge_members   list[str]    │ instance-only; inert for samples
    refined         bool         │
    rule_ops        list | None  ┘
    provenance      dict
```

`image_id` → `source_id` is a rename for honesty across modalities. To avoid a migration it is
**read with a fallback** (`d.get("source_id", d["image_id"])`) and written under the new name.

**Backward compatibility is total.** A project written by today's curator has neither new field;
`from_dict` defaults them to `("instance", "image")`, which is exactly what those projects are. No
migration script, no schema bump.

### 4.2 Records (the heavy collection)

`collection["records"][row]` stays a plain dict, but its required keys become modality-dependent:

| Key | instance/image | sample/image | sample/text | sample/video |
|---|---|---|---|---|
| `file_name` | ✓ | ✓ | — | ✓ |
| `rle`, `bbox`, `score` | ✓ | *(absent)* | — | — |
| `text` | — | — | ✓ | — |
| `preview` | derived crop | thumbnail | snippet | first-frame thumb |

Rendering dispatches on `(granularity, modality)`; the existing `crop()` / `mask_token()` path stays
the instance/image branch verbatim.

### 4.3 Project-level mode

A Chevron project declares a mode in `state.config`:

```
mode: "instance" | "sample"
modality: "image" | "text" | "video"
```

**Recommendation: enforce one mode per project for now**, even though the item model can express a
mix. Rationale: the feature matrices are dense and row-aligned across *all* items, so mixing
granularities in one project means a text item has no `raddino` row — which trips the global
feature-NaN check (see [hazard H1](#h1-the-global-feature-nan-check-is-the-biggest-hazard)). The data
model stays mix-capable so this can be relaxed later without another migration; the *enforcement* is a
config check, not a structural limit.

### 4.4 Where the two facets go

The curator already funnels every tab's visibility through one predicate:

```
_in_scope(iuid) = ingest-scope(batch_id) AND source-facet(proposing model)
```

`granularity` and `modality` fold into **that same predicate** — not into per-tab code. This is how the
source facet was added (one chip bar filtering every tab at once) and it is the pattern to follow. Any
new filter that does not fold into `_in_scope` is a design error.

---

## 5. Extractor registry (the model dropdown)

### 5.1 Protocol

```
Extractor (protocol):
    name       : str                      # the feats key
    modality   : {"image", "text", "video"}
    dim        : int
    space      : str | None               # shared-embedding-space tag, e.g. "clip"

    grid(images)  -> (B, C, g, g)  | None # dense patch tokens; None if not applicable
    embed(items)  -> (B, D)               # one vector per item
```

`grid()` is what enables instance pooling. `embed()` is the sample/text path. An extractor may
implement one or both; `RadDinoExtractor` today implements exactly this shape (`grid`, `grid_batch`).

### 5.2 Registry

| Name | Model | Modality | grid | Notes |
|---|---|---|---|---|
| `raddino` | `microsoft/rad-dino` | image | ✓ | current default; chest-X-ray domain-matched |
| `dinov2` | `facebook/dinov2-base` | image | ✓ | Spacewalker's general-purpose choice |
| `clip` | CLIP image + text | image, text | ✓ | **shared space** → cross-modal query |
| `siglip2` | SigLIP2 image + text | image, text | ✓ | shared space |
| `resnet50`, `vgg16`, `vit` | torchvision / HF | image | ✓ | Spacewalker parity; low priority |
| `decoder`, `maskpool`, `roialign`, `backbone` | the seg backend's own features | image | n/a | produced by the proposal backend, instance-only |
| `shape`, `shapecoord`, `coords` | handcrafted geometry | image | n/a | pure numpy/cv2, instance-only |

The image/text encoders of CLIP and SigLIP2 **share one `feats` key** because they share an embedding
space. That is what makes a text→image query a plain kNN in `feats["clip"]` with no special casing —
and it makes the query-projection in §7 work for text against an image map.

### 5.3 Why this is "just a dropdown"

`engine.available_features()` already returns the keys present in `collection["feats"]` and is
described in the code as *"single source of truth for the UI selectors"*. Adding an extractor adds a
key. The clustering spec, classifier feature picker, projection spec, and reference retrieval all read
from that one list. The UI change is a `<select>`; the backend change is a registry entry plus a
"compute this feature now" endpoint, which already exists in the shape of `/api/compute_raddino`.

Generalise that endpoint to `POST /api/compute_features {extractor}` and the existing
`/api/compute_raddino` becomes an alias.

---

## 6. Proposal backends

```
ProposalBackend (protocol):
    name : str
    propose(image_paths, cfg) -> collection      # records + feats, row-aligned
    supports_training : bool
    train(coco_path, cfg)     -> job             # optional
```

| Backend | Source of proposals | Requires |
|---|---|---|
| `qseg` | MaskDINO / Mask2Former inference | `chevron[qseg]`, detectron2, MaskDINO, CUDA |
| `coco` | an existing COCO json (any model) | nothing — already implemented as multi-source ingest |
| `sam_auto` | SAM automatic mask generation | `chevron[sam]` |
| `none` | no proposals; one full-image item per file | nothing — **this is sample mode** |

`sam_auto` matters for adoption: it makes Chevron useful on a fresh dataset with no trained
segmentation model at all. `none` is how §9 works.

The current qseg coupling is already narrow and **entirely lazy** — verified: no curator module
imports `qseg`, `qseg_playground`, `torch` or `detectron2` at module level, which is why the 273 tests
run CPU-only with none of it installed. The seams are:

| Seam | Surface | Destination |
|---|---|---|
| `qseg_playground` | 16 functions | 13 generic → `chevron.core`; 3 (`load_model`, `collect_instances`, `setup_env`) → `backends/qseg.py` |
| `qseg.ssl.refine_anatomy` | 346 LOC pure morphology | vendor into `chevron.refine` |
| `qseg.evaluation.shape_prior_model` | 138 LOC ConvDAE | vendor into `chevron.shape_prior` |
| `qseg.models.class_extend` | 220 LOC class-head surgery | stays in `backends/qseg.py` — MaskDINO-specific |
| `qseg-train` subprocess + MaskDINO `PYTHONPATH` | process launch | `backends/qseg.py:train()` |
| `_bootstrap.QSEG_ROOT = parents[2]` | hardcoded path | config / `CHEVRON_QSEG_ROOT` env |

---

## 7. Projection: fit, persist, transform

### 7.1 Today

`engine.project(spec, method, dims)` L2-normalises the fused feature matrix, calls
`HNNE(dim).fit_transform(X)` (falling back UMAP → PCA), min-max normalises the coordinates to `[0,1]`,
and caches the result keyed by `(coll_version, scope_token, spec, method, dims, n)`. The cache is
**label-independent** — it survives every assign/reject and only recomputes on ingest, re-cluster or
scope change. That property is valuable and must be preserved.

The fitted reducer itself is discarded. So a new point cannot be placed in the map.

### 7.2 Design

Keep the coord cache exactly as is. Additionally persist the **fitted reducer plus its normalisation**:

```
<project>/dr/<key>.joblib   ->  {
    reducer,                    # the fitted object
    method, dims,
    spec,                       # which feature space was fused
    coord_min, coord_max,       # the min/max used for the [0,1] normalisation AT FIT TIME
    n_fit, truncated,           # honesty about _PROJ_CAP
    coll_version, scope_token,
}
```

New endpoint:

```
POST /api/project_query
     {image | text | iuid, extractor, method, dims}
  -> {x, y[, z], neighbors: [iuid...]}
```

which embeds the query with the named extractor, projects it through the stored reducer, and applies
the **stored** `coord_min`/`coord_max`.

### 7.3 Two correctness details that are easy to get wrong

- **The normalisation must be the one from fit time.** Today the min-max is recomputed from the
  current coordinate set. If a query point were normalised against a recomputed range it would land
  in a different place than the map it is drawn on. Spacewalker gets this right by storing
  `scale_val` next to the reducer; Chevron stores `(coord_min, coord_max)` for the same reason.
- **Not every reducer can `.transform`.** `PCA`, `UMAP`, `Isomap` and **openTSNE** can;
  `sklearn.manifold.MDS` cannot, and `sklearn.manifold.TSNE` cannot either — which is exactly why
  Spacewalker depends on `openTSNE` rather than sklearn's TSNE. The DR menu must mark methods as
  query-capable, and `project_query` must return a clear "this DR method cannot place new points"
  error rather than silently refitting.

Also record `truncated` honestly: `project()` caps at `_PROJ_CAP` items, so a persisted reducer was
fitted on a subset. That is fine, but the UI should say so rather than imply the map is complete.

---

## 8. The unified viewer (2D + 3D)

### 8.1 One data contract, two renderers

Both renderers consume the existing endpoint, which already supports 3D:

```
GET /api/projection_points?dims=2|3&method=...&spec=...
 -> [{iuid, x, y[, z], state, cls, pid, score, source_id, modality, granularity}, ...]
```

| | 2D renderer | 3D renderer |
|---|---|---|
| Tech | existing HTML canvas | three.js `InstancedMesh` + `OrbitControls` |
| Navigation | pan / zoom | orbit / fly |
| Select | paint, lasso, box | paint via cursor sphere (radius slider), raycast pick |
| Best for | dense triage, precise work | structure, cluster geography |

They share **one selection model and one action bar**, so every verb works in both:
assign · reject · new class · add to reference bank · send to refine · gate.

### 8.2 Colour-by and hover

Colour-by: `state | class | partition | source | score | modality | granularity`
(the first five exist today; the last two are new and free).

Hover preview dispatches on the item:

| Item | Preview |
|---|---|
| instance / image | the mask crop (existing `/api/crop`, already batched via `/api/crops`) |
| sample / image | 128px thumbnail |
| sample / text | text snippet |
| sample / video | first-frame thumbnail |

This is the only place in the viewer where granularity is visible at all.

### 8.3 Query pin

A projected query (§7) is drawn as a distinct pinned marker with its k nearest neighbours highlighted
— the "upload an image or type a phrase and see where it lands" interaction. This subsumes the
curator's existing `match_image` (which returns a ranked list) by giving it a *position*.

### 8.4 No build step

Spacewalker uses Parcel and npm. The curator has no build step, deliberately, and that property is
worth more than the bundler. Ship three.js as a **vendored ESM file** plus an
`<script type="importmap">`:

```
web/vendor/three.module.js
web/vendor/OrbitControls.js
```

`node --check` on the JS stays the frontend "test", as today.

### 8.5 Performance

`InstancedMesh` with per-instance colour handles ~1e5–1e6 points. The existing `_PROJ_CAP` remains the
guard. Selection must resolve against the **live membership index** rather than an O(N) scan — the
curator already built that index for exactly this reason and it is the mechanism that keeps
interaction O(1) at 1M.

---

## 9. Sample mode

With §2 established, sample mode is a backend and a preview branch — not a parallel application.

1. `backends/none.py` walks a folder (or reads a CSV for text) and emits one item per file with
   `granularity="sample"`, no `rle`.
2. Feature extraction calls `extractor.embed()` (or globally pools `grid()`), writing the same
   `feats[name]` matrix.
3. Clustering, projection, viewer, classifier, kNN, reference retrieval, undo/redo, taxonomy, activity
   and stats work **unchanged** — none of them read masks.
4. Tabs that *are* mask-specific — Refine, Merge-rec, Substructure, In-image — hide in sample mode.
5. Export switches from COCO instances to a classification manifest
   (`{file, class}` CSV/JSON), since there are no masks to encode.

That is full Spacewalker parity: import a folder of images (or a CSV of text), embed with
DINOv2/CLIP/SigLIP2, project to 2D/3D, fly through it, paint labels, export.

---

## 10. Repository layout

```
chevron/
├── pyproject.toml
├── LICENSE                          # MIT
├── README.md
├── DESIGN.md                        # this document
├── chevron/
│   ├── core/
│   │   ├── model.py                 # ItemMeta, TaxonomyClass, Concept, Superclass, ProjectState
│   │   ├── store.py                 # on-disk project layout + atomic IO
│   │   ├── history.py               # undo/redo + audit log
│   │   ├── ids.py  metrics.py  cluster.py  sample.py  similar.py
│   │   └── geometry.py              # shape descriptors, pca_axes, radial_signature, contour_fourier
│   ├── engine/
│   │   ├── engine.py                # CuratorEngine (carve into services later, not during extraction)
│   │   ├── projection.py            # §7 — fit, persist, transform
│   │   ├── ingest.py  classify.py  merge_rec.py  match.py  reference_bank.py
│   │   ├── refine.py  autorefine.py  scale.py  contrastive.py
│   │   ├── shape_prior.py           # vendored from qseg.evaluation
│   │   └── export.py                # COCO + classification manifest
│   ├── extractors/
│   │   ├── base.py                  # protocol + registry
│   │   └── raddino.py  dinov2.py  clip.py  siglip2.py  torchvision.py  text.py
│   ├── backends/
│   │   ├── base.py  qseg.py  coco.py  sam_auto.py  none.py
│   ├── server.py                    # FastAPI
│   ├── cli.py                       # `chevron serve|init|ingest`
│   └── web/
│       ├── index.html  app.js
│       ├── map2d.js  map3d.js  mapshared.js
│       └── vendor/three.module.js  vendor/OrbitControls.js
├── tests/                           # the 273, plus new
└── docs/
```

The on-disk project format is inherited unchanged:

```
<project>/
  state.json            source of truth (config + taxonomy + item meta + order)
  manifest.json
  collection.pkl        records + per-extractor feats matrices (row-aligned)
  collection_shards/    append-only per-chunk ingest shards
  ingests.jsonl         ingest registry (enables view scoping)
  history.jsonl         append-only audit log
  refine/<iuid>.pkl     reversible refine overlays
  cluster_cache/*.npz   cached FINCH partitions
  dr/<key>.joblib       ← NEW: persisted fitted reducers (§7)
  thumbs/               ← NEW: sample-mode thumbnails
  snapshots/  exports/
```

---

## 11. Packaging, dependencies, licensing

```
chevron          numpy scipy scikit-learn opencv-python-headless pillow
                 fastapi uvicorn pycocotools scikit-image joblib
chevron[viz]     hnne umap-learn openTSNE          # DR menu
chevron[embed]   torch transformers                # extractor registry
chevron[sam]     segment-anything | segment-anything-hq
chevron[qseg]    detectron2 + MaskDINO             # documented, not pip-installable
chevron[faiss]   faiss-cpu                         # fast kNN at scale
```

The base install must stay CPU-only and light enough that the test suite runs without torch — that is
what makes the 273 tests fast and portable, and it is a property to defend.

**Licence: MIT**, matching Spacewalker (which is MIT and whose viewer code is being ported). qseg
currently has no LICENSE file; the curator code is first-party so relicensing is unencumbered. The
README should credit Spacewalker and cite arXiv:2409.16793.

---

## 12. Extraction mechanics from qseg

- **Preserve history.** `git subtree split --prefix=chevron` keeps the ~20 curator commits, then
  push that branch to the new repo. A plain copy loses provenance for no benefit.
- **Fix path references.** Tests and `_bootstrap.py` refer to `chevron` and
  `parents[2]`; these need rewriting after the split.
- **Extract verbatim first.** The single highest-risk move would be to carve the 229-method god object
  *while* merging Spacewalker. Move the code unchanged, get 273 tests green in the new repo, and only
  then restructure. The engine can be re-exported under `chevron.engine` with the module split
  deferred.
- **qseg keeps working.** Either qseg depends on `chevron[qseg]`, or `chevron` becomes a thin
  shim. Either way the qseg workflow does not break during the transition.

---

## 13. Sharp edges and design hazards

### H1. The global feature-NaN check is the biggest hazard

The curator disables a feature **everywhere** if *any* row of its matrix is non-finite —
`feature_nan_methods()` is documented as guarding sklearn and cosine-kNN, and imported COCO proposals
already work around it by zero-filling detector features.

Mixing granularities or modalities breaks this immediately: a text item has no `raddino` row, so
`raddino` would be disabled for every image item in the project.

**Mitigation, in order of preference:**
1. Enforce one mode per project (§4.3) — sidesteps it entirely for now.
2. Make the check **per row-subset** rather than global: a feature is usable for the *set of items that
   have it*, and consumers select the subset they can use.

Option 2 is the real fix and is already on the curator's own follow-up list. Option 1 buys time. Do not
attempt mixed-modality projects before option 2 lands.

### H2. The row-alignment invariant

`state.order` must stay index-identical to every `feats` matrix and to `collection["records"]`;
`assert_aligned()` enforces it. Sample items append to the same structures and must preserve it. Every
new ingest path needs the same append discipline.

### H3. Cache invalidation

There are ~13 hand-rolled caches keyed by bespoke tokens (`coll_version`, `mutation_serial`,
`scope_token`, `clf_version`) — the curator README names this the main correctness hazard. This merge
adds two more cache dimensions (`extractor`, persisted DR key). **Unify the versioning during the
extraction**, or the count grows to ~16 and the next feature becomes materially harder.

### H4. The mutation lock

21 mutating engine methods carry `@_mutating` (a reentrant lock) so concurrent sessions cannot corrupt
state. Every new mutating method — sample ingest, query pin, extractor compute — must carry it. This is
an easy thing to forget and a hard thing to debug.

### H5. Query-projection correctness

Stored-vs-recomputed normalisation, and DR methods without `.transform` — see §7.3.

### H6. Three.js without a bundler

Vendored ESM + importmap. Pin the three.js version explicitly; `OrbitControls` must come from the
matching release.

### H7. Scope creep during extraction

The god object (229 methods), 110 flat endpoints and ~571 JS module-globals are all real debt, but
attacking them concurrently with the merge risks the one asset that makes this safe: a green
273-test suite. Refactor after.

---

## 14. Phasing

Ordered by dependency. Nothing here is implementation; it is the sequence to design *to*.

| Phase | Content | Risk | Unblocks |
|---|---|---|---|
| **P0** | Extract to its own repo, verbatim. `pyproject`, LICENSE, CI, path fixes, vendored qseg modules, `backends/qseg.py`. **Exit criterion: 273 tests green in the new repo.** | Low | everything |
| **P1** | Data-model unification: `granularity` + `modality`, `_in_scope` folding, per-subset NaN fix (H1). | Medium | P4, P5 |
| **P2** | Extractor registry: protocol, `compute_features` endpoint, RAD-DINO ported as the reference implementation, then DINOv2 / CLIP / SigLIP2. | Low | P3, P5 |
| **P3** | Persisted DR + `project_query` + DR menu with query-capability flags. | Medium | P4 query pin |
| **P4** | Unified viewer: shared selection model, 3D renderer, colour-by extensions, query pin. Frontend-only; parallelisable with P1–P3. | Low | — |
| **P5** | Sample mode: `backends/none.py`, preview branches, tab gating, manifest export. | Low | — |

P4 is the most visible and the least entangled — it can proceed alongside P1–P3 since
`/api/projection_points?dims=3` already returns what it needs.

---

## 15. Open decisions

1. **Mixed-modality projects.** §4.3 recommends one mode per project initially, with the item model
   left mix-capable. Confirm, or commit to the per-subset NaN fix (H1 option 2) up front.
2. **Video depth.** Spacewalker samples 10 equidistant frames and pools them. Keep that, or treat
   video as out of scope for the first Chevron release?
3. **Torchvision-era encoders.** ResNet50 / VGG16 / plain ViT exist in Spacewalker's zoo for parity.
   Worth carrying, or start with RAD-DINO / DINOv2 / CLIP / SigLIP2 only?
4. **qseg's relationship.** Does qseg depend on Chevron, keep a shim, or drop `chevron`
   entirely once the new repo is live?
5. **Repo visibility and naming of the published artefact.** Chevron is the name used in the DCA-MI
   paper for the curation engine; Spacewalker has its own arXiv identity. Confirm the README framing
   (Chevron *incorporates* Spacewalker) is the intended public story.
