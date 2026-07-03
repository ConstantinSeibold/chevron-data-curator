# qseg curator — current state

A single-user (small-team over SSH) web app for **turning a segmentation model's raw predictions
into a curated, labelled dataset**: cluster the unassigned instances, assign/reject/merge/refine
them, and export COCO. Custom FastAPI backend + vanilla-JS frontend over an in-process
`CuratorEngine`; deliberately *not* Gradio (which does not scale to 25k+ instances).

Run: `python -m tools.curator.server --project DIR [--host 127.0.0.1] [--port 7870]`

Status (this snapshot): **~9.5k LOC**, **110 API endpoints**, **273 tests (34 files), green ×3**.

---

## Hosting / multi-user

The server binds `127.0.0.1:7870` by default (localhost only, **no built-in auth**). The intended
multi-user model is **one shared project accessed by a small team over SSH tunnels**:

```
ssh -L 7870:localhost:7870 you@this-pc      # each teammate; then open http://localhost:7870
```

The app stays localhost-bound (nothing exposed to the network), SSH is the auth, and all tunnels hit
the same shared engine.

- **Concurrent mutations are serialized** — a reentrant `self._mutate_lock` (`@_mutating` on the 21
  public state-mutating methods) + a lock-consistent background saver, so simultaneous requests can't
  corrupt state/caches/undo. Additive; no behavior change; no deadlock.
- **Live-refresh** — `GET /api/version` exposes a monotonic mutation stamp; the browser polls it (4s)
  and shows a non-disruptive "↻ updated elsewhere" banner when *another* session changed the data.
- Still **single-tenant / last-writer-wins** — two people editing the same partition don't get
  per-instance locks (the banner just makes it visible); no user identity (all sessions are
  `127.0.0.1` over the tunnel); one shared GPU serializes RAD-DINO/SAM/import/train.

For LAN/internet exposure you'd add a reverse proxy (Caddy/nginx) + auth — not built.

---

## Tabs / capabilities

- **Partitions** — FINCH-clustered unassigned pool; per-crop 1-NN class suggestion + gate markers,
  clickable class/reject subset filter, assign/merge/reject/→refine/→substructure.
- **In-image** — one image's instances; per-crop accept/reject with the classifier; **workload sort**
  (rank the image picker by classifier-estimated work left, with class-variety re-ranking).
- **Map** — latent-space projection (h-NNE→UMAP→PCA, label-independent & cached) on a 2D canvas:
  pan/zoom, **paint-select → assign/reject**, color by state/class/partition/**source**/score,
  hover-crop.
- **Refine** — per-instance mask refinement chain (contrast/threshold/vessel/line/grabcut/SAM/…),
  auto-refine search, class rules. Includes:
  - **Unified threshold op** — method (Otsu | manual | GHT/Barron) × region (in-mask | in-bb | any) ×
    direction (auto | above | below).
  - **Few-shot shape transfer** — use refined/drawn instances as reference masks; SAM/SAM-HQ refines
    the partition toward that shape within each bbox. **Auto-routes by shape-kind**: line partitions get
    a reference-calibrated vessel trace instead (no SAM). τ + agreement-IoU gates; preview-before-commit.
  - **Hand-draw mask editor** — brush/eraser (+ clear/invert/fill-holes) on a zoomed crop; the result
    becomes the effective mask, usable as a transfer reference.
- **Classifier** — train a per-class classifier over the labelled instances; apply to the pool.
- **Reference** — RAD-DINO retrieval; find a partition by an uploaded reference image.
- **Substructure / Classes / Release / Loop / Rejected / Activity / Stats / Config** — taxonomy,
  release gate (which images enter the final COCO), training launch, rejected bin, curation analytics,
  model + import config.

### Multi-model proposals + source facet
- Ingest **any model's proposals** as a tagged source: `Config → Import proposals (COCO)` (matched to
  the project's images by basename). Model-agnostic — detector features are 0-filled (never poisons the
  global NaN check), the cross-source space is `shapecoord` (+ `raddino` if present).
- **Source facet** — a header chip bar filters by proposing model in *every* tab at once (folded into
  the shared `_in_scope` predicate); composes with the ingest-scope selector; Map "color by source"
  shows agreement geography (mixed-source cluster = models agree).

---

## Architecture

- `engine.py` (~4.4k LOC) — `CuratorEngine`, the god object: data model, clustering, classifier,
  refine, SSL, ingest, projection, sources, release, training. State = `collection` (records + per-method
  `feats` matrices, row-aligned) + `state.meta` (per-instance `InstanceMeta`) + taxonomy.
- `server.py` — thin FastAPI layer (closures over one engine); windowed JSON + lazy/batched crops.
- `web/{index.html,app.js}` — vanilla-JS tabs; no framework/build.
- `refine.py` — pure cv2/skimage mask ops. `collect.py` — feature extraction + `concat_collections`.
  `state.py` — `InstanceMeta` + persistence. `match.py` — RAD-DINO correspondence (built, **unwired**).

**Key design properties**: reversible curation (history snapshots; refine overlays gated on
`meta.refined`); write-behind saves + append-only resumable ingest; label-*independent* projection cache;
row-alignment invariant across feats matrices (`assert_aligned`); scope/source folded into one
`_in_scope` predicate threaded through the live index.

---

## Known limitations (honest current state)

- **God object** — `CuratorEngine` is 229 methods; server is 110 flat endpoints; `app.js` is ~571
  module-globals. Fast to extend, hard to isolate (this session's features interleave across the same
  files).
- **~13 hand-rolled caches** with bespoke invalidation keys (coll_version / mutation_serial /
  scope_token / clf_version) — the main correctness hazard; a unified versioning abstraction would help.
- **Global feature-NaN check** — one bad feature row disables that feature everywhere (worked around by
  0-filling imported detector features).
- **Stringly-typed domain** (`pid` overloads class:/finch/singleton; `body: dict` endpoints, no Pydantic).
- **ML paths under-tested** — 273 tests are strong on pure-python orchestration but stub SAM / RAD-DINO /
  h-NNE / inference / training; **no frontend tests** (only `node --check`).
- **Single-tenant** — one shared project, one GPU; no per-user isolation, no auth, no soft-locks/presence.

## Suggested follow-ups
1. Soft-locks / presence (claim a partition, see who's editing) + user identity for attribution.
2. Cross-source **merge/consensus** (source-aware merge-rec to dedupe high-IoU proposals; "≥k agree").
3. Carve services out of the god object (`ProjectionService`/`IngestService`/…) + typed API models.
4. Wire `match.py` (RAD-DINO dense correspondence) to seed line transfers when a member mask is broken.
5. Unify the cache/versioning; add a mutation lock to any *new* mutating method (`@_mutating`).

---

## Tests

`pytest tools/curator/tests/ -q` — 273 tests / 34 files, CPU-only (SAM/RAD-DINO/model paths stubbed).
Run ×3 historically (thread-flakiness; the mutation lock should reduce it). Frontend: `node --check
tools/curator/web/app.js`.
