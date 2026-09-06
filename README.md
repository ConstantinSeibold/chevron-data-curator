# Chevron

**Turn a pile of unlabelled images into a labelled segmentation dataset, on your own machine.**

You point Chevron at some images. It gets mask proposals for them — from SAM, from an off-the-shelf
detector, or from a COCO file you already have — throws away whatever labels those models had, and
groups the masks by what they look like. You then label *groups* instead of individual masks, fix the
ones that are wrong, and export COCO.

Everything runs in one local process. No database, no object store, no inference server, no
containers, no accounts.

```bash
pip install -e ".[dev,viz,embed]"
chevron                       # → http://127.0.0.1:7870
```

---

## Quickstart

Ten minutes, starting from a folder of images and nothing else. Every command below is copy-pasteable.

**1. Start it.**

```bash
chevron                       # projects live in ~/.chevron/projects
```

Open <http://127.0.0.1:7870>. You get the launcher: project cards, and a **New project** button.

**2. Make a project.** Give it a name and your image folder. Or from the shell:

```bash
curl -s -X POST localhost:7870/api/projects -H 'Content-Type: application/json' \
  -d '{"name":"My first project","config":{"images":{"root":"/data/my_images"}},"open":true}'
```

**3. Get masks.** This is the one step with no button yet — it is an API call:

```bash
curl -s -X POST localhost:7870/api/propose -H 'Content-Type: application/json' \
  -d '{"backend":"sam_auto","image_root":"/data/my_images","limit":50}'
# → {"ok": true, "n_instances": 312, "n_images": 50}
```

`sam_auto` needs no trained model and downloads its checkpoint on first use. If you already have
masks, use `{"backend":"coco","coco_path":"/data/masks.json"}` instead — that needs no model at all.
See [Where masks come from](#where-masks-come-from).

**4. Describe the masks with an embedding model**, so similar things cluster together:

```bash
curl -s -X POST localhost:7870/api/compute_features -H 'Content-Type: application/json' \
  -d '{"extractor":"dinov2"}'
```

Or do it in the UI: **Settings → Instance features → Embedding model → Compute features**.

**5. Cluster.** In **Curate**, tick the features to cluster on and press **Cluster**. The scope rail
fills with partitions — groups of instances that look alike.

**6. Label.** Click a partition. Its crops appear in the grid. If the group is one thing, type a class
name in the inspector and hit **Assign** — the whole group is labelled at once. If it is junk, hit
**Reject**. If the group is mixed, select the good ones by clicking or drag-painting and assign only
those; the rest stay in the pool.

Repeat. Each pass shrinks the unassigned pool.

**7. Export.**

```bash
curl -s -X POST localhost:7870/api/export -H 'Content-Type: application/json' -d '{}'
# → exports/curated.json  (COCO: images, annotations with RLE segmentation, categories)
```

or **Ship → Export**.

---

## Working in the UI

Six areas across the top:

| Area | What you do there |
|---|---|
| **Curate** | The main workspace. A scope rail (partitions, classes, rejected bin, sub-clusters), a canvas, and an inspector that acts on your selection. |
| **Assist** | Let the machine propose: train a classifier on what you have labelled and apply it to the rest; a merge recommender; reference-image search. |
| **Classes** | Your taxonomy — create, rename, group, colour. |
| **Ship** | Export COCO, the release gate, and the retrain loop. |
| **Insights** | Statistics and an activity log. |
| **Settings** | Embedding model, feature computation, checkpoints, compute device. |

Curate's canvas has three views of the *same* selection, so switching never loses it:

- **Grid** — crops, with nearest-neighbour class suggestions on each.
- **Map** — a 2D/3D latent plot: orbit, paint-select, colour by state/class/partition/source/score.
  Type a phrase or an instance id to drop a query onto it and grab the neighbours.
- **Image** — one image and all its instances, ordered by how much work each image has left.

**Everything is reversible**: undo/redo covers every action, and mask edits are non-destructive
overlays — your original masks are never overwritten.

---

## Where masks come from

Labels are always discarded. A COCO detector's 80 classes are not your label space; you supply the
taxonomy.

| Backend | Needs | Use it when |
|---|---|---|
| `coco` | **nothing** | You already have masks in a COCO file. |
| `whole_image` | **nothing** | You want to label whole images, not masks (see [Sample mode](#sample-mode)). |
| `sam_auto` / `samhq_auto` | `pip install segment-anything` | You have no model at all. Checkpoint auto-downloads. |
| `torchvision_maskrcnn` | `torch` + `torchvision` | Quick generic proposals; runs fine on CPU. |
| `hf_seg` | `chevron[embed]` | Any HF `AutoModelForUniversalSegmentation` (default: Mask2Former-COCO). |
| `qseg` | a qseg checkout + CUDA | You have a trained qseg model; set `CHEVRON_QSEG_ROOT`. |

`GET /api/backends` lists them with availability and an install hint.

## Embedding models

The embedding decides what "looks alike" means, so it drives the clustering, the map, the classifier
and nearest-neighbour search. Pick one in **Settings → Instance features**:

| Model | Good for |
|---|---|
| `dinov2` | General purpose. **The sane default** for most data. |
| `clip`, `siglip2` | General purpose, plus a shared image-text space — needed for text queries on the map. |
| `raddino` | Chest X-rays specifically. |

You can compute several and mix them, with weights, wherever features are selected. Mask geometry
(`shape`, `shapecoord`, `coords`) is always available and needs no model.

## Speeding yourself up

Once you have a few hundred instances labelled:

- **Classifier** — trains on your labels, predicts the rest, and shows the predictions as badges on
  the crops for you to accept or ignore.
- **Merge recommender** — spots masks that are fragments of one object, and learns from your merges.
- **Reference search** — upload a picture of a thing, get the partitions that look like it.
- **Refine** — an op chain per instance (contrast, threshold, vessel trace, GrabCut, SAM/SAM-HQ),
  per-class rules, few-shot shape transfer, and a hand-draw editor for the stubborn ones.

## Sample mode

To label whole *images* rather than masks, create the project with `mode: "sample"` and ingest with
the `whole_image` backend. Clustering, the map, the classifier and search all work exactly as before;
the mask-only tools disappear from the nav, and Export becomes a classification manifest
(`{file, class}` as JSON and CSV) instead of COCO.

Text and video items are **not** implemented.

## Using it on your own data

The curation loop knows nothing about what your pixels depict — proposals in, clusters, labels, COCO
out. It has been used on chest X-rays because that is where it was written, not because of an
assumption in the core. For a surgical-video, microscopy or aerial dataset:

- Use `dinov2` as the embedding model (RAD-DINO is the chest-X-ray one).
- Ignore the shipped `taxonomy_seed.json` (chest foreign bodies) — it is applied only if you press
  **Seed**. Create your own classes, or supply your own seed JSON.
- The `vessel_extend` refine op is tuned for catheters; it is one op among many.

Nothing else changes.

---

## Configuration

```bash
chevron                             # launcher over ~/.chevron/projects
chevron --root /data/projects       # ...over a different folder
chevron --project /data/projects/p1 # skip the launcher, open one project
chevron --port 8080 --host 0.0.0.0
```

**Compute device** is automatic: CUDA → Apple MPS → CPU, whichever is present. **Settings → Instance
features** shows what was picked. To override:

```bash
CHEVRON_DEVICE=cpu chevron          # force CPU
CHEVRON_DEVICE=cuda:1 chevron       # pick a GPU
CHEVRON_AMP=1 chevron               # enable mixed precision on MPS (off by default)
```

**Sharing.** Chevron binds `127.0.0.1` and has **no authentication**. The intended multi-user setup is
one shared project reached over an SSH tunnel:

```bash
ssh -L 7870:localhost:7870 you@host
```

Concurrent edits are serialised and the browser shows an "updated elsewhere" banner, but it is
single-tenant and last-writer-wins. For any network exposure, put it behind an authenticating proxy.

## Troubleshooting

**All the crops are black.** The project was moved and the stored image paths no longer resolve.
Chevron loads images by absolute path and substitutes a black placeholder on a miss, so the UI looks
fine while showing nothing. Repoint the paths at the new location.

**RAD-DINO fails to download.** It forces `HF_HUB_OFFLINE=1`, so with no cached weights it fails
offline rather than fetching. `export HF_HUB_OFFLINE=0` before first use.

**The 3D map won't open.** It needs WebGL, which software/remote GL or a locked-down browser may not
provide. Chevron stays in 2D and says so.

**Clustering says a feature is unusable.** The NaN check is global: one non-finite row disables that
feature everywhere. Recompute it.

## What's on disk

A project is a plain directory — no database:

```
<project>/
  state.json            config + taxonomy + per-instance labels + row order
  collection.pkl        records + feature matrices
  collection_shards/    append-only ingest shards (crash-recoverable)
  history.jsonl         audit log, and what undo/redo reads
  refine/<iuid>.pkl     reversible mask edits
  cluster_cache/        cached clusterings
  exports/  snapshots/
```

Small writes are atomic (`tmp → os.replace`). Copy the directory to move a project.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q          # 418 tests, CPU-only — no model stack needed
```

Architecture, design decisions and their rationale live in `DESIGN.md`.

## Limitations

- **Ingest has no UI.** Getting masks into a project is an API call (`/api/propose`).
- **No authentication**, single-tenant, last-writer-wins.
- **Apple MPS has never run on Apple hardware.** Device selection, the precision policy and the
  operator-gap fallback are tested by simulation on every machine, but no Metal kernel has executed.
- Text and video items are not implemented.
- A project is held in RAM: the practical ceiling is roughly 1M instances on a 100 GB machine.
- ML paths are under-tested, and there are no browser tests.

## Credits

Chevron merges two projects:

- **qseg curator** — the instance-curation engine, extracted here with its history.
- **[Spacewalker](https://github.com/ConstantinSeibold/Spacewalker)**
  ([arXiv:2409.16793](https://arxiv.org/abs/2409.16793), MIT, © 2024 Lukas Heine) — the latent-space
  viewer, persisted dimensionality reduction with query projection, and the embedding/DR menus,
  generalised here so a point is an *instance* rather than only a whole sample.

MIT — see `LICENSE` and `THIRD_PARTY_NOTICES.md`.
