# Third-party notices

Chevron incorporates code from the projects below. Their licence terms are reproduced here as
required.

---

## Spacewalker

- **Upstream:** https://github.com/ConstantinSeibold/Spacewalker
- **Paper:** Heine et al., *Spacewalker: Traversing the Latent Space to Explore and Annotate Data*,
  [arXiv:2409.16793](https://arxiv.org/abs/2409.16793)
- **Licence:** MIT
- **Copyright:** Copyright (c) 2024 Lukas Heine

**What Chevron takes:** the latent-space viewer (three.js point cloud, orbit navigation, raycast
hover, paint-to-annotate), the persisted dimensionality-reduction approach that lets a *new* query be
projected into an already-fitted space, and the embedding/DR menu concept.

**What Chevron does not take:** Triton, Django, Postgres, MinIO, docker-compose, the ONNX
`model_repository` and the Parcel build — all replaced by an in-process, local-only stack.

**Ported in P7:** `chevron/web/map3d.js` — the 3D latent walk (instanced point cloud, orbit
navigation, paint-to-select). It carries the copyright line above in its header. Generalised so a
point is an *instance* (a mask crop) rather than only a whole sample, and so painting writes into
Chevron's shared selection.

```
MIT License

Copyright (c) 2024 Lukas Heine

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

---

## qseg

Chevron began as `qseg/tools/curator` and its git history is preserved here. First-party code by the
same author; no separate licence obligation. Modules vendored verbatim from the qseg tree, each
carrying a provenance note in its docstring:

| Chevron module | Origin in qseg |
|---|---|
| `chevron/core/collection.py` | `notebooks/qseg_playground.py` (generic collection/clustering/pair helpers) |
| `chevron/core/morphology.py` | `src/qseg/ssl/refine_anatomy.py` |
| `chevron/core/shape_prior.py` | `src/qseg/evaluation/shape_prior_model.py` |
| `chevron/core/class_head.py` | `src/qseg/models/class_extend.py` |
| `chevron/extractors/raddino.py` | `notebooks/qseg_playground.py` (`RadDinoExtractor`) |

---

## Runtime dependencies

Installed from PyPI under their own licences and not redistributed here — see each project for
terms: FINCH (`finch-clust`), h-NNE, UMAP, openTSNE, scikit-learn, scikit-image, OpenCV,
pycocotools, FastAPI, Uvicorn, PyTorch, Hugging Face Transformers, Segment Anything.

### three.js

- **Upstream:** https://github.com/mrdoob/three.js — **Licence:** MIT, © 2010-2025 three.js authors
- Vendored verbatim (licence header intact) at `chevron/web/vendor/three.module.min.js` and
  `chevron/web/vendor/OrbitControls.js`, and served by Chevron itself. Vendoring rather than using a
  CDN keeps the frontend free of both a bundler and a runtime network dependency.
