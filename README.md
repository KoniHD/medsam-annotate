# medsam-annotate

AI-assisted annotation for pediatric lower-leg ultrasound.

MedSAM2 is served through [MONAI Label](https://github.com/Project-MONAI/MONAILabel) to a
[3D Slicer](https://www.slicer.org/) client. The annotator box- or click-prompts, the model returns a
mask, she corrects it, and the corrections accumulate into a fine-tuning set so the model's
unprompted pre-labels improve round over round. Masks then flow into a per-structure measurement
table for downstream statistical analysis.

See [`design.md`](design.md) for the full design and rationale.

**Status:** pre-MVP. Implementing *Path A* (§8) — everything local, no cloud, no patient data.

## Requirements

- macOS (Apple Silicon) or Linux with CUDA
- [`uv`](https://docs.astral.sh/uv/)
- 3D Slicer 5.12+ (`brew install --cask slicer` on macOS)

## Setup

```bash
git clone --recurse-submodules gh:koni/medsam-annotate
cd medsam-annotate
./scripts/bootstrap.sh      # uv venv, deps, checkpoints
./scripts/start_server.sh   # MONAI Label on http://localhost:8000
```

Then in 3D Slicer: install the MONAI Label extension, point it at `http://localhost:8000`, and pick
the `medsam2_2d` or `medsam2_3d` model.

## Compute device

Inference and training pick a device at runtime — `cuda`, `mps`, or `cpu` — and never hardwire one.
The order is CUDA → MPS → CPU, and any failure on an accelerator demotes to CPU rather than erroring
at the annotator. Override with:

```bash
LEGUS_DEVICE=cpu ./scripts/start_server.sh
```

CPU is the guaranteed-correct reference path. MPS is the fast path on Apple Silicon. CUDA is what the
production deployment (design §8, Path C) will use, from this same code.

## Layout

| Path | What |
|---|---|
| `apps/legus/` | Our MONAI Label app — the only code we own |
| `apps/legus/lib/infers/medsam2.py` | The MedSAM2 adapter. This is the swap point for MedSAM3 (design §9) |
| `external/` | Pinned submodules: MONAI Label, MedSAM2. Never edited; pushes disabled |
| `scripts/` | Bootstrap, data fetch, server start |
| `data/`, `apps/legus/model/` | Gitignored. Imaging data and checkpoints never enter git |

## Data

The repo contains **no imaging data**, and must not — `data/` and `apps/legus/model/` are
gitignored. Real patient imaging stays out of this repository and off any machine without the
institutional data-processing agreement in place — see design §3.

Demo data: `scripts/fetch_data.py` fetches public **CAMUS** echocardiography images + masks
(`zeahub/camus-sample` on Hugging Face, CC BY-NC-SA 4.0) and converts them into the MONAI Label
datastore layout under `data/legus/`. `wanglab/RVENet-MedSAM2` was the originally-planned source
but turned out, on inspection, to contain only MedSAM2-generated mask PNGs with no source
images — the raw videos require an individual, non-redistributable Research Use Agreement with
RVENet — so it cannot populate an image+mask datastore; see the docstring in `fetch_data.py` for
the full story.

```bash
uv run scripts/fetch_data.py --limit 8   # small demo set; idempotent, re-running won't re-download
```

## Licenses

Apache-2.0 for MONAI Label and MedSAM2 code, BSD for 3D Slicer, BSD-3 for pyradiomics.
The MedSAM2 *weights* on Hugging Face are **CC-BY-SA-4.0** — share-alike, so derived fine-tuned
checkpoints inherit that obligation.
