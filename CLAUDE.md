# CLAUDE.md

Guidance for Claude Code working in this repo. See [`design.md`](design.md) for full rationale and
[`README.md`](README.md) for the demo runbook.

## What this is

AI-assisted annotation for pediatric lower-leg ultrasound. **MedSAM2** is served through
**MONAI Label** to a **3D Slicer** client: the annotator box/point-prompts, corrects the mask, and
corrections accumulate to fine-tune so unprompted pre-labels improve round over round. Masks flow
into a per-structure measurement CSV.

Current scope is **Path A** (design §8): local, offline, on an Apple Silicon Mac, 3D Slicer client.
Paths B (OHIF/browser) and C (cloud GPU) and the MedSAM3 upgrade (§9) are out of scope but the code
is built to stay swappable for them.

## Hard rules (do not violate)

- **`uv` only** for Python. Never bare `pip`, never conda. `uv sync`, `uv run`, `uv pip`.
- **Never edit anything under `external/`** — `external/MONAILabel` and `external/MedSAM2` are
  submodules pinned to SHAs with push URLs set to `DISABLE`. Read them freely; writing there is a
  hard failure. All our code lives in `apps/legus/`, `scripts/`, `tests/`.
- **Wrappers, not reimplementations.** Call what MONAI Label / MONAI / SAM2 / SimpleITK /
  pyradiomics already provide. Don't copy upstream method bodies to tweak a line.
- **Device is never hardwired.** Runtime-selectable cuda/mps/cpu via `available_devices()`; CPU is
  the reference path, MPS the fast path on this Mac, CUDA the future cloud path. Accelerator
  failures demote to CPU, they never 500 at the annotator.
- **No imaging data or checkpoints in git.** `data/` and `apps/legus/model/` are gitignored.
- Keep `.venv/bin/ruff check apps scripts tests` clean and `pytest tests/` green.

## Architecture / key files

- `apps/legus/main.py` — the MONAI Label app. Registers `medsam2_2d` / `medsam2_3d` infers and the
  fine-tune trainer. Knows no MedSAM2 specifics (checkpoint paths, device names) — those stay in the
  adapter, so the MedSAM3 swap (design §9) touches only that module.
- `apps/legus/lib/infers/medsam2.py` — **the model-serving seam** (`MedSAM2InferTask(Sam2InferTask)`).
  Subclasses upstream, reusing `run2d`/`run_3d`/`Writer` untouched. Owns checkpoint resolution
  (`_resolve_checkpoint` prefers `finetuned/latest.pt`), the device policy (`available_devices()`),
  two-layer accelerator→CPU fallback, and `segment(image, prompt) -> mask` (the documented swap
  point). MedSAM2 checkpoints load **strictly** under `configs/sam2.1/sam2.1_hiera_t.yaml` — no
  state-dict remapping.
- `apps/legus/lib/trainers/medsam2.py` — the fine-tune `TrainTask` (design §6). Trains the mask
  decoder on the annotator's **corrected** labels (FINAL tag), never on `labels/original/` (public
  ground truth). Stable per-image blake2b train/val split so the holdout survives datastore growth.
  Writes `finetuned/latest.pt`, which serving and the next round both prefer → the loop closes.
- `apps/legus/lib/measure/statistics.py` — pyradiomics wrapper → CSV, one row per subject×structure,
  `area_mm2` / echo intensity / first-order, with a `spacing_calibrated` flag (never fakes mm).
- `scripts/fetch_data.py` — fetches public CAMUS into the MONAI Label datastore layout as
  singleton-z uint8 NIfTI, real spacing preserved, ground truth under `labels/original/`.
- `scripts/bootstrap.sh` / `start_server.sh` / `check_demo.sh` — setup / run / preflight.
- `scripts/reset_demo.sh` — stop the server and clear submitted annotations (`labels/final/` +
  `datastore_v2.json`), keeping images and the `labels/original/` ground truth. `--stop-only` just
  stops the server.
- `scripts/legus_probe.py` — REST round-trip helper (uses the shipped `MONAILabelClient`).

## Running / testing

```bash
./scripts/bootstrap.sh                     # uv sync + submodules + checkpoints (idempotent)
uv run scripts/fetch_data.py --limit 8     # public CAMUS demo data
./scripts/check_demo.sh                     # 6-point preflight; exits non-zero on any failure
./scripts/start_server.sh                   # MONAI Label on :8000 (defaults LEGUS_DEVICE=cpu)
uv run python -m pytest tests/ -q           # fast suite; add -m slow for the real fine-tune test
```

## Gotchas learned the hard way

- `monailabel start_server` re-execs `python` from **PATH**, not the launching interpreter — dies
  with "No module named 'monailabel'" unless `.venv/bin` is on PATH. `start_server.sh` handles this.
- Upstream `run2d` on a 2D prompt with **no slice hint** takes a branch that never converts
  grayscale→RGB and 500s inside torchvision. Guarded centrally in `MedSAM2InferTask.__call__`
  (`_ensure_slice_hint`), because the REST path Slicer uses never calls `segment()`.
- `uv sync` will **prune** monailabel/monai/sam2 unless monailabel is a declared dependency sourced
  from the submodule — it is, via `[tool.uv.sources]` in `pyproject.toml`. Don't undo that.
- The MONAI Label Slicer plugin gates its box tool on `dimension`+`DEEPGROW` type, not the model
  name, so `medsam2_2d` gets the ROI widget with no renaming. Interactive models appear under
  **SmartEdit**, never Auto Segmentation (that section shows only for `segmentation`/`detection`
  models, which we don't register). The interactive **run button is labeled "Update"**, and the
  box tool is "ROI/BBOX Prompt".
- The task advertises `labels` (default `muscle,subcutaneous_fat,bone_surface`, override with
  `LEGUS_LABELS`). This is not optional polish: with `labels=None` the plugin auto-creates no
  segments and refuses to run an interactive model with no label selected. SAM2 is class-agnostic,
  so labels only name the output segment (→ one CSV row per structure). Real structures TBD at the
  demo (DEMO.md Q A2).
- Per-round `val_dice` uses a box prompt derived from the GT mask (an oracle prompt) — it measures
  segmentation-given-a-prompt, not unprompted pre-label quality. Read the numbers accordingly.

## Not machine-verified

The 3D Slicer GUI click-through (load study → draw box → mask → correct → submit). No headless way
to drive Slicer's Qt UI. A human must do one pass; the click-path is in README §2.
