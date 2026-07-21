# medsam-annotate — Demo Runbook

AI-assisted annotation for pediatric lower-leg ultrasound. MedSAM2 served through
[MONAI Label](https://github.com/Project-MONAI/MONAILabel) to a [3D Slicer](https://www.slicer.org/)
client. See [`design.md`](design.md) for the full design and rationale — this file is the
step-by-step runbook for actually running the **Path A** demo (design §8, §10): local, offline,
reliability-first, for a tech-averse domain-expert audience who will abandon the tool if it breaks.

**Read this top-to-bottom in order once. On the day of the demo, jump straight to "Before each
demo".**

---

## 0. One-time setup

```bash
git clone --recurse-submodules <repo-url>
cd medsam-annotate
./scripts/bootstrap.sh
uv run scripts/fetch_data.py --limit 8
```

**`./scripts/bootstrap.sh`** — idempotent, safe to re-run. It:
1. Initializes the pinned submodules (`external/MONAILabel`, `external/MedSAM2`) and disables
   their push remotes.
2. `uv sync`s the environment (uv only — never bare `pip`, never conda).
3. Downloads the MedSAM2 checkpoint (~149MB, CC-BY-SA-4.0) to `apps/legus/model/`, atomically
   (via a `.part` file + `mv`, so a network drop mid-download can never leave a corrupt file
   silently "present" on the next run).
4. Actually **loads** the checkpoint on CPU and asserts it comes back as the ~39.0M-parameter
   SAM 2.1 tiny model this whole demo was verified against — catches a corrupt/truncated
   checkpoint at setup time, not in front of the annotator.

**Expected output ends with:**
```
==> Verifying checkpoint loads (cpu, offline)
    loaded OK -- 39.0M params

Bootstrap complete.
```

**If it fails:**
| Symptom | Fix |
|---|---|
| `uv not found` | Install uv: https://docs.astral.sh/uv/ |
| Checkpoint download fails (network) | Re-run `./scripts/bootstrap.sh` — it's idempotent and only re-downloads what's missing/incomplete |
| `loaded OK` line never appears / assertion error | Delete `apps/legus/model/MedSAM2_latest.pt` and re-run — the download was corrupt |

**`uv run scripts/fetch_data.py --limit 8`** — fetches public CAMUS echocardiography images into
`data/legus/` as the demo dataset (see "Data" below for why CAMUS and not the originally-scoped
RVENet-MedSAM2). Idempotent; re-running doesn't re-download existing files. Ends by printing how
many image/mask pairs are now in the datastore.

---

## 1. Before each demo (run ~10 minutes ahead)

```bash
./scripts/check_demo.sh
```

This is the single command to run before the annotator sits down. It prints a `PASS`/`FAIL` line
for each of five things and **exits non-zero if anything is wrong** — do not proceed to the live
demo until it exits 0:

1. MedSAM2 checkpoint present (and not a truncated/corrupt file)
2. Datastore non-empty and has at least one unlabeled image (so there's active-learning work to
   show)
3. Server reachable — if nothing is listening yet, it starts one via `start_server.sh` (which
   pre-warms it) and **leaves it running** for the demo
4. A real box-prompt REST round trip against that server returns a non-empty mask
5. That mask flows through the measurement export into a real CSV

**Expected output:**
```
== LEGUS demo preflight ==

PASS  checkpoint present (.../MedSAM2_latest.pt, 156 MB)
PASS  datastore has 4 image(s) under .../data/legus, 4 unlabeled
PASS  started a server at http://localhost:8000 (left running for the demo -- pid 12345)
PASS  box-prompt REST round trip returned a non-empty mask (image=... elapsed=0.27s nonzero_voxels=71323)
PASS  measurement CSV generated (1 row(s), see $STUDIES-relative image=...)

ALL CHECKS PASSED. Server is up and pre-warmed at http://localhost:8000 -- point 3D Slicer at it.
```

**If any check fails**, it tells you exactly which of the four remediation commands to run
(`bootstrap.sh`, `fetch_data.py`, inspect the printed server log, or re-run once the above are
fixed). Fix it and re-run `check_demo.sh` — don't proceed until it's all green.

If it started the server for you, **leave that terminal window open** — closing it stops the
server mid-demo.

---

## 2. Live demo — Slicer click-path

> **Everything below this line involving 3D Slicer's GUI has NOT been machine-verified in this
> session** — there is no headless way to drive Slicer's Qt UI from here. See "What has NOT been
> verified" below for exactly what a human needs to click through once, before the real demo.

1. Open **3D Slicer**. First time only: **Extension Manager** → install "MONAI Label" (or, for the
   pinned dev version this repo builds against: **Developer → Extension Wizard → Select Extension**
   → `external/MONAILabel/plugins/slicer/MONAILabel`). Restart Slicer if prompted.
2. Open the **MONAI Label** module (module dropdown, top-left).
3. Server URL field → `http://localhost:8000` → **Fetch Info** button. The interactive models
   `medsam2_2d` / `medsam2_3d` and the advertised labels (`muscle`, `subcutaneous_fat`,
   `bone_surface`) load; those labels auto-create named segments per study.
4. **Active Learning → Next Sample** → loads an unlabeled image into the 2D slice views.
5. Open the **SmartEdit** section — this is where interactive models run. **There is no "Auto
   Segmentation" section, and that is correct:** the plugin only shows Auto Segmentation for
   whole-volume automatic models, and we deliberately register none (design §5 — MedSAM2 replaces
   that role). In SmartEdit:
   - **Model** → `medsam2_2d`.
   - **Label** → the structure you're about to outline (e.g. `muscle`). The mask lands in that
     segment, which becomes its row in the measurement CSV.
   - Click the **ROI/BBOX Prompt** place button, then drag a box around the structure in a slice
     view. Optionally add positive/negative points with the foreground/background place buttons.
6. Click **Update** — **this is the interactive run button** (there is no button literally called
   "Run" for interactive models). A mask appears as a colored overlay in ~1s (pre-warmed) — longer
   the first time if pre-warm was skipped.
7. Use Slicer's **Segment Editor** (Paint/Erase/Scissors) to correct the mask if needed.
8. **Active Learning → Submit Label** — writes the corrected mask into the datastore under
   `labels/final/`.
9. Repeat 4–8 on 2–3 images to show the prompt → correct → submit loop.

**Talking points while doing this (design §10):**
- *The annotation feel* — a box prompt returns a mask in ~1 second; contrast with drawing it by
  hand.
- *Pre-label quality, pitched honestly* — this checkpoint is a general echocardiography model, not
  fine-tuned on her leg-muscle images yet. State plainly that day-one quality on her actual data
  will be rougher, and that it improves round-over-round as her corrections accumulate (design §6)
  — do not oversell day-one quality.
- *Measurement → Excel* — after a mask is submitted, run the CSV export (next section) live and
  open the result, pointing at `area_mm2` and `echo_intensity_mean` as columns she'll recognize.

---

## 3. Measurement export (verified, CLI)

```bash
uv run apps/legus/lib/measure/statistics.py \
  --image data/legus/<subject>.nii.gz \
  --mask  data/legus/labels/final/<subject>.nii.gz \
  --subject-id <subject> \
  --out out/<subject>_measurements.csv
```

Produces one CSV row per labeled structure with real physical units (`area_mm2`,
`echo_intensity_mean`, full pyradiomics shape/first-order set) and a `spacing_calibrated` column
that is `False` instead of silently faking a millimetre value whenever the source image's spacing
can't be trusted (see `design.md` §7 and the docstring in
`apps/legus/lib/measure/statistics.py`). Open the resulting CSV in Excel/Numbers for the demo.

---

## 4. If something breaks live

| What breaks | Do this |
|---|---|
| Slicer can't reach the server | Check the terminal `check_demo.sh`/`start_server.sh` is running in — is it still alive? Re-run `curl http://localhost:8000/info/` |
| A box prompt returns an error / empty mask | Retry once (transient MPS op can demote to CPU automatically, see `lib/infers/medsam2.py`); if it persists, restart via `LEGUS_DEVICE=cpu ./scripts/start_server.sh` — the guaranteed-correct reference path |
| Ctrl-C doesn't stop `start_server.sh` (e.g. it was backgrounded/run under a supervisor) | `pkill -f monailabel.main` from another terminal — verified to stop it reliably regardless of how it was launched |
| Anything looks like it's about to touch the network live | It shouldn't — the whole loop is proven offline (§5 below). If in doubt, kill the demo server and restart it; do not try to debug network config live |
| General panic / anything not on this list | Fall back to a **screen recording** of a known-good run (design §8 Path C guidance) rather than improvising live |

---

## 5. The offline claim — what was actually verified, and how strongly

design.md's Path A requires the demo run fully offline. Rather than just asserting that, this was
tested empirically:

- **Code audit**: grepping `apps/legus/` and `scripts/` (excluding `external/`, the vendored
  submodules) for any network-capable call, the only hit is `huggingface_hub.snapshot_download` in
  `scripts/fetch_data.py` — a one-shot, bootstrap-time dataset fetch. Nothing in `main.py`,
  `lib/infers/medsam2.py`, or `lib/measure/statistics.py` makes a network call of any kind.
- **Empirical block**: started the server and ran a full box-prompt REST round trip + measurement
  CSV export with `HTTP_PROXY`/`HTTPS_PROXY` pointed at a closed local port (`127.0.0.1:9`) and
  `HF_HUB_OFFLINE=1` set, with only `localhost`/`127.0.0.1` exempted via `NO_PROXY` (so the
  server's own loopback traffic still works). Confirmed egress was actually blocked first
  (`curl https://huggingface.co` failed with connection refused in the same shell). The server
  booted, and the round trip + CSV export both succeeded, with no warnings/errors/retries in the
  server log and no anomalous slowdown versus a normal run (which would suggest a blocked call
  silently retrying).

**How strong this evidence is:** solid, not absolute. It proves no code path in this stack makes an
HTTP(S) request that respects proxy env vars during boot or inference, and a direct source audit of
our own code found nothing that even tries. It does **not** prove there is no raw-socket call
somewhere in the PyTorch/MONAI Label/SAM2 dependency graph that ignores proxy settings entirely —
that would need either a real firewall rule or a packet capture, neither of which this environment
should do (a firewall change here would cut this session's own connectivity, which the safety rules
this assistant follows explicitly avoid). Given the code audit found nothing and the proxy-blocked
run showed zero anomalies, this is good evidence for the offline claim, not a mathematical proof of
it.

---

## 6. What has NOT been machine-verified

Everything in **Section 2 (Live demo — Slicer click-path) above** is unverified in this session —
there is no headless way to drive 3D Slicer's Qt GUI here. Specifically, nobody has confirmed:
- Loading a study from the MONAI Label panel actually populates Slicer's slice views correctly.
- The box/ROI drawing tool produces a prompt in the format the server expects.
- The returned mask renders as a correct, aligned overlay.
- The Segment Editor correction tools work as expected on a MONAI-Label-sourced segmentation.
- **Submit Label** actually writes to `labels/final/` in the format `check_demo.sh` and
  `lib/measure/statistics.py` expect.

**A human needs to do the pass in Section 2 once before the real demo**, following the exact
click-path given there, ideally against the same server `check_demo.sh` just verified. Everything
verified by this runbook (bootstrap, checkpoint integrity, server start + pre-warm, the REST
round trip, the offline claim, and the measurement CSV) is proven up to the REST/CLI boundary —
the GUI layer on top of it is the one remaining unknown.

---

## Reference

### Compute device

Inference and training pick a device at runtime — `cuda`, `mps`, or `cpu` — and never hardwire
one. Order is CUDA → MPS → CPU; any accelerator failure demotes to CPU rather than erroring at the
annotator. Override with:

```bash
LEGUS_DEVICE=cpu ./scripts/start_server.sh
```

CPU is the guaranteed-correct reference path. MPS is the fast path on this Apple Silicon Mac (a
pre-warmed box prompt runs in well under half a second once the model is loaded — measured ~1–2s
for the very first, cold request after boot vs. ~0.2s for every request after, which is exactly the
gap `start_server.sh`'s pre-warm step is there to hide from the annotator). CUDA is what the
production deployment (design §8, Path C) will use, from this same code.

### Layout

| Path | What |
|---|---|
| `apps/legus/` | Our MONAI Label app — the only code we own |
| `apps/legus/lib/infers/medsam2.py` | The MedSAM2 adapter. The swap point for MedSAM3 (design §9) |
| `apps/legus/lib/trainers/` | Fine-tune `TrainTask` (design §6 step 3) |
| `apps/legus/lib/measure/statistics.py` | Mask → calibrated CSV export (design §7) |
| `external/` | Pinned submodules: MONAI Label, MedSAM2. Never edited; pushes disabled |
| `scripts/bootstrap.sh` | One-command setup: submodules, uv env, checkpoint(s), verification |
| `scripts/fetch_data.py` | Public demo dataset fetch → datastore layout |
| `scripts/start_server.sh` | Starts the server, waits for it, pre-warms it, then blocks (Ctrl-C to stop) |
| `scripts/check_demo.sh` | Preflight — run ~10 min before the demo, see Section 1 |
| `scripts/legus_probe.py` | Shared REST-round-trip helper used by both scripts above |
| `data/`, `apps/legus/model/` | Gitignored. Imaging data and checkpoints never enter git |

### Data

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

### Licenses

Apache-2.0 for MONAI Label and MedSAM2 code, BSD for 3D Slicer, BSD-3 for pyradiomics.
The MedSAM2 *weights* on Hugging Face are **CC-BY-SA-4.0** — share-alike, so derived fine-tuned
checkpoints inherit that obligation.

### Requirements

- macOS (Apple Silicon) or Linux with CUDA
- [`uv`](https://docs.astral.sh/uv/)
- 3D Slicer 5.12+ (`brew install --cask slicer` on macOS)
