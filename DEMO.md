# Demo — quick start & discovery questions

This file is for the **demo with your cousin**. Top half: how to get it running. Bottom half: the
questions the demo needs to answer so we know what to build next. The demo is as much a discovery
session as a show-and-tell — the answers below decide the whole next phase.

---

## Start it (do this ~10 minutes before she sits down)

```bash
# One-time on a fresh machine:
./scripts/bootstrap.sh                    # env + submodules + model checkpoints
uv run scripts/fetch_data.py --limit 8    # public CAMUS demo images

# Every demo — the single command that matters:
./scripts/check_demo.sh
```

`check_demo.sh` checks six things and **leaves a pre-warmed server running** if they pass. Do not
proceed until it prints `ALL CHECKS PASSED` and exits 0. Then:

1. Open **3D Slicer** → **MONAI Label** module.
2. Server URL `http://localhost:8000` → **Fetch Info** (the ⟳ button). The named segments
   (`muscle`, `subcutaneous_fat`, `bone_surface`) now load automatically with each study.
3. **Active Learning → Next Sample** → an unlabeled image loads.
4. **Lead with interactive** (day-one quality is good). Open the **SmartEdit** section:
   - **Model** → `medsam2_2d`.
   - **Label** → pick the structure you're about to outline (e.g. `muscle`).
   - Click the **ROI/BBOX Prompt** place button, then drag a box around that structure in a
     slice view. (Optionally add foreground/background points with their place buttons.)
   - Click **Update** — *this is the interactive run button* — a mask appears in ~1s.
5. **Then show automatic pre-labelling** (`medsam2`, in the **Auto Segmentation** section →
   **Run**; it also auto-runs when you load a Next Sample). Frame it honestly: this needs no
   clicks, but day-one on leg data it's *rough* and sharpens as her corrections fine-tune the
   model — the value is the trajectory, not the first mask (design §10).
6. Correct with the **Segment Editor** (Paint/Erase) if needed → **Active Learning → Submit Label**.
7. Show the measurement CSV: `README.md` §3 has the exact command.

> **Two modes, on purpose.** MedSAM2 is *promptable*: interactive (your box → mask, in SmartEdit,
> run button labelled **Update**) is its native, day-one-good mode. Automatic (`medsam2`, Auto
> Segmentation section) supplies a synthetic default box so it can pre-label with no prompt — the
> design §6 loop — but it's rough until fine-tuned. Lead with interactive; use auto to tell the
> "it gets better" story.

If anything breaks live, `README.md` §4 is the fallback table. The reliable fallback for a flaky
prompt is `LEGUS_DEVICE=cpu ./scripts/start_server.sh` — the guaranteed-correct path.

**After a practice run, reset before the real thing:**

```bash
./scripts/reset_demo.sh     # stops the server + clears the masks you submitted
```

This returns every image to "unlabeled" so `Next Sample` has work again. It keeps the images and
the ground truth; only your submitted annotations are cleared. Use `--stop-only` to just shut the
server down without touching annotations.

> ⚠️ **Do one full click-through yourself before she watches.** The server, the masks, and the CSV
> are all machine-verified — but the Slicer GUI steps (3–5 above) are the one part no automated test
> could reach. Rehearse them once.

**How to pitch it honestly (design §10):** the current model is a general echocardiography model, not
yet trained on her leg-muscle images. Say plainly that day-one quality on *her* data starts rougher
and climbs with each round of her corrections. Do not oversell day-one auto quality — that's the one
way this backfires with a tech-averse user.

---

## Questions the demo must answer

Grouped by what they unblock. Each notes **why it matters** and **what it changes** in the build.

### A. The research goal (unblocks: which measurements to compute)
1. **What does she ultimately want to predict or measure?** (e.g. distinguish healthy vs affected
   muscle, track change over time, correlate a measurement with an outcome.)
   *Changes:* which columns in the measurement CSV actually matter, and whether we wire the table
   into an interpretable model (logistic regression / random forest) for feature importance.
2. **Which structures in the lower leg?** Muscles? Individual muscles or the whole compartment?
   Bone surface? Vessels? Subcutaneous fat? A specific pathology?
   *Changes:* the label set, and therefore what she prompts and what we fine-tune toward.
3. **Which measurements are clinically meaningful to her?** Cross-sectional area, muscle thickness,
   fat thickness, echo intensity (muscle-quality biomarker), pennation angle / fascicle length?
   *Changes:* whether the pyradiomics wrapper is enough or we add muscle-architecture measures.

### B. The data (unblocks: the ingest path and whether measurements are in real units)
4. **How many studies, and in what format off the ultrasound machine?** DICOM straight from the
   scanner, or exported PNG/JPG?
   *Changes:* the fetch/convert pipeline. **Critical:** exported PNG loses physical scale, so
   measurements can't be in mm. If it's PNG-only we need per-image calibration some other way.
5. **Modality: 2D stills, cine loops, or 3D volumes?** (Resolve technically too — inspect 2–3 real
   files for single- vs multi-frame DICOM and the `SequenceOfUltrasoundRegions` calibration tag.)
   *Changes:* 2D uses the box→mask path as-is; cine/3D uses MedSAM2's frame-propagation (prompt one
   frame → propagate) — worth confirming propagation feels good on her actual clip lengths.
6. **Do her images carry pixel-spacing calibration** (the `SequenceOfUltrasoundRegions` tag)?
   *Changes:* without it, the CSV's `spacing_calibrated` column will read `False` and numbers stay
   in pixels, not mm. We'd need to capture depth/zoom per image to get comparable measurements.

### C. Constraints that shape hosting & licensing (unblocks: Path B/C decisions)
7. **Is 3D Slicer (a desktop app) acceptable for her, or does she need a browser?**
   *Changes:* desktop = stay on Path A; browser = build Path B (OHIF + a DICOMweb backend).
8. **Where will the real (patient) data live, and is the institutional data-processing agreement in
   place?** Pediatric imaging, GDPR/EU residency (design §3).
   *Changes:* real data cannot touch any machine — including this Mac — until that's signed. It also
   decides when we move to the EU cloud GPU (Path C).
9. **Does the fine-tuned model need to stay closed/proprietary?**
   *Changes:* MedSAM2's weights are **CC-BY-SA-4.0 (share-alike)**, so anything fine-tuned from them
   inherits that. If it must stay closed, we switch the warm start to base SAM2 (Apache-2.0) — a
   decision that's cheap now and expensive after training rounds accumulate.

### D. The loop (unblocks: how aggressively to schedule fine-tuning)
10. **Roughly how many images can she realistically correct per round, and how often?**
    *Changes:* the fine-tune cadence. Design §6 expects a first useful round at ~50–200 labeled
    slices; we trigger on *correction rate*, not a fixed count. Her throughput sets the pace.

**One thing to watch during the demo, not ask:** does the box-prompt-then-correct flow feel *faster*
to her than drawing masks by hand? That felt speed-up is the whole value proposition — if it doesn't
land, that's the most important signal of the session.
