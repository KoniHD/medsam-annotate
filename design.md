# AI-Assisted Pediatric Lower-Leg Ultrasound Annotation Tool — Design

> Working design doc. Intended to be dropped into a Claude Code session as `design.md`.
> Status: pre-MVP brainstorm consolidated into a plan. Several items are still open (see §11).

## 1. Purpose

Build a tool that lets a medical researcher (non-technical, tech-averse) annotate a dataset of
pediatric lower-leg ultrasound images with AI assistance, so that most of the manual masking is
replaced by "correct the model's suggestion." Masks then feed a downstream quantitative /
statistical analysis.

Two people, two roles:
- **Builder (me):** dev with HPC/cluster access. Owns setup, hosting, fine-tuning.
- **Annotator (cousin):** domain expert. Clicks/box-prompts, corrects, reviews. **Will abandon the
  tool if it is buggy or breaks.** Reliability > features.

## 2. Goals / Non-goals

**Goals**
- Interactive AI-assisted labeling (prompt → mask → correct) from the very first image.
- A human-in-the-loop loop where the model's *unprompted* pre-labels improve as corrections accumulate.
- Calibrated mask outputs + a per-structure measurement table exportable to CSV/Excel.
- Runs reliably; low cognitive load for the annotator.

**Non-goals (for MVP)**
- Text / concept prompting (park as phase-2 upgrade — see §9).
- Fully automatic, no-human segmentation.
- A custom from-scratch annotation UI (reuse mature OSS instead).
- Any handling of real patient data before data-governance sign-off (see §3).

## 3. Hard constraints

- **Reliability first.** Prefer mature, battle-tested components over new/cutting-edge. This alone
  defers SAM3-family models and argues against fragile bespoke UI.
- **Open source, permissive licenses** for building blocks (see component table in §5). One
  exception flagged there (MedSAM2 *weights*).
- **Production runs on rented cloud GPU** (chosen). MVP/demo may run locally (see §8).
- **GDPR / ethics.** Pediatric patient imaging. EU data residency required; institutional
  data-processing agreement expected before real images leave the clinic. **Demos use public /
  proxy data only.**

## 4. Model choice

**Primary: MedSAM2 (Wang Lab, arXiv:2504.03600).**
- SAM 2.1-based, promptable (box/point), native support for 2D, video (cine), and 3D volumes via
  memory-attention frame propagation.
- Mature ecosystem: HF weights, 3D Slicer plugin, released training scripts, existing ultrasound
  (echo) checkpoint to warm-start from.
- Degrades gracefully across modality — important because our data modality is not yet confirmed.

**Naming caution (do not confuse):**
- `MedSAM2` Wang Lab, arXiv:2504.03600 — **this is the one we use.**
- `Medical SAM 2` Zhu et al., arXiv:2408.00874 — different, older model.
- `SAMed-2` Yan et al., MICCAI 2025, arXiv:2507.03698 — different again.
  Track arXiv IDs, not display names.

**Not chosen (and why):**
- **SAMed-2** — its wins (noise-robust memory, multi-modality continual learning) solve problems we
  don't have (single clean modality). Thinner tooling. Skip.
- **Medical SAM 3 / SAM3** — text-promptable + ultrasound coverage is attractive, but brand-new,
  CUDA/Triton-only, heavier, less stable. Conflicts with reliability-first. Deferred, see §9.

## 5. Architecture

```
                 Cloud GPU (prod)  /  Local Mac (MVP)
   ┌─────────────────────────────────────────────────────┐
   │  MONAI Label server                                  │
   │   ├── interactive model  = MedSAM2 (custom adapter)  │  ← model behind a thin,
   │   ├── active-learning sample selection               │    replaceable interface
   │   ├── fine-tune job (between rounds)                  │
   │   └── datastore (images + masks + prompt metadata)   │
   └──────────────────────────┬──────────────────────────┘
                              │ (network / localhost)
        ┌─────────────────────┴─────────────────────┐
        │  Client (viewer UI the annotator uses)     │
        │   • 3D Slicer  (desktop, lowest-risk)      │
        │   • OHIF       (browser, gentler onboarding)│
        └────────────────────────────────────────────┘
                              │
              masks (NIfTI / DICOM-SEG, calibrated)
                              │
        Segment Statistics / pyradiomics → measurements → CSV / Excel
```

**Design principle:** the served model sits behind a thin internal adapter interface so
MedSAM2 → (later) MedSAM3 is a module/config swap, not a rewrite. Build it replaceable from day one.

**Components & licenses**

| Component | Role | License | Notes |
|---|---|---|---|
| MONAI Label | Server: loop, active learning, training orchestration | Apache-2.0 | Only turnkey option for the integrated loop |
| MedSAM2 (code) | Interactive segmentation model | Apache-2.0 | — |
| MedSAM2 (weights) | Pretrained checkpoint | **Research/education only** | ⚠️ not permissive; fine for academic research. Fallback: fine-tune base SAM2 (Apache-2.0 weights) |
| 3D Slicer | Desktop client | BSD-style (permissive) | Mature; runs great on macOS |
| OHIF Viewer | Browser client | MIT | Needs a DICOMweb backend (adds infra — see §8) |
| pyradiomics | Measurement extraction | BSD-3-Clause | Slicer's Segment Statistics is a lighter built-in alternative |

**DeepEdit note:** MONAI Label ships DeepEdit/DeepGrow as default interactive models. We **do not use
them** — MedSAM2 replaces that role. A small specialist (nnU-Net-style) is only worth training later
if bulk auto-labeling speed/cost becomes an issue.

## 6. The human-in-the-loop loop

1. **Interactive labeling (needs 0 training).** Annotator box/click-prompts; MedSAM2 returns a mask;
   she accepts/corrects. On cine/volume data, prompt one frame → propagate.
2. **Accumulate** corrected masks + prompt metadata in the datastore.
3. **Fine-tune offline** on the GPU between rounds.
   - First useful round: ~50–200 labeled slices (a handful of studies, since one volume/loop = many slices).
   - Subsequent rounds: ~50–100 new corrected samples; front-load early rounds, space out later.
   - Trigger on *correction rate*, not a fixed count.
4. **Improved pre-labels** → she corrects less → repeat until correction rate is acceptably low (<~10–20%).
5. **Hold out** ~10–20 images never used for training; track Dice per round so "is it improving" is objective.

## 7. Outputs

The mask is an intermediate, not the deliverable. Assume the annotator doesn't yet know what's
possible — so compute a sensible **superset** and let her pick what's clinically relevant.

**Primary artifacts**
- Segmentation masks in **NIfTI (.nii.gz)** or **DICOM-SEG**, carrying pixel spacing / calibration.
  (Raw PNG loses physical scale → measurements become non-comparable. Ultrasound depth/zoom varies
  per image, so calibration must be captured per image.)
- QA overlay PNGs (image + mask) for quick visual review.

**Derived per-structure measurements → CSV / Excel** (one row per subject × structure)
- Geometry: cross-sectional area, muscle thickness, subcutaneous fat thickness, (volume if 3D).
- Muscle architecture (if in scope): pennation angle, fascicle length.
- Intensity: **echo intensity** (mean grayscale — a validated muscle-quality biomarker), plus
  first-order intensity stats.
- Full radiomics feature set (shape / first-order / texture) via pyradiomics, optional.

**Downstream use.** For a "which measurements are predictive" question, feed the measurement table
into an interpretable model (logistic regression / random forest / gradient boosting) and read
feature importance. This is why CSV/Excel is the right interchange — the annotator's instinct was
correct; it just holds *measurements*, not masks.

## 8. Hosting: MVP vs production, with cost

### MVP / demo — reliability-first
Two viable paths. Recommendation: **local Mac (Path A) for the live demo**, because it removes
network, cold-start, and GPU-boot as failure modes — directly serving the "must not break" constraint.

**Path A — Local Mac (recommended for the live demo). Cost: $0.**
- macOS host (M-series), MedSAM2 **small/base** checkpoint.
- Run inference on **CPU** (Efficient MedSAM2 supports CPU) to avoid Apple MPS rough edges entirely;
  latency of a few seconds/image is fine for a demo. (MPS possible with `PYTORCH_ENABLE_MPS_FALLBACK=1`,
  but CPU is the zero-surprise choice.)
- Client: **3D Slicer** (single mature desktop app, offline, no DICOM server needed) — lowest number
  of moving parts.
- Fully offline, fully under builder's control.

**Path B — Local Mac + browser client. Cost: $0, but more fragile.**
- MONAI Label (CPU) + OHIF locally. Adds a DICOMweb backend (e.g., Orthanc — **GPLv3, copyleft**) and
  more services that can break live. Only if a browser experience is essential for the demo.

**Path C — Cloud (== production, but live-network risk at demo time).**
- Pre-warm and pre-load before the session; show as the "this is production" story, ideally via a
  short screen recording rather than live, to avoid cold-start/network risk in front of a tech-averse user.

### Production — rented cloud GPU (chosen)
- MedSAM2 inference is tiny; an RTX 4090 (24 GB) or L4/L40S is more than enough. No H100 needed.
- **Provider fit:** RunPod — EU regions (GDPR), 99% uptime Secure tier, per-second/serverless billing,
  ~$10 signup credit. (Vast.ai is cheaper but no SLA / no EU residency guarantee — unsuitable for
  patient data. Lambda is US-only.)

**Cost estimate (July 2026 rates, RunPod, RTX 4090):**
| Item | Rate | Est. |
|---|---|---|
| Secure Cloud RTX 4090 | ~$0.69/hr | reliable tier |
| Community/On-demand RTX 4090 | ~$0.34/hr | cheaper, interruptible |
| L40S (headroom for fine-tune) | ~$0.79/hr | optional |
| **MVP build+demo (~20 GPU-hrs over a week)** | — | **~$7–14, largely offset by the ~$10 signup credit → effectively ~$0–5** |
| Single pre-warmed demo session (~3 hrs) | ~$0.69/hr | **~$2** |

**Free-trial answer:** RunPod's ~$10 signup credit covers essentially the whole MVP. HF
Spaces / ZeroGPU exist but are flaky (dismissed). Colab free/Pro can host a MedSAM2 notebook but not a
clean persistent server+browser demo.

## 9. Upgrade path to MedSAM3 (Medical SAM 3) — difficulty

**Overall: low-to-moderate and *contained*, thanks to the MONAI Label seam.** The client (Slicer/OHIF),
the loop, the datastore, and the measurement pipeline **do not change**. The swap is isolated to the
model-serving adapter (§5 design principle).

What the upgrade actually costs:
1. **New inference adapter** for SAM3's different architecture + prompt interface (text/concept +
   exemplar, vs MedSAM2's box). Backend-only. *Low effort.*
2. **Heavier serving** — SAM3 ≈ 848M params, ~10–12 GB VRAM, CUDA/Triton-only. Fine on the cloud GPU,
   **not** on the Mac. Medical SAM3 weights ship as a ~10 GB full-finetune checkpoint (bundled
   optimizer state) → strip to inference weights (~1.7 GB in fp16). *Low effort, cloud only.*
3. **Text-prompt UI affordance** — if we want the actual text-prompting UX, the client needs a text
   input, which Slicer/OHIF don't provide by default. *This is the fiddliest part.* (A ready-made
   SAM3 text-prompt UX exists elsewhere in the OSS ecosystem, but adopting it trades away our
   medical-format + measurement strengths — not worth it. Prefer adding a minimal text field to the
   existing client.)
4. **Stability risk** — brand-new model; conflicts with reliability-first. Keep behind a feature flag
   as a phase-2 experiment, validated on a handful of her images before any rollout.

**Net:** building the model layer as a replaceable adapter now makes MedSAM3 a swap-plus-optional-UI
change later, not a re-architecture.

## 10. MVP / demo plan (build an MVP, not a slideshow)

Deliver a *working, minimal instance of the real stack* on the reliability-first Path A, on public
ultrasound data. The demo doubles as a discovery session (see §11).

Show three things, mapped to the annotator's stated priorities:
1. **The annotation feel** — live prompt → instant mask; on a clip, one-frame → propagation. Goal:
   she feels how much faster prompting is than drawing.
2. **Auto pre-label quality — pitched honestly.** On out-of-distribution leg ultrasound the base
   model's first pre-labels may be rough. Demo pre-labeling on a domain where MedSAM2 already works to
   convey the *concept and trajectory*, and state plainly that on her data it starts rougher and
   climbs with each round of her corrections. Do **not** oversell day-one auto quality — that's the one
   way this backfires with a tech-averse user.
3. **Measurement → Excel** — a mask flowing through Segment Statistics into a CSV with real units and
   a couple of columns she recognizes. This is the moment she connects "annotation tool" to "my
   research output."

Reliability rules for the demo: pre-warm/pre-load everything; run offline (Path A); no live cold
starts; have a fallback recording of any step that touches the network.

## 11. Open questions (bring to the demo)

- **Downstream research question** — what does she ultimately want to *predict or measure*?
  (Determines which measurement columns matter.)
- **Which structures** in the lower leg? (Muscles? bone surface? vessels? a specific pathology?)
- **The dataset** — roughly how many studies, and in what format off the machine (DICOM vs exported PNG)?
- **Modality (still unresolved)** — 2D stills vs cine loops vs 3D volumes. Resolve technically:
  inspect 2–3 real sample files for single- vs multi-frame DICOM and for the
  `SequenceOfUltrasoundRegions` calibration tag (needed for physical-unit measurements).

## 12. Build tasks (for Claude Code)

- [ ] Stand up MONAI Label locally (CPU) on macOS; confirm it serves and a client connects.
- [ ] Write the **MedSAM2 inference adapter** as a MONAI Label custom app, behind a thin interface
      (`segment(image, prompt) -> mask`) so the model is swappable.
- [ ] Wire **3D Slicer** as the client; verify interactive box-prompt → mask round-trip end to end.
- [ ] Implement the datastore convention: image + mask (NIfTI/DICOM-SEG) + prompt metadata JSON.
- [ ] Implement the **measurement export**: Segment Statistics / pyradiomics → CSV with units + captured
      pixel spacing.
- [ ] Wire the **fine-tune job** (MedSAM2 training script) + a held-out validation set with per-round Dice.
- [ ] Package a reliable **Path A demo**: preloaded public ultrasound sample, offline, one-command start.
- [ ] (Phase 2) Feature-flagged **MedSAM3 adapter** + minimal text-prompt UI; validate on sample images.
