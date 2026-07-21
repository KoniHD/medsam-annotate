# Copyright (c) 2026 MedSAM annotation tool contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""MedSAM2 adapter -- THE model-serving seam described in design.md Sec 5/9.

design.md Sec 5: "the served model sits behind a thin internal adapter interface so
MedSAM2 -> (later) MedSAM3 is a module/config swap, not a rewrite."

This module is the *only* place in the app that knows MedSAM2-specific facts: the checkpoint
filename, the packaged sam2.1 hydra config string, that loading must stay fully offline
(design.md Sec 8 Path A), and how to pick cuda/mps/cpu. main.py imports nothing from here but
the class name -- it never sees a checkpoint path, a config string, or a device name.

We subclass `monailabel.sam2.infer.Sam2InferTask` and reuse its `run2d` / `run_3d` / `Writer`
machinery completely untouched. We only override:

  * `__init__`   -- load a local checkpoint instead of downloading one, and skip the upstream
                    `hydra.initialize_config_dir(model_dir)` call (wrong for us: our config
                    string is resolved by the sam2 package's own packaged hydra search path,
                    not a directory of our own -- see established facts below).
  * `__call__`   -- two layers of accelerator-failure demotion, because a cuda/mps device can
                    fail at two different times:
                      1. *Build* time (device string is bogus, or the backend can't even
                         allocate/place the model). Handled by pre-populating
                         `self.predictors[device]` via `_ensure_predictor` before delegating,
                         so `run2d`/`run_3d` (which do `predictor = self.predictors.get(device)`
                         and only build+cache when that's `None`) always take the
                         "already built" branch for a demoted device.
                      2. *Call* time -- the predictor built fine, but the actual op inside
                         `predictor.predict()` / `add_new_points_or_box()` /
                         `propagate_in_video()` isn't implemented for the accelerator (the
                         realistic MPS failure mode: `NotImplementedError: The operator
                         'aten::...' is not currently implemented for the MPS device`), or the
                         accelerator OOMs. This can only be caught around the delegation itself,
                         so `__call__` wraps `super().__call__(...)` in a try/except and, on any
                         exception while not already on cpu, logs a warning and retries once on
                         cpu (rebuilding `self.inference_state` for 3D, since it's bound to
                         whichever predictor created it).
                    Belt-and-braces: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set (unless already set)
                    at import time so ops MPS hasn't implemented fall back to CPU transparently
                    -- see design.md Sec 8 Path A. That reduces how often layer 2 fires; it does
                    not replace it (fallback ops can still be slow, or MPS can still OOM).

Established facts this module relies on (see task brief / design.md Sec 8):
  * `build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", ckpt, device=...)` loads MedSAM2_latest.pt
    STRICTLY under upstream SAM 2.1 with no state-dict remapping. That config string is resolved
    by the `sam2` package's own hydra config module -- we must not call
    `hydra.initialize_config_dir` for it.
  * `monailabel.utils.others.generic.name_to_device` already maps "mps" -> "mps" and demotes
    "cuda*" -> "cpu" when CUDA is absent, so device selection flows through the MONAI Label
    request as normal; our own job is only to *offer* the right device list.
  * `monailabel.utils.others.generic.device_list()` is CUDA-only (`['cpu']` on a Mac); it does
    not know MPS exists, so we supply our own list via `available_devices()`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

# Belt-and-braces (design.md Sec 8 Path A): let MPS ops PyTorch hasn't implemented for the `mps`
# backend fall back to CPU transparently instead of raising `NotImplementedError`. Must be set
# before torch is imported/used. This does NOT replace the call-time demotion in
# `MedSAM2InferTask.__call__` below -- a fallback op can still be slow, and MPS can still OOM --
# it only shrinks how often that demotion needs to fire. `setdefault` so an operator/env override
# always wins.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import SimpleITK as sitk
import torch
from monai.transforms import LoadImaged
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.sam2.infer import Sam2InferTask
from monailabel.transform.writer import Writer
from monailabel.utils.others.generic import name_to_device
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

logger = logging.getLogger(__name__)

# Packaged sam2.1 hydra config resolved by the `sam2` package itself -- NOT a path on disk.
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_CHECKPOINT_FILENAME = "MedSAM2_latest.pt"

# Where `lib.trainers.medsam2.MedSAM2TrainTask` leaves the most recently completed fine-tune
# round (design.md Sec 6 step 4: "improved pre-labels -> she corrects less -> repeat"). Defined
# here (not in the trainer module) because this module owns checkpoint *resolution* for
# serving; the trainer imports these two names back rather than duplicating the path convention.
FINETUNED_DIRNAME = "finetuned"
FINETUNED_CHECKPOINT_FILENAME = "latest.pt"

ENV_CHECKPOINT = "LEGUS_MEDSAM2_CHECKPOINT"
ENV_CONFIG = "LEGUS_MEDSAM2_CONFIG"
ENV_DEVICE = "LEGUS_DEVICE"
ENV_LABELS = "LEGUS_LABELS"

# Structures the annotator can prompt, one named segment each. SAM2 is class-agnostic -- it
# segments whatever the box/points enclose -- so these names only decide which segment the mask
# lands in (and therefore which row it becomes in the measurement CSV, one per structure). They
# must be *non-empty* though: the MONAI Label Slicer plugin auto-creates a segment per advertised
# label when a study loads and refuses to run an interactive model with no label selected, so
# `labels=None` is exactly why a study opened with no segments and nothing to pick.
#
# These are demo placeholders for the lower-leg use case; the real structure list is an open
# discovery question for the demo (DEMO.md question A2). Override without touching code via
# LEGUS_LABELS="muscle,fat,bone" (comma-separated).
DEFAULT_LABELS = ["muscle", "subcutaneous_fat", "bone_surface"]


def _resolve_labels(labels: Any | None) -> Any:
    """An explicit `labels=` wins; else LEGUS_LABELS; else DEFAULT_LABELS. Never None/empty."""
    if labels:
        return labels
    env = os.environ.get(ENV_LABELS)
    if env:
        return [name.strip() for name in env.split(",") if name.strip()]
    return list(DEFAULT_LABELS)


def available_devices() -> list[str]:
    """Best-first list of usable torch devices: CUDA GPU(s), then MPS, then CPU (always last).

    `LEGUS_DEVICE` overrides the default by pinning it to the front of the list -- it does not
    have to already be one of the discovered devices (an unrecognised/unavailable pin will
    simply fail fast and get demoted to cpu at predictor-build time, see
    `MedSAM2InferTask._ensure_predictor`).

    This list is fed into `_config["device"]`, which is what MONAI Label / the Slicer plugin use
    to populate the device chooser.
    """
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")

    pinned = os.environ.get(ENV_DEVICE)
    if pinned:
        if pinned in devices:
            devices.remove(pinned)
        else:
            logger.warning(
                f"{ENV_DEVICE}={pinned!r} is not among the devices detected on this machine "
                f"({devices}); pinning it as the default anyway -- it will demote to cpu if it "
                "fails to build."
            )
        devices.insert(0, pinned)
    return devices


def _resolve_checkpoint(model_dir: str) -> str:
    """Resolve the MedSAM2 checkpoint path without ever downloading anything.

    Order:
      1. `LEGUS_MEDSAM2_CHECKPOINT` env var -- an explicit pin always wins (e.g. for pointing the
         demo at a known-good checkpoint regardless of what training has produced since).
      2. `<model_dir>/finetuned/latest.pt` -- the most recently completed `/train/medsam2` round
         (`lib.trainers.medsam2._save_checkpoint` writes this). This is what closes the
         human-in-the-loop (design.md Sec 6 step 4, "improved pre-labels -> she corrects less ->
         repeat"): without it, nothing ever consumed a fine-tuned checkpoint and every round's
         work was served to nobody.
      3. `<model_dir>/MedSAM2_latest.pt` -- the static pretrained checkpoint, the day-zero
         default before any fine-tune round has completed.

    The demo must work fully offline (design.md Sec 8 Path A), so a missing file at every step is
    a loud, immediate error instead of a network fetch.
    """
    override = os.environ.get(ENV_CHECKPOINT)
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(
                f"{ENV_CHECKPOINT}={override!r} does not exist. This adapter never downloads a "
                "checkpoint -- fix the path or unset the override."
            )
        return override

    finetuned = os.path.join(model_dir, FINETUNED_DIRNAME, FINETUNED_CHECKPOINT_FILENAME)
    if os.path.isfile(finetuned):
        logger.info(
            f"MedSAM2: serving {finetuned!r} -- the latest completed /train/medsam2 fine-tune "
            f"round -- instead of the static {DEFAULT_CHECKPOINT_FILENAME} (design.md Sec 6 "
            "step 4). Set LEGUS_MEDSAM2_CHECKPOINT to override."
        )
        return finetuned

    path = os.path.join(model_dir, DEFAULT_CHECKPOINT_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"MedSAM2 checkpoint not found at {path!r}. Set {ENV_CHECKPOINT} or place "
            f"{DEFAULT_CHECKPOINT_FILENAME} under the app's model directory. This adapter never "
            "downloads a checkpoint -- the demo runs fully offline (design.md Sec 8 Path A)."
        )
    return path


class MedSAM2InferTask(Sam2InferTask):
    """MONAI Label interactive-segmentation task backed by MedSAM2 (Wang Lab, SAM 2.1-based).

    Everything MedSAM2-specific (checkpoint resolution, config string, device fallback) lives
    here and nowhere else. A future MedSAM3 adapter is a sibling module implementing the same
    `segment(image, prompt) -> mask` seam (see below); `main.py` would only need to swap which
    class it instantiates.
    """

    def __init__(
        self,
        model_dir: str,
        type: InferType = InferType.DEEPGROW,
        dimension: int = 2,
        labels: Any | None = None,
        config: dict[str, Any] | None = None,
    ):
        checkpoint = _resolve_checkpoint(model_dir)
        config_path = os.environ.get(ENV_CONFIG, DEFAULT_CONFIG)
        devices = available_devices()
        labels = _resolve_labels(labels)

        # Deliberately call InferTask.__init__ (the grandparent), NOT Sam2InferTask.__init__:
        # the latter downloads settings.MONAI_SAM_MODEL_PT/_CFG and calls
        # hydra.initialize_config_dir(model_dir) -- neither applies to us. Our checkpoint is
        # already on disk (see _resolve_checkpoint) and our config string is resolved by the
        # sam2 package's own packaged hydra search path, not a directory we own.
        InferTask.__init__(
            self,
            type=type,
            dimension=dimension,
            labels=labels,
            description="MedSAM2 (interactive box/point prompt segmentation)",
            config={"device": devices, "reset_state": False, "largest_cc": False, "pylab": False},
        )
        if config:
            self._config.update(config)

        self.additional_info = None
        self.image_loader = LoadImaged(keys="image")
        self.post_trans = None
        self.writer = Writer(ref_image="image")

        self.path = checkpoint
        self.config_path = config_path

        self.predictors: dict[str, Any] = {}
        self.image_cache: dict[str, Any] = {}
        self.inference_state = None
        self._demoted: set[str] = set()  # devices that failed to build and now resolve to cpu

        self.default_device = devices[0]
        logger.info(
            f"MedSAM2InferTask[{type}/{dimension}d]: checkpoint={self.path} "
            f"config={self.config_path} devices={devices}"
        )

    # -- accelerator fallback: cuda/mps failure demotes to cpu, never raises to the user -----

    def _build_predictor(self, device: str):
        """Construct one fresh predictor on `device`. A thin wrapper around sam2's own
        builders -- no SAM2 internals are reimplemented here."""
        self._apply_cuda_tuning(device)
        if self.dimension == 2:
            model = build_sam2(self.config_path, self.path, device=device)
            return SAM2ImagePredictor(model)
        return build_sam2_video_predictor(self.config_path, self.path, device=device)

    @staticmethod
    def _apply_cuda_tuning(device: str) -> None:
        """Re-apply the CUDA bf16-autocast + TF32 setup that upstream does at predictor-build time.

        Upstream `Sam2InferTask.run2d`/`run_3d` enable these *inside* their `if predictor is None:`
        branch. Because `_ensure_predictor` pre-populates `self.predictors[device]`, that branch
        never runs, so without this the adapter would silently execute fp32 with TF32 off on a
        CUDA box -- correct results, but slower and more VRAM than upstream intends. That is
        design.md Sec 8 Path C, the one path with no hardware here to test on, so it must not
        depend on an inherited side effect we bypassed.
        """
        if not device.startswith("cuda") or not torch.cuda.is_available():
            return
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _ensure_predictor(self, device: str) -> str:
        """Guarantee `self.predictors[device]` is populated, demoting to cpu on failure.

        The inherited `run2d`/`run_3d` do `predictor = self.predictors.get(device)` and only
        build (+cache) when that's `None`. Pre-populating the cache here for the *requested*
        key means the inherited code always takes the "already cached" branch and never itself
        touches a broken accelerator. Returns the device actually usable, which may differ from
        `device` after a demotion.
        """
        # A device that already demoted stays demoted -- otherwise the second call would find the
        # cpu predictor cached under the *failed* key, take the early return below, and report the
        # broken device as usable, contradicting this method's contract.
        if device in self._demoted:
            return "cpu"
        if device in self.predictors:
            return device
        try:
            self.predictors[device] = self._build_predictor(device)
            return device
        except Exception as exc:  # accelerator failure must demote, never raise to the user
            logger.warning(
                f"MedSAM2: failed to build predictor on device={device!r} ({exc}); demoting to cpu"
            )
            cpu_predictor = self.predictors.get("cpu")
            if cpu_predictor is None:
                cpu_predictor = self._build_predictor("cpu")
                self.predictors["cpu"] = cpu_predictor
            self.predictors[device] = cpu_predictor  # cache under the requested key too
            self._demoted.add(device)
            return "cpu"

    def __call__(self, request, debug=False):
        request = self._ensure_slice_hint(request)
        device = name_to_device(request.get("device", self.default_device))
        actual = self._ensure_predictor(device)
        if actual != request.get("device"):
            request = dict(request)
            request["device"] = actual

        try:
            return super().__call__(request, debug=debug)
        except Exception as exc:
            # Layer 2 of the accelerator-fallback guard: `_ensure_predictor` above only catches
            # *build*-time failures. Real accelerator failures (MPS `NotImplementedError` on an
            # unimplemented `aten::` op, MPS/CUDA OOM) happen *inside* `run2d`/`run_3d`'s call
            # into `predictor.predict()` / `add_new_points_or_box()` / `propagate_in_video()`,
            # which only this try/except around the delegation itself can see. Demote to cpu and
            # retry exactly once; if cpu itself fails, let it raise -- there is nowhere left to
            # demote to.
            if actual == "cpu":
                raise
            logger.warning(
                f"MedSAM2: inference failed on device={actual!r} ({exc!r}); "
                "demoting to cpu and retrying once"
            )
            cpu_device = self._ensure_predictor("cpu")
            request = dict(request)
            request["device"] = cpu_device
            if self.dimension == 3:
                # `self.inference_state` (if any) was built against the failed device's video
                # predictor and its tensors live on that device. Drop it and force run_3d to
                # build a fresh one on the cpu predictor instead of calling
                # `predictor.reset_state(...)` across predictors/devices.
                self.inference_state = None
                request["reset_state"] = True
            return super().__call__(request, debug=debug)

    # -- the MedSAM3 swap point ----------------------------------------------------------------

    def segment(self, image: str, prompt: dict[str, Any] | None = None) -> np.ndarray:
        """THE seam design.md Sec 5/9 calls out: box/point prompt -> mask array.

        A future MedSAM3 adapter implements this exact `segment(image, prompt) -> mask`
        signature; nothing above this method (main.py, the client, the datastore convention)
        needs to change for that swap.

        This is a wrapper, not a reimplementation: it builds a MONAI Label request dict and
        delegates to the inherited `__call__` (-> `run2d`/`run_3d` -> `Writer`), asking the
        `Writer` to hand back an in-memory array (`result_write_to_file: False`) instead of a
        temp file on disk.

        Args:
            image: path to an image/volume readable by MONAI's `LoadImaged` (nii.gz, jpg, ...).
            prompt: dict with any of:
                - "box" / "roi": `[r1, c1, r2, c2]` (2D) or `[r1, c1, r2, c2, z1, z2]` (a 3D
                  volume read through a 2D task -- the z-range picks the prompted slice).
                - "points" / "foreground": list of `[row, col]` (+ slice for 3D) positive points.
                - "background": list of `[row, col]` (+ slice) negative points.
                - "slice": explicit slice index override.
                - "device": device string overriding this task's default.

        Returns:
            A numpy mask array.
        """
        prompt = prompt or {}
        request: dict[str, Any] = {
            "image": image,
            "foreground": prompt.get("points", prompt.get("foreground", [])),
            "background": prompt.get("background", []),
            "roi": prompt.get("box", prompt.get("roi", [])),
            "device": prompt.get("device", self.default_device),
            "cache_image": prompt.get("cache_image", False),
            "result_write_to_file": False,  # ask the Writer for an array, not a temp file
        }
        if "slice" in prompt:
            request["slice"] = prompt["slice"]
        # No slice hint is handled centrally in `__call__` (see `_ensure_slice_hint`), so the
        # REST path gets the same protection this seam does.

        mask, _result_json = self(request)
        return np.asarray(mask)

    def _ensure_slice_hint(self, request: dict[str, Any]) -> dict[str, Any]:
        """For a 2D task on a volume, pin a valid slice index -- defaulting to 0 when none is
        given, and *clamping* one that is out of range.

        Two distinct failures this prevents, both HTTP 500s in the annotator's face (design.md
        Sec 1/3), both reproduced over REST on the real CAMUS demo data:

          * No slice hint -> upstream `run2d` settles on `slice_idx = -1` and takes its whole-image
            branch, which (unlike the slice branch) never does `.convert("RGB")`; a single-channel
            frame then dies in torchvision Normalize ("[1,1024,1024] vs [3,1024,1024]").
          * Out-of-range slice -> the 3D Slicer plugin always sends `slice = <slice-view offset>`
            for a 2D model, and on our singleton-z (D=1) frames that offset is frequently >= 1, so
            `image_tensor[:, :, slice_idx]` raises `IndexError: index N is out of bounds for
            dimension 2 with size 1`. Clamping to the volume's real z-extent fixes it for our D=1
            data and stays correct for a genuine multi-slice volume served through a 2D task.

        Only touches a real z-axis image, so a genuine 2D file keeps upstream's behaviour. The
        header peek reads dimensions only, no pixels.
        """
        if self.dimension != 2:
            return request
        image = request.get("image")
        if not isinstance(image, str) or not os.path.isfile(image):
            return request
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(image)
            reader.ReadImageInformation()
            if reader.GetDimension() < 3:
                return request
            n_slices = int(reader.GetSize()[2])
        except Exception:  # unreadable header: leave upstream's behaviour untouched
            return request

        requested = request.get("slice")
        if requested is None or requested < 0:
            new_slice = 0
        else:
            new_slice = min(int(requested), n_slices - 1)  # clamp into [0, n_slices-1]

        if new_slice != requested:
            if requested is not None and requested >= n_slices:
                logger.info(
                    f"MedSAM2: requested slice {requested} is out of range for a {n_slices}-slice "
                    f"volume; clamping to {new_slice}."
                )
            request = dict(request)
            request["slice"] = new_slice
        return request


# --- automatic (unprompted) pre-labelling variant -------------------------------------------

ENV_AUTO_BOX = "LEGUS_AUTO_BOX_FRACTION"
DEFAULT_AUTO_BOX_FRACTION = 1.0


class MedSAM2AutoInferTask(MedSAM2InferTask):
    """MedSAM2 as an *automatic* pre-labeller: no human prompt, one mask per image.

    Why this exists, and its honest limits (design.md Sec 6 "unprompted pre-labels", Sec 10):
    SAM2/MedSAM2 is a *promptable* model -- it segments whatever a box/points enclose and cannot
    segment an image with no prompt at all. "Automatic" here therefore means *supplying a derived
    prompt* (a default bounding box) rather than a human one, which is exactly how MedSAM2's own
    auto pipelines work (they simulate a box on a slice). The output quality is only as good as
    (a) that default-box heuristic and (b) how well the weights already know the target structure,
    so on out-of-distribution leg ultrasound the day-one pre-label is rough and improves as the
    fine-tune loop (lib/trainers/medsam2.py) teaches the model the structure -- do not oversell
    day-one auto quality (design.md Sec 10 item 2).

    This is a thin wrapper: it injects a whole-image (or `LEGUS_AUTO_BOX_FRACTION`-sized, centred)
    ROI when the request carries no prompt of its own, then defers entirely to the interactive
    path. Registered as `InferType.SEGMENTATION` so the client's Auto-Segmentation affordance
    (Slicer's "Auto Segmentation" section / "Run") targets it, complementing the interactive
    `medsam2_2d`/`medsam2_3d` tasks rather than replacing them.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("type", InferType.SEGMENTATION)
        super().__init__(*args, **kwargs)
        frac = os.environ.get(ENV_AUTO_BOX)
        try:
            self.auto_box_fraction = float(frac) if frac else DEFAULT_AUTO_BOX_FRACTION
        except ValueError:
            logger.warning(f"{ENV_AUTO_BOX}={frac!r} is not a number; using {DEFAULT_AUTO_BOX_FRACTION}")
            self.auto_box_fraction = DEFAULT_AUTO_BOX_FRACTION
        self.auto_box_fraction = min(max(self.auto_box_fraction, 0.05), 1.0)

    def __call__(self, request, debug=False):
        request = self._ensure_default_box(request)
        return super().__call__(request, debug=debug)

    def _ensure_default_box(self, request: dict[str, Any]) -> dict[str, Any]:
        """Inject a centred default ROI covering `auto_box_fraction` of the frame, when the caller
        supplied no prompt (no roi and no points). Header peek reads dimensions only, no pixels."""
        has_prompt = (request.get("roi") or request.get("foreground") or request.get("background"))
        if has_prompt:
            return request
        image = request.get("image")
        if not isinstance(image, str) or not os.path.isfile(image):
            return request
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(image)
            reader.ReadImageInformation()
            size = reader.GetSize()  # (W, H[, D]) in sitk index order
        except Exception:
            return request
        width, height = int(size[0]), int(size[1])
        margin = (1.0 - self.auto_box_fraction) / 2.0
        r0, r1 = round(height * margin), round(height * (1.0 - margin))
        c0, c1 = round(width * margin), round(width * (1.0 - margin))
        request = dict(request)
        request["roi"] = [r0, c0, r1, c1]  # run2d expects [row0, col0, row1, col1]
        logger.info(
            f"MedSAM2 auto: no prompt supplied; using default box {request['roi']} "
            f"({self.auto_box_fraction:.0%} of frame). Pre-label quality improves with fine-tuning."
        )
        return request
