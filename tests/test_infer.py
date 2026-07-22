# Copyright (c) 2026 MedSAM annotation tool contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for the MedSAM2 adapter (design.md Sec 5/9 -- the model-serving seam).

The tests that need the real 149MB checkpoint skip cleanly when it is absent, so a fresh clone
can still run the suite. The device-selection and offline-guard tests need no checkpoint at all
and always run.

The box-prompt test is parametrised over *every* device the host advertises. That is the point:
design.md Sec 8 wants the same code on CPU (reference), MPS (fast path here) and CUDA (Path C),
so "it works on the one device I happened to try" is not good enough.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "apps" / "legus"))

from lib.infers.medsam2 import (  # noqa: E402
    DEFAULT_CHECKPOINT_FILENAME,
    ENV_DEVICE,
    MedSAM2AutoInferTask,
    MedSAM2InferTask,
    available_devices,
)
from monailabel.interfaces.tasks.infer_v2 import InferType  # noqa: E402

MODEL_DIR = REPO_ROOT / "apps" / "legus" / "model"
CHECKPOINT = MODEL_DIR / DEFAULT_CHECKPOINT_FILENAME
needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason=f"{DEFAULT_CHECKPOINT_FILENAME} not present; run scripts/bootstrap.sh"
)


# --- device selection (no checkpoint required) ------------------------------------------------


def test_cpu_is_always_offered_and_last():
    """CPU is the guaranteed-correct reference path, so it must always be present -- and last,
    since the list is best-first."""
    devices = available_devices()
    assert "cpu" in devices
    assert devices[-1] == "cpu"


def test_accelerator_precedes_cpu_when_present():
    import torch

    devices = available_devices()
    if torch.backends.mps.is_available():
        assert devices.index("mps") < devices.index("cpu")
    if torch.cuda.is_available():
        assert devices.index("cuda:0") < devices.index("cpu")


def test_legus_device_env_pins_the_default(monkeypatch):
    monkeypatch.setenv(ENV_DEVICE, "cpu")
    assert available_devices()[0] == "cpu"


def test_legus_device_env_accepts_unknown_device(monkeypatch):
    """An unrecognised pin must not crash at import/config time -- it is allowed through and
    demotes to cpu at build time instead (that path is exercised below)."""
    monkeypatch.setenv(ENV_DEVICE, "nonexistent-device")
    assert available_devices()[0] == "nonexistent-device"


# --- offline guard (no checkpoint required) ---------------------------------------------------


def test_missing_checkpoint_raises_instead_of_downloading(tmp_path, monkeypatch):
    """design.md Sec 8 Path A is fully offline: a missing checkpoint must fail loudly and never
    trigger a network fetch."""
    monkeypatch.delenv("LEGUS_MEDSAM2_CHECKPOINT", raising=False)
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        MedSAM2InferTask(model_dir=str(tmp_path), type=InferType.DEEPGROW, dimension=2)


# --- inference round trip (needs the checkpoint) ----------------------------------------------


@pytest.fixture(scope="module")
def frame(tmp_path_factory) -> Path:
    """A synthetic singleton-z uint8 volume, matching what scripts/fetch_data.py writes."""
    rng = np.random.default_rng(0)
    arr = rng.integers(40, 90, size=(1, 128, 160), dtype=np.uint8)
    arr[0, 40:90, 55:115] = 200  # a bright blob to prompt on
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((0.308, 0.308, 1.0))
    path = tmp_path_factory.mktemp("frames") / "frame.nii.gz"
    sitk.WriteImage(image, str(path), useCompression=True)
    return path


@pytest.fixture(scope="module")
def task():
    return MedSAM2InferTask(model_dir=str(MODEL_DIR), type=InferType.DEEPGROW, dimension=2)


@needs_checkpoint
@pytest.mark.parametrize("device", available_devices())
def test_box_prompt_returns_nonempty_mask(task, frame, device):
    mask = task.segment(str(frame), {"box": [40, 55, 90, 115], "device": device})
    assert np.asarray(mask).astype(bool).sum() > 0, f"empty mask on {device}"


@needs_checkpoint
def test_devices_agree(task, frame):
    """Accelerator and reference path must not disagree materially -- otherwise the annotator's
    results would depend on which machine served them."""
    devices = available_devices()
    if len(devices) < 2:
        pytest.skip("only one device available")
    box = {"box": [40, 55, 90, 115]}
    a = np.asarray(task.segment(str(frame), {**box, "device": devices[0]})).astype(bool)
    b = np.asarray(task.segment(str(frame), {**box, "device": devices[-1]})).astype(bool)
    iou = (a & b).sum() / max((a | b).sum(), 1)
    assert iou > 0.95, f"{devices[0]} vs {devices[-1]} IoU={iou:.3f}"


def test_normalize_roi_reorders_plugin_box():
    """The 3D Slicer plugin sends a box grouped by axis in IJK order [xmin,xmax,ymin,ymax,zmin,zmax];
    upstream's box = [roi[1],roi[0],roi[3],roi[2]] only forms a valid [x0,y0,x1,y1] box when roi is
    [r0,c0,r1,c1]. The reorder must turn the grouped order into a NON-inverted box."""
    out = MedSAM2InferTask._normalize_roi({"roi": [223, 492, 88, 497, 0, 1]})["roi"]
    box = [out[1], out[0], out[3], out[2]]  # exactly how run2d builds the SAM2 box
    assert box[0] < box[2] and box[1] < box[3], f"reordered box is still inverted: {box}"
    assert box == [223, 88, 492, 497]  # [xmin, ymin, xmax, ymax]


def test_normalize_roi_leaves_four_element_roi_untouched():
    """A 4-element ROI (segment()/tests, already in run2d order) must pass through unchanged."""
    req = {"roi": [40, 55, 90, 115]}
    assert MedSAM2InferTask._normalize_roi(req)["roi"] == [40, 55, 90, 115]


@needs_checkpoint
def test_plugin_grouped_roi_yields_nonempty_mask(task, frame):
    """End-to-end guard for the empty-mask bug: the plugin's grouped-IJK ROI must produce the same
    non-empty mask as the equivalent run2d-order 4-element ROI."""
    # frame blob is rows 40..90 (y), cols 55..115 (x); plugin order = [xmin,xmax,ymin,ymax,z0,z1].
    grouped = np.asarray(task.segment(str(frame), {"box": [55, 115, 40, 90, 0, 1], "slice": 0})).astype(bool)
    direct = np.asarray(task.segment(str(frame), {"box": [40, 55, 90, 115], "slice": 0})).astype(bool)
    assert grouped.sum() > 0, "plugin-grouped ROI produced an empty mask"
    assert np.array_equal(grouped, direct), "grouped ROI should match the equivalent row-major ROI"


@needs_checkpoint
def test_out_of_range_slice_is_clamped(task, frame):
    """Regression: the 3D Slicer plugin sends slice = <slice-view offset> for a 2D model, which on
    our singleton-z (D=1) frames is frequently >= 1, making run2d index a non-existent slice and
    500. The adapter must clamp the slice into range so any offset the plugin sends still works."""
    box = [40, 55, 90, 115]
    ref = np.asarray(task.segment(str(frame), {"box": box, "slice": 0})).astype(bool)
    for bad_slice in (1, 3, 99):
        out = np.asarray(task.segment(str(frame), {"box": box, "slice": bad_slice})).astype(bool)
        assert out.sum() > 0, f"slice={bad_slice} produced an empty mask"
        assert np.array_equal(out, ref), f"slice={bad_slice} should clamp to slice 0 and match it"


@needs_checkpoint
def test_box_prompt_without_slice_hint_does_not_crash(task, frame):
    """Regression: with no z hint, upstream run2d takes a branch that never converts grayscale to
    RGB and dies in torchvision Normalize ("[1,1024,1024] vs [3,1024,1024]"). Reproduced on the
    real CAMUS demo data before the fix."""
    mask = task.segment(str(frame), {"box": [40, 55, 90, 115]})
    assert np.asarray(mask).astype(bool).sum() > 0


@needs_checkpoint
def test_bogus_device_demotes_to_cpu_and_stays_demoted(task, frame):
    """An accelerator that cannot even build must degrade to a working mask, not a 500 at the
    annotator (design.md Sec 1/3, reliability-first) -- and must keep reporting cpu afterwards."""
    assert task._ensure_predictor("not-a-real-device") == "cpu"
    assert task._ensure_predictor("not-a-real-device") == "cpu"
    mask = task.segment(str(frame), {"box": [40, 55, 90, 115], "device": "not-a-real-device"})
    assert np.asarray(mask).astype(bool).sum() > 0


@needs_checkpoint
def test_mps_fallback_env_is_set():
    """design.md Sec 8 Path A calls for PYTORCH_ENABLE_MPS_FALLBACK=1; importing the adapter must
    establish it so unimplemented MPS ops fall back instead of raising."""
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


# --- automatic pre-labelling variant ----------------------------------------------------------


def test_auto_task_is_segmentation_type():
    """The auto variant must register as SEGMENTATION so the client's Auto-Segmentation affordance
    (and auto-run-on-next-sample) targets it -- registering it as DEEPGROW would leave the plugin
    with no valid segmentation model and 404 on next-sample auto-run."""
    task = MedSAM2AutoInferTask(model_dir=str(MODEL_DIR), dimension=2)
    assert task.type == InferType.SEGMENTATION
    assert task.labels, "auto task must advertise labels so segments are created and auto-run resolves to it"


@needs_checkpoint
def test_auto_segmentation_without_any_prompt(task_auto, frame):
    """The whole point of the auto variant: an unprompted request (what onClickSegmentation sends)
    still yields a non-empty, non-degenerate mask via the injected default box."""
    mask = np.asarray(task_auto.segment(str(frame), {})).astype(bool)
    frac = mask.mean()
    assert mask.sum() > 0, "auto segmentation produced an empty mask"
    assert frac < 0.95, "auto segmentation returned essentially the whole frame (degenerate)"


@pytest.fixture(scope="module")
def task_auto():
    return MedSAM2AutoInferTask(model_dir=str(MODEL_DIR), dimension=2)
