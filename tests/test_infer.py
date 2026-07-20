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
