# Copyright (c) 2026 MedSAM annotation tool contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""M5: the fine-tune loop (design.md Sec 6 step 3).

Cheap tests (no checkpoint, no torch model build) cover the parts that can go quietly wrong
without ever running a training step: warm-start resolution, the train/val split being
genuinely disjoint, the trainer reusing the infer adapter's device policy rather than growing a
second one, and -- the one design.md Sec 10 item 2 is emphatic about -- that training only ever
sees the annotator's corrected ('final') labels, never the pre-existing ('original') ones.

One real, `@pytest.mark.slow` end-to-end test proves the loop actually *closes*: it runs a real
1-epoch fine-tune over 3 tiny synthetic images against the real checkpoint and asserts a real
Dice number comes back and a checkpoint file lands on disk. It skips cleanly if no checkpoint is
present, same convention as tests/test_infer.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "apps" / "legus"))

from lib.infers.medsam2 import DEFAULT_CHECKPOINT_FILENAME  # noqa: E402
from lib.infers.medsam2 import available_devices as infer_available_devices  # noqa: E402
from lib.trainers.medsam2 import (  # noqa: E402
    DEFAULT_US_CHECKPOINT_FILENAME,
    ENV_TRAIN_CHECKPOINT,
    MedSAM2TrainTask,
    _bbox_from_mask,
    _dice_score,
    _resolve_warm_start,
    _train_val_split,
)
from lib.trainers.medsam2 import available_devices as trainer_available_devices  # noqa: E402
from monailabel.datastore.local import LocalDatastore  # noqa: E402
from monailabel.interfaces.datastore import DefaultLabelTag  # noqa: E402

MODEL_DIR = REPO_ROOT / "apps" / "legus" / "model"
US_CHECKPOINT = MODEL_DIR / DEFAULT_US_CHECKPOINT_FILENAME
LATEST_CHECKPOINT = MODEL_DIR / DEFAULT_CHECKPOINT_FILENAME
needs_checkpoint = pytest.mark.skipif(
    not (US_CHECKPOINT.is_file() or LATEST_CHECKPOINT.is_file()),
    reason="no MedSAM2 checkpoint present; run scripts/bootstrap.sh",
)


# --- warm-start resolution (no checkpoint required) --------------------------------------------


def test_env_override_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_TRAIN_CHECKPOINT, str(tmp_path / "nope.pt"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_warm_start(str(tmp_path))


def test_env_override_present_wins_over_everything(tmp_path, monkeypatch):
    override = tmp_path / "custom.pt"
    override.write_bytes(b"not a real checkpoint, just needs to exist")
    # Even with a real-looking US checkpoint sitting right there, the override must win.
    (tmp_path / DEFAULT_US_CHECKPOINT_FILENAME).write_bytes(b"decoy")
    monkeypatch.setenv(ENV_TRAIN_CHECKPOINT, str(override))

    path, source = _resolve_warm_start(str(tmp_path))
    assert path == str(override)
    assert "override" in source


def test_us_checkpoint_preferred_over_latest(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_TRAIN_CHECKPOINT, raising=False)
    (tmp_path / DEFAULT_US_CHECKPOINT_FILENAME).write_bytes(b"decoy us checkpoint")
    (tmp_path / DEFAULT_CHECKPOINT_FILENAME).write_bytes(b"decoy latest checkpoint")

    path, source = _resolve_warm_start(str(tmp_path))
    assert path == str(tmp_path / DEFAULT_US_CHECKPOINT_FILENAME)
    assert "ultrasound" in source


def test_falls_back_to_latest_when_us_checkpoint_absent(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_TRAIN_CHECKPOINT, raising=False)
    (tmp_path / DEFAULT_CHECKPOINT_FILENAME).write_bytes(b"decoy latest checkpoint")

    path, source = _resolve_warm_start(str(tmp_path))
    assert path == str(tmp_path / DEFAULT_CHECKPOINT_FILENAME)
    assert "fallback" in source


def test_missing_both_checkpoints_raises_instead_of_downloading(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_TRAIN_CHECKPOINT, raising=False)
    with pytest.raises(FileNotFoundError, match="No fine-tune warm-start checkpoint"):
        _resolve_warm_start(str(tmp_path))


# --- train/val split (design.md Sec 6 step 5: a holdout that is genuinely never trained on) ----


def test_split_is_disjoint_and_covers_everything():
    datalist = [{"image": f"img{i}.nii.gz", "label": f"lbl{i}.nii.gz"} for i in range(10)]
    train, val = _train_val_split(datalist, val_fraction=0.2, min_val=1, seed=0)

    train_images = {d["image"] for d in train}
    val_images = {d["image"] for d in val}
    assert train_images.isdisjoint(val_images)
    assert train_images | val_images == {d["image"] for d in datalist}
    assert len(val) == 2  # 20% of 10


def test_split_is_deterministic_given_a_seed():
    datalist = [{"image": f"img{i}.nii.gz", "label": f"lbl{i}.nii.gz"} for i in range(12)]
    a = _train_val_split(datalist, seed=42)
    b = _train_val_split(datalist, seed=42)
    assert a == b


def test_split_always_holds_out_at_least_min_val():
    datalist = [{"image": f"img{i}.nii.gz", "label": f"lbl{i}.nii.gz"} for i in range(3)]
    train, val = _train_val_split(datalist, val_fraction=0.01, min_val=1, seed=0)
    assert len(val) >= 1
    assert len(train) >= 1  # never hold out everything


def test_split_with_fewer_than_two_items_cannot_hold_out_anything():
    datalist = [{"image": "only_one.nii.gz", "label": "lbl.nii.gz"}]
    train, val = _train_val_split(datalist, seed=0)
    assert train == datalist
    assert val == []


# --- device policy: reused, not reinvented -----------------------------------------------------


def test_trainer_reuses_the_infer_adapters_device_policy():
    """design.md brief: 'reuse available_devices() from lib.infers.medsam2 rather than writing a
    second device policy.' Checked by identity, not just equal output, so a future edit that
    forks the function would fail this test even if both still happened to agree today."""
    assert trainer_available_devices is infer_available_devices


def test_cpu_is_always_available_for_training():
    assert "cpu" in trainer_available_devices()


# --- FINAL-vs-ORIGINAL label selection (design.md Sec 6 / Sec 10 item 2) -----------------------


def _write_frame(path: Path, arr: np.ndarray) -> None:
    image = sitk.GetImageFromArray(arr[np.newaxis, :, :])
    image.SetSpacing((0.308, 0.308, 1.0))
    sitk.WriteImage(image, str(path), useCompression=True)


def _build_datastore(tmp_path: Path) -> LocalDatastore:
    # auto_reload=False: this is a one-shot test datastore, no need for a filesystem watcher.
    return LocalDatastore(str(tmp_path), auto_reload=False)


def test_datalist_returns_only_final_labels_never_original(tmp_path):
    """The exact property MedSAM2TrainTask.__call__ relies on: `datastore.datalist()` -- called
    with no tag argument, no globbing -- must hand back only the annotator's corrected labels.
    Training on `labels/original/` (pre-existing/public ground truth) would be self-deception
    (design.md Sec 6, Sec 10 item 2)."""
    datastore = _build_datastore(tmp_path)
    rng = np.random.default_rng(0)

    src_dir = tmp_path / "src"
    src_dir.mkdir()

    image_original = src_dir / "original_only.nii.gz"
    image_final = src_dir / "final_labeled.nii.gz"
    label_original = src_dir / "label_original.nii.gz"
    label_final = src_dir / "label_final.nii.gz"
    _write_frame(image_original, rng.integers(0, 255, size=(32, 32), dtype=np.uint8))
    _write_frame(image_final, rng.integers(0, 255, size=(32, 32), dtype=np.uint8))
    _write_frame(label_original, np.ones((32, 32), dtype=np.uint8))
    _write_frame(label_final, np.ones((32, 32), dtype=np.uint8))

    datastore.add_image("original_only", str(image_original), {})
    datastore.add_image("final_labeled", str(image_final), {})
    # A pre-existing/public ground-truth label -- must NOT be trained on.
    datastore.save_label("original_only", str(label_original), DefaultLabelTag.ORIGINAL, {})
    # The annotator's own correction -- this is what training must see.
    datastore.save_label("final_labeled", str(label_final), DefaultLabelTag.FINAL, {})

    datalist = datastore.datalist()
    ids = {Path(d["image"]).stem.replace(".nii", "") for d in datalist}
    assert ids == {"final_labeled"}
    assert "original_only" not in ids


# --- mask -> box-prompt + Dice helpers (pure functions, no model needed) -----------------------


def test_bbox_from_mask_is_none_for_empty_mask():
    assert _bbox_from_mask(np.zeros((10, 10), dtype=bool)) is None


def test_bbox_from_mask_matches_hand_computed_box():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:9, 10:16] = True  # rows [5,9), cols [10,16)
    assert _bbox_from_mask(mask) == [5, 10, 9, 16]


def test_dice_score_perfect_overlap_is_one():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    assert _dice_score(mask, mask) == pytest.approx(1.0)


def test_dice_score_both_empty_is_one_not_undefined():
    empty = np.zeros((10, 10), dtype=bool)
    assert _dice_score(empty, empty) == 1.0


def test_dice_score_disjoint_masks_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b[7:10, 7:10] = True
    assert _dice_score(a, b) == 0.0


# --- the real thing: proves the loop closes (needs a real checkpoint) --------------------------


def _synthetic_pair(rng, size=(96, 96)):
    """A tiny image with a bright rectangular blob + the matching binary mask."""
    h, w = size
    image = rng.integers(30, 80, size=(h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    r0, c0, r1, c1 = h // 4, w // 4, 3 * h // 4, 3 * w // 4
    image[r0:r1, c0:c1] = 200
    mask[r0:r1, c0:c1] = 1
    return image, mask


@needs_checkpoint
@pytest.mark.slow
def test_tiny_finetune_round_closes_the_loop(tmp_path):
    """design.md Sec 6 step 3/5: run one real, tiny fine-tune round and prove it (a) trains only
    on FINAL labels, (b) reports a per-round Dice against a held-out split, and (c) writes a
    checkpoint back out -- the loop actually closes end to end, not just on paper."""
    datastore = _build_datastore(tmp_path / "datastore")
    rng = np.random.default_rng(0)

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    for i in range(3):  # >= 2 required to hold out a validation split
        image, mask = _synthetic_pair(rng)
        image_path = src_dir / f"case_{i}_image.nii.gz"
        label_path = src_dir / f"case_{i}_label.nii.gz"
        _write_frame(image_path, image)
        _write_frame(label_path, mask)
        datastore.add_image(f"case_{i}", str(image_path), {})
        datastore.save_label(f"case_{i}", str(label_path), DefaultLabelTag.FINAL, {})

    # output_checkpoint pins the write into tmp_path -- MODEL_DIR only needs to supply the
    # warm-start checkpoint here, not receive test-run artifacts.
    output_checkpoint = tmp_path / "finetuned.pt"
    task = MedSAM2TrainTask(
        model_dir=str(MODEL_DIR),
        config={"max_epochs": 1, "min_val": 1, "output_checkpoint": str(output_checkpoint)},
    )
    stats = task({"device": "cpu"}, datastore)

    assert stats["train_count"] >= 1
    assert stats["val_count"] >= 1
    assert stats["train_count"] + stats["val_count"] == 3
    assert len(stats["history"]) == 1
    assert 0.0 <= stats["val_dice"] <= 1.0
    assert Path(stats["checkpoint"]).is_file()

    # stats() must reflect the run that just happened (surfaced over GET /train/).
    assert task.stats()["checkpoint"] == stats["checkpoint"]
