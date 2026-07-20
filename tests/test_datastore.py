# Copyright (c) 2026 MedSAM annotation tool contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""M3: prove pixel spacing, dimensionality, and display range survive the write -> datastore ->
read round trip.

design.md Sec 7: "Raw PNG loses physical scale -> measurements become non-comparable.
Ultrasound depth/zoom varies per image, so calibration must be captured per image." This test
builds a synthetic volume with deliberately non-unit, anisotropic spacing, runs it through the
exact same `write_datastore_pair` used by `scripts/fetch_data.py`, and reads it back with a
fresh SimpleITK image read -- proving the NIfTI affine (not just an in-memory object) actually
carries the calibration. It also exercises the "record unknown, don't fake it" path for a
source with no coordinate grid at all.

M3 review findings (fixed here): the written image must (1) already be display-range uint8 --
not raw float source intensities that crash or blank out inside upstream `Sam2InferTask.run2d`
-- and (2) be a genuine 3-D volume with a singleton z-axis, not a 2-D NIfTI, because `run2d`
indexes `image_tensor[:, :, slice_idx]`. `test_written_image_is_3d_uint8_with_singleton_z` and
`test_to_display_uint8_windows_into_0_255_range` assert both directly, per the review's ask for
a test on "dimensionality and dtype/range, not just spacing".

No network access: this never calls `_download` / `snapshot_download`. Only synthetic arrays.
"""

from __future__ import annotations

import json
import math

import numpy as np
import SimpleITK as sitk
from monailabel.interfaces.datastore import DefaultLabelTag

from scripts.fetch_data import (
    SpacingResult,
    _spacing_mm_from_coordinates,
    to_display_uint8,
    write_datastore_pair,
)

# Deliberately non-unit, anisotropic, non-square spacing (mm) -- a 1.0mm/pixel bug would fail.
SPACING_XY_MM = (0.308, 0.417)
H, W = 40, 50


def _synthetic_frame():
    rng = np.random.default_rng(seed=0)
    # Deliberately dB-like: negative, well outside [0, 255] -- mirrors raw CAMUS intensities and
    # would crash/blank out downstream if written un-windowed (M3 review finding 1).
    image_2d = rng.uniform(-60.0, 0.0, size=(H, W)).astype(np.float32)
    label_2d = np.zeros((H, W), dtype=np.uint8)
    label_2d[10:25, 15:35] = 1
    return image_2d, label_2d


def test_known_spacing_survives_write_and_read(tmp_path):
    image_2d, label_2d = _synthetic_frame()
    display_image_2d, _window = to_display_uint8(image_2d)
    spacing = SpacingResult(values_mm=SPACING_XY_MM, known=True, source="unit test: synthetic known spacing")

    image_path, label_path, sidecar_path = write_datastore_pair(
        datastore_dir=tmp_path,
        image_id="synthetic_0001",
        image_2d=display_image_2d,
        label_2d=label_2d,
        spacing=spacing,
        sidecar={"image_id": "synthetic_0001", "spacing_known": True},
    )

    # Confirm the MONAI Label LocalDatastore layout (verified against
    # external/MONAILabel/monailabel/datastore/local.py): images at the datastore root, and the
    # (pre-existing, not annotator-produced) label under labels/original/ -- DefaultLabelTag.FINAL
    # is reserved for the annotator's own corrections, see write_datastore_pair's docstring.
    assert image_path == tmp_path / "synthetic_0001.nii.gz"
    assert label_path == tmp_path / "labels" / DefaultLabelTag.ORIGINAL.value / "synthetic_0001.nii.gz"
    assert sidecar_path == tmp_path / "synthetic_0001.json"

    # Read back with a *fresh* SimpleITK read -- proves spacing is in the file's own header/
    # affine, not merely preserved on the in-memory sitk.Image object.
    reread_image = sitk.ReadImage(str(image_path))
    reread_label = sitk.ReadImage(str(label_path))

    # NIfTI stores spacing (pixdim) as float32, so compare with float32-appropriate tolerance
    # rather than exact float64 equality. Spacing is 3-element (z=1.0) -- see next test for why.
    for got, want in zip(reread_image.GetSpacing(), (*SPACING_XY_MM, 1.0), strict=True):
        assert math.isclose(got, want, rel_tol=1e-6), (reread_image.GetSpacing(), SPACING_XY_MM)
    for got, want in zip(reread_label.GetSpacing(), (*SPACING_XY_MM, 1.0), strict=True):
        assert math.isclose(got, want, rel_tol=1e-6), (reread_label.GetSpacing(), SPACING_XY_MM)
    assert reread_image.GetSize() == (W, H, 1)

    # Pixel data round-trips too, not just the affine.
    np.testing.assert_array_equal(sitk.GetArrayFromImage(reread_label)[0], label_2d)

    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["spacing_known"] is True


def test_written_image_is_3d_uint8_with_singleton_z(tmp_path):
    """M3 review finding 2: a genuinely 2-D NIfTI breaks upstream `Sam2InferTask.run2d`, which
    indexes `image_tensor[:, :, slice_idx]`. The written image must be a real 3-D volume with a
    singleton z-axis, and already uint8 in [0, 255] (finding 1) -- not a raw float array."""
    image_2d, label_2d = _synthetic_frame()
    display_image_2d, _window = to_display_uint8(image_2d)
    spacing = SpacingResult(values_mm=SPACING_XY_MM, known=True, source="unit test")

    image_path, label_path, _sidecar_path = write_datastore_pair(
        datastore_dir=tmp_path,
        image_id="synthetic_3d",
        image_2d=display_image_2d,
        label_2d=label_2d,
        spacing=spacing,
        sidecar={"image_id": "synthetic_3d"},
    )

    reread_image = sitk.ReadImage(str(image_path))
    reread_label = sitk.ReadImage(str(label_path))

    assert reread_image.GetDimension() == 3
    assert reread_image.GetSize() == (W, H, 1)
    assert reread_image.GetSpacing()[2] == 1.0
    assert reread_image.GetPixelID() in (sitk.sitkUInt8, sitk.sitkUInt8)
    image_array = sitk.GetArrayFromImage(reread_image)
    assert image_array.dtype == np.uint8
    assert image_array.shape == (1, H, W)
    assert image_array.min() >= 0 and image_array.max() <= 255

    assert reread_label.GetDimension() == 3
    assert reread_label.GetSize() == (W, H, 1)


def test_to_display_uint8_windows_into_0_255_range():
    """M3 review finding 1: raw dB-range (e.g. [-60.0, 0.0]) intensities must be windowed into
    displayable uint8 before they ever reach `write_datastore_pair`, and the window recorded."""
    rng = np.random.default_rng(seed=1)
    db_image = rng.uniform(-60.0, 0.0, size=(H, W)).astype(np.float32)

    display, window = to_display_uint8(db_image)

    assert display.dtype == np.uint8
    assert display.shape == (H, W)
    assert display.min() >= 0 and display.max() <= 255
    # The synthetic frame spans nearly the full range, so windowing shouldn't collapse it.
    assert display.max() - display.min() > 200
    assert window["source_min"] < window["source_max"]
    assert window["mapped_range"] == [0, 255]


def test_to_display_uint8_degenerate_frame_does_not_divide_by_zero():
    """A uniform (constant-value) frame must not raise or NaN out -- map to flat mid-gray."""
    flat = np.full((H, W), -30.0, dtype=np.float32)
    display, window = to_display_uint8(flat)
    assert display.dtype == np.uint8
    assert np.all(display == 128)
    assert window["source_min"] == window["source_max"]


def test_unknown_spacing_is_recorded_not_faked(tmp_path):
    """A source with no calibration must record spacing_known=False, never a silent 1.0mm."""
    image_2d, label_2d = _synthetic_frame()
    spacing = _spacing_mm_from_coordinates(None)  # no coordinate grid available

    assert spacing.known is False
    assert spacing.values_mm == (1.0, 1.0)
    assert "assumed" in spacing.source

    _, _, sidecar_path = write_datastore_pair(
        datastore_dir=tmp_path,
        image_id="synthetic_unknown",
        image_2d=image_2d,
        label_2d=label_2d,
        spacing=spacing,
        sidecar={
            "image_id": "synthetic_unknown",
            "spacing_mm_xy": list(spacing.values_mm),
            "spacing_known": spacing.known,
            "spacing_source": spacing.source,
        },
    )

    sidecar = json.loads(sidecar_path.read_text())
    # The whole point: an assumed 1.0mm spacing must be distinguishable from a real one, because
    # a fake calibration that *looks* real is worse than a recorded unknown (design.md Sec 7).
    assert sidecar["spacing_known"] is False
    assert sidecar["spacing_mm_xy"] == [1.0, 1.0]


def test_spacing_from_coordinate_grid_matches_hand_computed_value():
    """Exercise the real conversion path `convert_camus` uses, with a synthetic coordinate grid."""
    h, w = 12, 16
    row_spacing_m = 0.000308  # 0.308 mm
    col_spacing_m = 0.000417  # 0.417 mm

    coords = np.zeros((h, w, 3), dtype=np.float32)
    row_idx, col_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords[..., 2] = row_idx * row_spacing_m  # depth varies with row (matches CAMUS convention)
    coords[..., 0] = col_idx * col_spacing_m  # lateral position varies with column

    spacing = _spacing_mm_from_coordinates(coords)

    assert spacing.known is True
    assert np.isclose(spacing.values_mm[0], col_spacing_m * 1000.0, rtol=1e-5)
    assert np.isclose(spacing.values_mm[1], row_spacing_m * 1000.0, rtol=1e-5)
    assert "coordinates" in spacing.source


def test_degenerate_coordinate_grid_falls_back_to_unknown():
    """A zero-step or too-small coordinate grid must not masquerade as a real calibration."""
    coords = np.zeros((1, 1, 3), dtype=np.float32)  # too small to difference
    spacing = _spacing_mm_from_coordinates(coords)
    assert spacing.known is False
    assert spacing.values_mm == (1.0, 1.0)
