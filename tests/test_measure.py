"""M4: prove the measurement export carries real physical units, not pixel counts.

Builds a synthetic image + a square mask of known pixel size with known, deliberately
anisotropic (non-1.0, non-square) spacing, and asserts the reported area equals the
hand-computed value in mm^2. This is the whole point of statistics.py -- see design.md
§7 ("the mask is an intermediate [...] it must carry REAL PHYSICAL UNITS").
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import SimpleITK as sitk

from apps.legus.lib.measure.statistics import (
    SpacingInfo,
    compute_structure_measurements,
    load_calibrated_image,
    main,
)

# Deliberately non-unit, anisotropic spacing (mm): x (columns) != y (rows) != 1.0.
SPACING_XY_MM = (0.2, 0.5)
MASK_ROWS = 20  # y extent
MASK_COLS = 30  # x extent
ROW_START, COL_START = 10, 10


def _make_image_and_mask() -> tuple[sitk.Image, sitk.Image]:
    rng = np.random.default_rng(seed=0)
    image_arr = rng.integers(0, 255, size=(100, 100)).astype(np.float32)
    mask_arr = np.zeros((100, 100), dtype=np.uint8)
    mask_arr[ROW_START : ROW_START + MASK_ROWS, COL_START : COL_START + MASK_COLS] = 1

    image = sitk.GetImageFromArray(image_arr)
    mask = sitk.GetImageFromArray(mask_arr)
    image.SetSpacing(SPACING_XY_MM)
    mask.SetSpacing(SPACING_XY_MM)
    return image, mask


def test_area_mm2_matches_hand_computed_value_for_known_spacing():
    image, mask = _make_image_and_mask()
    spacing = SpacingInfo(values=SPACING_XY_MM, source="image_header", calibrated=True)

    df = compute_structure_measurements(
        image, mask, subject_id="subj-001", spacing=spacing
    )

    assert len(df) == 1
    row = df.iloc[0]

    expected_area_mm2 = (MASK_ROWS * SPACING_XY_MM[1]) * (MASK_COLS * SPACING_XY_MM[0])
    assert math.isclose(row["area_mm2"], expected_area_mm2, rel_tol=1e-6)

    # Sanity: spacing was genuinely non-unit, so a pixel-count-as-mm bug would fail.
    pixel_count = MASK_ROWS * MASK_COLS
    assert not math.isclose(row["area_mm2"], pixel_count)
    assert row["pixel_count"] == pixel_count

    # Row is explicit about being physically calibrated.
    assert bool(row["spacing_calibrated"]) is True
    assert row["spacing_source"] == "image_header"
    assert row["dimensionality"] == "2D"


def test_uncalibrated_spacing_does_not_masquerade_as_millimetres():
    image, mask = _make_image_and_mask()
    # Same array, but this time we claim the spacing is *not* known to be real (e.g.
    # loaded from a PNG with no header calibration).
    spacing = SpacingInfo(values=image.GetSpacing(), source="assumed_unit", calibrated=False)

    df = compute_structure_measurements(
        image, mask, subject_id="subj-002", spacing=spacing
    )
    row = df.iloc[0]

    assert bool(row["spacing_calibrated"]) is False
    assert math.isnan(row["area_mm2"])
    assert row["pixel_count"] == MASK_ROWS * MASK_COLS

    # Every other mm/mm2-unit shape feature must be nulled too -- not just the two
    # renamed headline columns. Confirmed regression: pyradiomics still emits these as
    # real-looking numbers (e.g. shape2D_Perimeter=58.83) even when spacing is fake.
    for mm_col in ("shape2D_Perimeter", "shape2D_MeshSurface", "shape2D_MajorAxisLength"):
        assert mm_col in row.index
        assert math.isnan(row[mm_col]), f"{mm_col} must be NaN when spacing is uncalibrated"

    # Dimensionless shape ratios are spacing-independent and must survive uncorrupted.
    assert not math.isnan(row["shape2D_Sphericity"])


def test_uncalibrated_3d_nulls_all_mm_shape_features():
    rng = np.random.default_rng(seed=2)
    depth, rows, cols = 6, 12, 12
    image_arr = rng.integers(0, 255, size=(depth, rows, cols)).astype(np.float32)
    mask_arr = np.zeros((depth, rows, cols), dtype=np.uint8)
    mask_arr[1:4, 3:9, 3:9] = 1

    image = sitk.GetImageFromArray(image_arr)
    mask = sitk.GetImageFromArray(mask_arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    mask.SetSpacing((1.0, 1.0, 1.0))

    spacing = SpacingInfo(values=(1.0, 1.0, 1.0), source="assumed_unit", calibrated=False)
    df = compute_structure_measurements(image, mask, subject_id="subj-3d-uncal", spacing=spacing)
    row = df.iloc[0]

    assert math.isnan(row["volume_mm3"])
    for mm_col in (
        "shape_MeshVolume",
        "shape_SurfaceArea",
        "shape_Maximum3DDiameter",
        "shape_MajorAxisLength",
        "shape_MinorAxisLength",
    ):
        assert mm_col in row.index
        assert math.isnan(row[mm_col]), f"{mm_col} must be NaN when spacing is uncalibrated"

    assert not math.isnan(row["shape_Sphericity"])
    assert row["voxel_count"] == int(np.count_nonzero(mask_arr == 1))


def test_echo_intensity_and_structure_labelling():
    image, mask = _make_image_and_mask()
    spacing = SpacingInfo(values=SPACING_XY_MM, source="image_header", calibrated=True)

    df = compute_structure_measurements(
        image,
        mask,
        subject_id="subj-003",
        spacing=spacing,
        structure_names={1: "tibialis_anterior"},
    )
    row = df.iloc[0]

    mask_arr = sitk.GetArrayFromImage(mask)
    image_arr = sitk.GetArrayFromImage(image)
    expected_mean = image_arr[mask_arr == 1].mean()

    assert row["structure"] == "tibialis_anterior"
    assert math.isclose(row["echo_intensity_mean"], expected_mean, rel_tol=1e-6)
    # First-order stats beyond the mean are present (design.md §7: "plus first-order
    # intensity stats").
    assert "firstorder_Variance" in df.columns
    # Shape descriptors beyond raw area are present too.
    assert "shape2D_Sphericity" in df.columns


def test_volume_mm3_for_3d_mask():
    rng = np.random.default_rng(seed=1)
    depth, rows, cols = 10, 20, 20
    image_arr = rng.integers(0, 255, size=(depth, rows, cols)).astype(np.float32)
    mask_arr = np.zeros((depth, rows, cols), dtype=np.uint8)
    d0, r0, c0 = 2, 5, 5
    d_extent, r_extent, c_extent = 4, 10, 7
    mask_arr[d0 : d0 + d_extent, r0 : r0 + r_extent, c0 : c0 + c_extent] = 1

    spacing_xyz = (0.3, 0.4, 0.5)  # sitk order: (x, y, z)
    image = sitk.GetImageFromArray(image_arr)
    mask = sitk.GetImageFromArray(mask_arr)
    image.SetSpacing(spacing_xyz)
    mask.SetSpacing(spacing_xyz)

    spacing = SpacingInfo(values=spacing_xyz, source="image_header", calibrated=True)
    df = compute_structure_measurements(image, mask, subject_id="subj-004", spacing=spacing)
    row = df.iloc[0]

    expected_volume_mm3 = (
        d_extent * spacing_xyz[2] * r_extent * spacing_xyz[1] * c_extent * spacing_xyz[0]
    )
    assert math.isclose(row["volume_mm3"], expected_volume_mm3, rel_tol=1e-6)
    assert row["dimensionality"] == "3D"
    assert row["voxel_count"] == d_extent * r_extent * c_extent


def test_full_radiomics_flag_adds_texture_features():
    image, mask = _make_image_and_mask()
    spacing = SpacingInfo(values=SPACING_XY_MM, source="image_header", calibrated=True)

    core_df = compute_structure_measurements(image, mask, subject_id="s", spacing=spacing)
    full_df = compute_structure_measurements(
        image, mask, subject_id="s", spacing=spacing, full_radiomics=True
    )

    assert not any(c.startswith("glcm_") for c in core_df.columns)
    assert any(c.startswith("glcm_") for c in full_df.columns)


def _write_nifti_3d_singleton_z(path, rows=20, cols=30):
    arr = np.zeros((1, rows, cols), dtype=np.float32)  # sitk array order (z, y, x)
    image = sitk.GetImageFromArray(arr)
    sitk.WriteImage(image, str(path))
    return path


def test_2value_spacing_override_on_3d_singleton_z_nifti_does_not_crash(tmp_path):
    # design.md §7 names NIfTI as the primary calibrated format, and --spacing-mm's own
    # help text advertises a 2-value example ("e.g. 0.2 0.2"). A NIfTI written with a
    # singleton z axis (exactly what _squeeze_z exists to handle) must accept a 2-value
    # override instead of dying inside sitk.Image.SetSpacing.
    path = _write_nifti_3d_singleton_z(tmp_path / "img.nii.gz")
    image, spacing = load_calibrated_image(path, spacing_override=(0.2, 0.5))

    assert spacing.calibrated is True
    assert spacing.source == "cli_override"
    assert image.GetSpacing()[:2] == (0.2, 0.5)
    assert image.GetDimension() == 3


def test_spacing_override_wrong_arity_raises_clean_error_not_sitk_traceback(tmp_path):
    # A genuinely 3D (multi-slice) volume with a 2-value override is a real user
    # mistake and must fail with a readable message, not a raw SimpleITK exception.
    arr = np.zeros((5, 20, 30), dtype=np.float32)
    image = sitk.GetImageFromArray(arr)
    path = tmp_path / "vol.nii.gz"
    sitk.WriteImage(image, str(path))

    with pytest.raises(ValueError, match="expected 3 values"):
        load_calibrated_image(path, spacing_override=(0.2, 0.5))


def test_cli_reports_clean_error_for_bad_spacing_arity(tmp_path, capsys):
    image_path = tmp_path / "vol.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    image_arr = np.zeros((5, 20, 30), dtype=np.float32)
    mask_arr = np.zeros((5, 20, 30), dtype=np.uint8)
    mask_arr[1:3, 5:10, 5:10] = 1
    sitk.WriteImage(sitk.GetImageFromArray(image_arr), str(image_path))
    sitk.WriteImage(sitk.GetImageFromArray(mask_arr), str(mask_path))

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--image",
                str(image_path),
                "--mask",
                str(mask_path),
                "--subject-id",
                "subj-cli",
                "--out",
                str(tmp_path / "out.csv"),
                "--spacing-mm",
                "0.2",
                "0.5",
            ]
        )
    assert exc_info.value.code == 2  # argparse.error() exit code, not an uncaught traceback
    assert "expected 3 values" in capsys.readouterr().err


def test_exact_unit_header_spacing_is_treated_as_uncalibrated(tmp_path):
    # sitk.ReadImage default-fills spacing=(1,1,1) when a header carries no calibration
    # tag at all -- e.g. a DICOM whose real scale lives in SequenceOfUltrasoundRegions,
    # not PixelSpacing (design.md §11). That must not be trusted as real calibration
    # just because the suffix is a known calibrated-format suffix.
    arr = np.zeros((20, 30), dtype=np.float32)
    image = sitk.GetImageFromArray(arr)  # spacing defaults to (1.0, 1.0)
    path = tmp_path / "img.nii.gz"
    sitk.WriteImage(image, str(path))

    _, spacing = load_calibrated_image(path)

    assert spacing.calibrated is False
    assert spacing.source == "image_header_unit_default"


def test_genuinely_calibrated_non_unit_header_spacing_is_trusted(tmp_path):
    arr = np.zeros((20, 30), dtype=np.float32)
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((0.2, 0.5))
    path = tmp_path / "img.nii.gz"
    sitk.WriteImage(image, str(path))

    _, spacing = load_calibrated_image(path)

    assert spacing.calibrated is True
    assert spacing.source == "image_header"


def test_unrecognised_suffix_is_uncalibrated_not_silently_trusted(tmp_path):
    # A DICOM series directory (suffix "") or any other unfamiliar extension must not
    # fall through to the trusted "image_header" path -- calibration is an allowlist.
    arr = np.zeros((20, 30), dtype=np.float32)
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((0.2, 0.5))
    # .mnc (MINC) is a format sitk can read/write but is deliberately absent from both
    # _CALIBRATED_SUFFIXES and _UNCALIBRATED_SUFFIXES -- exercises the "unrecognised
    # suffix" branch on real, readable image bytes with real non-unit spacing.
    path = tmp_path / "img.mnc"
    sitk.WriteImage(image, str(path))

    _, spacing = load_calibrated_image(path)

    assert spacing.calibrated is False
    assert spacing.source == "unknown_format"


def test_explicit_unit_spacing_override_is_still_trusted(tmp_path):
    # The exact-unit-spacing distrust heuristic must not punish a user who deliberately
    # asserts 1.0mm spacing via --spacing-mm -- that's their explicit claim, not an
    # inferred/defaulted header value.
    arr = np.zeros((20, 30), dtype=np.float32)
    image = sitk.GetImageFromArray(arr)
    path = tmp_path / "img.nii.gz"
    sitk.WriteImage(image, str(path))

    _, spacing = load_calibrated_image(path, spacing_override=(1.0, 1.0))

    assert spacing.calibrated is True
    assert spacing.source == "cli_override"
